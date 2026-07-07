from __future__ import annotations
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import random
import torch
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from app.model import MNISTCNN, load_model
from app.dynamic_engine import DynamicState, predict, apply_decay
from app.schemas import (
    DynamicParams, SessionCreateRequest, ParamsUpdateRequest,
    HumanTrialResponse,
)
from app.data import get_image_tensor, tensor_to_base64, get_test_dataset
from app.sampler import sample_image
from app.db import (
    init_db, create_session, save_playback_event, save_human_trial,
    get_human_trials, save_calibration_result, get_best_calibration,
)
from app.human_training import HumanSession, get_base_prediction, compute_human_summary
from app.calibration import run_calibration

MODEL_PATH = "/app/models/mnist_cnn_base.pt"
AMBIGUITY_DB = "/app/data/ambiguity_bank.sqlite"

_base_model: MNISTCNN | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_sessions: dict[str, DynamicState] = {}
_human_sessions: dict[str, HumanSession] = {}
_playback_tasks: dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _base_model
    await init_db()
    _base_model = load_model(MODEL_PATH, _device)
    get_test_dataset()
    yield


app = FastAPI(title="Lingua Mentis MNIST", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="/app/app/static"), name="static")


@app.get("/")
async def index():
    return FileResponse("/app/app/static/index.html")


@app.get("/api/status")
async def status():
    return {
        "model_loaded": _base_model is not None,
        "active_sessions": len(_sessions),
        "device": str(_device),
    }


# ---- Playback sessions ----

@app.post("/api/session")
async def create_playback_session(req: SessionCreateRequest):
    sid = await create_session("playback", req.notes)
    state = DynamicState.create(sid, _base_model, req.params)
    _sessions[sid] = state
    return {"session_id": sid}


@app.post("/api/session/{session_id}/reset")
async def reset_session(session_id: str):
    state = _get_state(session_id)
    state.reset_delta()
    return {"ok": True}


@app.post("/api/session/{session_id}/params")
async def update_params(session_id: str, req: ParamsUpdateRequest):
    state = _get_state(session_id)
    state.params = req.params
    return {"ok": True}


@app.post("/api/session/{session_id}/step")
async def step(session_id: str):
    state = _get_state(session_id)
    p = state.params
    image_id = sample_image(p.sequence_mode, p.class_a, p.class_b, AMBIGUITY_DB)
    tensor, true_label = get_image_tensor(image_id)
    tensor = tensor.to(_device)
    if p.noise_std > 0:
        tensor = torch.clamp(tensor + torch.randn_like(tensor) * p.noise_std, -3.0, 3.0)
    result = predict(state, tensor, image_id, true_label)
    noise_seed = random.randint(0, 999999) if p.noise_std > 0 else None
    result.image_base64 = tensor_to_base64(image_id, {"noise_std": p.noise_std, "seed": noise_seed} if p.noise_std > 0 else None)
    await save_playback_event(
        session_id, 0, image_id, true_label,
        result.predicted_label, result.probabilities,
        result.confidence, result.margin, result.delta_norm,
        state.params.model_dump_json(),
    )
    return result


# ---- Images ----

@app.get("/api/images/{image_id}")
async def get_image(image_id: int):
    img_bytes = _image_id_to_png(image_id)
    return Response(content=img_bytes, media_type="image/png")


def _image_id_to_png(image_id: int) -> bytes:
    import io
    from PIL import Image
    ds = get_test_dataset()
    raw = ds.data[image_id].numpy()
    pil = Image.fromarray(raw, mode="L").resize((140, 140), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ---- Ambiguity bank ----

@app.get("/api/ambiguity-bank")
async def ambiguity_bank_info():
    import sqlite3
    try:
        conn = sqlite3.connect(AMBIGUITY_DB)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM ambiguity")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM ambiguity WHERE confidence < 0.75")
        low_conf = cur.fetchone()[0]
        conn.close()
        return {"total": total, "low_confidence": low_conf}
    except Exception as e:
        return {"total": 0, "error": str(e)}


# ---- Human training ----

@app.post("/api/human/session")
async def create_human_session(body: dict):
    mode = body.get("mode", "3_vs_8")
    class_a = body.get("class_a")
    class_b = body.get("class_b")
    degrade = body.get("degrade_params")
    sid = await create_session("human_training")
    hs = HumanSession.create(sid, mode, class_a, class_b, degrade)
    _human_sessions[sid] = hs
    return {"session_id": sid, "total_trials": len(hs.sequence)}


@app.post("/api/human/{session_id}/next")
async def human_next(session_id: str):
    hs = _get_human_session(session_id)
    result = hs.advance()
    if result is None:
        raise HTTPException(404, "Session complete")
    image_id, true_label = result
    base_pred, base_probs = get_base_prediction(_base_model, image_id)
    img_b64 = tensor_to_base64(image_id, hs.degrade_params)
    return {
        "trial_index": hs.trial_index - 1,
        "image_id": image_id,
        "true_label": true_label,
        "image_base64": img_b64,
        "base_pred": base_pred,
        "base_probs": base_probs,
        "total": len(hs.sequence),
    }


@app.post("/api/human/{session_id}/response")
async def human_response(session_id: str, body: dict):
    hs = _get_human_session(session_id)
    user_label = body.get("user_label")
    rt = body.get("response_time_ms")
    image_id = hs.current_image_id
    true_label = hs.current_true_label
    base_pred, base_probs = get_base_prediction(_base_model, image_id)
    await save_human_trial(
        session_id, hs.trial_index - 1, image_id, true_label,
        user_label, rt,
        json.dumps(hs.degrade_params) if hs.degrade_params else None,
        base_pred, json.dumps(base_probs),
    )
    return {"ok": True}


@app.get("/api/human/{session_id}/summary")
async def human_summary(session_id: str):
    trials = await get_human_trials(session_id)
    return compute_human_summary(trials)


# ---- Calibration ----

@app.post("/api/calibrate/{human_session_id}")
async def calibrate(human_session_id: str):
    trials = await get_human_trials(human_session_id)
    if not trials:
        raise HTTPException(400, "No trials found")
    score, params = run_calibration(trials, _base_model)
    await save_calibration_result(human_session_id, score, params.model_dump_json())
    return {"score": score, "params": params.model_dump()}


@app.get("/api/calibrate/{human_session_id}/best")
async def best_calibration(human_session_id: str):
    row = await get_best_calibration(human_session_id)
    if not row:
        raise HTTPException(404, "No calibration found")
    return {"score": row["score"], "params": json.loads(row["params_json"])}


# ---- WebSocket playback ----

@app.websocket("/ws/play/{session_id}")
async def ws_play(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in _sessions:
        await websocket.close(code=4004)
        return

    state = _sessions[session_id]
    playing = False
    step_index = 0

    async def do_step():
        nonlocal step_index
        p = state.params
        image_id = sample_image(p.sequence_mode, p.class_a, p.class_b, AMBIGUITY_DB)
        tensor, true_label = get_image_tensor(image_id)
        tensor = tensor.to(_device)
        if p.noise_std > 0:
            tensor = torch.clamp(tensor + torch.randn_like(tensor) * p.noise_std, -3.0, 3.0)
        result = predict(state, tensor, image_id, true_label)
        noise_seed = random.randint(0, 999999) if p.noise_std > 0 else None
        result.image_base64 = tensor_to_base64(image_id, {"noise_std": p.noise_std, "seed": noise_seed} if p.noise_std > 0 else None)
        await save_playback_event(
            session_id, step_index, image_id, true_label,
            result.predicted_label, result.probabilities,
            result.confidence, result.margin, result.delta_norm,
            state.params.model_dump_json(),
        )
        step_index += 1
        msg = {
            "type": "prediction",
            "step": step_index,
            "image_id": result.image_id,
            "image_base64": result.image_base64,
            "true_label": result.true_label,
            "predicted_label": result.predicted_label,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "vanilla_probabilities": result.vanilla_probabilities,
            "margin": result.margin,
            "delta_norm": result.delta_norm,
            "last_update_norm": result.last_update_norm,
        }
        await websocket.send_json(msg)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.05)
                t = msg.get("type")
                if t == "play":
                    playing = True
                elif t == "pause":
                    playing = False
                elif t == "step":
                    await do_step()
                elif t == "reset":
                    state.reset_delta()
                    step_index = 0
                    await websocket.send_json({"type": "reset_ack", "delta_norm": 0.0})
                elif t == "update_params":
                    state.params = DynamicParams(**msg.get("params", {}))
            except asyncio.TimeoutError:
                pass

            if playing:
                await do_step()
                await asyncio.sleep(state.params.interval_ms / 1000.0)

    except WebSocketDisconnect:
        pass


# ---- Helpers ----

def _get_state(session_id: str) -> DynamicState:
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    return _sessions[session_id]


def _get_human_session(session_id: str) -> HumanSession:
    if session_id not in _human_sessions:
        raise HTTPException(404, "Human session not found")
    return _human_sessions[session_id]
