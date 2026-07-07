from __future__ import annotations
import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any

import random
import torch
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

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

# Password auth — reads from env, disabled when APP_PASSWORD is unset
_APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
_COOKIE_NAME = "lm_auth"
_PUBLIC_PATHS = {"/login", "/login/check"}

def _make_token(password: str) -> str:
    return hmac.new(password.encode(), b"lingua-mentis-session", hashlib.sha256).hexdigest()

def _auth_ok(request: Request) -> bool:
    if not _APP_PASSWORD:
        return True
    token = request.cookies.get(_COOKIE_NAME, "")
    return hmac.compare_digest(token, _make_token(_APP_PASSWORD))

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lingua Mentis · Login</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🧠</text></svg>">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d0f;color:#e8e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  min-height:100vh;display:flex;align-items:center;justify-content:center}
.card{background:#141418;border:1px solid #2a2a36;border-radius:12px;padding:40px 36px;
  width:100%;max-width:340px;display:flex;flex-direction:column;gap:20px}
h1{font-size:18px;font-weight:600;display:flex;align-items:center;gap:10px}
h1 span{font-size:28px}
p{font-size:12px;color:#5a5a70}
input{background:#1c1c22;border:1px solid #2a2a36;color:#e8e8f0;border-radius:6px;
  padding:10px 12px;width:100%;font-size:14px;outline:none}
input:focus{border-color:#7c6af7}
button{background:#7c6af7;border:none;color:#fff;font-weight:600;border-radius:6px;
  padding:10px;width:100%;font-size:14px;cursor:pointer}
button:hover{background:#a78bfa}
.err{color:#f87171;font-size:12px;display:none}
.err.show{display:block}
</style>
</head>
<body>
<div class="card">
  <h1><span>🧠</span> Lingua Mentis</h1>
  <p>Dynamic Belief · MNIST Experiment</p>
  <form method="post" action="/login/check" id="f">
    <div style="display:flex;flex-direction:column;gap:10px">
      <input type="password" name="password" placeholder="Access password" autofocus autocomplete="current-password">
      <div class="err {err_class}" id="err">Incorrect password</div>
      <button type="submit">Enter</button>
    </div>
  </form>
</div>
</body>
</html>"""

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _APP_PASSWORD:
            return await call_next(request)
        path = request.url.path
        if path in _PUBLIC_PATHS or path.startswith("/static/"):
            return await call_next(request)
        if not _auth_ok(request):
            if path.startswith("/api/") or path.startswith("/ws/"):
                return Response(status_code=401, content="Unauthorized")
            return RedirectResponse("/login", status_code=302)
        return await call_next(request)

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
app.add_middleware(AuthMiddleware)
app.mount("/static", StaticFiles(directory="/app/app/static"), name="static")


@app.get("/login", response_class=HTMLResponse)
async def login_page(error: str = ""):
    html = _LOGIN_HTML.replace("{err_class}", "show" if error else "")
    return HTMLResponse(html)


@app.post("/login/check")
async def login_check(request: Request):
    form = await request.form()
    password = form.get("password", "")
    if _APP_PASSWORD and not hmac.compare_digest(password, _APP_PASSWORD):
        return RedirectResponse("/login?error=1", status_code=302)
    token = _make_token(_APP_PASSWORD) if _APP_PASSWORD else ""
    resp = RedirectResponse("/", status_code=302)
    resp.set_cookie(_COOKIE_NAME, token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
    return resp


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
