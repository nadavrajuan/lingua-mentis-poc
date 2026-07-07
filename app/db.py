from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
import aiosqlite

DB_PATH = "/app/data/experiment.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS human_trials (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    trial_index INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    true_label INTEGER NOT NULL,
    user_label INTEGER,
    response_time_ms INTEGER,
    degrade_json TEXT,
    base_pred INTEGER,
    base_probs_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS playback_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    true_label INTEGER,
    predicted_label INTEGER NOT NULL,
    probs_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    margin REAL NOT NULL,
    delta_norm REAL NOT NULL,
    params_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calibration_results (
    id TEXT PRIMARY KEY,
    human_session_id TEXT NOT NULL,
    score REAL NOT NULL,
    params_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


async def create_session(session_type: str, notes: str = "") -> str:
    sid = new_id()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO sessions (id, type, created_at, notes) VALUES (?, ?, ?, ?)",
            (sid, session_type, _now(), notes),
        )
        await db.commit()
    return sid


async def save_playback_event(
    session_id: str, step_index: int, image_id: int, true_label: int,
    predicted_label: int, probs: list[float], confidence: float, margin: float,
    delta_norm: float, params_json: str,
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO playback_events
               (id, session_id, step_index, image_id, true_label, predicted_label,
                probs_json, confidence, margin, delta_norm, params_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                new_id(), session_id, step_index, image_id, true_label,
                predicted_label, json.dumps(probs), confidence, margin,
                delta_norm, params_json, _now(),
            ),
        )
        await db.commit()


async def save_human_trial(
    session_id: str, trial_index: int, image_id: int, true_label: int,
    user_label: int | None, response_time_ms: int | None,
    degrade_json: str | None, base_pred: int | None, base_probs_json: str | None,
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO human_trials
               (id, session_id, trial_index, image_id, true_label, user_label,
                response_time_ms, degrade_json, base_pred, base_probs_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                new_id(), session_id, trial_index, image_id, true_label,
                user_label, response_time_ms, degrade_json, base_pred,
                base_probs_json, _now(),
            ),
        )
        await db.commit()


async def get_human_trials(session_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM human_trials WHERE session_id=? ORDER BY trial_index", (session_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def save_calibration_result(human_session_id: str, score: float, params_json: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO calibration_results (id, human_session_id, score, params_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (new_id(), human_session_id, score, params_json, _now()),
        )
        await db.commit()


async def get_best_calibration(human_session_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM calibration_results WHERE human_session_id=? ORDER BY score ASC LIMIT 1",
            (human_session_id,),
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None
