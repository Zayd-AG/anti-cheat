"""FastAPI entry point: validate telemetry, queue scoring, retrieve cached results."""

from __future__ import annotations

import json

import redis
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError

from config import REDIS_URL
from db import check_database_connection, get_player_risk_summary, get_recent_flagged_sessions, initialize_database
from schemas import QueuedResponse, TelemetryBatch
from tasks import celery_app

app = FastAPI(title="Game Anti-Cheat Detection System", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        redis_client.ping()
        initialize_database()
        check_database_connection()
    except (redis.RedisError, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="A required data service is unavailable") from exc
    return {"status": "ok", "redis": "ok", "postgres": "ok"}


@app.post("/events", response_model=QueuedResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest_events(batch: TelemetryBatch) -> QueuedResponse:
    # Passing only JSON-safe primitives makes messages portable across workers.
    task = celery_app.send_task("score_session", args=[batch.session_id, batch.player_id, batch.model_dump(exclude={"session_id", "player_id"})])
    return QueuedResponse(session_id=batch.session_id, player_id=batch.player_id, task_id=task.id)


@app.get("/cheat_risk/{session_id}")
def get_cheat_risk(session_id: str) -> dict:
    try:
        result = redis_client.get(f"cheat_risk:{session_id}")
    except redis.RedisError as exc:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from exc
    if result is None:
        # 404 covers a genuinely unknown ID and a valid task that is still queued;
        # clients can poll after POST /events rather than tying up a request.
        raise HTTPException(status_code=404, detail="Cheat risk score not available yet")
    return json.loads(result)


@app.get("/players/{player_id}/cheat_risk")
def get_player_cheat_risk(player_id: str) -> dict:
    try:
        summary = get_player_risk_summary(player_id)
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable") from exc
    if summary is None:
        raise HTTPException(status_code=404, detail="Player risk history not found")
    return {
        "player_id": summary.player_id,
        "player_cheat_risk_score": round(summary.rolling_cheat_risk_score, 3),
        "sessions_considered": summary.sessions_considered,
        "window_size": 20,
    }


def flag_reasons(features: dict[str, float]) -> list[str]:
    """Turn feature values into short reviewer-facing explanations."""
    reasons: list[str] = []
    if features["avg_reaction_time_ms"] < 150:
        reasons.append("Fast Reactions")
    if features["max_movement_speed"] > 9.5:
        reasons.append("Movement Above Human Cap")
    if features["click_interval_std"] < 12:
        reasons.append("Highly Regular Clicks")
    if features["aim_snap_ratio"] > 0.55:
        reasons.append("High Aim Snapping")
    return reasons or ["Multiple Unusual Patterns"]


@app.get("/review/flagged-sessions")
def get_flagged_sessions(limit: int = 20, minimum_score: float = 50) -> list[dict]:
    if not 1 <= limit <= 100 or not 0 <= minimum_score <= 100:
        raise HTTPException(status_code=422, detail="limit must be 1-100 and minimum_score must be 0-100")
    try:
        sessions = get_recent_flagged_sessions(limit, minimum_score)
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable") from exc
    return [
        {
            "session_id": item.session_id,
            "player_id": item.player_id,
            "cheat_risk_score": round(item.cheat_risk_score, 3),
            "anomaly_score": round(item.anomaly_score, 6),
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "features": item.features,
            "reasons": flag_reasons(item.features),
        }
        for item in sessions
    ]
