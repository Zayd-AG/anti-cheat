"""FastAPI entry point: validate telemetry, queue scoring, retrieve cached results."""

from __future__ import annotations

import json

import redis
from fastapi import FastAPI, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError

from config import REDIS_URL
from db import check_database_connection, get_player_risk_summary, initialize_database
from schemas import QueuedResponse, TelemetryBatch
from tasks import celery_app

app = FastAPI(title="Game Anti-Cheat Detection System", version="1.1.0")
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
