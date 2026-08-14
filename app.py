"""FastAPI entry point: validate telemetry, queue scoring, retrieve cached results."""

from __future__ import annotations

import json

import redis
from fastapi import FastAPI, HTTPException, status

from config import REDIS_URL
from schemas import QueuedResponse, TelemetryBatch
from tasks import celery_app

app = FastAPI(title="Anti-Cheat / Trust & Safety Pipeline", version="1.0.0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        redis_client.ping()
    except redis.RedisError as exc:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from exc
    return {"status": "ok"}


@app.post("/events", response_model=QueuedResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest_events(batch: TelemetryBatch) -> QueuedResponse:
    # Passing only JSON-safe primitives makes messages portable across workers.
    task = celery_app.send_task("score_session", args=[batch.session_id, batch.model_dump(exclude={"session_id"})])
    return QueuedResponse(session_id=batch.session_id, task_id=task.id)


@app.get("/trust_score/{session_id}")
def get_trust_score(session_id: str) -> dict:
    try:
        result = redis_client.get(f"trust_score:{session_id}")
    except redis.RedisError as exc:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from exc
    if result is None:
        # 404 covers a genuinely unknown ID and a valid task that is still queued;
        # clients can poll after POST /events rather than tying up a request.
        raise HTTPException(status_code=404, detail="Score not available yet")
    return json.loads(result)
