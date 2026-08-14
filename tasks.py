"""Celery worker that turns an accepted batch into a Redis-backed trust score."""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import redis
from celery import Celery
from sqlalchemy.exc import SQLAlchemyError

from config import MODEL_PATH, REDIS_URL, TRUST_SCORE_TTL_SECONDS
# Import the model class before unpickling the calibrated directional model.
# This also makes a missing image/source file fail immediately at worker startup
# instead of after the first scoring request.
from directional_detector import DirectionalSignalDetector
from db import record_player_risk
from telemetry import FEATURE_COLUMNS, extract_features

# Redis serves two separate roles: Celery's broker transports durable-ish task
# messages, while the direct client below caches short-lived API lookup results.
celery_app = Celery("trust_safety", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(task_serializer="json", accept_content=["json"], result_serializer="json")


@lru_cache(maxsize=1)
def get_model():
    """Load once per worker process, avoiding disk I/O on every player session."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


@celery_app.task(name="score_session", autoretry_for=(redis.RedisError, SQLAlchemyError), retry_backoff=True, max_retries=3)
def score_session(session_id: str, player_id: str, telemetry: dict) -> dict:
    """Score asynchronously so slow model/cache work never consumes API workers.

    Keeping scoring off the FastAPI request path lets the API acknowledge bursts
    quickly. Celery supplies independent worker scaling and retry handling for
    transient Redis failures, which is much safer than holding client connections.
    """
    features = extract_features(**telemetry)
    vector = np.array([[features[column] for column in FEATURE_COLUMNS]])
    model = get_model()
    anomaly_score = float(-model.score_samples(vector)[0])
    # A score near 100 means the session looks highly suspicious. It is a useful
    # review-prioritization signal, not a calibrated probability of cheating.
    decision_margin = float(model.decision_function(vector)[0])
    cheat_risk_score = float(100 / (1 + np.exp(20 * decision_margin)))
    rolling_score, sessions_considered = record_player_risk(
        session_id, player_id, cheat_risk_score, anomaly_score, features
    )
    payload = {
        "session_id": session_id,
        "player_id": player_id,
        "cheat_risk_score": round(cheat_risk_score, 3),
        "player_cheat_risk_score": round(rolling_score, 3),
        "player_sessions_considered": sessions_considered,
        "anomaly_score": round(anomaly_score, 6),
        "features": features,
    }
    get_redis_client().setex(f"cheat_risk:{session_id}", TRUST_SCORE_TTL_SECONDS, json.dumps(payload))
    return payload
