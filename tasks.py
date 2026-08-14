"""Celery worker that turns an accepted batch into a Redis-backed trust score."""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import redis
from celery import Celery

from config import MODEL_PATH, REDIS_URL, TRUST_SCORE_TTL_SECONDS
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


@celery_app.task(name="score_session", autoretry_for=(redis.RedisError,), retry_backoff=True, max_retries=3)
def score_session(session_id: str, telemetry: dict) -> dict:
    """Score asynchronously so slow model/cache work never consumes API workers.

    Keeping scoring off the FastAPI request path lets the API acknowledge bursts
    quickly. Celery supplies independent worker scaling and retry handling for
    transient Redis failures, which is much safer than holding client connections.
    """
    features = extract_features(**telemetry)
    vector = np.array([[features[column] for column in FEATURE_COLUMNS]])
    model = get_model()
    anomaly_score = float(-model.score_samples(vector)[0])
    # `decision_function` is centered on the Isolation Forest's learned
    # contamination threshold: positive means inlier, negative means outlier.
    # Its sign is therefore a safer trust-score midpoint than a hard-coded raw
    # score (whose scale varies by fitted model). This remains a display score,
    # not a calibrated probability that a player cheated.
    decision_margin = float(model.decision_function(vector)[0])
    trust_score = float(100 / (1 + np.exp(-20 * decision_margin)))
    payload = {"session_id": session_id, "trust_score": round(trust_score, 3), "anomaly_score": round(anomaly_score, 6), "features": features}
    get_redis_client().setex(f"trust_score:{session_id}", TRUST_SCORE_TTL_SECONDS, json.dumps(payload))
    return payload
