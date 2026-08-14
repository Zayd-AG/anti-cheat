"""Centralized runtime settings; environment variables keep containers portable."""

import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_PATH = os.getenv("MODEL_PATH", "models/isolation_forest.joblib")
TRUST_SCORE_TTL_SECONDS = int(os.getenv("TRUST_SCORE_TTL_SECONDS", "86400"))
