"""Centralized runtime settings; environment variables keep containers portable."""

import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_PATH = os.getenv("MODEL_PATH", "models/directional_signal_detector.joblib")
TRUST_SCORE_TTL_SECONDS = int(os.getenv("TRUST_SCORE_TTL_SECONDS", "86400"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://anti_cheat:anti_cheat@localhost:5432/anti_cheat")
PLAYER_HISTORY_WINDOW = int(os.getenv("PLAYER_HISTORY_WINDOW", "20"))
