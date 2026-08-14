"""Durable player risk history and rolling-score persistence."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from config import DATABASE_URL, PLAYER_HISTORY_WINDOW


class Base(DeclarativeBase):
    pass


class PlayerSessionRisk(Base):
    """One durable audit record per scored session."""

    __tablename__ = "player_session_risks"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    player_id: Mapped[str] = mapped_column(String(128), index=True)
    cheat_risk_score: Mapped[float] = mapped_column(Float)
    anomaly_score: Mapped[float] = mapped_column(Float)
    features: Mapped[dict[str, float]] = mapped_column(JSON)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PlayerRiskSummary(Base):
    """Latest rolling score for quick durable player-level review queries."""

    __tablename__ = "player_risk_summaries"

    player_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    rolling_cheat_risk_score: Mapped[float] = mapped_column(Float)
    sessions_considered: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_session_factory():
    return sessionmaker(get_engine(), expire_on_commit=False)


@lru_cache(maxsize=1)
def initialize_database() -> None:
    """Create demo tables once, safely across the API and worker processes.

    FastAPI and Celery boot independently. A PostgreSQL advisory lock prevents
    both processes from issuing CREATE TABLE at the same time during startup.
    Production systems should use versioned migrations instead.
    """
    with get_engine().begin() as connection:
        connection.execute(text("SELECT pg_advisory_lock(8472031)"))
        Base.metadata.create_all(connection)
        connection.execute(text("ALTER TABLE player_session_risks ADD COLUMN IF NOT EXISTS features JSON DEFAULT '{}'::json"))
        connection.execute(text("SELECT pg_advisory_unlock(8472031)"))


def check_database_connection() -> None:
    with get_engine().connect() as connection:
        connection.execute(text("SELECT 1"))


def record_player_risk(
    session_id: str,
    player_id: str,
    cheat_risk_score: float,
    anomaly_score: float,
    features: dict[str, float],
) -> tuple[float, int]:
    """Store a session and calculate an exponentially weighted recent-player risk.

    Newer sessions receive the greatest weight, but a single unusual session does
    not completely overwrite a player's history. Session IDs are primary keys, so
    a Celery retry does not double-count the same session.
    """
    initialize_database()
    session_factory = get_session_factory()
    with session_factory() as database:
        existing = database.get(PlayerSessionRisk, session_id)
        if existing is None:
            database.add(PlayerSessionRisk(
                session_id=session_id,
                player_id=player_id,
                cheat_risk_score=cheat_risk_score,
                anomaly_score=anomaly_score,
                features=features,
            ))
            database.flush()
        elif existing.player_id != player_id:
            raise ValueError("A session ID cannot be reassigned to a different player.")

        recent = list(database.scalars(
            select(PlayerSessionRisk)
            .where(PlayerSessionRisk.player_id == player_id)
            .order_by(PlayerSessionRisk.created_at.desc(), PlayerSessionRisk.session_id.desc())
            .limit(PLAYER_HISTORY_WINDOW)
        ))
        scores = np.asarray([entry.cheat_risk_score for entry in recent], dtype=float)
        weights = 0.85 ** np.arange(len(scores))
        rolling_score = float(np.average(scores, weights=weights))
        summary = database.get(PlayerRiskSummary, player_id)
        if summary is None:
            summary = PlayerRiskSummary(
                player_id=player_id,
                rolling_cheat_risk_score=rolling_score,
                sessions_considered=len(scores),
            )
            database.add(summary)
        else:
            summary.rolling_cheat_risk_score = rolling_score
            summary.sessions_considered = len(scores)
        database.commit()
    return rolling_score, len(scores)


def get_player_risk_summary(player_id: str) -> PlayerRiskSummary | None:
    """Return the latest durable rolling score for one player."""
    initialize_database()
    with get_session_factory()() as database:
        return database.get(PlayerRiskSummary, player_id)


def get_recent_flagged_sessions(limit: int, minimum_score: float) -> list[PlayerSessionRisk]:
    """Return the highest-risk recent sessions for the reviewer queue."""
    initialize_database()
    with get_session_factory()() as database:
        return list(database.scalars(
            select(PlayerSessionRisk)
            .where(PlayerSessionRisk.cheat_risk_score >= minimum_score)
            .order_by(PlayerSessionRisk.cheat_risk_score.desc(), PlayerSessionRisk.created_at.desc())
            .limit(limit)
        ))
