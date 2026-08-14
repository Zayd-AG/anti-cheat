"""Shared telemetry schemas and feature extraction for offline and live paths."""

from __future__ import annotations

from typing import Sequence

import numpy as np

FEATURE_COLUMNS = [
    "avg_reaction_time_ms",
    "reaction_time_std",
    "avg_movement_speed",
    "max_movement_speed",
    "click_interval_mean",
    "click_interval_std",
    "aim_snap_ratio",
]


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Return stable population statistics, including for a one-item session."""
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        raise ValueError("A session must contain at least one value for each required stream.")
    return float(array.mean()), float(array.std(ddof=0))


def extract_features(
    reaction_times_ms: Sequence[float],
    movement_speeds: Sequence[float],
    click_timestamps_ms: Sequence[float],
    aim_movements: Sequence[bool],
) -> dict[str, float]:
    """Convert raw event streams to the exact feature vector used during training.

    Keeping this code in one place prevents *training-serving skew*: a model trained
    on one definition of a feature should never receive a subtly different definition
    from the API later.
    """
    reaction_mean, reaction_std = _mean_std(reaction_times_ms)
    movement_mean, _ = _mean_std(movement_speeds)
    if len(click_timestamps_ms) < 2:
        raise ValueError("At least two click timestamps are required to calculate intervals.")
    click_times = np.asarray(click_timestamps_ms, dtype=float)
    intervals = np.diff(click_times)
    if np.any(intervals <= 0):
        raise ValueError("Click timestamps must be strictly increasing.")
    click_mean, click_std = _mean_std(intervals)
    if not aim_movements:
        raise ValueError("At least one aim movement is required.")

    return {
        "avg_reaction_time_ms": reaction_mean,
        "reaction_time_std": reaction_std,
        "avg_movement_speed": movement_mean,
        "max_movement_speed": float(np.max(np.asarray(movement_speeds, dtype=float))),
        "click_interval_mean": click_mean,
        "click_interval_std": click_std,
        "aim_snap_ratio": float(np.mean(np.asarray(aim_movements, dtype=bool))),
    }
