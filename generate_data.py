"""Generate reproducible, labeled synthetic player-session feature data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from telemetry import FEATURE_COLUMNS


def generate_sessions(
    n_sessions: int = 5_000,
    cheat_rate: float = 0.05,
    elite_rate: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Return labeled sessions with typical, elite, and cheating behavior.

    Labels exist only to evaluate detectors. The later Isolation Forest never sees
    them during fitting, mirroring a setting where confirmed cheats are scarce.
    """
    if n_sessions <= 0 or not 0 <= cheat_rate <= 1 or not 0 <= elite_rate <= 1:
        raise ValueError("n_sessions must be positive; cheat_rate and elite_rate must be in [0, 1].")
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_sessions) < cheat_rate).astype(int)
    normal = labels == 0
    # Elite players are a subset of legitimate sessions, so they retain label 0.
    # Keeping this profile only as evaluation metadata lets us quantify whether the
    # detector mistakes unusually strong (but human) play for cheating.
    elite = np.zeros(n_sessions, dtype=bool)
    normal_indices = np.flatnonzero(normal)
    elite_count = round(len(normal_indices) * elite_rate)
    elite[rng.choice(normal_indices, size=elite_count, replace=False)] = True
    typical = normal & ~elite
    data = np.empty((n_sessions, len(FEATURE_COLUMNS)), dtype=float)

    # Typical players vary between and within sessions. Clip physical measures to
    # avoid impossible synthetic outliers that would teach unrealistic rules.
    n = typical.sum()
    data[typical] = np.column_stack((
        np.clip(rng.normal(250, 40, n), 120, 450),
        np.clip(rng.normal(55, 18, n), 12, 125),
        np.clip(rng.normal(5.2, 0.9, n), 2.0, 7.5),
        np.clip(rng.normal(7.3, 0.8, n), 4.0, 9.5),
        np.clip(rng.normal(210, 45, n), 75, 500),
        np.clip(rng.normal(55, 20, n), 12, 150),
        rng.beta(2.0, 10.0, n),
    ))

    # Elite legitimate players react and aim better than average, but never with
    # bot-like consistency or impossible speed. In particular, their reaction
    # times remain in a plausible human range and their within-session variance,
    # click timing, and movement caps all preserve natural human variability.
    e = elite.sum()
    data[elite] = np.column_stack((
        np.clip(rng.normal(185, 24, e), 130, 270),
        np.clip(rng.normal(38, 12, e), 14, 85),
        np.clip(rng.normal(6.3, 0.65, e), 3.5, 7.5),
        np.clip(rng.normal(8.5, 0.55, e), 5.5, 9.5),
        np.clip(rng.normal(175, 32, e), 85, 360),
        np.clip(rng.normal(34, 13, e), 10, 105),
        rng.beta(4.0, 6.0, e),
    ))

    # Cheats are intentionally conspicuous for this first portfolio baseline:
    # ultra-low stable reaction times, impossible speed, periodic clicks, and snaps.
    c = (~normal).sum()
    data[~normal] = np.column_stack((
        np.clip(rng.normal(85, 8, c), 45, 110),
        np.clip(rng.normal(4, 2, c), 0.3, 10),
        np.clip(rng.normal(10.8, 0.9, c), 9.2, 14),
        np.clip(rng.normal(14.2, 1.0, c), 11, 18),
        np.clip(rng.normal(95, 10, c), 65, 125),
        np.clip(rng.normal(3, 1.5, c), 0.2, 8),
        np.clip(rng.beta(12.0, 1.5, c), 0, 1),
    ))
    frame = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    frame["label"] = labels
    frame["player_profile"] = np.where(labels == 1, "cheater", np.where(elite, "elite_legitimate", "typical_legitimate"))
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


def print_summary(data: pd.DataFrame) -> None:
    print(f"Sessions: {len(data):,}; normal: {(data.label == 0).sum():,}; cheater: {(data.label == 1).sum():,}")
    if "player_profile" in data:
        print("Player profiles:")
        print(data["player_profile"].value_counts().to_string())
    print("\nFeature means by ground-truth label (0=normal, 1=cheater):")
    print(data.groupby("label")[FEATURE_COLUMNS].mean().round(2).to_string())
    if "player_profile" in data:
        print("\nFeature means by player profile:")
        print(data.groupby("player_profile")[FEATURE_COLUMNS].mean().round(2).to_string())
    print("\nFeature standard deviations by label:")
    print(data.groupby("label")[FEATURE_COLUMNS].std().round(2).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("sessions.csv"))
    parser.add_argument("--sessions", type=int, default=5_000)
    parser.add_argument("--cheat-rate", type=float, default=0.05)
    parser.add_argument("--elite-rate", type=float, default=0.10, help="share of legitimate sessions modeled as elite players")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sessions = generate_sessions(args.sessions, args.cheat_rate, args.elite_rate, args.seed)
    sessions.to_csv(args.output, index=False)
    print(f"Saved {args.output}")
    print_summary(sessions)
