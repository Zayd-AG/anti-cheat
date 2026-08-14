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
    lag_rate: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """Return labeled sessions with typical, elite, and cheating behavior.

    Labels exist only to evaluate detectors. The later Isolation Forest never sees
    them during fitting, mirroring a setting where confirmed cheats are scarce.
    """
    if n_sessions <= 0 or not all(0 <= rate <= 1 for rate in (cheat_rate, elite_rate, lag_rate)):
        raise ValueError("n_sessions must be positive; cheat_rate, elite_rate, and lag_rate must be in [0, 1].")
    if elite_rate + lag_rate + 0.12 > 1:
        raise ValueError("elite_rate + lag_rate + controller rate (0.12) cannot exceed 1.")
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_sessions) < cheat_rate).astype(int)
    normal = labels == 0
    # Profiles are evaluation metadata, not model inputs. They let us measure
    # whether the detector harms legitimate players with different skill, device,
    # or network conditions.
    elite = np.zeros(n_sessions, dtype=bool)
    normal_indices = np.flatnonzero(normal)
    elite_count = round(len(normal_indices) * elite_rate)
    elite[rng.choice(normal_indices, size=elite_count, replace=False)] = True
    remaining_legitimate = np.flatnonzero(normal & ~elite)
    controller = np.zeros(n_sessions, dtype=bool)
    lag_affected = np.zeros(n_sessions, dtype=bool)
    controller_count = round(len(normal_indices) * 0.12)
    lag_count = round(len(normal_indices) * lag_rate)
    controller_indices = rng.choice(remaining_legitimate, size=controller_count, replace=False)
    controller[controller_indices] = True
    lag_candidates = np.setdiff1d(remaining_legitimate, controller_indices, assume_unique=True)
    lag_affected[rng.choice(lag_candidates, size=lag_count, replace=False)] = True
    typical = normal & ~elite & ~controller & ~lag_affected
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

    # Controller players have a different but legitimate input profile: slower
    # target acquisition and lower aim snapping, with ordinary variation.
    k = controller.sum()
    data[controller] = np.column_stack((
        np.clip(rng.normal(285, 42, k), 160, 440),
        np.clip(rng.normal(58, 17, k), 15, 125),
        np.clip(rng.normal(4.7, 0.8, k), 2.0, 7.0),
        np.clip(rng.normal(6.8, 0.75, k), 3.5, 9.2),
        np.clip(rng.normal(235, 50, k), 85, 500),
        np.clip(rng.normal(65, 23, k), 12, 160),
        rng.beta(1.4, 12.0, k),
    ))

    # Lag-affected sessions have irregular timing and occasional movement-sample
    # spikes. They are legitimate and stay within the human movement cap, making
    # them realistic false-positive edge cases rather than disguised cheaters.
    l = lag_affected.sum()
    data[lag_affected] = np.column_stack((
        np.clip(rng.normal(320, 55, l), 170, 480),
        np.clip(rng.normal(88, 22, l), 25, 150),
        np.clip(rng.normal(5.0, 0.95, l), 2.0, 7.5),
        np.clip(rng.normal(8.7, 0.55, l), 5.0, 9.5),
        np.clip(rng.normal(260, 65, l), 90, 550),
        np.clip(rng.normal(92, 28, l), 20, 180),
        rng.beta(2.2, 9.0, l),
    ))

    # Cheats use varied strategies. Some remain obviously inhuman; others alter
    # only one signal, so the evaluation tests realistic, difficult trade-offs.
    cheat_indices = np.flatnonzero(~normal)
    cheat_profiles = rng.choice(
        ["hard_cheat", "subtle_aim_assist", "speed_cheat", "click_bot"],
        size=len(cheat_indices),
        p=[0.30, 0.30, 0.20, 0.20],
    )

    def populate(profile: str, columns: tuple[np.ndarray, ...]) -> None:
        mask = np.zeros(n_sessions, dtype=bool)
        mask[cheat_indices[cheat_profiles == profile]] = True
        data[mask] = np.column_stack(columns)

    c = (cheat_profiles == "hard_cheat").sum()
    populate("hard_cheat", (
        np.clip(rng.normal(85, 8, c), 45, 110), np.clip(rng.normal(4, 2, c), 0.3, 10),
        np.clip(rng.normal(10.8, 0.9, c), 9.2, 14), np.clip(rng.normal(14.2, 1.0, c), 11, 18),
        np.clip(rng.normal(95, 10, c), 65, 125), np.clip(rng.normal(3, 1.5, c), 0.2, 8),
        np.clip(rng.beta(12.0, 1.5, c), 0, 1),
    ))
    c = (cheat_profiles == "subtle_aim_assist").sum()
    populate("subtle_aim_assist", (
        np.clip(rng.normal(185, 26, c), 130, 290), np.clip(rng.normal(30, 10, c), 12, 75),
        np.clip(rng.normal(6.1, 0.75, c), 3.0, 7.5), np.clip(rng.normal(8.4, 0.6, c), 5.0, 9.5),
        np.clip(rng.normal(175, 35, c), 80, 350), np.clip(rng.normal(30, 12, c), 10, 100),
        rng.beta(12.0, 3.0, c),
    ))
    c = (cheat_profiles == "speed_cheat").sum()
    populate("speed_cheat", (
        np.clip(rng.normal(230, 38, c), 130, 390), np.clip(rng.normal(48, 16, c), 12, 110),
        np.clip(rng.normal(7.8, 0.55, c), 6.5, 9.2), np.clip(rng.normal(11.5, 0.55, c), 10.0, 13.0),
        np.clip(rng.normal(205, 42, c), 85, 450), np.clip(rng.normal(50, 18, c), 12, 135),
        rng.beta(2.0, 10.0, c),
    ))
    c = (cheat_profiles == "click_bot").sum()
    populate("click_bot", (
        np.clip(rng.normal(235, 38, c), 130, 390), np.clip(rng.normal(48, 16, c), 12, 110),
        np.clip(rng.normal(5.3, 0.85, c), 2.2, 7.5), np.clip(rng.normal(7.4, 0.75, c), 4.5, 9.5),
        np.clip(rng.normal(108, 10, c), 75, 145), np.clip(rng.normal(4, 1.5, c), 0.5, 8),
        rng.beta(2.0, 10.0, c),
    ))
    frame = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    frame["label"] = labels
    profiles = np.full(n_sessions, "typical_legitimate", dtype=object)
    profiles[elite] = "elite_legitimate"
    profiles[controller] = "controller_legitimate"
    profiles[lag_affected] = "lag_affected_legitimate"
    profiles[cheat_indices] = cheat_profiles
    frame["player_profile"] = profiles
    frame["input_device"] = np.where(controller, "controller", "keyboard_mouse")
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
    parser.add_argument("--lag-rate", type=float, default=0.08, help="share of legitimate sessions modeled as lag-affected players")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sessions = generate_sessions(
        args.sessions,
        cheat_rate=args.cheat_rate,
        elite_rate=args.elite_rate,
        lag_rate=args.lag_rate,
        seed=args.seed,
    )
    sessions.to_csv(args.output, index=False)
    print(f"Saved {args.output}")
    print_summary(sessions)
