"""Stream simulated sessions into the API and measure queue-to-cache latency."""

from __future__ import annotations

import argparse
import time

import httpx
import numpy as np

from generate_data import generate_sessions
from telemetry import FEATURE_COLUMNS


def telemetry_from_features(row: dict[str, float], seed: int) -> dict[str, list[float] | list[bool]]:
    """Create raw streams whose aggregate properties approximate generated features.

    This deliberately derives events from Phase 1 session distributions so the load
    test exercises the same behavioral population rather than arbitrary HTTP data.
    """
    rng = np.random.default_rng(seed)

    def values(mean: float, std: float, count: int) -> list[float]:
        raw = rng.normal(0, 1, count)
        normalized = (raw - raw.mean()) / raw.std()
        return np.maximum(0.01, mean + normalized * max(std, 0.01)).round(3).tolist()

    reactions = values(row["avg_reaction_time_ms"], row["reaction_time_std"], 24)
    speeds = values(row["avg_movement_speed"], 0.5, 30)
    speeds[0] = row["max_movement_speed"]
    intervals = values(row["click_interval_mean"], row["click_interval_std"], 24)
    timestamps = np.cumsum([0.0, *intervals]).round(3).tolist()
    snap_count = round(row["aim_snap_ratio"] * 30)
    aim = [True] * snap_count + [False] * (30 - snap_count)
    rng.shuffle(aim)
    return {"reaction_times_ms": reactions, "movement_speeds": speeds, "click_timestamps_ms": timestamps, "aim_movements": aim}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--sessions", type=int, default=100)
    parser.add_argument("--rate", type=float, default=10, help="sessions per second")
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.rate <= 0:
        raise ValueError("rate must be positive")
    sessions = generate_sessions(args.sessions, seed=123)
    latencies: list[float] = []
    with httpx.Client(timeout=args.timeout) as client:
        for index, (_, row) in enumerate(sessions.iterrows()):
            session_id = f"load-{int(time.time() * 1000)}-{index}"
            payload = {"session_id": session_id, **telemetry_from_features(row.to_dict(), index)}
            started = time.perf_counter()
            response = client.post(f"{args.url}/events", json=payload)
            response.raise_for_status()
            while True:
                score = client.get(f"{args.url}/trust_score/{session_id}")
                if score.status_code == 200:
                    latencies.append(time.perf_counter() - started)
                    break
                if time.perf_counter() - started > args.timeout:
                    raise TimeoutError(f"Timed out awaiting score for {session_id}")
                time.sleep(args.poll_interval)
            # Pace requests by their ingestion start; the queue-to-cache latency
            # remains independently measured above.
            next_start = started + 1 / args.rate
            time.sleep(max(0, next_start - time.perf_counter()))
    values = np.array(latencies) * 1000
    print(f"Completed {len(values)} sessions at requested {args.rate:g}/s")
    print(f"End-to-end latency ms: mean={values.mean():.1f}, p50={np.percentile(values, 50):.1f}, p95={np.percentile(values, 95):.1f}, max={values.max():.1f}")


if __name__ == "__main__":
    main()
