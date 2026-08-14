"""Measure how Isolation Forest precision and recall vary with cheat prevalence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from generate_data import generate_sessions
from ml_detector import metrics, train_and_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=5_000)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20])
    parser.add_argument("--output", type=Path, default=Path("graphs/precision_recall_vs_cheat_rate.png"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = []
    for index, rate in enumerate(args.rates):
        data = generate_sessions(args.sessions, cheat_rate=rate, seed=args.seed + index)
        # Matching contamination to the experimental prevalence isolates the impact
        # of class balance. In deployment it should be tuned from validated traffic.
        _, _, predictions = train_and_score(data, contamination=rate, seed=args.seed)
        precision, recall, f1 = metrics(data["label"], predictions)
        rows.append({"cheat_rate": rate, "precision": precision, "recall": recall, "f1": f1})
    results = pd.DataFrame(rows)
    print(results.to_string(index=False, formatters={column: "{:.3f}".format for column in ["cheat_rate", "precision", "recall", "f1"]}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(results.cheat_rate * 100, results.precision, marker="o", label="Precision")
    plt.plot(results.cheat_rate * 100, results.recall, marker="o", label="Recall")
    plt.ylim(0, 1.05)
    plt.xlabel("Cheat rate (%)")
    plt.ylabel("Score")
    plt.title("Isolation Forest performance vs cheat prevalence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    plt.close()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
