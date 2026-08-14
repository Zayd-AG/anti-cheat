"""Train and evaluate an unsupervised Isolation Forest session detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from baseline_detector import run as run_baseline
from telemetry import FEATURE_COLUMNS


def metrics(labels: pd.Series, predictions: pd.Series) -> tuple[float, float, float]:
    return precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)[:3]


def train_and_score(data: pd.DataFrame, contamination: float, seed: int) -> tuple[IsolationForest, pd.Series, pd.Series]:
    """Fit only feature values; labels remain strictly evaluation-only.

    Isolation Forest is useful here because it isolates rare patterns with random
    feature splits, without needing a large trusted corpus of confirmed cheats.
    `contamination` tells it the expected outlier prevalence for its threshold.
    """
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=300, n_jobs=-1)
    model.fit(data[FEATURE_COLUMNS])
    # sklearn returns larger scores for more normal samples, so negate for an
    # intuitive "larger means more suspicious" value stored by the API later.
    scores = pd.Series(-model.score_samples(data[FEATURE_COLUMNS]), index=data.index, name="anomaly_score")
    predictions = pd.Series((model.predict(data[FEATURE_COLUMNS]) == -1).astype(int), index=data.index)
    return model, scores, predictions


def print_evaluation(name: str, labels: pd.Series, predictions: pd.Series) -> tuple[float, float, float]:
    precision, recall, f1 = metrics(labels, predictions)
    false_positive_rate = float(((predictions == 1) & (labels == 0)).sum() / (labels == 0).sum())
    print(f"{name}: precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}, false-positive rate={false_positive_rate:.3%}")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))
    return precision, recall, f1


def print_profile_false_positive_rates(data: pd.DataFrame, predictions: pd.Series) -> None:
    """Show whether rare-but-legitimate skill is disproportionately flagged."""
    if "player_profile" not in data:
        return
    print("False-positive rate by legitimate player profile:")
    for profile in ("typical_legitimate", "elite_legitimate"):
        profile_mask = data["player_profile"] == profile
        if profile_mask.any():
            print(f"{profile}: {(predictions[profile_mask] == 1).mean():.3%}")


def plot_score_distribution(scores: pd.Series, labels: pd.Series, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.hist(scores[labels == 0], bins=40, alpha=0.7, density=True, label="Normal")
    plt.hist(scores[labels == 1], bins=40, alpha=0.7, density=True, label="Cheater")
    plt.xlabel("Isolation Forest anomaly score (higher = more suspicious)")
    plt.ylabel("Density")
    plt.title("Anomaly-score distribution by ground-truth label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_comparison(results: dict[str, tuple[float, float, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results, index=["Precision", "Recall", "F1"]).T
    ax = table[["Precision", "Recall"]].plot.bar(rot=0, figsize=(7, 4.5), ylim=(0, 1.05))
    ax.set_title("Baseline vs Isolation Forest")
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("sessions.csv"))
    parser.add_argument("--model-output", type=Path, default=Path("models/isolation_forest.joblib"))
    parser.add_argument("--graphs-dir", type=Path, default=Path("graphs"))
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-threshold", type=float, default=1.5)
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    labels = data["label"]
    model, scores, predictions = train_and_score(data, args.contamination, args.seed)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    print(f"Saved trained model to {args.model_output}")
    ml_result = print_evaluation("Isolation Forest", labels, predictions)
    print_profile_false_positive_rates(data, predictions)
    baseline_labels, baseline_predictions, _, _ = run_baseline(args.data, args.baseline_threshold)
    baseline_result = print_evaluation("Z-score baseline", baseline_labels, baseline_predictions)
    plot_score_distribution(scores, labels, args.graphs_dir / "anomaly_score_distribution.png")
    plot_comparison({"Z-score baseline": baseline_result, "Isolation Forest": ml_result}, args.graphs_dir / "baseline_vs_ml.png")
    print(f"Saved graphs to {args.graphs_dir}")
