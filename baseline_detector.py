"""Explainable z-score baseline for synthetic session anomaly detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from telemetry import FEATURE_COLUMNS


def fit_zscore_baseline(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Fit population parameters without consulting labels."""
    means = train[FEATURE_COLUMNS].mean()
    # ddof=0 matches population standardization; protect against a constant feature.
    stds = train[FEATURE_COLUMNS].std(ddof=0).replace(0, 1.0)
    return means, stds


def anomaly_scores(data: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.Series:
    """Combine per-feature deviations into a scale-independent anomaly score.

    RMS z-score emphasizes a session abnormal on several dimensions while keeping the
    score understandable. Signed deviations are squared because both unusual lows
    (reaction variance) and highs (movement speed) can be suspicious.
    """
    z_scores = (data[FEATURE_COLUMNS] - means) / stds
    return np.sqrt((z_scores**2).mean(axis=1))


def evaluate(labels: pd.Series, predictions: np.ndarray, scores: pd.Series, data: pd.DataFrame) -> None:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    normal_mask = labels.to_numpy() == 0
    false_positive_rate = float(np.mean(predictions[normal_mask] == 1))
    print(f"Precision:           {precision:.3f}\nRecall:              {recall:.3f}\nF1:                  {f1:.3f}\nFalse-positive rate: {false_positive_rate:.3%}")
    print("\nConfusion matrix [[true normal, false positive], [false negative, true cheater]]:")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))
    if "player_profile" in data:
        print("\nFalse-positive rate by legitimate player profile:")
        for profile in sorted(data.loc[data["label"] == 0, "player_profile"].unique()):
            profile_mask = ((data["player_profile"] == profile) & (data["label"] == 0)).to_numpy()
            if profile_mask.any():
                print(f"{profile}: {(predictions[profile_mask] == 1).mean():.3%}")
    for title, mask in (
        ("False positives", (labels == 0) & (predictions == 1)),
        ("False negatives", (labels == 1) & (predictions == 0)),
    ):
        examples = data.loc[mask, FEATURE_COLUMNS].copy()
        examples["anomaly_score"] = scores[mask]
        print(f"\n{title} (up to 5):")
        print(examples.head(5).round(2).to_string(index=False) if not examples.empty else "None")


def run(data_path: Path, threshold: float) -> tuple[pd.Series, np.ndarray, pd.Series, pd.DataFrame]:
    data = pd.read_csv(data_path)
    required = set(FEATURE_COLUMNS + ["label"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {data_path}: {sorted(missing)}")
    means, stds = fit_zscore_baseline(data)
    scores = anomaly_scores(data, means, stds)
    predictions = (scores >= threshold).astype(int).to_numpy()
    return data["label"], predictions, scores, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("sessions.csv"))
    parser.add_argument("--threshold", type=float, default=1.5, help="RMS z-score at which to flag a session")
    args = parser.parse_args()
    labels, predictions, scores, data = run(args.data, args.threshold)
    evaluate(labels, predictions, scores, data)
