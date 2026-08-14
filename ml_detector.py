"""Train and evaluate an unsupervised Isolation Forest session detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline_detector import anomaly_scores, fit_zscore_baseline
from directional_detector import DirectionalSignalDetector
from telemetry import FEATURE_COLUMNS


def metrics(labels: pd.Series, predictions: pd.Series) -> tuple[float, float, float]:
    return precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)[:3]


def train_and_score(data: pd.DataFrame, contamination: float, seed: int) -> tuple[IsolationForest, pd.Series, pd.Series]:
    """Fit only feature values; labels remain strictly evaluation-only.

    Isolation Forest is useful here because it isolates rare patterns with random
    feature splits, without needing a large trusted corpus of confirmed cheats.
    `contamination` tells it the expected outlier prevalence for its threshold.
    """
    # Scaling prevents high-range measurements (milliseconds) from dominating
    # random split selection over bounded features such as aim_snap_ratio.
    model = make_pipeline(
        StandardScaler(),
        IsolationForest(contamination=contamination, random_state=seed, n_estimators=500, n_jobs=-1),
    )
    model.fit(data[FEATURE_COLUMNS])
    # sklearn returns larger scores for more normal samples, so negate for an
    # intuitive "larger means more suspicious" value stored by the API later.
    scores = pd.Series(-model.score_samples(data[FEATURE_COLUMNS]), index=data.index, name="anomaly_score")
    predictions = pd.Series((model.predict(data[FEATURE_COLUMNS]) == -1).astype(int), index=data.index)
    return model, scores, predictions


def split_data(data: pd.DataFrame, test_size: float, validation_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create disjoint, label-stratified splits without using labels for fitting.

    Labels preserve the rare cheat rate in every split, but are never passed to the
    model. The validation split is reserved for operating-point selection; final
    metrics are computed once on the untouched test set.
    """
    train_validation, test = train_test_split(data, test_size=test_size, stratify=data["label"], random_state=seed)
    validation_share = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=validation_share,
        stratify=train_validation["label"],
        random_state=seed,
    )
    return train, validation, test


def threshold_for_target_fpr(scores: pd.Series, labels: pd.Series, target_fpr: float) -> float:
    """Select a score threshold from validation normals at a declared FPR budget."""
    normal_scores = scores[labels == 0]
    if normal_scores.empty:
        raise ValueError("Validation data must contain legitimate sessions.")
    return float(np.quantile(normal_scores, 1 - target_fpr))


def predict_from_threshold(scores: pd.Series, threshold: float) -> pd.Series:
    return pd.Series((scores >= threshold).astype(int), index=scores.index)


def print_evaluation(name: str, labels: pd.Series, predictions: pd.Series) -> tuple[float, float, float]:
    precision, recall, f1 = metrics(labels, predictions)
    false_positive_rate = float(((predictions == 1) & (labels == 0)).sum() / (labels == 0).sum())
    print(f"{name}: precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}, false-positive rate={false_positive_rate:.3%}")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))
    return precision, recall, f1


def metric_report(labels: pd.Series, predictions: pd.Series) -> dict[str, float]:
    """Return JSON-safe final-test metrics for reproducible reporting."""
    precision, recall, f1 = metrics(labels, predictions)
    return {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "false_positive_rate": round(float(((predictions == 1) & (labels == 0)).sum() / (labels == 0).sum()), 6),
    }


def print_profile_error_rates(data: pd.DataFrame, predictions: pd.Series) -> None:
    """Show safety impact across legitimate profiles and recall by cheat style."""
    if "player_profile" not in data:
        return
    print("False-positive rate by legitimate player profile:")
    for profile in sorted(data.loc[data["label"] == 0, "player_profile"].unique()):
        profile_mask = (data["player_profile"] == profile) & (data["label"] == 0)
        if profile_mask.any():
            print(f"{profile}: {(predictions[profile_mask] == 1).mean():.3%}")
    print("Recall by cheat style:")
    for profile in sorted(data.loc[data["label"] == 1, "player_profile"].unique()):
        profile_mask = (data["player_profile"] == profile) & (data["label"] == 1)
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
    parser.add_argument("--model-output", type=Path, default=Path("models/directional_signal_detector.joblib"))
    parser.add_argument("--graphs-dir", type=Path, default=Path("graphs"))
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--target-fpr", type=float, default=0.01, help="maximum validation false-positive rate used to set score thresholds")
    args = parser.parse_args()
    if not 0 < args.validation_size < 1 or not 0 < args.test_size < 1 or args.validation_size + args.test_size >= 1:
        raise ValueError("validation-size and test-size must be positive and sum to less than 1.")
    if not 0 < args.target_fpr < 1:
        raise ValueError("target-fpr must be between 0 and 1.")

    data = pd.read_csv(args.data)
    train, validation, test = split_data(data, args.test_size, args.validation_size, args.seed)
    print(f"Split sizes: train={len(train):,}, validation={len(validation):,}, test={len(test):,}")
    print(f"Threshold policy: select the highest score allowed by a {args.target_fpr:.1%} validation false-positive-rate budget.")

    # Fit only the training features. The known labels remain evaluation metadata.
    model = make_pipeline(
        StandardScaler(),
        IsolationForest(contamination=args.contamination, random_state=args.seed, n_estimators=500, n_jobs=-1),
    )
    model.fit(train[FEATURE_COLUMNS])
    validation_scores = pd.Series(-model.score_samples(validation[FEATURE_COLUMNS]), index=validation.index)
    test_scores = pd.Series(-model.score_samples(test[FEATURE_COLUMNS]), index=test.index, name="anomaly_score")
    ml_threshold = threshold_for_target_fpr(validation_scores, validation["label"], args.target_fpr)
    ml_predictions = predict_from_threshold(test_scores, ml_threshold)

    print("\nFinal held-out test-set results:")
    print(f"Isolation Forest threshold: {ml_threshold:.6f}")
    ml_result = print_evaluation("Isolation Forest", test["label"], ml_predictions)
    print_profile_error_rates(test, ml_predictions)

    # The z-score baseline receives the same train/validation/test treatment.
    baseline_means, baseline_stds = fit_zscore_baseline(train)
    validation_baseline_scores = anomaly_scores(validation, baseline_means, baseline_stds)
    test_baseline_scores = anomaly_scores(test, baseline_means, baseline_stds)
    baseline_threshold = threshold_for_target_fpr(validation_baseline_scores, validation["label"], args.target_fpr)
    baseline_predictions = predict_from_threshold(test_baseline_scores, baseline_threshold)
    print(f"Z-score baseline threshold: {baseline_threshold:.6f}")
    baseline_result = print_evaluation("Z-score baseline", test["label"], baseline_predictions)
    print_profile_error_rates(test, baseline_predictions)

    # A directional model complements general anomaly detection. It focuses on
    # signals that should only be suspicious on one side (for example, highly
    # regular clicks), so normal slow or controller input is not penalized.
    directional_model = DirectionalSignalDetector().fit(train)
    validation_directional_scores = pd.Series(-directional_model.score_samples(validation), index=validation.index)
    directional_threshold = threshold_for_target_fpr(validation_directional_scores, validation["label"], args.target_fpr)
    directional_model.set_threshold(directional_threshold)
    test_directional_scores = pd.Series(-directional_model.score_samples(test), index=test.index)
    directional_predictions = predict_from_threshold(test_directional_scores, directional_threshold)
    print(f"Directional signal threshold: {directional_threshold:.6f}")
    directional_result = print_evaluation("Directional signal detector", test["label"], directional_predictions)
    print_profile_error_rates(test, directional_predictions)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(directional_model, args.model_output)
    print(f"Saved calibrated directional model to {args.model_output}")

    report = {
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "target_validation_false_positive_rate": args.target_fpr,
        "isolation_forest": {"threshold": ml_threshold, **metric_report(test["label"], ml_predictions)},
        "z_score_baseline": {"threshold": baseline_threshold, **metric_report(test["label"], baseline_predictions)},
        "directional_signal_detector": {"threshold": directional_threshold, **metric_report(test["label"], directional_predictions)},
    }
    args.graphs_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.graphs_dir / "held_out_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_score_distribution(test_scores, test["label"], args.graphs_dir / "anomaly_score_distribution.png")
    plot_comparison({"Z-score baseline": baseline_result, "Isolation Forest": ml_result, "Directional signals": directional_result}, args.graphs_dir / "baseline_vs_ml.png")
    print(f"Saved graphs and held-out report to {args.graphs_dir}")
