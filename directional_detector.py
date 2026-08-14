"""A transparent detector for signals that are suspicious in one direction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from telemetry import FEATURE_COLUMNS


class DirectionalSignalDetector:
    """Score the strongest human-limit violation in a session.

    Unlike a two-sided RMS z-score, this model does not flag a player merely for
    being slower than average. It only considers directions associated with
    automation: unusually low timing variance, unusually fast movement/clicks,
    and unusually high aim snapping.
    """

    # -1 marks suspiciously low values; +1 marks suspiciously high values.
    directions = np.array([-1, -1, 1, 1, -1, -1, 1], dtype=float)

    def fit(self, data: pd.DataFrame | np.ndarray) -> "DirectionalSignalDetector":
        values = self._values(data)
        self.means_ = values.mean(axis=0)
        self.stds_ = np.maximum(values.std(axis=0), 1e-6)
        self.threshold_ = None
        return self

    def set_threshold(self, threshold: float) -> "DirectionalSignalDetector":
        self.threshold_ = float(threshold)
        return self

    def suspiciousness(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = self._values(data)
        z_scores = ((values - self.means_) / self.stds_) * self.directions
        one_sided = np.maximum(z_scores, 0)

        # A click bot's most reliable fingerprint is the combination of short,
        # highly regular click intervals. Each value alone can occur in human
        # play, but both together are a stronger and still explainable signal.
        click_pattern = np.hypot(one_sided[:, 4], one_sided[:, 5])
        return np.maximum(one_sided.max(axis=1), click_pattern)

    def score_samples(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Match scikit-learn convention: larger scores mean more normal.
        return -self.suspiciousness(data)

    def decision_function(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise ValueError("Set a calibrated threshold before calling decision_function.")
        return self.threshold_ - self.suspiciousness(data)

    @staticmethod
    def _values(data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data[FEATURE_COLUMNS].to_numpy(dtype=float)
        return np.asarray(data, dtype=float)
