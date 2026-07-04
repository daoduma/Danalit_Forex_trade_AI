"""Probability calibration: isotonic regression per class, fit on validation folds.

Position-taking thresholds depend on '62% means 62%', so raw LightGBM scores
are never used directly. Calibrators are fit one-vs-rest on the fold's
validation split and renormalised to sum to 1.
"""

from __future__ import annotations

import numpy as np


def fit_calibration(proba_val: np.ndarray, y_val: np.ndarray):
    """Fit per-class isotonic calibrators. Returns list of 3 fitted models (or None)."""
    from sklearn.isotonic import IsotonicRegression

    calibrators = []
    for k in range(proba_val.shape[1]):
        target = (y_val == k).astype(float)
        if target.sum() < 10 or len(np.unique(proba_val[:, k])) < 5:
            calibrators.append(None)  # too little signal to calibrate
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(proba_val[:, k], target)
        calibrators.append(iso)
    return calibrators


def apply_calibration(proba: np.ndarray, calibrators) -> np.ndarray:
    """Apply per-class calibrators and renormalise rows to sum to 1."""
    out = proba.copy()
    for k, cal in enumerate(calibrators):
        if cal is not None:
            out[:, k] = cal.predict(proba[:, k])
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return out / row_sum


def reliability_table(proba: np.ndarray, y: np.ndarray, class_k: int, bins: int = 10) -> list[dict]:
    """Binned predicted-vs-observed frequencies for reliability diagrams."""
    p = proba[:, class_k]
    hit = (y == class_k).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if mask.sum() > 0:
            rows.append({
                "bin_low": float(edges[i]), "bin_high": float(edges[i + 1]),
                "n": int(mask.sum()),
                "predicted": float(p[mask].mean()),
                "observed": float(hit[mask].mean()),
            })
    return rows
