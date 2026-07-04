"""LightGBM signal-model training over purged walk-forward folds.

The model for fold k trains on fold k's train split with early stopping on the
validation split — it never sees its own test period in any form. Shallow
trees, high min_child_samples and feature subsampling keep the model honest on
noisy financial data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from danalit.features.dataset import _git_commit, load_dataset
from danalit.logging_setup import setup_logging
from danalit.models import registry
from danalit.models.calibrate import fit_calibration
from danalit.timeutil import utc_now

log = setup_logging("train")

DEFAULT_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 7,
}


def class_weights(y: np.ndarray) -> np.ndarray:
    """Balanced sample weights for the 3-class problem."""
    counts = np.bincount(y, minlength=3).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (3 * counts)
    return w[y]


def train_fold(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    params: Optional[dict] = None,
    num_boost_round: int = 400,
    early_stopping_rounds: int = 30,
):
    """Train one fold's booster with early stopping on its validation split."""
    import lightgbm as lgb

    p = {**DEFAULT_PARAMS, **(params or {})}
    dtrain = lgb.Dataset(X_tr.to_numpy(), label=y_tr, weight=class_weights(y_tr))
    dval = lgb.Dataset(X_val.to_numpy(), label=y_val, reference=dtrain)
    booster = lgb.train(
        p, dtrain, num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return booster


def train_instrument(
    instrument: str,
    dataset_version: str,
    params: Optional[dict] = None,
    version: Optional[str] = None,
    dataset_dir: Optional[Path] = None,
    models_base: Optional[Path] = None,
    db_path: Optional[Path] = None,
    save: bool = True,
):
    """Train + calibrate every fold of a dataset; register the result.

    Returns (version, boosters, calibrators, feature_cols, frames, manifest).
    """
    frames, manifest = load_dataset(instrument, dataset_version, base_dir=dataset_dir)
    feature_cols = manifest["feature_cols"]
    boosters, calibrators = [], []
    for fname in sorted(frames):
        tr, va = frames[fname]["train"], frames[fname]["validate"]
        if len(tr) < 100 or len(va) < 50:
            raise ValueError(f"{instrument}/{fname}: too little data (train={len(tr)}, val={len(va)})")
        booster = train_fold(
            tr[feature_cols], tr["label"].to_numpy(),
            va[feature_cols], va["label"].to_numpy(),
            params=params,
        )
        proba_val = np.atleast_2d(booster.predict(va[feature_cols].to_numpy()))
        calibrators.append(fit_calibration(proba_val, va["label"].to_numpy()))
        boosters.append(booster)
        log.info("%s %s: trained %d trees", instrument, fname, booster.num_trees())

    version = version or f"m{utc_now().strftime('%Y%m%d_%H%M%S')}"
    if save:
        registry.save_model(
            instrument, version, boosters, calibrators, feature_cols,
            metrics={}, dataset_version=dataset_version,
            base=models_base, db_path=db_path, git_commit=_git_commit(),
        )
    return version, boosters, calibrators, feature_cols, frames, manifest
