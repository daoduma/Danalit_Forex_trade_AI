"""Optuna hyperparameter search — validation splits ONLY, logged, resumable.

HARD RULES enforced here: the objective only ever evaluates on the fold's
validation split (the FoldIsolationGuard trips on any test-period timestamp);
every trial is logged to SQLite so the search is auditable and resumable; test
periods are touched exactly once, by run_walkforward with the final config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from danalit.config import load_config
from danalit.db import connect
from danalit.logging_setup import setup_logging
from danalit.models.calibrate import apply_calibration, fit_calibration
from danalit.models.train import train_fold
from danalit.timeutil import utc_now_iso

log = setup_logging("tuning")


def _val_expectancy(booster, calibrators, va: pd.DataFrame, feature_cols: list[str],
                    tau: float) -> tuple[float, int]:
    """Expectancy (ATR units, net of label costs) of thresholded decisions on validation."""
    proba = np.atleast_2d(booster.predict(va[feature_cols].to_numpy()))
    proba = apply_calibration(proba, calibrators)
    p_long, p_short = proba[:, 1], proba[:, 2]
    conf = np.maximum(p_long, p_short)
    act = conf > tau
    n = int(act.sum())
    if n < 20:
        return -1.0, n  # too few signals to trust; penalise
    direction = np.where(p_long >= p_short, 1, 2)
    ret = np.where(direction == 1, va["ret_long"].to_numpy(), va["ret_short"].to_numpy())
    atr = va["atr"].to_numpy()
    ret_atr = np.divide(ret, atr, out=np.zeros_like(ret), where=atr > 0)
    return float(ret_atr[act].mean()), n


def tune_fold(
    instrument: str,
    tr: pd.DataFrame,
    va: pd.DataFrame,
    feature_cols: list[str],
    guard=None,
    n_trials: int = 20,
    study_name: str = "study",
    db_path: Optional[Path] = None,
    storage_dir: Optional[Path] = None,
) -> dict:
    """Search (tau, lgbm depth/leaves/min_child) on train->validate only.

    Returns {'tau', 'lgbm_params', 'value', 'n_trials'}. Resumable: the Optuna
    study persists to models_store/optuna/{study_name}.db; trials are also
    logged to the danalit optuna_trials table for auditability.
    """
    import optuna

    if guard is not None:
        guard.check(tr, "tuning train split")
        guard.check(va, "tuning validate split")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = load_config()
    storage_dir = storage_dir or cfg.settings.paths.absolute("models_store") / "optuna"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(storage_dir / (study_name + '.db')).as_posix()}"
    study = optuna.create_study(direction="maximize", study_name=study_name,
                                storage=storage, load_if_exists=True)
    journal_db = db_path or cfg.settings.paths.db_path

    y_tr, y_va = tr["label"].to_numpy(), va["label"].to_numpy()

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63]),
            "min_child_samples": trial.suggest_categorical("min_child_samples", [100, 200, 400]),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        }
        tau = trial.suggest_float("tau", 0.45, 0.70, step=0.05)
        booster = train_fold(tr[feature_cols], y_tr, va[feature_cols], y_va,
                             params=params, num_boost_round=200, early_stopping_rounds=20)
        cal = fit_calibration(
            np.atleast_2d(booster.predict(va[feature_cols].to_numpy())), y_va)
        value, n_signals = _val_expectancy(booster, cal, va, feature_cols, tau)
        trial.set_user_attr("n_signals", n_signals)
        con = connect(journal_db)
        try:
            with con:
                con.execute(
                    "INSERT INTO optuna_trials (study, trial, params, value, state, ts_utc)"
                    " VALUES (?,?,?,?,?,?)",
                    (study_name, trial.number,
                     json.dumps({**params, "tau": tau}), value, "COMPLETE", utc_now_iso()),
                )
        finally:
            con.close()
        return value

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    lgbm_params = {k: v for k, v in best.params.items() if k != "tau"}
    log.info("%s: best val expectancy %.4f ATR at tau=%.2f (%d total trials)",
             study_name, best.value, best.params["tau"], len(study.trials))
    return {"tau": best.params["tau"], "lgbm_params": lgbm_params,
            "value": best.value, "n_trials": len(study.trials)}
