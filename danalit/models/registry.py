"""Model registry: every trained model persisted with full provenance.

Layout: models_store/{instrument}/{version}/
    fold_{k}.lgb          — LightGBM booster per walk-forward fold
    calibrator_{k}.pkl    — per-fold isotonic calibrators
    features.json         — exact feature list (order matters)
    metrics.json          — evaluation metrics
The model_registry table tracks versions; the champion pointer (is_champion)
is updated atomically — exactly one champion per instrument.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from danalit.config import load_config
from danalit.db import connect
from danalit.timeutil import utc_now_iso


def store_dir(instrument: str, version: str, base: Optional[Path] = None) -> Path:
    base = base or load_config().settings.paths.absolute("models_store")
    return base / instrument / version


@dataclass
class ModelBundle:
    instrument: str
    version: str
    boosters: list  # lightgbm.Booster per fold
    calibrators: list  # per fold: list of 3 calibrators or None
    feature_cols: list[str]
    metrics: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame, fold: int = -1) -> np.ndarray:
        """Calibrated [P_none, P_long, P_short] using the given fold's model
        (default: last fold — the one trained on the most recent data)."""
        booster = self.boosters[fold]
        raw = booster.predict(X[self.feature_cols].to_numpy())
        raw = np.atleast_2d(raw)
        cal = self.calibrators[fold] if self.calibrators else None
        if cal is None:
            return raw
        from danalit.models.calibrate import apply_calibration

        return apply_calibration(raw, cal)


def save_model(
    instrument: str,
    version: str,
    boosters: list,
    calibrators: list,
    feature_cols: list[str],
    metrics: dict,
    dataset_version: str,
    base: Optional[Path] = None,
    db_path: Optional[Path] = None,
    git_commit: str = "unknown",
) -> Path:
    d = store_dir(instrument, version, base)
    d.mkdir(parents=True, exist_ok=True)
    for k, booster in enumerate(boosters):
        booster.save_model(str(d / f"fold_{k}.lgb"))
    with open(d / "calibrators.pkl", "wb") as f:
        pickle.dump(calibrators, f)
    (d / "features.json").write_text(json.dumps(feature_cols), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    con = connect(db_path or load_config().settings.paths.db_path)
    try:
        with con:
            con.execute(
                """INSERT INTO model_registry
                   (instrument, version, dataset_version, path, metrics, git_commit, created_utc, is_champion)
                   VALUES (?,?,?,?,?,?,?,0)
                   ON CONFLICT (instrument, version) DO UPDATE SET
                     metrics=excluded.metrics, path=excluded.path""",
                (instrument, version, dataset_version, str(d),
                 json.dumps(metrics, default=str), git_commit, utc_now_iso()),
            )
    finally:
        con.close()
    return d


def load_model(instrument: str, version: str, base: Optional[Path] = None) -> ModelBundle:
    import lightgbm as lgb

    d = store_dir(instrument, version, base)
    boosters = []
    for k in range(64):
        p = d / f"fold_{k}.lgb"
        if not p.exists():
            break
        boosters.append(lgb.Booster(model_file=str(p)))
    if not boosters:
        raise FileNotFoundError(f"no models under {d}")
    with open(d / "calibrators.pkl", "rb") as f:
        calibrators = pickle.load(f)
    feature_cols = json.loads((d / "features.json").read_text(encoding="utf-8"))
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8")) if (d / "metrics.json").exists() else {}
    return ModelBundle(instrument, version, boosters, calibrators, feature_cols, metrics)


def set_champion(instrument: str, version: str, db_path: Optional[Path] = None) -> None:
    """Atomically move the champion pointer."""
    con = connect(db_path or load_config().settings.paths.db_path)
    try:
        with con:  # single transaction: clear + set
            con.execute("UPDATE model_registry SET is_champion=0 WHERE instrument=?", (instrument,))
            n = con.execute(
                "UPDATE model_registry SET is_champion=1 WHERE instrument=? AND version=?",
                (instrument, version),
            ).rowcount
            if n != 1:
                raise ValueError(f"version {version} not registered for {instrument}")
    finally:
        con.close()


def champion_version(instrument: str, db_path: Optional[Path] = None) -> Optional[str]:
    con = connect(db_path or load_config().settings.paths.db_path)
    try:
        row = con.execute(
            "SELECT version FROM model_registry WHERE instrument=? AND is_champion=1",
            (instrument,),
        ).fetchone()
        return row["version"] if row else None
    finally:
        con.close()


def load_latest(instrument: str, base: Optional[Path] = None,
                db_path: Optional[Path] = None) -> ModelBundle:
    """Champion if set, else most recently registered version."""
    con = connect(db_path or load_config().settings.paths.db_path)
    try:
        row = con.execute(
            """SELECT version FROM model_registry WHERE instrument=?
               ORDER BY is_champion DESC, created_utc DESC LIMIT 1""",
            (instrument,),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise LookupError(f"no registered models for {instrument}")
    return load_model(instrument, row["version"], base)
