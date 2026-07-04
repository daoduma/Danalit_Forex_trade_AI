"""Prompt 7: planted-pattern recovery, registry round-trip, calibration monotonicity."""

import numpy as np
import pandas as pd
import pytest

from danalit.models import registry
from danalit.models.calibrate import apply_calibration, fit_calibration
from danalit.models.train import train_fold

lgb = pytest.importorskip("lightgbm")


def planted_dataset(n=3000, seed=3):
    """x0 strongly determines the class; x1..x3 are noise."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, (n, 3))
    y = np.where(x0 > 0.6, 1, np.where(x0 < -0.6, 2, 0))
    X = pd.DataFrame(np.column_stack([x0, noise]), columns=[f"x{i}" for i in range(4)])
    return X, y


def test_training_recovers_planted_pattern():
    X, y = planted_dataset()
    cut = 2400
    booster = train_fold(X[:cut], y[:cut], X[cut:], y[cut:], num_boost_round=100)
    proba = booster.predict(X[cut:].to_numpy())
    acc = (proba.argmax(axis=1) == y[cut:]).mean()
    assert acc > 0.9, f"planted pattern not recovered (acc={acc:.3f})"


def test_calibration_monotonic_and_normalised():
    X, y = planted_dataset()
    cut = 2400
    booster = train_fold(X[:cut], y[:cut], X[cut:], y[cut:], num_boost_round=60)
    proba_val = booster.predict(X[cut:].to_numpy())
    cals = fit_calibration(proba_val, y[cut:])
    out = apply_calibration(proba_val, cals)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)
    # isotonic must be monotone non-decreasing in the raw score
    grid = np.linspace(0, 1, 50)
    for cal in cals:
        if cal is not None:
            pred = cal.predict(grid)
            assert (np.diff(pred) >= -1e-12).all()


def test_registry_roundtrip_and_champion(tmp_path):
    from danalit.db import init_db

    db = tmp_path / "t.db"
    init_db(db)
    X, y = planted_dataset(1200)
    booster = train_fold(X[:900], y[:900], X[900:], y[900:], num_boost_round=30)
    cals = [fit_calibration(booster.predict(X[900:].to_numpy()), y[900:])]

    registry.save_model("EURUSD", "vtest1", [booster], cals, list(X.columns),
                        {"m": 1}, "ds1", base=tmp_path, db_path=db)
    bundle = registry.load_model("EURUSD", "vtest1", base=tmp_path)
    assert bundle.feature_cols == list(X.columns)
    proba = bundle.predict_proba(X[:10])
    assert proba.shape == (10, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    assert registry.champion_version("EURUSD", db_path=db) is None
    registry.set_champion("EURUSD", "vtest1", db_path=db)
    assert registry.champion_version("EURUSD", db_path=db) == "vtest1"
    # champion pointer is exclusive
    registry.save_model("EURUSD", "vtest2", [booster], cals, list(X.columns),
                        {}, "ds1", base=tmp_path, db_path=db)
    registry.set_champion("EURUSD", "vtest2", db_path=db)
    assert registry.champion_version("EURUSD", db_path=db) == "vtest2"
    with pytest.raises(ValueError):
        registry.set_champion("EURUSD", "does-not-exist", db_path=db)
