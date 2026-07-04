"""Prompt 4: technical features — NO-LOOKAHEAD proof, as-of joins, completeness."""

import numpy as np
import pandas as pd
import pytest

from danalit.data import price_store
from danalit.data.synthetic import generate_m1
from danalit.features import technical


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Six months of synthetic EURUSD, all timeframes, in an isolated store."""
    root = tmp_path_factory.mktemp("prices")
    m1 = generate_m1("2024-01-01", "2024-07-01", s0=1.10, seed=42)
    price_store.write_bars("EURUSD", "M1", m1, root=root)
    price_store.build_all_timeframes("EURUSD", root=root)
    return root


@pytest.fixture(scope="module")
def features(store):
    return technical.build_features("EURUSD", root=store)


def test_no_lookahead_truncation_proof(store, features):
    """THE correctness test: features at T computed on full history must equal
    features at T computed on history truncated at T's close."""
    t_cut = features.index[len(features) // 2]
    cut_close = t_cut + pd.Timedelta(minutes=15)

    # Rebuild an isolated truncated store
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        m1 = price_store.read_bars("EURUSD", "M1", root=store)
        m1_trunc = m1[m1["time_utc"] < cut_close]  # only data known by T's close
        price_store.write_bars("EURUSD", "M1", m1_trunc, root=tmp)
        price_store.build_all_timeframes("EURUSD", root=tmp)
        trunc = technical.build_features("EURUSD", root=tmp)

    assert t_cut in trunc.index
    full_row = features.loc[t_cut]
    trunc_row = trunc.loc[t_cut]
    diffs = (full_row - trunc_row).abs()
    offenders = diffs[diffs > 1e-9]
    assert offenders.empty, f"lookahead detected in: {offenders.index.tolist()}"


def test_higher_tf_asof_join_no_peeking(features):
    """H4 features may only change at M15 rows whose open >= an H4 bar close.

    H4 bars close at 00/04/08/12/16/20 UTC — h4_rsi must be constant across
    every M15 row strictly inside an H4 window and change only at the boundary.
    """
    h4rsi = features["h4_rsi"]
    changes = h4rsi.loc[h4rsi.diff().abs() > 1e-12]
    assert len(changes) > 10  # it does change...
    # A change is legitimate at an H4 boundary, or at the first bar after a
    # session gap (weekend reopen) where merge_asof catches up.
    idx = features.index
    pos = {t: i for i, t in enumerate(idx)}
    bad = []
    for t in changes.index:
        at_boundary = (t.hour % 4 == 0) and t.minute == 0
        i = pos[t]
        after_gap = i > 0 and (t - idx[i - 1]) > pd.Timedelta(minutes=15)
        if not (at_boundary or after_gap):
            bad.append(t)
    assert bad == [], f"h4_rsi changed inside a forming H4 bar at {bad[:5]}"


def test_feature_matrix_complete_and_registered(features):
    assert len(features) > 5000
    assert features.isna().sum().sum() == 0, "NaNs after warmup drop"
    unregistered = [c for c in features.columns if c not in technical.FEATURE_REGISTRY]
    assert unregistered == [], f"unregistered features: {unregistered}"
    assert len(features.columns) >= 45


def test_time_encodings_sane(features):
    assert features["hod_sin"].between(-1, 1).all()
    assert ((features["sess_london"] == 0) | (features["sess_london"] == 1)).all()
    lon = features[features["sess_london"] > 0]
    assert (lon.index.hour >= 7).all() and (lon.index.hour < 16).all()
    # Friday 20:45 bar is 1 bar from the weekend
    fri = features[(features.index.weekday == 4) & (features.index.hour == 20)
                   & (features.index.minute == 45)]
    if len(fri):
        assert (fri["bars_to_weekend"] == 1.0).all()


def test_indicator_primitives_known_values():
    close = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    e = technical.ema(close, 3)
    assert e.iloc[-1] == pytest.approx(9.0, abs=0.15)
    r = technical.rsi(pd.Series(np.linspace(1, 2, 30)), 14)
    assert r.iloc[-1] > 95  # monotonic rise -> RSI ~ 100
    r2 = technical.rsi(pd.Series(np.linspace(2, 1, 30)), 14)
    assert r2.iloc[-1] < 5


def test_rolling_percentile_bounds():
    s = pd.Series(np.random.default_rng(1).random(500))
    p = technical.rolling_percentile(s, 100)
    assert p.dropna().between(0, 1).all()
