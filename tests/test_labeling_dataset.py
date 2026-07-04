"""Prompt 6: triple-barrier correctness on scripted paths; purge/embargo; determinism."""

import numpy as np
import pandas as pd
import pytest

from danalit.features import labeling
from danalit.features.dataset import Fold, dataset_hash, split_fold
from danalit.features.labeling import LABEL_LONG, LABEL_NONE, LABEL_SHORT, triple_barrier


def path_bars(closes, start="2024-01-02 00:00", wick=0.0):
    """Bars from a scripted close path; open = previous close; symmetric wicks."""
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "time_utc": pd.date_range(start, periods=len(closes), freq="15min", tz="UTC"),
        "open": opens,
        "high": np.maximum(opens, closes) + wick,
        "low": np.minimum(opens, closes) - wick,
        "close": closes,
        "tick_volume": 10,
        "spread": 0.0,
    })


def warmup(level=100.0, n=20):
    """Flat warmup section so ATR(14) is defined and equals the wick range."""
    return [level] * n


def test_tp_hit_first_labels_long():
    # flat 100 (ATR from wick=0.5 -> TR=1.0), then steady rise
    closes = warmup() + list(np.linspace(100, 110, 30))
    bars = path_bars(closes, wick=0.5)
    out = triple_barrier(bars, spread=0.0, k_tp=2.0, k_sl=1.0, horizon=20)
    t_dec = bars["time_utc"].iloc[19]  # last flat bar: entry at first rising bar's open
    row = out.loc[t_dec]
    assert row["label"] == LABEL_LONG
    assert row["label_long"] == 1
    assert row["ret_long"] == pytest.approx(row["tp_long"] - row["entry_long"])
    assert row["hit_bar_long"] < 20


def test_sl_hit_first_labels_short():
    closes = warmup() + list(np.linspace(100, 90, 30))
    bars = path_bars(closes, wick=0.5)
    out = triple_barrier(bars, spread=0.0, k_tp=2.0, k_sl=1.0, horizon=20)
    row = out.loc[bars["time_utc"].iloc[19]]
    assert row["label"] == LABEL_SHORT
    assert row["label_long"] == -1  # long framing stopped out
    assert row["label_short"] == 1


def test_gap_through_barrier_fills_at_gapped_open():
    # flat, then bar 20 OPENS far below the long SL (weekend-style gap)
    closes = warmup(100.0, 40)
    bars = path_bars(closes, wick=0.5)
    bars.loc[20:, ["open", "high", "low", "close"]] = [90.0, 90.5, 89.5, 90.0]
    out = triple_barrier(bars, spread=0.0, k_tp=2.0, k_sl=1.0, horizon=15)
    row = out.loc[bars["time_utc"].iloc[18]]  # entry at bar 19 open (=100), gap at bar 20
    assert row["label_long"] == -1
    # exit at the gapped open (90), NOT the barrier price -> loss far exceeds k_sl*ATR
    assert row["ret_long"] == pytest.approx(90.0 - row["entry_long"])
    assert row["ret_long"] < (row["sl_long"] - row["entry_long"]) - 1


def test_pessimistic_rule_both_barriers_in_one_bar():
    # ATR ~1 from wick; entry bar has a huge range covering both barriers
    closes = warmup() + [100.0] * 25
    bars = path_bars(closes, wick=0.5)
    i = 21  # bar after entry
    bars.loc[i, "high"] = 105.0
    bars.loc[i, "low"] = 95.0
    out = triple_barrier(bars, spread=0.0, k_tp=2.0, k_sl=1.0, horizon=15)
    row = out.loc[bars["time_utc"].iloc[19]]
    assert row["label_long"] == -1  # SL assumed first
    assert row["label_short"] == -1  # short SL (ask high) also assumed first
    assert row["label"] == LABEL_NONE


def test_timeout_dead_zone():
    closes = warmup(100.0, 40)  # perfectly flat forever
    bars = path_bars(closes, wick=0.5)
    out = triple_barrier(bars, spread=0.0, k_tp=5.0, k_sl=5.0, horizon=10, dead_zone_atr=0.25)
    row = out.iloc[15]
    assert row["label"] == LABEL_NONE
    assert row["hit_bar_long"] == 10  # timed out at horizon


def test_spread_is_paid():
    closes = warmup() + list(np.linspace(100, 110, 30))
    bars = path_bars(closes, wick=0.5)
    out = triple_barrier(bars, spread=0.5, k_tp=2.0, k_sl=1.0, horizon=20)
    row = out.loc[bars["time_utc"].iloc[19]]
    no_spread = triple_barrier(bars, spread=0.0, k_tp=2.0, k_sl=1.0, horizon=20).loc[row.name]
    assert row["entry_long"] == pytest.approx(no_spread["entry_long"] + 0.5)


def test_purge_and_embargo():
    idx = pd.date_range("2024-01-01", "2024-12-31", freq="1D", tz="UTC")
    df = pd.DataFrame({"x": np.arange(len(idx)), "label": 0}, index=idx)
    fold = Fold(("2024-01-01", "2024-07-01"), ("2024-07-01", "2024-10-01"),
                ("2024-10-01", "2025-01-01"))
    span = pd.Timedelta(days=1)
    embargo = pd.Timedelta(days=5)
    s = split_fold(df, fold, span, embargo)
    # last train sample: T + 1d <= Jul1 - 5d  ->  T <= Jun 25
    assert s["train"].index.max() == pd.Timestamp("2024-06-25", tz="UTC")
    # last validate sample: T + 1d <= Oct1 - 5d -> T <= Sep 25
    assert s["validate"].index.max() == pd.Timestamp("2024-09-25", tz="UTC")
    assert s["validate"].index.min() == pd.Timestamp("2024-07-01", tz="UTC")
    assert s["test"].index.min() == pd.Timestamp("2024-10-01", tz="UTC")
    # no overlap anywhere
    assert not set(s["train"].index) & set(s["validate"].index)
    assert not set(s["validate"].index) & set(s["test"].index)


def test_dataset_hash_deterministic():
    idx = pd.date_range("2024-01-01", periods=100, freq="1D", tz="UTC")
    df = pd.DataFrame({"x": np.arange(100.0), "label": 0}, index=idx)
    fold = Fold(("2024-01-01", "2024-03-01"), ("2024-03-01", "2024-04-01"),
                ("2024-04-01", "2024-05-01"))
    s1 = {"fold_0": split_fold(df, fold, pd.Timedelta(days=1), pd.Timedelta(days=2))}
    s2 = {"fold_0": split_fold(df.copy(), fold, pd.Timedelta(days=1), pd.Timedelta(days=2))}
    assert dataset_hash(s1, {"p": 1}) == dataset_hash(s2, {"p": 1})
    df2 = df.copy()
    df2.iloc[0, 0] = 999.0
    s3 = {"fold_0": split_fold(df2, fold, pd.Timedelta(days=1), pd.Timedelta(days=2))}
    assert dataset_hash(s1, {"p": 1}) != dataset_hash(s3, {"p": 1})


def test_label_span():
    assert labeling.label_span(96) == pd.Timedelta(minutes=97 * 15)
