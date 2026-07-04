"""Prompt 2: price store — roundtrip, dedup, resampling correctness, UTC/DST."""

import numpy as np
import pandas as pd
import pytest

from danalit.data import price_store


def make_bars(times, base=1.0):
    n = len(times)
    close = base + np.arange(n) * 0.001
    return pd.DataFrame(
        {
            "time_utc": pd.DatetimeIndex(times, tz="UTC"),
            "open": close - 0.0005,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "tick_volume": np.arange(1, n + 1, dtype="int64"),
            "spread": 0.0001,
        }
    )


def test_write_read_roundtrip_and_year_split(tmp_path):
    times = pd.date_range("2023-12-31 23:58", "2024-01-01 00:02", freq="1min")
    df = make_bars(times)
    n = price_store.write_bars("EURUSD", "M1", df, root=tmp_path)
    assert n == 5
    assert (tmp_path / "EURUSD/M1/2023.parquet").exists()
    assert (tmp_path / "EURUSD/M1/2024.parquet").exists()

    out = price_store.read_bars("EURUSD", "M1", root=tmp_path)
    assert len(out) == 5
    assert str(out["time_utc"].dt.tz) == "UTC"
    pd.testing.assert_series_equal(out["close"], df["close"], check_names=False)

    part = price_store.read_bars("EURUSD", "M1", start="2024-01-01", root=tmp_path)
    assert len(part) == 3


def test_duplicate_timestamps_newest_wins(tmp_path):
    times = pd.date_range("2024-03-01 10:00", periods=3, freq="1min")
    df1 = make_bars(times, base=1.0)
    price_store.write_bars("EURUSD", "M1", df1, root=tmp_path)
    df2 = make_bars(times[1:], base=2.0)  # overlaps last two bars
    price_store.write_bars("EURUSD", "M1", df2, root=tmp_path)

    out = price_store.read_bars("EURUSD", "M1", root=tmp_path)
    assert len(out) == 3
    assert out["close"].iloc[0] == pytest.approx(1.0)
    assert out["close"].iloc[1] == pytest.approx(2.0)  # replaced by newer write


def test_naive_timestamps_rejected(tmp_path):
    df = make_bars(pd.date_range("2024-01-01", periods=2, freq="1min"))
    df["time_utc"] = df["time_utc"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        price_store.write_bars("EURUSD", "M1", df, root=tmp_path)


def test_resample_m1_to_m15_known_values(tmp_path):
    # 30 known M1 bars starting exactly on a quarter hour -> exactly 2 M15 bars
    times = pd.date_range("2024-05-06 09:00", periods=30, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "time_utc": times,
            "open": np.arange(30, dtype=float) + 100,
            "high": np.arange(30, dtype=float) + 100.5,
            "low": np.arange(30, dtype=float) + 99.5,
            "close": np.arange(30, dtype=float) + 100.2,
            "tick_volume": np.ones(30, dtype="int64"),
            "spread": 0.1,
        }
    )
    price_store.write_bars("XAUUSD", "M1", df, root=tmp_path)
    out = price_store.resample("XAUUSD", "M1", "M15", root=tmp_path)

    assert len(out) == 2
    b0 = out.iloc[0]
    assert b0["time_utc"] == pd.Timestamp("2024-05-06 09:00", tz="UTC")  # labelled by OPEN
    assert b0["open"] == 100.0          # first minute's open
    assert b0["high"] == 114.5          # max of minutes 0..14
    assert b0["low"] == 99.5
    assert b0["close"] == pytest.approx(114.2)  # minute 14's close
    assert b0["tick_volume"] == 15
    # resampled bars were persisted
    assert (tmp_path / "XAUUSD/M15/2024.parquet").exists()


def test_resample_skips_empty_weekend_bins(tmp_path):
    fri = pd.date_range("2024-05-03 20:00", periods=15, freq="1min", tz="UTC")
    mon = pd.date_range("2024-05-06 00:00", periods=15, freq="1min", tz="UTC")
    df = make_bars(fri.append(mon))
    price_store.write_bars("EURUSD", "M1", df, root=tmp_path)
    out = price_store.resample("EURUSD", "M1", "M15", root=tmp_path)
    assert len(out) == 2  # no NaN weekend bars in between


def test_utc_continuity_across_dst_change(tmp_path):
    # US DST change 2024-03-10: UTC data must remain strictly monotonic 1-minute steps
    times = pd.date_range("2024-03-10 06:55", periods=10, freq="1min", tz="UTC")
    price_store.write_bars("EURUSD", "M1", make_bars(times), root=tmp_path)
    out = price_store.read_bars("EURUSD", "M1", root=tmp_path)
    deltas = out["time_utc"].diff().dropna().dt.total_seconds()
    assert (deltas == 60).all()


def test_invalid_resample_direction_rejected(tmp_path):
    with pytest.raises(ValueError):
        price_store.resample("EURUSD", "M15", "M1", root=tmp_path)
