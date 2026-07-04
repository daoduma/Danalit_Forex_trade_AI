"""Prompt 2: quality metrics — gap detection, weekend handling, sanity counters."""

import pandas as pd

from danalit.data import quality
from danalit.data.synthetic import generate_m1
from tests.test_price_store import make_bars


def test_gap_detection_flags_intraweek_gap_only():
    a = pd.date_range("2024-05-06 09:00", periods=60, freq="1min", tz="UTC")
    b = pd.date_range("2024-05-06 12:00", periods=60, freq="1min", tz="UTC")  # 2h intraweek gap
    df = make_bars(a.append(b))
    m = quality.analyze(df, timeframe_minutes=1)
    assert m["gap_count"] == 1
    assert m["gaps"][0]["minutes"] == 121.0


def test_weekend_break_not_reported_as_gap():
    fri = pd.date_range("2024-05-03 20:30", periods=30, freq="1min", tz="UTC")
    sun = pd.date_range("2024-05-05 22:00", periods=30, freq="1min", tz="UTC")
    df = make_bars(fri.append(sun))
    m = quality.analyze(df, timeframe_minutes=1)
    assert m["gap_count"] == 0


def test_counters_zero_volume_ohlc_weekend():
    times = pd.date_range("2024-05-04 10:00", periods=5, freq="1min", tz="UTC")  # Saturday
    df = make_bars(times)
    df.loc[0, "tick_volume"] = 0
    df.loc[1, "high"] = df.loc[1, "low"] - 1  # OHLC violation
    m = quality.analyze(df)
    assert m["zero_volume_bars"] == 1
    assert m["ohlc_violations"] == 1
    assert m["weekend_bars"] == 5


def test_synthetic_data_is_clean_and_report_writes(tmp_path):
    df = generate_m1("2024-06-03", "2024-06-10")
    m = quality.analyze(df)
    assert m["duplicates"] == 0
    assert m["ohlc_violations"] == 0
    assert m["weekend_bars"] == 0 or m["weekend_bars"] < 200  # Sunday open bars allowed
    path = quality.write_report("TEST", m, out_path=tmp_path / "q.md")
    text = path.read_text(encoding="utf-8")
    assert "Data quality — TEST" in text
    assert "n_bars" in text
