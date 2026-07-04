"""Prompt 2: dukascopy CSV parsing and MT5 ingest pure-logic helpers."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from danalit.data import dukascopy_ingest, mt5_history_ingest


def test_parse_dukascopy_csv(tmp_path):
    # dukascopy-node schema: ms epoch timestamps, includes one bad OHLC row
    csv = tmp_path / "eurusd-m1-bid-2024.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1717401600000,1.0850,1.0852,1.0849,1.0851,120\n"
        "1717401660000,1.0851,1.0853,1.0850,1.0852,95\n"
        "1717401720000,1.0852,1.0840,1.0851,1.0853,80\n",  # high < low: dropped
        encoding="utf-8",
    )
    df = dukascopy_ingest.parse_csv(csv, spread_estimate=0.00012)
    assert len(df) == 2
    assert df["time_utc"].iloc[0] == pd.Timestamp("2024-06-03 08:00:00", tz="UTC")
    assert df["tick_volume"].tolist() == [120, 95]
    assert (df["source"] == "dukascopy").all()


def test_ingest_directory_matches_by_instrument_id(tmp_path):
    csv = tmp_path / "raw" / "xauusd-m1-bid-2024.csv"
    csv.parent.mkdir()
    csv.write_text(
        "timestamp,open,high,low,close,volume\n1717401600000,2300,2301,2299,2300.5,50\n",
        encoding="utf-8",
    )
    store = tmp_path / "store"
    n = dukascopy_ingest.ingest_directory(csv.parent, "XAUUSD", root=store)
    assert n == 1
    assert dukascopy_ingest.ingest_directory(csv.parent, "EURUSD", root=store) == 0


def test_round_offset_to_half_hour():
    f = mt5_history_ingest.round_offset_to_half_hour
    assert f(7205.0) == 7200        # UTC+2 with 5s clock skew
    assert f(-3595.0) == -3600      # UTC-1
    assert f(9000.0) == 9000        # UTC+2:30 stays
    assert f(200.0) == 0


def test_chunk_ranges_cover_interval_exactly():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 3, 15, tzinfo=timezone.utc)
    chunks = list(mt5_history_ingest.chunk_ranges(start, end, days=30))
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    for (a, b), (c, _) in zip(chunks, chunks[1:]):
        assert b == c  # contiguous


def test_fetch_m1_with_fake_copy_rates(tmp_path):
    # Simulated broker at UTC+2: server epoch = real epoch + 7200
    base_utc = pd.Timestamp("2024-06-03 08:00", tz="UTC")
    dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
             ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")]

    def fake_copy(s, e):
        rows = []
        for i in range(3):
            t = int(base_utc.timestamp()) + 7200 + i * 60
            rows.append((t, 1.1, 1.101, 1.099, 1.1005, 42, 12, 0))
        return np.array(rows, dtype=dtype) if s <= base_utc.to_pydatetime() <= e else np.array([], dtype=dtype)

    n = mt5_history_ingest.fetch_m1(
        "EURUSD", "EURUSD",
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 5, tzinfo=timezone.utc),
        root=tmp_path,
        _copy_rates=fake_copy,
        _offset=7200,
        point=0.00001,
    )
    assert n == 3
    from danalit.data import price_store

    out = price_store.read_bars("EURUSD", "M1", root=tmp_path)
    assert out["time_utc"].iloc[0] == base_utc          # server time converted back to UTC
    assert out["spread"].iloc[0] == 12 * 0.00001        # points -> price units
    assert (out["source"] == "broker").all()
