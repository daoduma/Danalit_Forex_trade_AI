"""Pull M1 history from the local MT5 terminal into the price store.

Broker-timezone handling: MT5 `copy_rates_*` returns bar times as epoch seconds
whose calendar rendering matches the BROKER SERVER clock, not UTC. We estimate
the server's UTC offset by comparing the latest tick time against the local UTC
clock and rounding to the nearest 30 minutes (all real broker offsets are whole
or half hours), then subtract it. The offset is logged; verify it once against
your broker's published server timezone.

Requires a running, logged-in MT5 terminal. The MetaTrader5 package is
lazy-imported so the rest of the system (and the test suite) works without it.
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from danalit.config import load_config
from danalit.data import price_store
from danalit.logging_setup import setup_logging

log = setup_logging("mt5_history_ingest")

CHUNK_DAYS = 30


def _mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "The MetaTrader5 package is not installed (pip install MetaTrader5). "
            "It requires Windows and a running MT5 terminal."
        ) from e
    return mt5


def round_offset_to_half_hour(seconds: float) -> int:
    """Round a raw clock delta to the nearest 30 minutes (broker offsets are :00/:30)."""
    half_hours = round(seconds / 1800.0)
    return int(half_hours * 1800)


def estimate_server_utc_offset(mt5, symbol: str) -> int:
    """Server UTC offset in seconds, from the latest tick vs the local UTC clock."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"no tick for {symbol}; is the symbol visible and market open?")
    return round_offset_to_half_hour(tick.time - _time.time())


def chunk_ranges(start: datetime, end: datetime, days: int = CHUNK_DAYS):
    """Yield (chunk_start, chunk_end) pairs covering [start, end], newest last."""
    cur = start
    step = timedelta(days=days)
    while cur < end:
        nxt = min(cur + step, end)
        yield cur, nxt
        cur = nxt


def rates_to_frame(rates, server_offset_sec: int, spread_points_to_price: float) -> pd.DataFrame:
    """Convert an MT5 rates array to the canonical bar schema (UTC)."""
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    df["time_utc"] = pd.to_datetime(df["time"] - server_offset_sec, unit="s", utc=True)
    out = pd.DataFrame(
        {
            "time_utc": df["time_utc"],
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "tick_volume": df["tick_volume"].astype("int64"),
            "spread": df["spread"].astype(float) * spread_points_to_price,
        }
    )
    out["source"] = "broker"
    return out


def fetch_m1(
    instrument: str,
    broker_symbol: str,
    start: datetime,
    end: Optional[datetime] = None,
    root: Optional[Path] = None,
    _copy_rates: Optional[Callable] = None,  # test hook
    _offset: Optional[int] = None,  # test hook
    point: Optional[float] = None,
) -> int:
    """Fetch M1 bars in chunks and write them to the store. Returns rows written."""
    mt5 = None
    if _copy_rates is None:
        mt5 = _mt5()
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()} — is the terminal running?")
        info = mt5.symbol_info(broker_symbol)
        if info is None:
            raise RuntimeError(f"symbol {broker_symbol} not found at broker")
        mt5.symbol_select(broker_symbol, True)
        point = info.point
        _copy_rates = lambda s, e: mt5.copy_rates_range(  # noqa: E731
            broker_symbol, mt5.TIMEFRAME_M1, s, e
        )
        _offset = estimate_server_utc_offset(mt5, broker_symbol)
        log.info("%s: server UTC offset estimated at %+d min", broker_symbol, _offset // 60)

    end = end or datetime.now(timezone.utc)
    total = 0
    empty_streak = 0
    for c_start, c_end in chunk_ranges(start, end):
        rates = _copy_rates(c_start, c_end)
        if rates is None or len(rates) == 0:
            empty_streak += 1
            if empty_streak >= 6:  # ~6 months of nothing: history exhausted
                log.info("%s: no data for 6 consecutive chunks — assuming history starts later", instrument)
                empty_streak = 0
            continue
        empty_streak = 0
        df = rates_to_frame(rates, _offset or 0, point or 0.0)
        total += price_store.write_bars(instrument, "M1", df, root=root)
    log.info("%s: wrote %d M1 bars from broker history", instrument, total)
    return total


def ingest_all(start: datetime, root: Optional[Path] = None) -> dict[str, int]:
    """Fetch broker history for every enabled instrument in instruments.yaml."""
    cfg = load_config()
    counts = {}
    for name in cfg.enabled_instruments():
        inst = cfg.instruments[name]
        counts[name] = fetch_m1(name, inst.broker_symbol, start, root=root)
    return counts
