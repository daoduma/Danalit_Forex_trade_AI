"""Parquet-backed price store.

Layout: data_store/prices/{instrument}/{timeframe}/{year}.parquet
Columns: time_utc (tz-aware UTC), open, high, low, close, tick_volume, spread[, source]
Bars are labelled by their OPEN time. M1 is the only ingested base timeframe;
M5/M15/H1/H4/D1 are produced by resample() and never ingested directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from danalit.config import load_config
from danalit.constants import PANDAS_FREQ, TIMEFRAMES

REQUIRED_COLUMNS = ["time_utc", "open", "high", "low", "close", "tick_volume", "spread"]

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "tick_volume": "sum",
    "spread": "mean",
}


def default_root() -> Path:
    return load_config().settings.paths.absolute("data_store") / "prices"


def _dir(instrument: str, timeframe: str, root: Optional[Path]) -> Path:
    root = root or default_root()
    return root / instrument / timeframe


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    df = df.copy()
    ts = pd.to_datetime(df["time_utc"])
    if ts.dt.tz is None:
        raise ValueError("time_utc must be timezone-aware (UTC)")
    df["time_utc"] = ts.dt.tz_convert("UTC")
    df = df.sort_values("time_utc").drop_duplicates(subset="time_utc", keep="last")
    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    )
    if bad.any():
        df = df[~bad]
    return df.reset_index(drop=True)


def write_bars(
    instrument: str,
    timeframe: str,
    df: pd.DataFrame,
    root: Optional[Path] = None,
) -> int:
    """Merge bars into the store (newest wins on duplicate timestamps). Returns rows written."""
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {timeframe}")
    df = _normalize(df)
    if df.empty:
        return 0
    out_dir = _dir(instrument, timeframe, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for year, chunk in df.groupby(df["time_utc"].dt.year):
        path = out_dir / f"{year}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            merged = pd.concat([existing, chunk], ignore_index=True)
            merged = merged.sort_values("time_utc").drop_duplicates(
                subset="time_utc", keep="last"
            )
        else:
            merged = chunk
        merged.reset_index(drop=True).to_parquet(path, index=False)
        total += len(chunk)
    return total


def read_bars(
    instrument: str,
    timeframe: str,
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """Read bars [start, end] inclusive; returns empty frame with schema if none."""
    d = _dir(instrument, timeframe, root)
    start = _ts(start)
    end = _ts(end)
    frames = []
    if d.exists():
        for path in sorted(d.glob("*.parquet")):
            year = int(path.stem)
            if start is not None and year < start.year:
                continue
            if end is not None and year > end.year:
                continue
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = pd.concat(frames, ignore_index=True).sort_values("time_utc")
    if start is not None:
        df = df[df["time_utc"] >= start]
    if end is not None:
        df = df[df["time_utc"] <= end]
    return df.reset_index(drop=True)


def resample(
    instrument: str,
    from_tf: str,
    to_tf: str,
    root: Optional[Path] = None,
    write: bool = True,
) -> pd.DataFrame:
    """Aggregate from_tf bars into to_tf bars (labelled by open time, UTC)."""
    if TIMEFRAMES[to_tf] <= TIMEFRAMES[from_tf]:
        raise ValueError(f"cannot resample {from_tf} -> {to_tf}")
    if TIMEFRAMES[to_tf] % TIMEFRAMES[from_tf] != 0:
        raise ValueError(f"{to_tf} is not a multiple of {from_tf}")
    df = read_bars(instrument, from_tf, root=root)
    if df.empty:
        return df
    out = resample_frame(df, to_tf)
    if write:
        write_bars(instrument, to_tf, out, root=root)
    return out


def resample_frame(df: pd.DataFrame, to_tf: str) -> pd.DataFrame:
    """Pure resampling of a bar frame to a higher timeframe."""
    agg_cols = {k: v for k, v in _AGG.items() if k in df.columns}
    out = (
        df.set_index("time_utc")
        .resample(PANDAS_FREQ[to_tf], label="left", closed="left")
        .agg(agg_cols)
    )
    out = out.dropna(subset=["open"]).reset_index()
    out["tick_volume"] = out["tick_volume"].fillna(0).astype("int64")
    return out


def build_all_timeframes(
    instrument: str,
    targets: Iterable[str] = ("M5", "M15", "H1", "H4", "D1"),
    root: Optional[Path] = None,
) -> dict[str, int]:
    """Resample M1 up into every target timeframe; returns rows per timeframe."""
    counts = {}
    for tf in targets:
        out = resample(instrument, "M1", tf, root=root, write=True)
        counts[tf] = len(out)
    return counts


def _ts(x) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
