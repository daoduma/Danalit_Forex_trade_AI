"""Ingest CSVs produced by the open-source `dukascopy-node` CLI.

We do NOT reimplement the downloader — run the CLI once per instrument, then
point this module at its output directory. One-time download commands
(requires Node.js; run from the repo root, output lands in data_raw/dukascopy):

    npx dukascopy-node -i eurusd       -from 2014-01-01 -to 2026-07-01 -t m1 -f csv -dir data_raw/dukascopy -p bid -v true
    npx dukascopy-node -i xauusd       -from 2014-01-01 -to 2026-07-01 -t m1 -f csv -dir data_raw/dukascopy -p bid -v true
    npx dukascopy-node -i usatecidxusd -from 2014-01-01 -to 2026-07-01 -t m1 -f csv -dir data_raw/dukascopy -p bid -v true

CSV schema (dukascopy-node): timestamp (ms epoch, UTC), open, high, low, close, volume.
Filenames contain the instrument id, e.g. eurusd-m1-bid-2014-01-01-2026-07-01.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from danalit.data import price_store
from danalit.logging_setup import setup_logging

log = setup_logging("dukascopy_ingest")

# canonical instrument -> dukascopy-node instrument id (used to match filenames)
DUKASCOPY_IDS = {
    "EURUSD": "eurusd",
    "XAUUSD": "xauusd",
    "US100": "usatecidxusd",
}


def parse_csv(path: Path, spread_estimate: float = float("nan")) -> pd.DataFrame:
    """Parse one dukascopy-node CSV into the canonical bar schema."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    ts_col = cols.get("timestamp") or cols.get("time")
    if ts_col is None:
        raise ValueError(f"{path}: no timestamp column (columns: {list(df.columns)})")
    out = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(df[ts_col], unit="ms", utc=True),
            "open": df[cols["open"]].astype(float),
            "high": df[cols["high"]].astype(float),
            "low": df[cols["low"]].astype(float),
            "close": df[cols["close"]].astype(float),
            "tick_volume": df[cols["volume"]].fillna(0).astype("int64")
            if "volume" in cols
            else 0,
        }
    )
    out["spread"] = spread_estimate
    out["source"] = "dukascopy"
    n0 = len(out)
    bad = (out["high"] < out[["open", "close", "low"]].max(axis=1)) | (
        out["low"] > out[["open", "close", "high"]].min(axis=1)
    )
    out = out[~bad]
    if n0 - len(out):
        log.warning("%s: dropped %d rows failing OHLC sanity", path.name, n0 - len(out))
    return out.reset_index(drop=True)


def ingest_directory(
    csv_dir: Path,
    instrument: str,
    root: Optional[Path] = None,
    spread_estimate: float = float("nan"),
) -> int:
    """Ingest every CSV in csv_dir matching the instrument's dukascopy id. Returns rows written."""
    duk_id = DUKASCOPY_IDS.get(instrument)
    if duk_id is None:
        raise ValueError(f"no dukascopy id mapped for {instrument}")
    files = sorted(p for p in Path(csv_dir).glob("*.csv") if duk_id in p.name.lower())
    if not files:
        log.warning("no CSVs matching '%s' under %s", duk_id, csv_dir)
        return 0
    total = 0
    for path in files:
        df = parse_csv(path, spread_estimate=spread_estimate)
        n = price_store.write_bars(instrument, "M1", df, root=root)
        log.info("%s: ingested %d M1 bars for %s", path.name, n, instrument)
        total += n
    return total
