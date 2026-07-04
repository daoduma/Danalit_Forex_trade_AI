"""End-to-end price data build: ingest -> merge -> M1 Parquet -> resample -> quality report.

Usage:
    python scripts/build_price_data.py --dukascopy-dir data_raw/dukascopy          # ingest CSVs
    python scripts/build_price_data.py --with-mt5 --since 2016-01-01               # broker history
    python scripts/build_price_data.py --synthetic                                  # demo/test data

Merge policy: Dukascopy is the deep base; broker bars are written AFTER Dukascopy
so they win on overlapping timestamps (write_bars keeps the newest write).

One-time Dukascopy download (requires Node.js) — see danalit/data/dukascopy_ingest.py:
    npx dukascopy-node -i eurusd       -from 2014-01-01 -to 2026-07-01 -t m1 -f csv -dir data_raw/dukascopy -p bid -v true
    npx dukascopy-node -i xauusd       -from 2014-01-01 -to 2026-07-01 -t m1 -f csv -dir data_raw/dukascopy -p bid -v true
    npx dukascopy-node -i usatecidxusd -from 2014-01-01 -to 2026-07-01 -t m1 -f csv -dir data_raw/dukascopy -p bid -v true
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danalit.config import load_config  # noqa: E402
from danalit.data import price_store, quality  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="all")
    ap.add_argument("--dukascopy-dir", type=Path, default=None)
    ap.add_argument("--with-mt5", action="store_true", help="also pull broker MT5 history")
    ap.add_argument("--since", default="2016-01-01", help="MT5 history start date")
    ap.add_argument("--synthetic", action="store_true", help="generate synthetic demo data instead")
    args = ap.parse_args()

    cfg = load_config()
    instruments = cfg.enabled_instruments() if args.instrument == "all" else [args.instrument]

    for name in instruments:
        inst = cfg.instruments[name]
        print(f"=== {name} ===")

        if args.synthetic:
            from danalit.data.synthetic import generate_m1

            df = generate_m1("2024-01-01", "2026-01-01", s0=_s0(name), seed=hash(name) % 2**32)
            n = price_store.write_bars(name, "M1", df)
            print(f"  synthetic M1 bars written: {n}")

        if args.dukascopy_dir:
            from danalit.data import dukascopy_ingest

            n = dukascopy_ingest.ingest_directory(
                args.dukascopy_dir, name,
                spread_estimate=inst.spread_estimate_pips * inst.pip_size,
            )
            print(f"  dukascopy M1 bars written: {n}")

        if args.with_mt5:
            from danalit.data import mt5_history_ingest

            start = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
            n = mt5_history_ingest.fetch_m1(name, inst.broker_symbol, start)
            print(f"  broker M1 bars written: {n}")

        counts = price_store.build_all_timeframes(name)
        print("  resampled:", ", ".join(f"{tf}={n}" for tf, n in counts.items()))

        m1 = price_store.read_bars(name, "M1")
        if m1.empty:
            print("  no M1 data yet — skipping quality report")
            continue
        metrics = quality.analyze(m1, timeframe_minutes=1)
        path = quality.write_report(name, metrics)
        print(f"  quality report: {path} (bars={metrics['n_bars']}, gaps={metrics['gap_count']})")
    return 0


def _s0(name: str) -> float:
    return {"EURUSD": 1.10, "XAUUSD": 2300.0, "US100": 18500.0}.get(name, 100.0)


if __name__ == "__main__":
    raise SystemExit(main())
