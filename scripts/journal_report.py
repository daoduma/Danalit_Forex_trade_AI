"""Forward-test analytics report for a date range.

    python scripts/journal_report.py [--start 2026-07-01] [--end 2026-10-01]
    python scripts/journal_report.py --export     # journal -> Parquet snapshot
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danalit.journal import analytics  # noqa: E402
from danalit.timeutil import utc_now  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()

    if args.export:
        for p in analytics.export_parquet():
            print("exported", p)
        return 0

    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp(utc_now())
    start = pd.Timestamp(args.start, tz="UTC") if args.start else end - pd.Timedelta(weeks=12)
    stats = analytics.gather(None, start, end)
    path = analytics.write_report(stats)
    print(f"report: {path}")
    for c in stats["checklist"]:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['item']} (value={c['value']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
