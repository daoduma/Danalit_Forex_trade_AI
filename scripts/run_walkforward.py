"""Honest walk-forward backtest -> reports/walkforward_{instrument}_{version}.html

Usage: python scripts/run_walkforward.py [--instrument EURUSD] [--tune] [--trials 20]
       [--cost-scale 1.0]   (1.5 / 2.0 for the cost-stress reruns)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danalit.backtest.walkforward import run_walkforward  # noqa: E402
from danalit.config import load_config  # noqa: E402


def latest_dataset_version(instrument: str) -> str:
    base = load_config().settings.paths.absolute("data_store") / "datasets" / instrument
    versions = sorted(p.name for p in base.iterdir() if (p / "manifest.json").exists())
    if not versions:
        raise SystemExit(f"no datasets for {instrument}; run scripts/build_dataset.py first")
    return versions[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="EURUSD")
    ap.add_argument("--dataset-version", default="latest")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.55)
    ap.add_argument("--cost-scale", type=float, default=1.0)
    args = ap.parse_args()

    ds = args.dataset_version if args.dataset_version != "latest" \
        else latest_dataset_version(args.instrument)
    res = run_walkforward(args.instrument, ds, tune=args.tune, n_trials=args.trials,
                          tau=args.tau, cost_scale=args.cost_scale)
    s = res["summary"]
    print(f"{args.instrument} {ds}: {s['n_trades']} OOS trades, "
          f"net {s.get('net_profit', 0):+.2f}, PF {s.get('profit_factor')}, "
          f"maxDD {s.get('max_drawdown_equity', 0):.1%}")
    if "report_path" in res:
        print("report:", res["report_path"])

    if args.cost_scale == 1.0:
        for scale in (1.5, 2.0):
            stressed = run_walkforward(args.instrument, ds, tune=False, tau=args.tau,
                                       cost_scale=scale, write_html=False)
            ss = stressed["summary"]
            print(f"  cost x{scale}: net {ss.get('net_profit', 0):+.2f}, "
                  f"PF {ss.get('profit_factor')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
