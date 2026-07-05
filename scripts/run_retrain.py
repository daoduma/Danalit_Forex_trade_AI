"""Monthly retraining run (or manual / drift-recommended).

    python scripts/run_retrain.py [--instrument all] [--probation-check]

Windows Task Scheduler (installed by scripts/deploy/install_tasks.ps1):
  monthly, 1st of month 02:00 — retrain
  daily 03:00 — probation check
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danalit.config import load_config  # noqa: E402
from danalit.models import retrain  # noqa: E402
from danalit.monitor.notifier import make_notifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="all")
    ap.add_argument("--probation-check", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    names = cfg.enabled_instruments() if args.instrument == "all" else [args.instrument]
    notifier = make_notifier()

    if args.probation_check:
        for name in names:
            print(name, "->", retrain.check_probation(name, notifier=notifier))
        return 0

    for name in names:
        try:
            d = retrain.retrain_instrument(name, notifier=notifier)
            print(f"{name}: {'PROMOTED ' + d['challenger'] if d['promote'] else 'kept champion'}"
                  f" — report written")
        except Exception as e:
            print(f"{name}: retrain failed — {e}")
            notifier.notify("WARNING", f"Retrain failed: {name}", str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
