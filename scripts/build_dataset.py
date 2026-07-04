"""Build versioned training datasets (features + triple-barrier labels, purged folds).

Usage: python scripts/build_dataset.py [--instrument all] [--news]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danalit.config import load_config  # noqa: E402
from danalit.features import dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="all")
    ap.add_argument("--news", action="store_true", help="include news/calendar features")
    args = ap.parse_args()

    cfg = load_config()
    names = cfg.enabled_instruments() if args.instrument == "all" else [args.instrument]
    for name in names:
        version, manifest = dataset.build_and_save(name, include_news=args.news)
        print(f"=== {name} -> {version} ===")
        for fname, meta in manifest["fold_meta"].items():
            row = "  ".join(
                f"{split}: {m['rows']:>7,} rows  {m['class_balance']}"
                for split, m in meta.items()
            )
            print(f"  {fname}: {row}")
            for split, m in meta.items():
                if 0 < m["rows"] < 500:
                    print(f"    WARNING: {fname}/{split} has {m['rows']} rows (<500 — "
                          f"statistically thin, documented per roadmap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
