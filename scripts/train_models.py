"""Train + calibrate + evaluate LightGBM signal models for all instruments.

Usage: python scripts/train_models.py [--instrument all] [--dataset-version latest]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danalit.config import load_config  # noqa: E402
from danalit.models import evaluate, registry, train  # noqa: E402


def latest_dataset_version(instrument: str) -> str:
    base = load_config().settings.paths.absolute("data_store") / "datasets" / instrument
    versions = sorted(p.name for p in base.iterdir() if (p / "manifest.json").exists())
    if not versions:
        raise SystemExit(f"no datasets for {instrument}; run scripts/build_dataset.py first")
    return versions[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="all")
    ap.add_argument("--dataset-version", default="latest")
    args = ap.parse_args()

    cfg = load_config()
    names = cfg.enabled_instruments() if args.instrument == "all" else [args.instrument]
    for name in names:
        ds = args.dataset_version if args.dataset_version != "latest" else latest_dataset_version(name)
        print(f"=== {name} on dataset {ds} ===")
        version, boosters, calibrators, feature_cols, frames, _ = train.train_instrument(name, ds)
        metrics = evaluate.evaluate(name, version, boosters, calibrators, feature_cols, frames)
        # persist metrics next to the model + registry row
        d = registry.store_dir(name, version)
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        if registry.champion_version(name) is None:
            registry.set_champion(name, version)
            print(f"  set as initial champion: {version}")
        exp = metrics["best_expectancy_atr"]
        print(f"  model {version}: best tau={metrics['best_tau']}, "
              f"OOS expectancy={exp if exp is not None else float('nan'):.4f} ATR/signal, "
              f"report: reports/model_eval_{name}_{version}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
