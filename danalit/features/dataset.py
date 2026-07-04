"""Versioned training datasets with purged, embargoed walk-forward splits.

Purging: a training/validation sample whose label window [T, T + label_span]
crosses into the next split's period is removed — label leakage cannot inflate
results. Embargo: an additional gap (default 5 days) is left before the next
split. No random shuffles anywhere. Datasets persist with a manifest (features,
label params, folds, row counts, class balance, git commit, content hash) and
are deterministic: same inputs -> identical dataset hash.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from danalit.config import load_config
from danalit.data import price_store
from danalit.features.labeling import LABEL_COLUMNS, label_span, triple_barrier
from danalit.features.technical import build_features
from danalit.timeutil import utc_now


@dataclass
class Fold:
    train: tuple[str, str]
    validate: tuple[str, str]
    test: tuple[str, str]


def make_folds(index: pd.DatetimeIndex, n_folds: int = 0) -> list[Fold]:
    """Derive folds from the available span: 60/20/20 single fold under 4 years
    of data, otherwise rolling yearly folds (train 3y, validate 1y, test 1y)."""
    start, end = index.min(), index.max()
    years = (end - start).days / 365.25
    if years < 4 or n_folds == 1:
        t0 = start + (end - start) * 0.6
        v0 = start + (end - start) * 0.8
        return [Fold((str(start), str(t0)), (str(t0), str(v0)), (str(v0), str(end)))]
    folds = []
    y = start.year
    while pd.Timestamp(f"{y + 4}-01-01", tz="UTC") <= end + pd.Timedelta(days=366):
        folds.append(Fold(
            (f"{y}-01-01", f"{y + 3}-01-01"),
            (f"{y + 3}-01-01", f"{y + 4}-01-01"),
            (f"{y + 4}-01-01", str(min(pd.Timestamp(f'{y + 5}-01-01', tz='UTC'), end))),
        ))
        y += 1
    return folds


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def split_fold(
    df: pd.DataFrame,
    fold: Fold,
    span: pd.Timedelta,
    embargo: pd.Timedelta,
) -> dict[str, pd.DataFrame]:
    """Purged + embargoed split of a labelled feature frame."""
    t0, t1 = _ts(fold.train[0]), _ts(fold.train[1])
    v0, v1 = _ts(fold.validate[0]), _ts(fold.validate[1])
    s0, s1 = _ts(fold.test[0]), _ts(fold.test[1])
    idx = df.index
    train = df[(idx >= t0) & (idx < t1) & (idx + span <= v0 - embargo)]
    val = df[(idx >= v0) & (idx < v1) & (idx + span <= s0 - embargo)]
    test = df[(idx >= s0) & (idx < s1)]
    return {"train": train, "validate": val, "test": test}


def build_labelled_frame(
    instrument: str,
    include_news: bool = False,
    root: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Features + labels joined on the decision bar. Returns (df, feature_cols, label_params)."""
    cfg = load_config()
    lab_cfg = cfg.settings.labeling
    inst = cfg.instruments[instrument]
    features = build_features(instrument, root=root, include_news=include_news, db_path=db_path)
    bars = price_store.read_bars(instrument, cfg.settings.trading.primary_timeframe, root=root)
    labels = triple_barrier(
        bars,
        spread=inst.spread_estimate_pips * inst.pip_size,
        k_tp=lab_cfg.k_tp, k_sl=lab_cfg.k_sl,
        horizon=lab_cfg.horizon_bars, dead_zone_atr=lab_cfg.dead_zone_atr,
    )
    df = features.join(labels, how="inner")
    feature_cols = [c for c in features.columns]
    params = {"k_tp": lab_cfg.k_tp, "k_sl": lab_cfg.k_sl, "horizon_bars": lab_cfg.horizon_bars,
              "dead_zone_atr": lab_cfg.dead_zone_atr,
              "spread": inst.spread_estimate_pips * inst.pip_size,
              "include_news": include_news}
    return df, feature_cols, params


def dataset_hash(frames: dict[str, dict[str, pd.DataFrame]], params: dict) -> str:
    h = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode())
    for fold_name in sorted(frames):
        for split in ("train", "validate", "test"):
            vals = pd.util.hash_pandas_object(frames[fold_name][split], index=True).to_numpy()
            h.update(fold_name.encode())
            h.update(split.encode())
            h.update(vals.tobytes())
    return h.hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2], timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_and_save(
    instrument: str,
    folds: Optional[list[Fold]] = None,
    include_news: bool = False,
    version: Optional[str] = None,
    root: Optional[Path] = None,
    db_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> tuple[str, dict]:
    """Build, split, persist. Returns (version, manifest)."""
    cfg = load_config()
    df, feature_cols, params = build_labelled_frame(instrument, include_news, root, db_path)
    folds = folds or make_folds(df.index)
    span = label_span(cfg.settings.labeling.horizon_bars)
    embargo = pd.Timedelta(days=cfg.settings.dataset.embargo_days)

    frames = {f"fold_{i}": split_fold(df, fold, span, embargo) for i, fold in enumerate(folds)}
    content = dataset_hash(frames, params)
    version = version or f"v{utc_now().strftime('%Y%m%d')}_{content}"

    out_dir = out_dir or (cfg.settings.paths.absolute("data_store") / "datasets" / instrument / version)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_meta = {}
    for fname, splits in frames.items():
        fold_meta[fname] = {}
        for split, frame in splits.items():
            frame.reset_index().to_parquet(out_dir / f"{fname}_{split}.parquet", index=False)
            balance = frame["label"].value_counts(normalize=True).round(4).to_dict() if len(frame) else {}
            fold_meta[fname][split] = {
                "rows": len(frame),
                "start": str(frame.index.min()) if len(frame) else None,
                "end": str(frame.index.max()) if len(frame) else None,
                "class_balance": {str(k): v for k, v in balance.items()},
            }

    manifest = {
        "instrument": instrument,
        "version": version,
        "dataset_hash": content,
        "created_utc": utc_now().isoformat(),
        "git_commit": _git_commit(),
        "label_params": params,
        "embargo_days": cfg.settings.dataset.embargo_days,
        "feature_cols": feature_cols,
        "folds": [vars(f) for f in folds],
        "fold_meta": fold_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return version, manifest


def load_dataset(instrument: str, version: str, base_dir: Optional[Path] = None) -> tuple[dict, dict]:
    """Load a persisted dataset. Returns ({fold: {split: df}}, manifest)."""
    cfg = load_config()
    d = (base_dir or cfg.settings.paths.absolute("data_store") / "datasets" / instrument) / version
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    for fname in manifest["fold_meta"]:
        frames[fname] = {}
        for split in ("train", "validate", "test"):
            df = pd.read_parquet(d / f"{fname}_{split}.parquet")
            frames[fname][split] = df.set_index("time_utc")
    return frames, manifest
