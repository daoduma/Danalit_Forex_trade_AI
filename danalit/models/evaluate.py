"""Model evaluation with the metrics that matter for trading.

For each fold and pooled out-of-sample: log loss, per-class precision/recall,
and the thresholded decision economics — for act-when-max(P_long,P_short)>tau,
the signal count, hit rate, average labelled return AFTER the label's cost
model, and expectancy in ATR units. Reports honestly: if OOS expectancy is
negative or near zero, the summary says so plainly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from danalit.config import load_config
from danalit.models.calibrate import apply_calibration, reliability_table

TAUS = [round(t, 2) for t in np.arange(0.45, 0.71, 0.05)]
CLASS_NAMES = {0: "no-trade", 1: "long", 2: "short"}


def fold_predictions(booster, calibrators, frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    proba = np.atleast_2d(booster.predict(frame[feature_cols].to_numpy()))
    if calibrators is not None:
        proba = apply_calibration(proba, calibrators)
    out = frame[["label", "ret_long", "ret_short", "atr"]].copy()
    out["p_none"], out["p_long"], out["p_short"] = proba[:, 0], proba[:, 1], proba[:, 2]
    return out


def threshold_economics(pred: pd.DataFrame, taus: list[float] = TAUS) -> list[dict]:
    """Trading economics of thresholded decisions on labelled outcomes (net of label costs)."""
    rows = []
    p_dir = pred[["p_long", "p_short"]].to_numpy()
    direction = np.where(p_dir[:, 0] >= p_dir[:, 1], 1, 2)
    conf = p_dir.max(axis=1)
    ret_price = np.where(direction == 1, pred["ret_long"], pred["ret_short"])
    ret_atr = ret_price / pred["atr"].replace(0, np.nan).to_numpy()
    hit = np.where(direction == pred["label"].to_numpy(), 1.0, 0.0)
    for tau in taus:
        act = conf > tau
        n = int(act.sum())
        rows.append({
            "tau": tau,
            "signals": n,
            "hit_rate": float(hit[act].mean()) if n else None,
            "avg_ret_atr": float(np.nanmean(ret_atr[act])) if n else None,
            "expectancy_atr": float(np.nansum(ret_atr[act]) / n) if n else None,
        })
    return rows


def per_class_pr(pred: pd.DataFrame) -> dict:
    y = pred["label"].to_numpy()
    yhat = pred[["p_none", "p_long", "p_short"]].to_numpy().argmax(axis=1)
    out = {}
    for k, name in CLASS_NAMES.items():
        tp = int(((yhat == k) & (y == k)).sum())
        fp = int(((yhat == k) & (y != k)).sum())
        fn = int(((yhat != k) & (y == k)).sum())
        out[name] = {
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "support": int((y == k).sum()),
        }
    return out


def log_loss(pred: pd.DataFrame) -> float:
    p = pred[["p_none", "p_long", "p_short"]].to_numpy().clip(1e-9, 1)
    y = pred["label"].to_numpy()
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def feature_importance_groups(boosters, feature_cols: list[str]) -> pd.DataFrame:
    """Gain importance averaged over folds, with the registry's feature groups."""
    from danalit.features.technical import FEATURE_REGISTRY

    gains = np.mean([b.feature_importance(importance_type="gain") for b in boosters], axis=0)
    df = pd.DataFrame({"feature": feature_cols, "gain": gains})
    df["group"] = df["feature"].map(
        lambda f: FEATURE_REGISTRY.get(f, {}).get("group", "unknown"))
    return df.sort_values("gain", ascending=False).reset_index(drop=True)


def evaluate(
    instrument: str,
    model_version: str,
    boosters,
    calibrators,
    feature_cols: list[str],
    frames: dict,
    split: str = "test",
    out_dir: Optional[Path] = None,
) -> dict:
    """Full evaluation -> metrics dict + markdown report."""
    per_fold, pooled_frames = {}, []
    for i, fname in enumerate(sorted(frames)):
        frame = frames[fname][split]
        if frame.empty:
            continue
        pred = fold_predictions(boosters[i], calibrators[i] if calibrators else None,
                                frame, feature_cols)
        pooled_frames.append(pred)
        per_fold[fname] = {
            "log_loss": log_loss(pred),
            "economics": threshold_economics(pred),
        }
    pooled = pd.concat(pooled_frames)
    econ = threshold_economics(pooled)
    best = max((r for r in econ if r["signals"] >= 30), default=econ[0],
               key=lambda r: (r["expectancy_atr"] or -9e9))
    imp = feature_importance_groups(boosters, feature_cols)
    reliability = reliability_table(
        pooled[["p_none", "p_long", "p_short"]].to_numpy(), pooled["label"].to_numpy(), 1)

    metrics = {
        "split": split,
        "pooled_log_loss": log_loss(pooled),
        "per_class": per_class_pr(pooled),
        "economics_by_tau": econ,
        "best_tau": best["tau"],
        "best_expectancy_atr": best["expectancy_atr"],
        "n_oos_samples": int(len(pooled)),
        "per_fold": per_fold,
        "importance_by_group": imp.groupby("group")["gain"].sum().sort_values(ascending=False).to_dict(),
        "top_features": imp.head(30)[["feature", "group", "gain"]].to_dict("records"),
        "reliability_long": reliability,
    }
    write_report(instrument, model_version, metrics, out_dir)
    return metrics


def write_report(instrument: str, model_version: str, metrics: dict,
                 out_dir: Optional[Path] = None) -> Path:
    out_dir = out_dir or load_config().settings.paths.absolute("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"model_eval_{instrument}_{model_version}.md"

    exp = metrics["best_expectancy_atr"]
    if exp is None or exp <= 0:
        verdict = ("**NO EDGE at this stage.** Pooled out-of-sample expectancy is "
                   f"{exp if exp is not None else 'undefined'} ATR per signal — negative or nil. "
                   "Do NOT tune on test data to fix this; see the roadmap Chapter 14 fallbacks "
                   "(feature enrichment, H1 horizon, session filters, meta-labeling).")
    elif exp < 0.05:
        verdict = (f"**MARGINAL.** Best OOS expectancy {exp:.4f} ATR/signal at tau="
                   f"{metrics['best_tau']} — near zero; treat as no-edge until walk-forward "
                   "backtesting (Prompts 8-9) says otherwise.")
    else:
        verdict = (f"Best OOS expectancy {exp:.4f} ATR/signal at tau={metrics['best_tau']} "
                   f"over {metrics['n_oos_samples']} samples. Subject to full cost-realistic "
                   "walk-forward confirmation (Prompts 8-9).")

    lines = [
        f"# Model evaluation — {instrument} {model_version}",
        "", f"_Split: {metrics['split']}, pooled across folds. All returns are net of the "
        "label cost model (spread paid on entry)._", "",
        "## Verdict", "", verdict, "",
        f"- pooled log loss: {metrics['pooled_log_loss']:.4f}",
        f"- OOS samples: {metrics['n_oos_samples']:,}", "",
        "## Economics by threshold", "",
        "| tau | signals | hit rate | avg ret (ATR) | expectancy (ATR) |",
        "|---|---|---|---|---|",
    ]
    for r in metrics["economics_by_tau"]:
        hr = f"{r['hit_rate']:.3f}" if r["hit_rate"] is not None else "—"
        ar = f"{r['avg_ret_atr']:.4f}" if r["avg_ret_atr"] is not None else "—"
        ex = f"{r['expectancy_atr']:.4f}" if r["expectancy_atr"] is not None else "—"
        lines.append(f"| {r['tau']} | {r['signals']} | {hr} | {ar} | {ex} |")

    lines += ["", "## Per-class precision / recall", ""]
    for name, m in metrics["per_class"].items():
        p = f"{m['precision']:.3f}" if m["precision"] is not None else "—"
        rc = f"{m['recall']:.3f}" if m["recall"] is not None else "—"
        lines.append(f"- **{name}**: precision {p}, recall {rc}, support {m['support']:,}")

    lines += ["", "## Feature importance by group (gain share)", ""]
    total = sum(metrics["importance_by_group"].values()) or 1
    for g, v in metrics["importance_by_group"].items():
        lines.append(f"- {g}: {100 * v / total:.1f}%")
    lines += ["", "## Top 30 features", ""]
    for r in metrics["top_features"]:
        lines.append(f"- {r['feature']} ({r['group']}): {r['gain']:.0f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
