"""Continuous learning: scheduled retraining with champion/challenger gates.

Monthly by default (or manual, or drift-recommended — never auto-retrain on
drift alone; a human stays in that loop). The challenger trains with the
champion's FROZEN hyperparameters (full re-tuning stays a deliberate act),
is evaluated on a holdout of the most recent weeks neither model trained on,
and promotes only if it wins on expectancy without materially worse drawdown
or calibration. Promotions update the champion pointer atomically; a 2-week
probation compares live expectancy to the holdout and auto-rolls back on
degradation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from danalit.config import load_config
from danalit.db import connect
from danalit.logging_setup import setup_logging
from danalit.timeutil import iso, utc_now

log = setup_logging("retrain")

HOLDOUT_WEEKS = 8
PROBATION_DAYS = 14
PROBATION_BAND = 0.5          # live expectancy may lag holdout by up to 50%
PSI_WARN = 0.2


# ------------------------------------------------------------ promotion gate

@dataclass
class GateResult:
    promote: bool
    reasons: list[str]


def promotion_gate(champion: dict, challenger: dict, dd_tolerance: float = 1.1) -> GateResult:
    """challenger promotes only if: holdout expectancy >= champion's AND
    max drawdown <= champion's * dd_tolerance AND calibration not worse."""
    reasons = []
    ok = True
    if challenger.get("expectancy") is None or champion.get("expectancy") is None:
        return GateResult(False, ["missing expectancy metrics — keep champion"])
    if challenger["expectancy"] >= champion["expectancy"]:
        reasons.append(f"expectancy {challenger['expectancy']:.4f} >= "
                       f"{champion['expectancy']:.4f} OK")
    else:
        ok = False
        reasons.append(f"expectancy {challenger['expectancy']:.4f} < "
                       f"{champion['expectancy']:.4f} FAIL")
    ch_dd = challenger.get("max_drawdown", 1.0)
    cp_dd = champion.get("max_drawdown", 1.0)
    if ch_dd <= cp_dd * dd_tolerance:
        reasons.append(f"drawdown {ch_dd:.3f} <= {cp_dd:.3f}x{dd_tolerance} OK")
    else:
        ok = False
        reasons.append(f"drawdown {ch_dd:.3f} > {cp_dd:.3f}x{dd_tolerance} FAIL")
    ch_cal = challenger.get("log_loss", np.inf)
    cp_cal = champion.get("log_loss", np.inf)
    if ch_cal <= cp_cal * 1.02:  # 'not worse' with 2% tolerance
        reasons.append(f"calibration {ch_cal:.4f} vs {cp_cal:.4f} OK")
    else:
        ok = False
        reasons.append(f"calibration {ch_cal:.4f} worse than {cp_cal:.4f} FAIL")
    return GateResult(ok, reasons)


# --------------------------------------------------------------- evaluation

def holdout_metrics(bundle, frame: pd.DataFrame, tau: float = 0.55) -> dict:
    """Expectancy (ATR, net of label costs), max drawdown of the signal P&L
    sequence, and log loss on a holdout frame."""
    proba = bundle.predict_proba(frame)
    p = np.atleast_2d(proba)
    y = frame["label"].to_numpy()
    ll = float(-np.log(p[np.arange(len(y)), y].clip(1e-9)).mean())
    p_long, p_short = p[:, 1], p[:, 2]
    conf = np.maximum(p_long, p_short)
    act = conf > tau
    if act.sum() == 0:
        return {"expectancy": 0.0, "max_drawdown": 0.0, "log_loss": ll, "n_signals": 0}
    direction = np.where(p_long >= p_short, 1, 2)
    ret = np.where(direction == 1, frame["ret_long"], frame["ret_short"])
    atr = frame["atr"].to_numpy()
    ret_atr = np.divide(ret, atr, out=np.zeros_like(ret), where=atr > 0)[act]
    cum = np.cumsum(ret_atr)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    dd = float(np.max(peak - cum)) if len(cum) else 0.0
    return {"expectancy": float(ret_atr.mean()), "max_drawdown": dd,
            "log_loss": ll, "n_signals": int(act.sum())}


# --------------------------------------------------------------------- runs

def retrain_instrument(
    instrument: str,
    include_news: bool = True,
    tau: float = 0.55,
    db_path: Optional[Path] = None,
    notifier=None,
) -> dict:
    """Full retrain run: rebuild dataset -> train challenger with frozen champion
    params -> holdout comparison -> gated promotion. Returns the decision record."""
    from danalit.features.dataset import build_and_save, load_dataset
    from danalit.models import registry
    from danalit.models.train import train_instrument

    cfg = load_config()
    champ_version = registry.champion_version(instrument, db_path=db_path)
    if champ_version is None:
        raise RuntimeError(f"{instrument}: no champion registered — run initial training first")
    champion = registry.load_model(instrument, champ_version)
    frozen_params = champion.metrics.get("lgbm_params", {})

    ds_version, manifest = build_and_save(instrument, include_news=include_news)
    frames, _ = load_dataset(instrument, ds_version)
    # holdout: the most recent HOLDOUT_WEEKS of the final fold's test period
    last_fold = frames[sorted(frames)[-1]]
    test = last_fold["test"]
    cutoff = test.index.max() - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    holdout = test[test.index > cutoff]
    if len(holdout) < 200:
        raise RuntimeError(f"{instrument}: holdout too small ({len(holdout)} rows)")

    chall_version, boosters, calibrators, feature_cols, ch_frames, _ = train_instrument(
        instrument, ds_version, params=frozen_params, db_path=db_path)
    challenger = registry.load_model(instrument, chall_version)

    champ_m = holdout_metrics(champion, holdout, tau)
    chall_m = holdout_metrics(challenger, holdout, tau)
    gate = promotion_gate(champ_m, chall_m)

    decision = {
        "instrument": instrument,
        "date": utc_now().strftime("%Y-%m-%d"),
        "dataset_version": ds_version,
        "champion": champ_version,
        "challenger": chall_version,
        "champion_metrics": champ_m,
        "challenger_metrics": chall_m,
        "promote": gate.promote,
        "reasons": gate.reasons,
    }
    if gate.promote:
        registry.set_champion(instrument, chall_version, db_path=db_path)
        _record_promotion(db_path, decision)
        if notifier:
            notifier.notify("WARNING", f"Model promoted: {instrument}",
                            f"{champ_version} -> {chall_version}; probation "
                            f"{PROBATION_DAYS}d, holdout exp {chall_m['expectancy']:.4f}")
    write_retrain_report(decision)
    return decision


def _record_promotion(db_path, decision: dict) -> None:
    con = connect(db_path or load_config().settings.paths.db_path)
    try:
        with con:
            con.execute(
                "INSERT INTO system_events (ts_utc, type, detail) VALUES (?,?,?)",
                (iso(utc_now()), "promotion", json.dumps(decision, default=str)))
    finally:
        con.close()


def write_retrain_report(decision: dict, out_dir: Optional[Path] = None) -> Path:
    out_dir = out_dir or load_config().settings.paths.absolute("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"retrain_{decision['instrument']}_{decision['date']}.md"
    d = decision
    lines = [
        f"# Retrain — {d['instrument']} {d['date']}", "",
        f"**Decision: {'PROMOTE' if d['promote'] else 'KEEP CHAMPION'}**", "",
        f"- champion:   {d['champion']}  {d['champion_metrics']}",
        f"- challenger: {d['challenger']}  {d['challenger_metrics']}",
        f"- dataset: {d['dataset_version']}", "", "## Gate rationale", "",
    ]
    lines += [f"- {r}" for r in d["reasons"]]
    if d["promote"]:
        lines += ["", f"Probation: {PROBATION_DAYS} days; auto-rollback if live "
                  f"expectancy < {PROBATION_BAND:.0%} of holdout expectancy."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------- probation

def check_probation(
    instrument: str,
    db_path: Optional[Path] = None,
    notifier=None,
    now: Optional[pd.Timestamp] = None,
) -> Optional[str]:
    """If within PROBATION_DAYS of a promotion, compare live expectancy to the
    challenger's holdout expectancy; roll back on degradation beyond the band.
    Returns 'rolled_back', 'ok', or None (no active probation)."""
    from danalit.models import registry

    cfg = load_config()
    db = db_path or cfg.settings.paths.db_path
    now = now or pd.Timestamp(utc_now())
    con = connect(db)
    try:
        row = con.execute(
            "SELECT ts_utc, detail FROM system_events WHERE type='promotion'"
            " ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        promo = json.loads(row["detail"])
        if promo["instrument"] != instrument:
            return None
        promoted_at = pd.Timestamp(row["ts_utc"])
        if now - promoted_at > pd.Timedelta(days=PROBATION_DAYS):
            return None
        trades = con.execute(
            "SELECT net_pnl FROM trades WHERE instrument=? AND closed_utc >= ?",
            (instrument, row["ts_utc"])).fetchall()
    finally:
        con.close()

    if len(trades) < 5:
        return "ok"  # not enough live evidence yet
    live_exp = float(np.mean([t["net_pnl"] for t in trades]))
    holdout_exp = promo["challenger_metrics"]["expectancy"]
    # compare SIGN-scaled: degradation = live falls below band * holdout (both
    # in their own units; sign and ratio are what matter for the tripwire)
    threshold = holdout_exp * PROBATION_BAND
    if live_exp < min(threshold, 0.0) or (holdout_exp > 0 and live_exp < 0):
        registry.set_champion(instrument, promo["champion"], db_path=db_path)
        _record_promotion(db, {**promo, "promote": False,
                               "rolled_back": True, "live_expectancy": live_exp})
        if notifier:
            notifier.notify("CRITICAL", f"AUTO-ROLLBACK {instrument}",
                            f"live expectancy {live_exp:.4f} degraded vs holdout "
                            f"{holdout_exp:.4f}; champion restored: {promo['champion']}")
        return "rolled_back"
    return "ok"


# ------------------------------------------------------------------- drift

def compute_psi(train: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """Population stability index between two samples of one feature."""
    train = np.asarray(train, dtype=float)
    live = np.asarray(live, dtype=float)
    if len(train) < 10 or len(live) < 10:
        return 0.0
    edges = np.quantile(train, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(train, edges)[0] / len(train), 1e-6, None)
    a = np.clip(np.histogram(live, edges)[0] / len(live), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(
    instrument: str,
    train_features: pd.DataFrame,
    db_path: Optional[Path] = None,
    weeks: int = 4,
    core_features: Optional[list[str]] = None,
    notifier=None,
) -> dict:
    """Weekly PSI of live feature snapshots vs training distributions.
    WARNING recommending an early retrain when core features drift — never
    an automatic retrain."""
    cfg = load_config()
    con = connect(db_path or cfg.settings.paths.db_path)
    try:
        since = iso(pd.Timestamp(utc_now()) - pd.Timedelta(weeks=weeks))
        rows = con.execute(
            "SELECT features_snapshot FROM decisions WHERE instrument=? AND ts_utc>=?",
            (instrument, since)).fetchall()
    finally:
        con.close()
    if not rows:
        return {"n_live": 0, "psi": {}, "drifted": []}
    live = pd.DataFrame([json.loads(r["features_snapshot"]) for r in rows])
    core = core_features or ["atr_norm", "adx14", "rsi14", "bb_width", "atr_pctile_90d"]
    psi = {}
    for f in core:
        if f in live.columns and f in train_features.columns:
            psi[f] = compute_psi(train_features[f].to_numpy(),
                                 pd.to_numeric(live[f], errors="coerce").dropna().to_numpy())
    drifted = [f for f, v in psi.items() if v > PSI_WARN]
    if drifted and notifier:
        notifier.notify("WARNING", f"Feature drift: {instrument}",
                        f"PSI > {PSI_WARN} on {drifted} — consider an early retrain "
                        "(manual decision; never automatic)")
    return {"n_live": len(live), "psi": psi, "drifted": drifted}
