"""Walk-forward harness: the honest pipeline that produces THE headline result.

Per fold: train on the train split, (optionally) tune on the validation split
ONLY, then run the backtester over the fold's TEST period with an ML strategy
that enters when max(P_long, P_short) > tau with SL/TP at the label barriers.
Fold test periods are stitched into one continuous out-of-sample equity curve.

Fold isolation is enforced by FoldIsolationGuard — an explicit tripwire, not
just discipline: any frame whose timestamps intersect the test period raises
before it can reach training or tuning code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from danalit.backtest.costs import CostModel
from danalit.backtest.engine import Backtester, Context
from danalit.backtest.metrics import (
    bootstrap_drawdowns,
    deflated_sharpe_ratio,
    summarize,
)
from danalit.backtest.report import build_report
from danalit.config import load_config
from danalit.data import price_store
from danalit.features.dataset import load_dataset
from danalit.logging_setup import setup_logging
from danalit.models.calibrate import apply_calibration, fit_calibration
from danalit.models.train import train_fold

log = setup_logging("walkforward")


class FoldLeakError(RuntimeError):
    pass


@dataclass
class FoldIsolationGuard:
    """Raises if a frame containing test-period timestamps reaches train/tune code."""

    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def check(self, frame: pd.DataFrame, context: str) -> pd.DataFrame:
        idx = frame.index
        if len(idx) and (idx.max() >= self.test_start) and (idx.min() <= self.test_end):
            overlap = idx[(idx >= self.test_start) & (idx <= self.test_end)]
            if len(overlap):
                raise FoldLeakError(
                    f"{context}: {len(overlap)} test-period timestamps "
                    f"({overlap.min()}..{overlap.max()}) reached a training/tuning code path"
                )
        return frame


class MLSignalStrategy:
    """Enter on calibrated model probability; SL/TP at the label's ATR barriers.

    Sizing here is a PLACEHOLDER fixed-fractional sizer — the real risk manager
    (Prompt 10) replaces it via the Backtester risk_check/sizer hook.
    """

    def __init__(
        self,
        instrument: str,
        probs: pd.DataFrame,  # index time_utc; p_long, p_short, atr[, blackout]
        tau: float,
        k_tp: float,
        k_sl: float,
        horizon_bars: int,
        contract_size: float,
        risk_frac: float = 0.0075,
        min_lot: float = 0.01,
        lot_step: float = 0.01,
        max_lot: float = 100.0,
        bar_minutes: int = 15,
        sizer=None,  # optional callable(equity, sl_distance) -> lots (Prompt 10)
    ):
        self.instrument = instrument
        self.probs = probs
        self.tau, self.k_tp, self.k_sl = tau, k_tp, k_sl
        self.hold = pd.Timedelta(minutes=horizon_bars * bar_minutes)
        self.contract_size = contract_size
        self.risk_frac, self.min_lot, self.lot_step, self.max_lot = (
            risk_frac, min_lot, lot_step, max_lot)
        self.sizer = sizer

    def _size(self, equity: float, sl_distance: float) -> float:
        if self.sizer is not None:
            return self.sizer(equity, sl_distance)
        if sl_distance <= 0:
            return 0.0
        lots = (equity * self.risk_frac) / (sl_distance * self.contract_size)
        lots = np.floor(lots / self.lot_step) * self.lot_step
        return float(min(max(lots, 0.0), self.max_lot))

    def on_bar(self, ctx: Context):
        orders = []
        mine = [p for p in ctx.positions if p.instrument == self.instrument]
        # time exit first
        for p in mine:
            if ctx.time - p.entry_time >= self.hold:
                orders.append({"type": "close", "position_id": p.id,
                               "fraction": 1.0, "reason": "time_exit"})
                mine = [q for q in mine if q.id != p.id]
        if mine or ctx.time not in self.probs.index:
            return orders
        row = self.probs.loc[ctx.time]
        if float(row.get("blackout", 0.0)) > 0:
            return orders
        p_long, p_short = float(row["p_long"]), float(row["p_short"])
        conf = max(p_long, p_short)
        if conf <= self.tau:
            return orders
        side = 1 if p_long >= p_short else -1
        atr = float(row["atr"])
        close = float(ctx.bars[self.instrument]["close"])
        sl_dist, tp_dist = self.k_sl * atr, self.k_tp * atr
        lots = self._size(ctx.equity, sl_dist)
        if lots < self.min_lot:
            return orders
        sl = close - side * sl_dist
        tp = close + side * tp_dist
        orders.append({"type": "open", "instrument": self.instrument, "side": side,
                       "lots": lots, "sl": sl, "tp": tp,
                       "tag": f"ml:{conf:.3f}"})
        return orders


def _fold_probs(booster, calibrators, frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    proba = np.atleast_2d(booster.predict(frame[feature_cols].to_numpy()))
    if calibrators is not None:
        proba = apply_calibration(proba, calibrators)
    out = pd.DataFrame(index=frame.index)
    out["p_long"], out["p_short"] = proba[:, 1], proba[:, 2]
    out["atr"] = frame["atr"]
    if "blackout" in frame.columns:
        out["blackout"] = frame["blackout"]
    return out


def run_walkforward(
    instrument: str,
    dataset_version: str,
    tune: bool = False,
    n_trials: int = 20,
    tau: float = 0.55,
    initial_balance: float = 1000.0,
    cost_scale: float = 1.0,
    root: Optional[Path] = None,
    db_path: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
    write_html: bool = True,
    out_path: Optional[Path] = None,
) -> dict:
    """Full honest pipeline for one instrument. Returns summary + artifacts."""
    cfg = load_config()
    inst = cfg.instruments[instrument]
    lab = cfg.settings.labeling
    frames, manifest = load_dataset(instrument, dataset_version, base_dir=dataset_dir)
    feature_cols = manifest["feature_cols"]
    bars = price_store.read_bars(instrument, cfg.settings.trading.primary_timeframe, root=root)
    bars_by_time = bars.set_index("time_utc")

    cost = CostModel.from_instrument(inst)
    if cost_scale != 1.0:
        cost = CostModel(
            spread=cost.spread * cost_scale,
            commission_per_lot=cost.commission_per_lot * cost_scale,
            slippage=cost.slippage * cost_scale,
            news_slippage_extra=cost.news_slippage_extra * cost_scale,
            swap_long=cost.swap_long, swap_short=cost.swap_short,
        )

    all_trades, curves = [], []
    balance = initial_balance
    total_trials = 0
    fold_params: dict[str, dict] = {}

    for fname in sorted(frames):
        tr, va, te = (frames[fname][s] for s in ("train", "validate", "test"))
        if te.empty or len(tr) < 200:
            continue
        guard = FoldIsolationGuard(te.index.min(), te.index.max())
        guard.check(tr, f"{fname} train split")
        guard.check(va, f"{fname} validate split")

        params = {}
        fold_tau, k_tp, k_sl, horizon = tau, lab.k_tp, lab.k_sl, lab.horizon_bars
        if tune:
            from danalit.models.tuning import tune_fold

            best = tune_fold(instrument, tr, va, feature_cols, guard=guard,
                             n_trials=n_trials, study_name=f"{instrument}_{dataset_version}_{fname}",
                             db_path=db_path)
            params = best["lgbm_params"]
            fold_tau = best["tau"]
            total_trials += best["n_trials"]
        else:
            total_trials += 1
        fold_params[fname] = {"tau": fold_tau, "k_tp": k_tp, "k_sl": k_sl, **params}

        booster = train_fold(guard.check(tr, "train X")[feature_cols], tr["label"].to_numpy(),
                             guard.check(va, "val X")[feature_cols], va["label"].to_numpy(),
                             params=params)
        cal = fit_calibration(
            np.atleast_2d(booster.predict(va[feature_cols].to_numpy())), va["label"].to_numpy())

        probs = _fold_probs(booster, cal, te, feature_cols)
        # engine needs raw bars over the test window (entry fills at next open)
        te_bars = bars[(bars["time_utc"] >= te.index.min())
                       & (bars["time_utc"] <= te.index.max() + pd.Timedelta(days=3))]
        strategy = MLSignalStrategy(
            instrument, probs, fold_tau, k_tp, k_sl, horizon,
            contract_size=inst.contract_size,
            risk_frac=cfg.settings.risk.risk_per_trade,
            min_lot=inst.min_lot, lot_step=inst.lot_step, max_lot=inst.max_lot,
        )
        bt = Backtester({instrument: te_bars}, {instrument: cost},
                        {instrument: inst.contract_size},
                        initial_balance=balance, leverage=cfg.settings.broker.leverage)
        res = bt.run(strategy)
        all_trades.extend(res["trades"])
        curves.append(res["equity_curve"])
        balance = res["final_balance"]
        log.info("%s %s: %d trades, balance %.2f", instrument, fname,
                 len(res["trades"]), balance)

    equity_curve = pd.concat(curves) if curves else pd.DataFrame()
    summary = summarize(all_trades, equity_curve, initial_balance)
    years = max((equity_curve.index[-1] - equity_curve.index[0]).days / 365.25, 1e-9) \
        if not equity_curve.empty else 1.0
    bars_per_year = len(equity_curve) / years if years else 1.0
    summary["deflated_sharpe_p"] = deflated_sharpe_ratio(
        summary.get("sharpe"), max(total_trials, 1), len(equity_curve), bars_per_year)
    summary["n_trials"] = total_trials
    summary["monte_carlo"] = bootstrap_drawdowns(
        [t.net_pnl for t in all_trades], initial_balance)
    summary["fold_params"] = fold_params

    result = {"trades": all_trades, "equity_curve": equity_curve,
              "rejections": [], "final_balance": balance,
              "summary": summary, "manifest": manifest}
    if write_html:
        out_path = out_path or (cfg.settings.paths.absolute("reports")
                                / f"walkforward_{instrument}_{dataset_version}.html")
        build_report(result, summary, f"Walk-forward — {instrument} ({dataset_version})",
                     out_path, extra_sections=[verdict_section(summary)])
        result["report_path"] = out_path
    return result


def verdict_section(summary: dict) -> str:
    """Plain-language verdict: edge or no edge, and confidence."""
    pf = summary.get("profit_factor")
    exp = summary.get("expectancy")
    dd = summary.get("max_drawdown_equity", 1.0)
    dsr = summary.get("deflated_sharpe_p")
    mc = summary.get("monte_carlo", {})
    n = summary.get("n_trades", 0)

    if n < 100:
        headline = (f"INSUFFICIENT DATA: only {n} out-of-sample trades — no statistical claim "
                    "possible (roadmap wants 500+).")
    elif exp is not None and exp > 0 and (pf or 0) >= 1.15 and dd < 0.25 and (dsr or 0) > 0.9:
        headline = (f"PROVISIONAL EDGE: PF {pf:.2f}, expectancy {exp:.4f}/trade over {n} trades, "
                    f"max DD {dd:.1%}, deflated-Sharpe confidence {dsr:.2f}. "
                    "Proceed to demo forward test — nothing is proven until it survives one.")
    elif exp is not None and exp > 0:
        headline = (f"WEAK/UNPROVEN: positive expectancy ({exp:.4f}) but PF {pf if pf else 0:.2f} "
                    f"/ DD {dd:.1%} / deflated-Sharpe {dsr if dsr else 0:.2f} do not clear the bar. "
                    "Treat as no-edge.")
    else:
        headline = ("NO EDGE net of costs. Do NOT keep tuning until it looks good. "
                    "Roadmap Chapter 14 fallbacks, in order: (1) enrich features "
                    "(higher-TF context, cross-asset: DXY/yields for gold, VIX for US100); "
                    "(2) change the trade definition (H1 horizon, barrier multiples); "
                    "(3) session-specific scope (e.g. London open only); "
                    "(4) meta-labeling over a rule-based signal, ML as filter only.")

    mc_line = (f"Monte Carlo: median max-DD {mc.get('dd_p50', 0):.1%}, "
               f"95th pct {mc.get('dd_p95', 0):.1%}, P(lose half) {mc.get('p_ruin_half', 0):.1%}."
               if mc else "")
    return (f"<h2>Verdict</h2><p><b>{headline}</b></p><p>{mc_line}</p>"
            f"<p>Configurations tried (deflation basis): {summary.get('n_trials')}. "
            "All numbers are net of spread/commission/slippage/swap on stitched, "
            "never-touched test periods.</p>")
