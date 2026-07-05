"""The single decision point: fuses model probabilities, sentiment, calendar
state and regime filters into one explainable trading decision.

decide() is PURE and deterministic given its inputs — no side effects, no
clock reads, no DB access — so it is identically testable in backtest and
live. Every call, including NONE decisions and vetoes, returns the full
features snapshot for journaling.

Pipeline (each stage can veto; vetoes carry reasons):
  1. data freshness gate    -> NONE("stale data: ...")
  2. calendar blackout gate -> NONE("news blackout")
  3. regime filter          -> NONE("regime: ...")
  4. model signal           -> NONE("below tau") unless max(P_long,P_short) > tau
  5. sentiment veto (gated) -> NONE("sentiment veto: ...")
  6. confluence bonus       -> confidence += bonus (journal analytics only —
                               sizing NEVER uses confidence; the risk manager sizes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from danalit.constants import LONG, NONE, SHORT


@dataclass(frozen=True)
class Decision:
    action: str                    # LONG | SHORT | NONE
    confidence: float
    sl_price: Optional[float]
    tp_price: Optional[float]
    explanation: str
    signal_id: str
    veto_reason: Optional[str] = None
    features_snapshot: dict = field(default_factory=dict, hash=False, compare=False)


@dataclass
class SignalParams:
    max_bar_age_intervals: float = 2.0
    max_collector_age_sec: float = 1800.0
    regime_enabled: bool = True
    min_adx: float = 20.0
    atr_band: tuple[float, float] = (0.15, 0.95)
    sentiment_veto_enabled: bool = True
    sentiment_veto_threshold: float = 1.0
    confluence_bonus: float = 0.05


def _fmt_mins(mins: float) -> str:
    m = int(mins)
    return f"{m // 60}h{m % 60:02d}m"


def _net_sentiment_4h(row: pd.Series) -> Optional[float]:
    cols = [c for c in row.index if c.startswith("sent_") and c.endswith("_4h")]
    if not cols:
        return None
    return float(sum(row[c] for c in cols))


class SignalEngine:
    def __init__(self, params: Optional[SignalParams] = None,
                 k_tp: float = 2.0, k_sl: float = 1.0):
        self.p = params or SignalParams()
        self.k_tp, self.k_sl = k_tp, k_sl

    def decide(
        self,
        instrument: str,
        now: pd.Timestamp,
        features_row: pd.Series,
        close: float,
        model,                       # .predict_proba(DataFrame) -> [[p_none,p_long,p_short]]
        tau: float,
        bar_age_intervals: float = 0.0,
        collector_age_sec: float = 0.0,
    ) -> Decision:
        snapshot = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in features_row.items()}
        signal_id = f"{instrument}-{now.strftime('%Y%m%d%H%M%S')}"

        def none(reason: str, confidence: float = 0.0) -> Decision:
            return Decision(NONE, confidence, None, None,
                            f"NONE {instrument} | {reason}", signal_id,
                            veto_reason=reason, features_snapshot=snapshot)

        # 1 — data freshness
        if bar_age_intervals > self.p.max_bar_age_intervals:
            return none(f"stale data: bar {bar_age_intervals:.1f} intervals old")
        if collector_age_sec > self.p.max_collector_age_sec:
            return none(f"stale data: collector heartbeat {collector_age_sec:.0f}s old")

        # 2 — calendar blackout
        if float(features_row.get("blackout", 0.0)) > 0:
            return none("news blackout")

        # 3 — regime filter: tradable if trending (ADX) OR vol in the earning band
        adx = float(features_row.get("adx14", 0.0))
        atr_pct = float(features_row.get("atr_pctile_90d", 0.5))
        if self.p.regime_enabled:
            lo, hi = self.p.atr_band
            if not (adx > self.p.min_adx or lo <= atr_pct <= hi):
                return none(f"regime: ADX {adx:.0f} <= {self.p.min_adx:.0f} "
                            f"and ATR pct {atr_pct:.2f} outside [{lo},{hi}]")

        # 4 — model signal
        proba = np.atleast_2d(model.predict_proba(features_row.to_frame().T))[0]
        p_long, p_short = float(proba[1]), float(proba[2])
        conf = max(p_long, p_short)
        if conf <= tau:
            return none(f"below tau: max(P)={conf:.3f} <= {tau:.2f}", confidence=conf)
        side = 1 if p_long >= p_short else -1
        action = LONG if side > 0 else SHORT

        # 5 — sentiment veto (config-gated; logged distinctly for later evaluation)
        net4h = _net_sentiment_4h(features_row)
        if self.p.sentiment_veto_enabled and net4h is not None:
            thr = self.p.sentiment_veto_threshold
            if side > 0 and net4h < -thr:
                return none(f"sentiment veto: net4h {net4h:+.2f} < -{thr} blocks LONG",
                            confidence=conf)
            if side < 0 and net4h > thr:
                return none(f"sentiment veto: net4h {net4h:+.2f} > +{thr} blocks SHORT",
                            confidence=conf)

        # 6 — confluence bonus (journal analytics only; never sizing)
        h4 = float(features_row.get("h4_trend", 0.0))
        sent_sign = np.sign(net4h) if net4h else 0.0
        if h4 == side and sent_sign == side:
            conf = min(conf + self.p.confluence_bonus, 1.0)

        atr = float(features_row.get("atr", features_row.get("atr_norm", 0.0) * close))
        sl = close - side * self.k_sl * atr
        tp = close + side * self.k_tp * atr
        h4_txt = {1.0: "H4 up", -1.0: "H4 down"}.get(h4, "H4 flat")
        sent_txt = f"sent {net4h:+.2f}" if net4h is not None else "sent n/a"
        nxt = features_row.get("mins_to_next_high")
        nxt_txt = f"next high-impact {_fmt_mins(float(nxt))}" if nxt is not None else ""
        explanation = (f"{action} {instrument} p={max(p_long, p_short):.2f} tau={tau:.2f} | "
                       f"{h4_txt} | {sent_txt} | ADX {adx:.0f}"
                       + (f" | {nxt_txt}" if nxt_txt else ""))
        return Decision(action, conf, sl, tp, explanation, signal_id,
                        features_snapshot=snapshot)


class DecisionStrategy:
    """Backtester adapter running the FULL signal engine per bar — the same
    code path the live orchestrator calls, so backtest == live behavior.
    Sizing comes from the injected sizer (the risk manager in Prompt 10+)."""

    def __init__(self, instrument: str, engine: SignalEngine, features: pd.DataFrame,
                 model, tau: float, horizon_bars: int, contract_size: float,
                 sizer=None, risk_frac: float = 0.0075,
                 min_lot: float = 0.01, lot_step: float = 0.01, bar_minutes: int = 15):
        self.instrument, self.engine, self.features = instrument, engine, features
        self.model, self.tau = model, tau
        self.hold = pd.Timedelta(minutes=horizon_bars * bar_minutes)
        self.contract_size, self.sizer = contract_size, sizer
        self.risk_frac, self.min_lot, self.lot_step = risk_frac, min_lot, lot_step
        self.decisions: list[Decision] = []

    def on_bar(self, ctx):
        orders = []
        mine = [p for p in ctx.positions if p.instrument == self.instrument]
        for p in mine:
            if ctx.time - p.entry_time >= self.hold:
                orders.append({"type": "close", "position_id": p.id, "fraction": 1.0,
                               "reason": "time_exit"})
                mine = [q for q in mine if q.id != p.id]
        if mine or ctx.time not in self.features.index:
            return orders
        row = self.features.loc[ctx.time]
        close = float(ctx.bars[self.instrument]["close"])
        d = self.engine.decide(self.instrument, ctx.time, row, close, self.model, self.tau)
        self.decisions.append(d)
        if d.action == NONE:
            return orders
        side = 1 if d.action == LONG else -1
        sl_dist = abs(close - d.sl_price)
        if self.sizer is not None:
            lots = self.sizer(ctx.equity, sl_dist)
        else:
            lots = np.floor((ctx.equity * self.risk_frac) / (sl_dist * self.contract_size)
                            / self.lot_step) * self.lot_step
        if lots < self.min_lot:
            return orders
        orders.append({"type": "open", "instrument": self.instrument, "side": side,
                       "lots": float(lots), "sl": d.sl_price, "tp": d.tp_price,
                       "tag": d.signal_id})
        return orders
