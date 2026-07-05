"""Rule-based management of OPEN positions (entries are the signal engine's job).

Runs identically inside the backtester (each bar) and live (each loop tick),
operating on the normalized Position object. Emits engine-compatible action
dicts plus a management log for the journal.

RULE ORDERING (the contract, tested):
  1. NEWS PROTECTION overrides everything: inside a blackout window the only
     action taken is tighten-SL-to-0.5*ATR or flatten (per config); no other
     rule fires that tick.
  2. WEEKEND RULE: flatten at the configured Friday UTC time.
  3. BREAKEVEN before trailing: at >= +1.0*ATR unrealized, SL -> entry + spread.
  4. PARTIAL TP: close 50% at +1.0*ATR (skipped if position < 2*min_lot).
  5. ATR TRAILING (only after breakeven): SL trails 1.5*ATR behind the best
     price reached — monotonic, never widens.
  6. TIME EXIT last: close anything older than hold_bars.

Initial protection (SL = k_sl*ATR, TP = k_tp*ATR) is set at entry by the signal
engine using the SAME barriers as the training labels — that consistency is the
point of the whole design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ManageParams:
    breakeven_trigger_atr: float = 1.0
    partial_trigger_atr: float = 1.0
    partial_fraction: float = 0.5
    trail_atr: float = 1.5
    news_action: str = "tighten"        # 'tighten' | 'flatten' | 'off'
    news_tighten_atr: float = 0.5
    weekend_flatten: bool = True
    weekend_flatten_utc: str = "20:30"  # Friday
    hold_bars: int = 96
    bar_minutes: int = 15


@dataclass
class BarInfo:
    """Market context for one management tick."""
    time: pd.Timestamp
    close: float           # current bid close
    atr: float
    spread: float
    blackout: bool = False


class TradeManager:
    def __init__(self, params: Optional[ManageParams] = None, min_lot: float = 0.01):
        self.p = params or ManageParams()
        self.min_lot = min_lot
        self._partial_taken: set[int] = set()
        self._breakeven_done: set[int] = set()

    def adopt_state(self, pos) -> None:
        """After a live restart, re-derive per-position flags from broker facts."""
        if pos.sl is not None and pos.side * (pos.sl - pos.entry_price) >= 0:
            self._breakeven_done.add(pos.id)
        if pos.initial_lots and pos.lots < pos.initial_lots - 1e-9:
            self._partial_taken.add(pos.id)

    def manage(self, pos, bar: BarInfo) -> tuple[list[dict], list[dict]]:
        """Returns (engine-compatible orders, management log rows)."""
        orders: list[dict] = []
        logs: list[dict] = []
        side = pos.side
        profit = side * (bar.close - pos.entry_price)
        atr = bar.atr

        def log(rule: str, before, after) -> None:
            logs.append({"ts_utc": bar.time, "position_id": pos.id,
                         "instrument": pos.instrument, "rule": rule,
                         "before": before, "after": after})

        # 1 — news protection overrides everything this tick
        if bar.blackout and self.p.news_action != "off":
            if self.p.news_action == "flatten":
                orders.append({"type": "close", "position_id": pos.id, "fraction": 1.0,
                               "reason": "news_flatten"})
                log("news_flatten", {"lots": pos.lots}, {"lots": 0})
                return orders, logs
            tight = bar.close - side * self.p.news_tighten_atr * atr
            if pos.sl is None or side * (tight - pos.sl) > 0:
                orders.append({"type": "modify", "position_id": pos.id, "sl": tight})
                log("news_tighten", {"sl": pos.sl}, {"sl": tight})
            return orders, logs  # nothing else fires during a blackout

        # 2 — weekend flatten
        if self.p.weekend_flatten and bar.time.weekday() == 4:
            hh, mm = map(int, self.p.weekend_flatten_utc.split(":"))
            if (bar.time.hour, bar.time.minute) >= (hh, mm):
                orders.append({"type": "close", "position_id": pos.id, "fraction": 1.0,
                               "reason": "weekend_flatten"})
                log("weekend_flatten", {"lots": pos.lots}, {"lots": 0})
                return orders, logs

        # 3 — breakeven before trailing  (tiny epsilon: fp-safe >= at thresholds)
        eps = 1e-9 * max(atr, 1e-9)
        new_sl = pos.sl
        if pos.id not in self._breakeven_done \
                and profit >= self.p.breakeven_trigger_atr * atr - eps:
            be = pos.entry_price + side * bar.spread
            if new_sl is None or side * (be - new_sl) > 0:
                new_sl = be
                log("breakeven", {"sl": pos.sl}, {"sl": be})
            self._breakeven_done.add(pos.id)

        # 4 — partial take-profit
        if (pos.id not in self._partial_taken
                and profit >= self.p.partial_trigger_atr * atr - eps):
            if pos.lots >= 2 * self.min_lot:
                orders.append({"type": "close", "position_id": pos.id,
                               "fraction": self.p.partial_fraction, "reason": "partial_tp"})
                log("partial_tp", {"lots": pos.lots},
                    {"lots": pos.lots * (1 - self.p.partial_fraction)})
                self._partial_taken.add(pos.id)
            else:
                log("partial_skip_min_lot", {"lots": pos.lots}, {"lots": pos.lots})
                self._partial_taken.add(pos.id)  # do not re-log every bar

        # 5 — ATR trailing, only after breakeven, monotonic only
        if pos.id in self._breakeven_done:
            best = pos.best_price if pos.best_price else bar.close
            trail = best - side * self.p.trail_atr * atr
            if side * (trail - (new_sl if new_sl is not None else -side * 1e18)) > 0:
                log("trail", {"sl": new_sl}, {"sl": trail})
                new_sl = trail

        if new_sl is not None and new_sl != pos.sl:
            orders.append({"type": "modify", "position_id": pos.id, "sl": new_sl})

        # 6 — time exit last
        age = bar.time - pos.entry_time
        if age >= pd.Timedelta(minutes=self.p.hold_bars * self.p.bar_minutes):
            orders.append({"type": "close", "position_id": pos.id, "fraction": 1.0,
                           "reason": "time_exit"})
            log("time_exit", {"age_min": age.total_seconds() / 60}, {"lots": 0})

        return orders, logs


class ManagedStrategy:
    """Backtester adapter: entry strategy + TradeManager on every bar.

    atr/blackout looked up from the entry strategy's probs frame so backtest
    management sees exactly what the live loop will see.
    """

    def __init__(self, entry_strategy, manager: TradeManager, instrument: str,
                 probs: pd.DataFrame):
        self.entry = entry_strategy
        self.manager = manager
        self.instrument = instrument
        self.probs = probs
        self.management_log: list[dict] = []

    def on_bar(self, ctx):
        orders = []
        row = self.probs.loc[ctx.time] if ctx.time in self.probs.index else None
        if row is not None:
            bar = BarInfo(
                time=ctx.time,
                close=float(ctx.bars[self.instrument]["close"]),
                atr=float(row["atr"]),
                spread=float(row.get("spread", 0.0)) or 0.0,
                blackout=bool(row.get("blackout", 0.0)),
            )
            for pos in [p for p in ctx.positions if p.instrument == self.instrument]:
                o, logs = self.manager.manage(pos, bar)
                orders.extend(o)
                self.management_log.extend(logs)
        closed = {o.get("position_id") for o in orders
                  if o["type"] == "close" and o.get("fraction", 1.0) >= 1.0}
        if closed:
            remaining = [p for p in ctx.positions if p.id not in closed]
            ctx = type(ctx)(time=ctx.time, bars=ctx.bars, history=ctx.history,
                            positions=remaining, equity=ctx.equity, balance=ctx.balance)
        orders.extend(self.entry.on_bar(ctx) or [])
        return orders
