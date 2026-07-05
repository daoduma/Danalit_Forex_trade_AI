"""Event-driven, multi-instrument backtester.

Lifecycle semantics deliberately match MT5 (Prompts 11-15 reuse them live):
- Orders created at bar close execute at the NEXT bar's open +/- slippage.
- SL/TP live "at the broker": resolved intrabar from OHLC with the pessimistic
  rule (both inside one bar => SL first). Gaps fill at the gapped open.
- Swap accrues per UTC-midnight crossing, tripled leaving Wednesday.
- Margin from contract specs at configured leverage; orders that don't fit are
  rejected.

Strategy interface:  strategy.on_bar(ctx) -> list[order dict]
  {"type": "open", "instrument", "side": +1|-1, "lots", "sl", "tp", "tag"}
  {"type": "modify", "position_id", "sl": px|None, "tp": px|None}
  {"type": "close", "position_id", "fraction": 0..1}
  {"type": "close_all"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from danalit.backtest.costs import CostModel


@dataclass
class Position:
    id: int
    instrument: str
    side: int  # +1 long, -1 short
    lots: float
    entry_price: float
    entry_time: pd.Timestamp
    sl: Optional[float]
    tp: Optional[float]
    contract_size: float
    tag: str = ""
    swap_accrued: float = 0.0  # price units
    initial_lots: float = 0.0
    best_price: float = 0.0    # best BID reached (for MAE/MFE + trailing)
    worst_price: float = 0.0
    spread_at_entry: float = 0.0  # price units, one spread per round turn

    def unrealized(self, bid: float, spread: float) -> float:
        px = bid if self.side > 0 else bid + spread
        return self.side * (px - self.entry_price + self.side * self.swap_accrued) \
            * self.lots * self.contract_size


@dataclass
class Trade:
    position_id: int
    instrument: str
    side: int
    lots: float
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    gross_pnl: float
    commission: float
    swap: float
    net_pnl: float
    exit_reason: str
    tag: str = ""
    mae: float = 0.0
    mfe: float = 0.0
    spread_cost: float = 0.0  # account ccy, implicit in the fill prices


@dataclass
class Context:
    """What a strategy sees at one bar close."""
    time: pd.Timestamp
    bars: dict            # instrument -> bar row (Series) at this timestamp
    history: dict         # instrument -> DataFrame of bars up to and incl. this one
    positions: list
    equity: float
    balance: float


class Backtester:
    def __init__(
        self,
        bars: dict[str, pd.DataFrame],
        costs: dict[str, CostModel],
        contract_sizes: dict[str, float],
        initial_balance: float = 20.0,
        leverage: float = 500.0,
        risk_check=None,  # optional: callable(order, equity, positions) -> (ok, reason)
    ):
        for name, df in bars.items():
            if not df["time_utc"].is_monotonic_increasing:
                raise ValueError(f"{name} bars not sorted")
        self.bars = {k: df.reset_index(drop=True) for k, df in bars.items()}
        self.costs = costs
        self.contract_sizes = contract_sizes
        self.balance = initial_balance
        self.leverage = leverage
        self.risk_check = risk_check
        self.positions: list[Position] = []
        self.trades: list[Trade] = []
        self.pending: list[dict] = []
        self.equity_curve: list[dict] = []
        self.rejections: list[dict] = []
        self._next_id = 1

    # ------------------------------------------------------------------ utils
    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _margin_used(self, prices: dict[str, float]) -> float:
        return sum(
            p.lots * p.contract_size * prices.get(p.instrument, p.entry_price) / self.leverage
            for p in self.positions
        )

    def equity(self, bids: dict[str, float]) -> float:
        eq = self.balance
        for p in self.positions:
            spread = self.costs[p.instrument].spread
            eq += p.unrealized(bids.get(p.instrument, p.entry_price), spread)
        return eq

    # ------------------------------------------------------------- lifecycle
    def _fill_pending(self, t: pd.Timestamp, opens: dict[str, pd.Series]) -> None:
        still = []
        for order in self.pending:
            inst = order.get("instrument")
            if order["type"] == "open" and inst in opens:
                self._open_position(order, t, opens[inst])
            elif order["type"] in ("modify", "close", "close_all"):
                self._apply_management(order, t, opens)
            else:
                still.append(order)  # instrument had no bar this timestamp
        self.pending = still

    def _open_position(self, order: dict, t: pd.Timestamp, bar: pd.Series) -> None:
        inst = order["instrument"]
        cost = self.costs[inst]
        side = int(order["side"])
        spread = cost.spread_at(bar.get("spread"))
        slip = cost.slippage_at(t)
        open_bid = float(bar["open"])
        price = open_bid + spread + slip if side > 0 else open_bid - slip

        if self.risk_check is not None:
            ok, reason = self.risk_check(order, self.equity({inst: open_bid}), self.positions)
            if not ok:
                self.rejections.append({"time": t, "order": order, "reason": reason})
                return
        # margin check
        need = order["lots"] * self.contract_sizes[inst] * open_bid / self.leverage
        eq = self.equity({inst: open_bid})
        if need + self._margin_used({inst: open_bid}) > eq:
            self.rejections.append({"time": t, "order": order, "reason": "insufficient margin"})
            return

        pos = Position(
            id=self._alloc_id(), instrument=inst, side=side, lots=float(order["lots"]),
            entry_price=price, entry_time=t, sl=order.get("sl"), tp=order.get("tp"),
            contract_size=self.contract_sizes[inst], tag=order.get("tag", ""),
            initial_lots=float(order["lots"]), best_price=open_bid, worst_price=open_bid,
            spread_at_entry=spread,
        )
        self.positions.append(pos)

    def _apply_management(self, order: dict, t: pd.Timestamp, opens: dict[str, pd.Series]) -> None:
        if order["type"] == "close_all":
            for p in list(self.positions):
                bar = opens.get(p.instrument)
                if bar is not None:
                    self._close(p, t, float(bar["open"]), 1.0, "close_all")
            return
        pos = next((p for p in self.positions if p.id == order.get("position_id")), None)
        if pos is None:
            return
        if order["type"] == "modify":
            if order.get("sl") is not None:
                pos.sl = order["sl"]
            if order.get("tp") is not None:
                pos.tp = order["tp"]
        elif order["type"] == "close":
            bar = opens.get(pos.instrument)
            if bar is not None:
                self._close(pos, t, float(bar["open"]), float(order.get("fraction", 1.0)),
                            order.get("reason", "manual"))

    def _close(self, pos: Position, t: pd.Timestamp, bid: float, fraction: float,
               reason: str) -> None:
        cost = self.costs[pos.instrument]
        lots = round(min(pos.lots, pos.lots if fraction >= 1.0 else pos.lots * fraction), 4)
        if lots <= 0:
            return
        slip = cost.slippage_at(t) if reason in ("manual", "close_all", "time_exit") else 0.0
        exit_px = (bid - slip) if pos.side > 0 else (bid + cost.spread + slip)
        gross = pos.side * (exit_px - pos.entry_price) * lots * pos.contract_size
        swap = pos.side * pos.swap_accrued * lots * pos.contract_size
        commission = cost.commission_per_lot * lots
        net = gross + swap - commission
        self.balance += net
        atr_denom = pos.contract_size * pos.lots or 1.0
        self.trades.append(Trade(
            position_id=pos.id, instrument=pos.instrument, side=pos.side, lots=lots,
            entry_time=pos.entry_time, entry_price=pos.entry_price,
            exit_time=t, exit_price=exit_px,
            gross_pnl=gross, commission=commission, swap=swap, net_pnl=net,
            exit_reason=reason, tag=pos.tag,
            mae=pos.side * (pos.worst_price - pos.entry_price),
            mfe=pos.side * (pos.best_price - pos.entry_price),
            spread_cost=pos.spread_at_entry * lots * pos.contract_size,
        ))
        pos.lots = round(pos.lots - lots, 4)
        if pos.lots <= 1e-9:
            self.positions.remove(pos)

    def _check_stops(self, t: pd.Timestamp, inst: str, bar: pd.Series) -> None:
        """Intrabar SL/TP with pessimistic rule and gap-at-open fills."""
        cost = self.costs[inst]
        spread = cost.spread_at(bar.get("spread"))
        o, h, l = float(bar["open"]), float(bar["high"]), float(bar["low"])
        for pos in [p for p in self.positions if p.instrument == inst]:
            if pos.side > 0:  # exits at bid
                bid_o, bid_h, bid_l = o, h, l
                sl, tp = pos.sl, pos.tp
                if sl is not None and bid_o <= sl:
                    self._close(pos, t, bid_o, 1.0, "sl_gap"); continue
                if tp is not None and bid_o >= tp:
                    self._close(pos, t, bid_o, 1.0, "tp_gap"); continue
                if sl is not None and bid_l <= sl:  # pessimistic: SL before TP
                    self._close(pos, t, sl, 1.0, "sl"); continue
                if tp is not None and bid_h >= tp:
                    self._close(pos, t, tp, 1.0, "tp"); continue
            else:  # short exits at ask = bid + spread
                ask_o, ask_h, ask_l = o + spread, h + spread, l + spread
                sl, tp = pos.sl, pos.tp
                if sl is not None and ask_o >= sl:
                    self._close(pos, t, ask_o - spread, 1.0, "sl_gap"); continue
                if tp is not None and ask_o <= tp:
                    self._close(pos, t, ask_o - spread, 1.0, "tp_gap"); continue
                if sl is not None and ask_h >= sl:
                    self._close(pos, t, sl - spread, 1.0, "sl"); continue
                if tp is not None and ask_l <= tp:
                    self._close(pos, t, tp - spread, 1.0, "tp"); continue

    def _accrue_swap(self, prev_t: Optional[pd.Timestamp], t: pd.Timestamp) -> None:
        if prev_t is None:
            return
        nights = pd.date_range(prev_t.normalize() + pd.Timedelta(days=1), t.normalize(), freq="1D")
        for midnight in nights:
            from_weekday = (midnight - pd.Timedelta(days=1)).weekday()
            for pos in self.positions:
                pos.swap_accrued += pos.side * self.costs[pos.instrument].swap_for_night(
                    pos.side, from_weekday)

    # ------------------------------------------------------------------- run
    def run(self, strategy) -> dict:
        frames = {k: df.set_index("time_utc") for k, df in self.bars.items()}
        all_times = sorted(set().union(*[set(df.index) for df in frames.values()]))
        prev_t = None
        for t in all_times:
            bars_now = {k: df.loc[t] for k, df in frames.items() if t in df.index}
            self._accrue_swap(prev_t, t)
            # 1) fills of orders queued at the previous close, at this open
            self._fill_pending(t, bars_now)
            # 2) intrabar SL/TP resolution + excursion tracking
            for inst, bar in bars_now.items():
                self._check_stops(t, inst, bar)
                for p in self.positions:
                    if p.instrument == inst:
                        hi, lo = float(bar["high"]), float(bar["low"])
                        if p.side > 0:
                            p.best_price = max(p.best_price, hi)
                            p.worst_price = min(p.worst_price, lo)
                        else:
                            p.best_price = min(p.best_price or hi, lo)
                            p.worst_price = max(p.worst_price, hi)
            # 3) strategy decides at bar close -> orders fill next bar
            closes = {k: float(b["close"]) for k, b in bars_now.items()}
            ctx = Context(
                time=t, bars=bars_now,
                history={k: frames[k].loc[:t] for k in bars_now},
                positions=list(self.positions),
                equity=self.equity(closes), balance=self.balance,
            )
            orders = strategy.on_bar(ctx) or []
            self.pending.extend(orders)
            # 4) snapshot
            self.equity_curve.append({
                "time_utc": t, "balance": self.balance, "equity": ctx.equity,
                "margin": self._margin_used(closes), "n_positions": len(self.positions),
            })
            prev_t = t

        # liquidate anything left at final close for a clean accounting
        if self.positions and all_times:
            t = all_times[-1]
            for p in list(self.positions):
                last_bid = float(frames[p.instrument].iloc[-1]["close"])
                self._close(p, t, last_bid, 1.0, "end_of_data")

        eq = pd.DataFrame(self.equity_curve).set_index("time_utc") if self.equity_curve else pd.DataFrame()
        return {"trades": self.trades, "equity_curve": eq,
                "rejections": self.rejections, "final_balance": self.balance}


# --------------------------------------------------------------------- toys
class BuyAndHold:
    """Open one long on the first bar; hold to the end."""

    def __init__(self, instrument: str, lots: float):
        self.instrument, self.lots, self.done = instrument, lots, False

    def on_bar(self, ctx: Context):
        if not self.done and self.instrument in ctx.bars:
            self.done = True
            return [{"type": "open", "instrument": self.instrument, "side": 1,
                     "lots": self.lots, "sl": None, "tp": None, "tag": "buyhold"}]
        return []


class MACross:
    """Toy validation strategy: fast/slow SMA cross, one position at a time."""

    def __init__(self, instrument: str, lots: float, fast: int = 20, slow: int = 50):
        self.instrument, self.lots, self.fast, self.slow = instrument, lots, fast, slow

    def on_bar(self, ctx: Context):
        hist = ctx.history.get(self.instrument)
        if hist is None or len(hist) < self.slow + 2:
            return []
        c = hist["close"]
        fast_now, slow_now = c.iloc[-self.fast:].mean(), c.iloc[-self.slow:].mean()
        fast_prev = c.iloc[-self.fast - 1:-1].mean()
        slow_prev = c.iloc[-self.slow - 1:-1].mean()
        mine = [p for p in ctx.positions if p.instrument == self.instrument]
        orders = []
        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_dn = fast_prev >= slow_prev and fast_now < slow_now
        if crossed_up or crossed_dn:
            for p in mine:
                orders.append({"type": "close", "position_id": p.id, "fraction": 1.0})
            orders.append({"type": "open", "instrument": self.instrument,
                           "side": 1 if crossed_up else -1, "lots": self.lots,
                           "sl": None, "tp": None, "tag": "macross"})
        return orders
