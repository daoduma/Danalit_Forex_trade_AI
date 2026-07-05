"""The stateful risk gatekeeper — the most safety-critical code in Danalit.

Every order the system ever sends passes through check_order(), in backtest and
live identically. State (high-water mark, daily/weekly anchors, loss streak,
halts, breaker) persists to SQLite so a restart can never reset a limit.

Manual breaker reset (after the mandatory review, see RUNBOOK):
    python -m danalit.risk.risk_manager --reset-breaker
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from danalit.config import AppConfig, CapitalTier, load_config
from danalit.db import connect
from danalit.logging_setup import setup_logging
from danalit.risk.position_sizing import SizingResult, size_position
from danalit.timeutil import iso, parse_iso, utc_now

log = setup_logging("risk")

STATE_KEYS = ("hwm", "day_anchor_date", "day_anchor_equity", "week_anchor_key",
              "week_anchor_equity", "consec_losses", "brake_until",
              "daily_halt_until", "weekly_halt_until", "breaker_active")


@dataclass
class Approved:
    sizing: SizingResult
    risk_frac_used: float

    ok = True


@dataclass
class Rejected:
    reason: str

    ok = False


class RiskManager:
    def __init__(
        self,
        cfg: Optional[AppConfig] = None,
        db_path: Optional[Path] = None,
        now_fn: Callable = utc_now,
    ):
        self.cfg = cfg or load_config()
        self.db_path = db_path or self.cfg.settings.paths.db_path
        self.now = now_fn
        self.r = self.cfg.settings.risk
        s = self._load_state()
        self.hwm: float = s.get("hwm") or 0.0
        self.day_anchor_date: Optional[str] = s.get("day_anchor_date")
        self.day_anchor_equity: float = s.get("day_anchor_equity") or 0.0
        self.week_anchor_key: Optional[str] = s.get("week_anchor_key")
        self.week_anchor_equity: float = s.get("week_anchor_equity") or 0.0
        self.consec_losses: int = int(s.get("consec_losses") or 0)
        self.brake_until = parse_iso(s.get("brake_until"))
        self.daily_halt_until = parse_iso(s.get("daily_halt_until"))
        self.weekly_halt_until = parse_iso(s.get("weekly_halt_until"))
        self.breaker_active: bool = bool(int(s.get("breaker_active") or 0))

    # ------------------------------------------------------------ persistence
    def _load_state(self) -> dict:
        con = connect(self.db_path)
        try:
            rows = con.execute("SELECT key, value FROM risk_state").fetchall()
            return {r["key"]: json.loads(r["value"]) for r in rows}
        finally:
            con.close()

    def _save(self) -> None:
        state = {
            "hwm": self.hwm,
            "day_anchor_date": self.day_anchor_date,
            "day_anchor_equity": self.day_anchor_equity,
            "week_anchor_key": self.week_anchor_key,
            "week_anchor_equity": self.week_anchor_equity,
            "consec_losses": self.consec_losses,
            "brake_until": iso(self.brake_until) if self.brake_until is not None else None,
            "daily_halt_until": iso(self.daily_halt_until) if self.daily_halt_until is not None else None,
            "weekly_halt_until": iso(self.weekly_halt_until) if self.weekly_halt_until is not None else None,
            "breaker_active": int(self.breaker_active),
        }
        con = connect(self.db_path)
        try:
            with con:
                for k, v in state.items():
                    con.execute(
                        "INSERT INTO risk_state (key, value, updated_utc) VALUES (?,?,?)"
                        " ON CONFLICT (key) DO UPDATE SET value=excluded.value,"
                        " updated_utc=excluded.updated_utc",
                        (k, json.dumps(v), iso(self.now())),
                    )
        finally:
            con.close()

    def _log_event(self, type_: str, detail: str) -> None:
        log.warning("%s: %s", type_, detail)
        con = connect(self.db_path)
        try:
            with con:
                con.execute("INSERT INTO system_events (ts_utc, type, detail) VALUES (?,?,?)",
                            (iso(self.now()), type_, detail))
        finally:
            con.close()

    # ------------------------------------------------------------------ tiers
    def tier(self, equity: float) -> CapitalTier:
        """Tier boundaries are USD; equity arrives in account units (maybe cents)."""
        equity_usd = equity / self.cfg.settings.broker.units_per_usd
        for t in self.cfg.settings.capital.tiers:
            if t.max_equity is None or equity_usd < t.max_equity:
                return t
        return self.cfg.settings.capital.tiers[-1]

    def risk_frac(self, equity: float) -> float:
        base = self.tier(equity).risk_per_trade
        now = pd.Timestamp(self.now())
        if self.brake_until is not None and now < self.brake_until:
            return base * self.r.brake_risk_factor
        return base

    # -------------------------------------------------------------- lifecycle
    def on_equity_snapshot(self, equity: float, ts=None) -> list[str]:
        """Update anchors/HWM; returns any newly-triggered halt events."""
        now = pd.Timestamp(ts or self.now())
        events: list[str] = []
        today = now.strftime("%Y-%m-%d")
        week = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"

        if self.day_anchor_date != today:
            self.day_anchor_date, self.day_anchor_equity = today, equity
        if self.week_anchor_key != week:
            self.week_anchor_key, self.week_anchor_equity = week, equity
        if equity > self.hwm:
            self.hwm = equity

        if (self.day_anchor_equity > 0
                and equity <= self.day_anchor_equity * (1 - self.r.daily_loss_limit)
                and (self.daily_halt_until is None or now >= self.daily_halt_until)):
            self.daily_halt_until = (now.normalize() + pd.Timedelta(days=1))
            events.append("daily_halt")
            self._log_event("daily_halt", f"equity {equity:.2f} breached "
                            f"{100 * self.r.daily_loss_limit}% daily loss from "
                            f"{self.day_anchor_equity:.2f}; halted until {self.daily_halt_until}")
        if (self.week_anchor_equity > 0
                and equity <= self.week_anchor_equity * (1 - self.r.weekly_loss_limit)
                and (self.weekly_halt_until is None or now >= self.weekly_halt_until)):
            days_to_monday = (7 - now.weekday()) % 7 or 7
            self.weekly_halt_until = now.normalize() + pd.Timedelta(days=days_to_monday)
            events.append("weekly_halt")
            self._log_event("weekly_halt", f"weekly loss limit hit; halted until "
                            f"{self.weekly_halt_until}")
        if self.hwm > 0 and equity <= self.hwm * (1 - self.r.max_drawdown) \
                and not self.breaker_active:
            self.breaker_active = True
            events.append("FLATTEN_AND_HALT")
            self._log_event("drawdown_breaker",
                            f"equity {equity:.2f} is {100 * self.r.max_drawdown}% below "
                            f"HWM {self.hwm:.2f} — FLATTEN_AND_HALT; manual reset required")
        self._save()
        return events

    def on_trade_closed(self, net_pnl: float, ts=None) -> None:
        now = pd.Timestamp(ts or self.now())
        if net_pnl < 0:
            self.consec_losses += 1
            if self.consec_losses >= self.r.consecutive_loss_brake:
                self.brake_until = now + pd.Timedelta(hours=self.r.brake_hours)
                self.consec_losses = 0
                self._log_event("loss_brake",
                                f"{self.r.consecutive_loss_brake} straight losses — risk halved "
                                f"until {self.brake_until}")
        else:
            self.consec_losses = 0
        self._save()

    def halted_reason(self, ts=None) -> Optional[str]:
        now = pd.Timestamp(ts or self.now())
        if self.breaker_active:
            return "drawdown breaker active — manual reset required (see RUNBOOK)"
        if self.daily_halt_until is not None and now < self.daily_halt_until:
            return f"daily loss limit — halted until {self.daily_halt_until}"
        if self.weekly_halt_until is not None and now < self.weekly_halt_until:
            return f"weekly loss limit — halted until {self.weekly_halt_until}"
        return None

    # ------------------------------------------------------------ gatekeeper
    def check_order(
        self,
        instrument: str,
        side: int,
        price: float,
        sl_distance: float,
        equity: float,
        open_positions: Optional[list[dict]] = None,  # [{'instrument', 'risk_amount'}]
        margin_available: Optional[float] = None,
    ):
        open_positions = open_positions or []
        reason = self.halted_reason()
        if reason:
            return self._reject(instrument, reason)

        tier = self.tier(equity)
        if instrument not in tier.instruments:
            return self._reject(
                instrument, f"instrument gated by equity tier "
                f"(equity {equity:.2f}: {tier.instruments} only)")

        if len(open_positions) >= self.r.max_positions:
            return self._reject(instrument, f"max concurrent positions ({self.r.max_positions})")
        same = sum(1 for p in open_positions if p["instrument"] == instrument)
        if same >= self.r.max_positions_per_instrument:
            return self._reject(instrument, "max positions per instrument")

        inst = self.cfg.instruments[instrument]
        frac = self.risk_frac(equity)
        sizing = size_position(
            equity=equity, risk_frac=frac, sl_distance=sl_distance,
            value_per_price_unit_per_lot=inst.contract_size,
            min_lot=inst.min_lot, lot_step=inst.lot_step, max_lot=inst.max_lot,
            price=price, leverage=self.cfg.settings.broker.leverage,
            margin_available=margin_available,
            hard_cap_mult=self.r.min_lot_risk_cap_mult,
        )
        if sizing.refused:
            return self._reject(instrument, sizing.reason)

        open_risk = sum(p.get("risk_amount", 0.0) for p in open_positions)
        if (open_risk + sizing.risk_amount) / equity > self.r.max_total_risk + 1e-12:
            return self._reject(
                instrument,
                f"total open risk {(open_risk + sizing.risk_amount) / equity:.2%} would exceed "
                f"{self.r.max_total_risk:.2%}")

        return Approved(sizing=sizing, risk_frac_used=frac)

    def _reject(self, instrument: str, reason: str) -> Rejected:
        self._log_event("order_rejected", f"{instrument}: {reason}")
        return Rejected(reason=reason)

    # ------------------------------------------------------------------ admin
    def reset_breaker(self) -> None:
        self.breaker_active = False
        self._save()
        self._log_event("breaker_reset", "drawdown breaker manually reset")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Risk manager admin")
    ap.add_argument("--reset-breaker", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)
    rm = RiskManager()
    if args.reset_breaker:
        print("Before resetting, complete the RUNBOOK review: cause identified? "
              "Data feeds healthy? Model behaving? Costs in line?")
        rm.reset_breaker()
        print("Breaker reset. Trading may resume at next startup.")
        return 0
    print(json.dumps({
        "hwm": rm.hwm, "breaker_active": rm.breaker_active,
        "halted": rm.halted_reason(), "consec_losses": rm.consec_losses,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
