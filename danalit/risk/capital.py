"""Capital policy: equity-tier scaling with hysteresis, profit set-aside
ledger, and monthly withdrawal recommendations.

- Tier changes require N consecutive trading days beyond the boundary
  (no flapping), apply only at day rollover, never mid-trade, and notify.
- The set-aside ledger is VIRTUAL: money stays at the broker until manually
  withdrawn — by design no API can move funds. Reserved cash is excluded from
  the equity used for position sizing (working equity = equity - set-aside).
- Set-aside credits pause while equity is below the high-water mark:
  recover first, skim later.

CLI:  python -m danalit.risk.capital status | report | withdraw <amount>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from danalit.config import AppConfig, load_config
from danalit.db import connect
from danalit.logging_setup import setup_logging
from danalit.timeutil import iso, utc_now

log = setup_logging("capital")

_TIER_KEYS = ("tier_index", "tier_candidate", "tier_streak", "tier_last_day")


# ------------------------------------------------------------------ ledger

def set_aside_balance(db_path: Optional[Path] = None) -> float:
    con = connect(db_path or load_config().settings.paths.db_path)
    try:
        row = con.execute(
            "SELECT balance FROM set_aside_ledger ORDER BY id DESC LIMIT 1").fetchone()
        return float(row["balance"]) if row else 0.0
    finally:
        con.close()


def working_equity(broker_equity: float, db_path: Optional[Path] = None) -> float:
    """Sizing equity = broker equity minus virtually-reserved profits."""
    return broker_equity - set_aside_balance(db_path)


def month_end_close(
    month: str,                 # 'YYYY-MM'
    net_pnl: float,
    equity: float,
    hwm: float,
    set_aside_pct: float,
    db_path: Optional[Path] = None,
    min_withdrawal: Optional[float] = None,
) -> dict:
    """Credit the ledger at month end. Returns the ledger row + recommendation."""
    cfg = load_config()
    min_withdrawal = min_withdrawal or cfg.settings.capital.min_withdrawal
    paused = equity < hwm
    credit = set_aside_pct * net_pnl if (net_pnl > 0 and not paused) else 0.0
    balance = set_aside_balance(db_path) + credit
    note = ("credit paused: equity below HWM (recover first, skim later)"
            if paused and net_pnl > 0 else "")
    con = connect(db_path or cfg.settings.paths.db_path)
    try:
        with con:
            con.execute(
                "INSERT INTO set_aside_ledger (month, net_pnl, credit, withdrawal,"
                " balance, note) VALUES (?,?,?,?,?,?)",
                (month, net_pnl, credit, 0.0, balance, note))
    finally:
        con.close()
    units = cfg.settings.broker.units_per_usd
    return {"month": month, "credit": credit, "balance": balance, "paused": paused,
            "withdrawal_recommended": balance / units >= min_withdrawal}


def record_withdrawal(amount: float, db_path: Optional[Path] = None) -> float:
    """Record a manual broker withdrawal against the ledger; returns new balance."""
    balance = set_aside_balance(db_path)
    if amount <= 0 or amount > balance + 1e-9:
        raise ValueError(f"withdrawal {amount} exceeds set-aside balance {balance}")
    new_balance = balance - amount
    con = connect(db_path or load_config().settings.paths.db_path)
    try:
        with con:
            con.execute(
                "INSERT INTO set_aside_ledger (month, net_pnl, credit, withdrawal,"
                " balance, note) VALUES (?,?,?,?,?,?)",
                (utc_now().strftime("%Y-%m"), 0.0, 0.0, amount, new_balance,
                 "manual withdrawal recorded"))
    finally:
        con.close()
    log.info("withdrawal %.2f recorded; set-aside balance now %.2f", amount, new_balance)
    return new_balance


# ------------------------------------------------------------------- tiers

class TierManager:
    """Hysteresis-managed tier state, persisted in risk_state."""

    def __init__(self, cfg: Optional[AppConfig] = None, db_path: Optional[Path] = None,
                 notifier=None):
        self.cfg = cfg or load_config()
        self.db_path = db_path or self.cfg.settings.paths.db_path
        self.notifier = notifier
        s = self._load()
        self.tier_index = int(s.get("tier_index") or 0)
        self.candidate = s.get("tier_candidate")
        self.streak = int(s.get("tier_streak") or 0)
        self.last_day = s.get("tier_last_day")

    def _load(self) -> dict:
        con = connect(self.db_path)
        try:
            rows = con.execute(
                f"SELECT key, value FROM risk_state WHERE key IN "
                f"({','.join('?' * len(_TIER_KEYS))})", _TIER_KEYS).fetchall()
            return {r["key"]: json.loads(r["value"]) for r in rows}
        finally:
            con.close()

    def _save(self) -> None:
        state = {"tier_index": self.tier_index, "tier_candidate": self.candidate,
                 "tier_streak": self.streak, "tier_last_day": self.last_day}
        con = connect(self.db_path)
        try:
            with con:
                for k, v in state.items():
                    con.execute(
                        "INSERT INTO risk_state (key, value, updated_utc) VALUES (?,?,?)"
                        " ON CONFLICT (key) DO UPDATE SET value=excluded.value,"
                        " updated_utc=excluded.updated_utc",
                        (k, json.dumps(v), iso(utc_now())))
        finally:
            con.close()

    def raw_tier_index(self, equity: float) -> int:
        equity_usd = equity / self.cfg.settings.broker.units_per_usd
        for i, t in enumerate(self.cfg.settings.capital.tiers):
            if t.max_equity is None or equity_usd < t.max_equity:
                return i
        return len(self.cfg.settings.capital.tiers) - 1

    def daily_update(self, equity: float, day: str) -> Optional[int]:
        """Call once per trading-day rollover. Applies a tier change only after
        tier_hysteresis_days consecutive days beyond the boundary. Returns the
        new tier index if changed, else None."""
        if day == self.last_day:
            return None  # idempotent within a day
        self.last_day = day
        raw = self.raw_tier_index(equity)
        changed = None
        if raw == self.tier_index:
            self.candidate, self.streak = None, 0
        elif raw == self.candidate:
            self.streak += 1
            if self.streak >= self.cfg.settings.capital.tier_hysteresis_days:
                old = self.tier_index
                self.tier_index, self.candidate, self.streak = raw, None, 0
                changed = raw
                msg = (f"tier change {old} -> {raw}: instruments "
                       f"{self.cfg.settings.capital.tiers[raw].instruments}, risk "
                       f"{self.cfg.settings.capital.tiers[raw].risk_per_trade:.2%}")
                log.warning(msg)
                if self.notifier:
                    self.notifier.notify("WARNING", "Capital tier change", msg)
        else:
            self.candidate, self.streak = raw, 1
        self._save()
        return changed

    @property
    def tier(self):
        return self.cfg.settings.capital.tiers[self.tier_index]


# ------------------------------------------------------------------ report

def months_to_next_tier(equity: float, monthly_growth: float,
                        cfg: Optional[AppConfig] = None) -> Optional[float]:
    import math

    cfg = cfg or load_config()
    equity_usd = equity / cfg.settings.broker.units_per_usd
    for t in cfg.settings.capital.tiers:
        if t.max_equity is not None and equity_usd < t.max_equity:
            if monthly_growth <= 0:
                return None
            return math.log(t.max_equity / equity_usd) / math.log(1 + monthly_growth)
    return None


def projection_table(equity: float, months: int = 12,
                     rates: Optional[dict] = None) -> list[dict]:
    """Compounding projections — clearly labeled projections, not promises."""
    rates = rates or {"conservative": 0.01, "expected": 0.03, "optimistic": 0.05}
    rows = []
    for m in (3, 6, 12) if months >= 12 else range(1, months + 1):
        rows.append({"months": m, **{name: equity * (1 + r) ** m
                                     for name, r in rates.items()}})
    return rows


def monthly_report(equity: float, hwm: float, monthly_expectancy: float = 0.02,
                   db_path: Optional[Path] = None,
                   out_dir: Optional[Path] = None) -> Path:
    cfg = load_config()
    tm = TierManager(cfg, db_path)
    balance = set_aside_balance(db_path)
    work = equity - balance
    nxt = months_to_next_tier(equity, monthly_expectancy, cfg)
    month = utc_now().strftime("%Y-%m")
    units = cfg.settings.broker.units_per_usd

    lines = [
        f"# Capital policy report — {month}", "",
        f"- equity: {equity:.2f} (HWM {hwm:.2f}"
        + (", BELOW HWM — set-aside paused)" if equity < hwm else ")"),
        f"- tier: {tm.tier_index} — instruments {tm.tier.instruments}, "
        f"risk/trade {tm.tier.risk_per_trade:.2%}, set-aside {tm.tier.set_aside_pct:.0%}",
        f"- working equity (sizing basis): {work:.2f}",
        f"- set-aside balance: {balance:.2f} "
        f"({'WITHDRAWAL RECOMMENDED' if balance / units >= cfg.settings.capital.min_withdrawal else 'below withdrawal minimum'})",
        f"- months to next tier at {monthly_expectancy:.0%}/mo: "
        + (f"{nxt:.1f}" if nxt else "n/a"), "",
        "## Compounding projections (projections, not promises)", "",
        "| months | conservative | expected | optimistic |", "|---|---|---|---|",
    ]
    for r in projection_table(equity):
        lines.append(f"| {r['months']} | {r['conservative']:.2f} | "
                     f"{r['expected']:.2f} | {r['optimistic']:.2f} |")
    out_dir = out_dir or cfg.settings.paths.absolute("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"capital_{month}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Capital policy admin")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status")
    w = sub.add_parser("withdraw")
    w.add_argument("amount", type=float)
    r = sub.add_parser("report")
    r.add_argument("--equity", type=float, required=True)
    r.add_argument("--hwm", type=float, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "withdraw":
        print(f"new set-aside balance: {record_withdrawal(args.amount):.2f} "
              "(execute the actual broker withdrawal manually)")
    elif args.cmd == "report":
        print("report:", monthly_report(args.equity, args.hwm))
    else:
        tm = TierManager()
        print(json.dumps({"tier": tm.tier_index, "instruments": tm.tier.instruments,
                          "set_aside_balance": set_aside_balance()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
