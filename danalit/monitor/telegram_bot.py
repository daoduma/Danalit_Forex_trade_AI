"""Telegram remote control: /status /positions /halt /halt_flat /resume /report.

Runs as a SIDECAR process (scripts/run_telegram_bot.py), not inside the
orchestrator: a bot crash then cannot take down the trading loop, and the bot
keeps answering /status even while the orchestrator is halted. It talks to the
system only through the journal DB and the kill-switch/RESUME files — the same
mechanisms a human at the keyboard would use.

Only the whitelisted DANALIT_TG_CHAT_ID may issue commands; everyone else is
rejected and logged. All text-building is in pure functions, testable without
the python-telegram-bot dependency (which is lazy-imported).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from danalit.config import load_config
from danalit.data.collector_daemon import heartbeat_age_seconds
from danalit.db import connect
from danalit.logging_setup import setup_logging
from danalit.timeutil import parse_iso, utc_now

log = setup_logging("telegram_bot")

REPO_ROOT = Path(__file__).resolve().parents[2]


def is_authorized(chat_id, allowed: Optional[str] = None) -> bool:
    allowed = allowed or os.environ.get("DANALIT_TG_CHAT_ID", "")
    return bool(allowed) and str(chat_id) == str(allowed)


# ------------------------------------------------------------- command logic

def cmd_halt(kill_dir: Path = REPO_ROOT, flat: bool = False) -> str:
    name = "HALT_FLAT" if flat else "HALT"
    (kill_dir / name).write_text("via telegram", encoding="utf-8")
    return (f"{name} file created — no new entries"
            + ("; open positions will be flattened" if flat else "")
            + ". Resume is explicit: /resume")


def cmd_resume(kill_dir: Path = REPO_ROOT) -> str:
    removed = []
    for name in ("HALT", "HALT_FLAT"):
        p = kill_dir / name
        if p.exists():
            p.unlink()
            removed.append(name)
    (kill_dir / "RESUME").write_text("via telegram", encoding="utf-8")
    return (f"removed {removed or 'no kill files'}; RESUME requested — "
            "the orchestrator will re-reconcile and resume on its next tick")


def build_status(db_path=None, kill_dir: Path = REPO_ROOT,
                 orchestrator_heartbeat: Optional[Path] = None) -> str:
    cfg = load_config()
    con = connect(db_path or cfg.settings.paths.db_path)
    try:
        state_row = con.execute(
            "SELECT ts_utc, detail FROM system_events WHERE type='state_transition'"
            " ORDER BY id DESC LIMIT 1").fetchone()
        eq_row = con.execute(
            "SELECT * FROM equity_snapshots ORDER BY id DESC LIMIT 1").fetchone()
        today = utc_now().strftime("%Y-%m-%d")
        day_pnl = con.execute(
            "SELECT COALESCE(SUM(net_pnl),0) s FROM trades WHERE closed_utc LIKE ?",
            (today + "%",)).fetchone()["s"]
        n_dec = con.execute(
            "SELECT COUNT(*) c FROM decisions WHERE ts_utc LIKE ?", (today + "%",)
        ).fetchone()["c"]
    finally:
        con.close()

    hb_path = orchestrator_heartbeat or cfg.settings.paths.absolute("logs") / "orchestrator.heartbeat"
    orch_age = "n/a"
    if hb_path.exists():
        ts = parse_iso(hb_path.read_text(encoding="utf-8").strip())
        if ts is not None:
            orch_age = f"{(pd.Timestamp(utc_now()) - ts).total_seconds():.0f}s"
    col_age = heartbeat_age_seconds()
    kill = [n for n in ("HALT", "HALT_FLAT") if (kill_dir / n).exists()]

    lines = ["Danalit status"]
    lines.append(f"state: {state_row['detail'] if state_row else 'never started'}")
    if eq_row:
        lines.append(f"equity: {eq_row['equity']:.2f} (bal {eq_row['balance']:.2f}, "
                     f"open risk {eq_row['open_risk']:.2f}) [{eq_row['mode']}]")
    lines.append(f"today: P&L {day_pnl:+.2f}, {n_dec} decisions")
    lines.append(f"heartbeats: orchestrator {orch_age}, "
                 f"collector {f'{col_age:.0f}s' if col_age is not None else 'n/a'}")
    lines.append(f"kill switch: {kill or 'none'}")
    return "\n".join(lines)


def build_positions(db_path=None) -> str:
    """Open positions per the journal (the bot never touches the gateway)."""
    cfg = load_config()
    con = connect(db_path or cfg.settings.paths.db_path)
    try:
        rows = con.execute(
            """SELECT o.instrument, o.side, o.lots, o.sl, o.tp, o.filled_price, o.ts_utc
               FROM orders o WHERE o.status='filled'
               AND NOT EXISTS (SELECT 1 FROM trades t WHERE t.signal_id=o.client_id
                               AND t.closed_utc IS NOT NULL)""").fetchall()
    finally:
        con.close()
    if not rows:
        return "no open positions (per journal)"
    return "\n".join(
        f"{r['instrument']} {r['side']} {r['lots']} @ {r['filled_price']} "
        f"SL {r['sl']} TP {r['tp']} since {r['ts_utc']}" for r in rows)


def build_digest(db_path=None, date: Optional[str] = None) -> str:
    """Daily digest: equity, P&L, trades with explanations, risk headroom, health."""
    cfg = load_config()
    date = date or utc_now().strftime("%Y-%m-%d")
    con = connect(db_path or cfg.settings.paths.db_path)
    try:
        eq = con.execute("SELECT * FROM equity_snapshots ORDER BY id DESC LIMIT 1").fetchone()
        trades = con.execute("SELECT * FROM trades WHERE closed_utc LIKE ?",
                             (date + "%",)).fetchall()
        decisions = con.execute(
            "SELECT action, COUNT(*) c FROM decisions WHERE ts_utc LIKE ? GROUP BY action",
            (date + "%",)).fetchall()
        expl = con.execute(
            """SELECT d.explanation FROM decisions d JOIN orders o
               ON o.signal_id = d.signal_id WHERE d.ts_utc LIKE ? LIMIT 10""",
            (date + "%",)).fetchall()
    finally:
        con.close()
    day_pnl = sum(t["net_pnl"] or 0 for t in trades)
    r = cfg.settings.risk
    lines = [f"Danalit daily digest — {date}"]
    if eq:
        lines.append(f"equity {eq['equity']:.2f} | balance {eq['balance']:.2f} "
                     f"| open risk {eq['open_risk']:.2f}")
        lines.append(f"risk headroom: daily halt at -{r.daily_loss_limit:.0%}, "
                     f"weekly at -{r.weekly_loss_limit:.0%} of anchors")
    lines.append(f"closed trades: {len(trades)}, day P&L {day_pnl:+.2f}")
    for t in trades[:10]:
        lines.append(f"  {t['instrument']} {t['side']} {t['lots']} -> {t['net_pnl']:+.2f}"
                     f" ({t['exit_reason'] or 'closed'})")
    if decisions:
        lines.append("decisions: " + ", ".join(f"{d['action']}={d['c']}" for d in decisions))
    for e in expl[:5]:
        lines.append(f"  {e['explanation']}")
    col_age = heartbeat_age_seconds()
    lines.append(f"collector heartbeat: {f'{col_age:.0f}s' if col_age is not None else 'MISSING'}")
    try:
        import shutil
        free_gb = shutil.disk_usage(str(REPO_ROOT)).free / 1e9
        lines.append(f"disk free: {free_gb:.1f} GB")
    except Exception:
        pass
    return "\n".join(lines)


def save_digest(text: str, date: Optional[str] = None) -> Path:
    cfg = load_config()
    date = date or utc_now().strftime("%Y-%m-%d")
    out = cfg.settings.paths.absolute("reports") / "digests" / f"digest_{date}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


# ---------------------------------------------------------------- bot runner

def run_bot() -> None:  # pragma: no cover — needs telegram + network
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes

    token = os.environ.get("DANALIT_TG_TOKEN")
    if not token:
        raise SystemExit("DANALIT_TG_TOKEN not set")

    async def guard(update: Update) -> bool:
        if not is_authorized(update.effective_chat.id):
            log.warning("rejected chat id %s", update.effective_chat.id)
            await update.message.reply_text("not authorized")
            return False
        return True

    def handler(fn):
        async def h(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if await guard(update):
                await update.message.reply_text(fn())
        return h

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("status", handler(build_status)))
    app.add_handler(CommandHandler("positions", handler(build_positions)))
    app.add_handler(CommandHandler("halt", handler(lambda: cmd_halt(flat=False))))
    app.add_handler(CommandHandler("halt_flat", handler(lambda: cmd_halt(flat=True))))
    app.add_handler(CommandHandler("resume", handler(cmd_resume)))
    app.add_handler(CommandHandler("report", handler(build_digest)))
    log.info("telegram bot polling")
    app.run_polling()
