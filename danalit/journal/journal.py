"""Trade journal: every decision, order intent/result, action and snapshot.

Append-only by convention. Order intents are written BEFORE the gateway send
with a client-side id (the signal_id), so a crash between send and ack is
recoverable: restart reconciliation matches intents to broker deals and can
never double-fire (the two-generals window is closed by the journal, tested in
Prompt 15). Prompt 16 adds lifecycle stitching and analytics on top.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from danalit.config import load_config
from danalit.db import connect
from danalit.timeutil import iso, utc_now


class Journal:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or load_config().settings.paths.db_path

    def _con(self):
        return connect(self.db_path)

    # ---------------------------------------------------------------- writes
    def record_decision(self, decision, instrument: str, mode: str) -> None:
        con = self._con()
        try:
            with con:
                con.execute(
                    """INSERT OR IGNORE INTO decisions
                       (ts_utc, instrument, action, confidence, sl_price, tp_price,
                        explanation, veto_reason, features_snapshot, mode, signal_id)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (iso(utc_now()), instrument, decision.action, decision.confidence,
                     decision.sl_price, decision.tp_price, decision.explanation,
                     decision.veto_reason,
                     json.dumps(decision.features_snapshot, default=str),
                     mode, decision.signal_id),
                )
        finally:
            con.close()

    def record_order_intent(self, client_id: str, signal_id: str, instrument: str,
                            side: str, lots: float, sl: Optional[float],
                            tp: Optional[float], intended_price: Optional[float],
                            mode: str) -> None:
        con = self._con()
        try:
            with con:
                con.execute(
                    """INSERT INTO orders (client_id, signal_id, ts_utc, instrument,
                                            side, lots, sl, tp, status, intended_price)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (client_id, signal_id, iso(utc_now()), instrument, side, lots,
                     sl, tp, f"intent_{mode}", intended_price),
                )
        finally:
            con.close()

    def update_order_result(self, client_id: str, status: str,
                            retcode: Optional[int] = None,
                            ticket: Optional[int] = None,
                            filled_price: Optional[float] = None,
                            error: str = "") -> None:
        con = self._con()
        try:
            with con:
                con.execute(
                    """UPDATE orders SET status=?, retcode=?, broker_ticket=?,
                       filled_price=?, error=? WHERE client_id=?""",
                    (status, retcode, ticket, filled_price, error, client_id),
                )
        finally:
            con.close()

    def record_managed_action(self, log_row: dict) -> None:
        con = self._con()
        try:
            with con:
                con.execute(
                    "INSERT INTO managed_actions (ts_utc, trade_id, instrument, rule,"
                    " before_state, after_state) VALUES (?,?,?,?,?,?)",
                    (iso(log_row.get("ts_utc", utc_now())), log_row.get("position_id"),
                     log_row.get("instrument"), log_row["rule"],
                     json.dumps(log_row.get("before"), default=str),
                     json.dumps(log_row.get("after"), default=str)),
                )
        finally:
            con.close()

    def record_equity(self, balance: float, equity: float, margin: float,
                      open_risk: float, mode: str) -> None:
        con = self._con()
        try:
            with con:
                con.execute(
                    "INSERT INTO equity_snapshots (ts_utc, balance, equity, margin,"
                    " open_risk, mode) VALUES (?,?,?,?,?,?)",
                    (iso(utc_now()), balance, equity, margin, open_risk, mode),
                )
        finally:
            con.close()

    def record_system_event(self, type_: str, detail: str) -> None:
        con = self._con()
        try:
            with con:
                con.execute("INSERT INTO system_events (ts_utc, type, detail)"
                            " VALUES (?,?,?)", (iso(utc_now()), type_, detail))
        finally:
            con.close()

    # ----------------------------------------------------------------- reads
    def unacked_intents(self) -> list[dict]:
        """Orders journaled as sent but never acknowledged (crash window)."""
        con = self._con()
        try:
            rows = con.execute(
                "SELECT * FROM orders WHERE status IN ('sending','intent_live','intent_demo')"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()

    def open_trades(self) -> list[dict]:
        """Journal's view of open positions: filled orders without a close row."""
        con = self._con()
        try:
            rows = con.execute(
                """SELECT o.client_id AS signal_id, o.broker_ticket AS ticket,
                          o.instrument
                   FROM orders o
                   WHERE o.status = 'filled'
                     AND NOT EXISTS (SELECT 1 FROM trades t
                                     WHERE t.signal_id = o.client_id
                                       AND t.closed_utc IS NOT NULL)"""
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()

    def decisions_since(self, since: pd.Timestamp) -> list[dict]:
        con = self._con()
        try:
            rows = con.execute("SELECT * FROM decisions WHERE ts_utc >= ? ORDER BY ts_utc",
                               (iso(since),)).fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()
