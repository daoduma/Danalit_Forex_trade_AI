"""Live orchestrator: an explicit state machine tying data, signals, risk,
trade management and the gateway into a safe, restartable process.

    STARTING -> RECONCILING -> TRADING -> HALTED (recoverable) -> STOPPED

Design rules baked in and tested:
- never trade on a broken start: ANY startup failure -> HALTED with reason;
- kill switch: a HALT file stops new entries (HALT_FLAT also flattens);
  deleting the file does NOT resume — resume() is an explicit action;
- crash safety: every intended order is journaled BEFORE sending with a
  client-side id; startup reconciliation matches intents to broker positions
  so a crash between send and ack can never double-fire;
- dry-run mode: full pipeline, gateway send path never touched, decisions
  journaled with mode='dry_run' — the default until deliberately changed;
- a global exception handler notifies CRITICAL and transitions to HALTED —
  never a silent dead process (the Prompt 20 watchdog covers hangs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from danalit.config import AppConfig, load_config
from danalit.constants import (
    HALTED,
    LONG,
    MODE_DEMO,
    MODE_DRY_RUN,
    NONE,
    RECONCILING,
    STARTING,
    STOPPED,
    TRADING,
)
from danalit.logging_setup import setup_logging
from danalit.timeutil import utc_now, utc_now_iso

log = setup_logging("orchestrator")


class NullNotifier:
    def __init__(self):
        self.sent: list[tuple] = []

    def notify(self, level: str, title: str, body: str = "") -> None:
        self.sent.append((level, title, body))
        log.info("notify[%s] %s %s", level, title, body)


class Orchestrator:
    def __init__(
        self,
        cfg: Optional[AppConfig] = None,
        gateway=None,
        risk_manager=None,
        signal_engine=None,
        trade_manager=None,
        journal=None,
        notifier=None,
        feature_provider: Optional[Callable] = None,
        # feature_provider(instrument) -> dict(bar_time, features_row, close,
        #                                      bar_age_intervals, collector_age_sec)
        model_loader: Optional[Callable] = None,   # (instrument) -> model bundle
        taus: Optional[dict[str, float]] = None,
        kill_dir: Optional[Path] = None,
        heartbeat_path: Optional[Path] = None,
        now_fn: Callable = utc_now,
        orphan_policy: str = "adopt",  # 'adopt' | 'flatten'
    ):
        self.cfg = cfg or load_config()
        self.gw = gateway
        self.rm = risk_manager
        self.engine = signal_engine
        self.tm = trade_manager
        self.journal = journal
        self.notifier = notifier or NullNotifier()
        self.features = feature_provider
        self.model_loader = model_loader
        self.taus = taus or {}
        self.kill_dir = kill_dir or Path(__file__).resolve().parents[2]
        self.heartbeat_path = heartbeat_path or (
            self.cfg.settings.paths.absolute("logs") / "orchestrator.heartbeat")
        self.now = now_fn
        self.orphan_policy = orphan_policy
        self.mode = MODE_DRY_RUN if self.cfg.settings.trading.dry_run else MODE_DEMO
        self.state = STARTING
        self.models: dict = {}
        self._last_bar: dict[str, pd.Timestamp] = {}
        self._halt_reason = ""

    # -------------------------------------------------------------- plumbing
    def _transition(self, new_state: str, reason: str = "") -> None:
        old, self.state = self.state, new_state
        msg = f"{old} -> {new_state}" + (f" ({reason})" if reason else "")
        log.info("state %s", msg)
        self.journal.record_system_event("state_transition", msg)
        if new_state == HALTED:
            self._halt_reason = reason
            self.notifier.notify("WARNING", "Orchestrator HALTED", reason)

    def kill_switch(self) -> Optional[str]:
        if (self.kill_dir / "HALT_FLAT").exists():
            return "HALT_FLAT"
        if (self.kill_dir / "HALT").exists():
            return "HALT"
        return None

    def _heartbeat(self) -> None:
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.write_text(utc_now_iso(), encoding="utf-8")

    # --------------------------------------------------------------- startup
    def startup(self) -> str:
        """Full startup sequence; any failure lands in HALTED, never TRADING."""
        try:
            self._transition(STARTING, "startup")
            for name in self.cfg.enabled_instruments():
                self.models[name] = self.model_loader(name)
            self.gw.connect()
            self.gw.validate_symbols()
            self._transition(RECONCILING)
            self._reconcile()
            self._verify_freshness()
            self._transition(TRADING, f"mode={self.mode}")
        except Exception as e:
            self._transition(HALTED, f"startup failure: {e}")
        return self.state

    def _reconcile(self) -> None:
        # 1) crash-window intents: sent but never acknowledged
        broker_positions = self.gw.get_open_positions()
        by_signal = {p.signal_id: p for p in broker_positions if p.signal_id}
        for intent in self.journal.unacked_intents():
            p = by_signal.get(intent["client_id"])
            if p is not None:
                self.journal.update_order_result(intent["client_id"], "filled",
                                                 ticket=p.id, filled_price=p.entry_price)
                self.journal.record_system_event(
                    "reconcile", f"intent {intent['client_id']} matched broker "
                    f"position {p.id} (crash between send and ack)")
            else:
                self.journal.update_order_result(intent["client_id"], "failed",
                                                 error="no broker position found")
                self.journal.record_system_event(
                    "reconcile", f"intent {intent['client_id']}: no broker fill — marked failed")
        # 2) orphans / ghosts
        report = self.gw.reconcile(self.journal.open_trades())
        for orphan in report.orphans:
            if self.orphan_policy == "flatten":
                self.gw.close_position(orphan.id, 1.0, comment="danalit:orphan_flatten")
                self.journal.record_system_event("reconcile",
                                                 f"orphan {orphan.id} flattened")
            else:
                self.journal.record_system_event("reconcile",
                                                 f"orphan {orphan.id} adopted")
        for ghost in report.ghosts:
            self.journal.record_system_event(
                "reconcile", f"ghost {ghost.get('signal_id')} marked closed")

    def _verify_freshness(self) -> None:
        for name in self.cfg.enabled_instruments():
            info = self.features(name)
            if info["bar_age_intervals"] > 4:
                raise RuntimeError(f"{name}: price data stale "
                                   f"({info['bar_age_intervals']:.1f} bar intervals)")

    # ------------------------------------------------------------- main loop
    def tick(self) -> None:
        """One loop iteration; scheduled every loop_interval_sec live."""
        try:
            self._heartbeat()
            ks = self.kill_switch()
            if ks and self.state == TRADING:
                if ks == "HALT_FLAT":
                    for p in self.gw.get_open_positions():
                        self.gw.close_position(p.id, 1.0, comment="danalit:halt_flat")
                self._transition(HALTED, f"kill switch {ks}")
                return
            if self.state != TRADING:
                # explicit resume requested (e.g. Telegram /resume writes RESUME)
                resume_file = self.kill_dir / "RESUME"
                if self.state == HALTED and resume_file.exists():
                    resume_file.unlink()
                    self.resume()
                return

            account = self.gw.get_account()
            open_positions = self.gw.get_open_positions()
            open_risk = sum(self._position_risk(p) for p in open_positions)
            self.journal.record_equity(account.balance, account.equity,
                                       account.margin, open_risk, self.mode)
            halts = self.rm.on_equity_snapshot(account.equity)
            if "FLATTEN_AND_HALT" in halts:
                for p in open_positions:
                    self.gw.close_position(p.id, 1.0, comment="danalit:breaker")
                self._transition(HALTED, "drawdown breaker — manual reset required")
                self.notifier.notify("CRITICAL", "DRAWDOWN BREAKER",
                                     "flattened everything; manual reset required")
                return

            for name in self.cfg.enabled_instruments():
                self._process_instrument(name, account, open_positions)
        except Exception as e:  # global handler: never a silent dead process
            log.exception("tick failed")
            self.journal.record_system_event("error", f"tick: {e}")
            self.notifier.notify("CRITICAL", "Orchestrator exception", str(e))
            self._transition(HALTED, f"unhandled exception: {e}")

    def _process_instrument(self, name: str, account, open_positions) -> None:
        info = self.features(name)
        bar_time = info["bar_time"]
        if self._last_bar.get(name) == bar_time:
            return  # no new completed bar
        self._last_bar[name] = bar_time

        from danalit.trading.trade_manager import BarInfo

        # 1) manage open positions FIRST
        mine = [p for p in open_positions if p.instrument == name]
        bar = BarInfo(time=bar_time, close=info["close"],
                      atr=float(info["features_row"].get("atr", 0.0)),
                      spread=float(info.get("spread", 0.0)),
                      blackout=bool(info["features_row"].get("blackout", 0.0)))
        for p in mine:
            orders, logs = self.tm.manage(p, bar)
            for row in logs:
                self.journal.record_managed_action(row)
            if not self.cfg.settings.trading.dry_run:
                self._execute_management(p, orders)

        # 2) decide
        decision = self.engine.decide(
            name, bar_time, info["features_row"], info["close"],
            self.models[name], self.taus.get(name, 0.55),
            bar_age_intervals=info.get("bar_age_intervals", 0.0),
            collector_age_sec=info.get("collector_age_sec", 0.0),
        )
        self.journal.record_decision(decision, name, self.mode)
        if decision.action == NONE or mine:
            return

        # 3) risk gate
        side = 1 if decision.action == LONG else -1
        sl_dist = abs(info["close"] - decision.sl_price)
        check = self.rm.check_order(
            name, side, info["close"], sl_dist, account.equity,
            open_positions=[{"instrument": p.instrument,
                             "risk_amount": self._position_risk(p)}
                            for p in open_positions],
            margin_available=account.margin_free,
        )
        if not check.ok:
            self.journal.record_system_event("risk_rejected",
                                             f"{decision.signal_id}: {check.reason}")
            return

        # 4) execute — intent journaled BEFORE any send (two-generals safety)
        lots = check.sizing.lots
        self.journal.record_order_intent(
            decision.signal_id, decision.signal_id, name, decision.action, lots,
            decision.sl_price, decision.tp_price, info["close"],
            "dry_run" if self.cfg.settings.trading.dry_run else "live")
        if self.cfg.settings.trading.dry_run:
            self.journal.update_order_result(decision.signal_id, "dry_run")
            return
        result = self.gw.market_order(name, side, lots, decision.sl_price,
                                      decision.tp_price,
                                      comment=f"danalit:{decision.signal_id}")
        self.journal.update_order_result(
            decision.signal_id, "filled" if result.ok else "failed",
            retcode=result.retcode, ticket=result.ticket,
            filled_price=result.price, error=result.error)
        if result.ok:
            self.notifier.notify("INFO", f"OPEN {decision.action} {name} {lots}",
                                 decision.explanation)
        else:
            self.notifier.notify("WARNING", f"order failed {name}", result.error)

    def _execute_management(self, pos, orders: list[dict]) -> None:
        for o in orders:
            if o["type"] == "modify":
                self.gw.modify_position_sltp(pos.id, o.get("sl"), o.get("tp"))
            elif o["type"] == "close":
                self.gw.close_position(pos.id, o.get("fraction", 1.0),
                                       comment=f"danalit:{o.get('reason', 'manage')}")

    def _position_risk(self, p) -> float:
        if p.sl is None or not p.contract_size:
            return 0.0
        return abs(p.entry_price - p.sl) * p.lots * p.contract_size

    # ----------------------------------------------------------------- admin
    def resume(self) -> str:
        """Explicit resume from HALTED: kill-switch files must be gone, then a
        full re-reconciliation runs before trading again (no flapping)."""
        if self.state != HALTED:
            return self.state
        if self.kill_switch():
            log.warning("resume refused: kill-switch file still present")
            return self.state
        reason = self.rm.halted_reason()
        if reason:
            log.warning("resume refused: %s", reason)
            return self.state
        self._transition(RECONCILING, "resume")
        try:
            self._reconcile()
            self._verify_freshness()
            self._transition(TRADING, "resumed")
        except Exception as e:
            self._transition(HALTED, f"resume failure: {e}")
        return self.state

    def stop(self) -> None:
        self._transition(STOPPED, "clean shutdown")

    def run_forever(self) -> None:  # pragma: no cover — exercised in ops
        from apscheduler.schedulers.blocking import BlockingScheduler

        sched = BlockingScheduler(timezone="UTC")
        sched.add_job(self.tick, "interval",
                      seconds=self.cfg.settings.trading.loop_interval_sec,
                      max_instances=1, coalesce=True, next_run_time=self.now())
        log.info("orchestrator loop starting (%ss interval, mode=%s)",
                 self.cfg.settings.trading.loop_interval_sec, self.mode)
        try:
            sched.start()
        except (KeyboardInterrupt, SystemExit):
            self.stop()
