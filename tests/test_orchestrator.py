"""Prompt 15: state machine, kill switch, two-generals crash window, new-bar
detection with weekend gaps, dry-run isolation from the gateway send path."""

import numpy as np
import pandas as pd
import pytest

from danalit.config import load_config
from danalit.constants import HALTED, TRADING
from danalit.db import init_db
from danalit.journal.journal import Journal
from danalit.risk.risk_manager import RiskManager
from danalit.trading.mt5_gateway import MT5Gateway
from danalit.trading.orchestrator import NullNotifier, Orchestrator
from danalit.trading.signal_engine import SignalEngine, SignalParams
from danalit.trading.trade_manager import ManageParams, TradeManager
from tests.fake_mt5 import FakeMT5
from tests.test_signal_engine import FakeModel, row

BAR_T = pd.Timestamp("2026-07-06 10:00", tz="UTC")


def make_orch(tmp_path, dry_run=True, x=0.0, fake=None, bar_time=BAR_T):
    db = tmp_path / "orch.db"
    init_db(db)
    cfg = load_config()
    cfg.settings.trading.dry_run = dry_run
    fake = fake or FakeMT5()
    if not fake.symbols:
        for name, inst in cfg.instruments.items():
            fake.add_symbol(inst.broker_symbol, contract_size=inst.contract_size,
                            volume_min=inst.min_lot, volume_step=inst.lot_step)
        fake.set_tick("EURUSD", 1.1000, 1.1002)
        fake.set_tick("XAUUSD", 2300.0, 2300.3)
        fake.set_tick("US100", 18500.0, 18501.0)
    state = {"bar_time": bar_time, "x": x}

    def provider(instrument):
        return {
            "bar_time": state["bar_time"],
            "features_row": row(x=state["x"] if instrument == "EURUSD" else 0.0,
                                atr=0.0010),
            "close": 1.1000,
            "spread": 0.0001,
            "bar_age_intervals": 0.5,
            "collector_age_sec": 60.0,
        }

    orch = Orchestrator(
        cfg=cfg,
        gateway=MT5Gateway(cfg, mt5=fake, sleep=lambda s: None),
        risk_manager=RiskManager(cfg, db_path=db),
        signal_engine=SignalEngine(SignalParams(sentiment_veto_enabled=False)),
        trade_manager=TradeManager(ManageParams()),
        journal=Journal(db),
        notifier=NullNotifier(),
        feature_provider=provider,
        model_loader=lambda name: FakeModel(),
        kill_dir=tmp_path,
        heartbeat_path=tmp_path / "orch.heartbeat",
    )
    orch._test_state = state
    return orch


def test_startup_reaches_trading(tmp_path):
    orch = make_orch(tmp_path)
    assert orch.startup() == TRADING
    assert orch.mode == "dry_run"


def test_startup_failure_lands_in_halted_never_trading(tmp_path):
    orch = make_orch(tmp_path)
    orch.model_loader = lambda name: (_ for _ in ()).throw(RuntimeError("no champion"))
    assert orch.startup() == HALTED
    assert "no champion" in orch._halt_reason


def test_stale_data_at_startup_halts(tmp_path):
    orch = make_orch(tmp_path)
    real = orch.features
    orch.features = lambda n: {**real(n), "bar_age_intervals": 99}
    assert orch.startup() == HALTED
    assert "stale" in orch._halt_reason


def test_dry_run_never_touches_gateway_send_path(tmp_path):
    orch = make_orch(tmp_path, dry_run=True, x=3.0)  # strong long signal
    orch.startup()
    orch.tick()
    assert orch.gw.mt5.order_send_calls == []  # nothing sent, ever
    con_rows = orch.journal.decisions_since(pd.Timestamp("2000-01-01", tz="UTC"))
    assert any(r["action"] == "LONG" and r["mode"] == "dry_run" for r in con_rows)
    intents = Journal(orch.journal.db_path)._con().execute(
        "SELECT status FROM orders").fetchall()
    assert [r["status"] for r in intents] == ["dry_run"]


def test_live_mode_sends_and_journals_before_ack(tmp_path):
    orch = make_orch(tmp_path, dry_run=False, x=3.0)
    orch.startup()
    orch.tick()
    calls = orch.gw.mt5.order_send_calls
    assert len(calls) == 1 and calls[0]["comment"].startswith("danalit:EURUSD-")
    con = orch.journal._con()
    orders = con.execute("SELECT * FROM orders").fetchall()
    con.close()
    assert len(orders) == 1 and orders[0]["status"] == "filled"
    assert orders[0]["broker_ticket"] is not None


def test_new_bar_detection_including_weekend_gap(tmp_path):
    epoch = pd.Timestamp("2000-01-01", tz="UTC")
    orch = make_orch(tmp_path, x=3.0)
    orch.startup()
    orch.tick()
    n1 = len(orch.journal.decisions_since(epoch))
    orch.tick()  # same bar_time again -> no new decision
    assert len(orch.journal.decisions_since(epoch)) == n1
    orch._test_state["bar_time"] = BAR_T + pd.Timedelta(days=2, minutes=15)  # weekend gap
    orch.tick()
    assert len(orch.journal.decisions_since(epoch)) == n1 + 1


def test_kill_switch_halts_and_requires_explicit_resume(tmp_path):
    orch = make_orch(tmp_path)
    orch.startup()
    (tmp_path / "HALT").write_text("stop", encoding="utf-8")
    orch.tick()
    assert orch.state == HALTED
    # deleting the file does NOT auto-resume
    (tmp_path / "HALT").unlink()
    orch.tick()
    assert orch.state == HALTED
    # explicit resume does
    assert orch.resume() == TRADING


def test_halt_flat_flattens_positions(tmp_path):
    fake = FakeMT5()
    orch = make_orch(tmp_path, fake=fake, dry_run=False)
    orch.startup()
    fake.add_position(7001, "EURUSD", 1, 0.05, 1.0990,
                      magic=load_config().settings.broker.magic_number)
    (tmp_path / "HALT_FLAT").write_text("flat", encoding="utf-8")
    orch.tick()
    assert orch.state == HALTED
    close_calls = [c for c in fake.order_send_calls if c.get("position") == 7001]
    assert len(close_calls) == 1


def test_two_generals_crash_between_send_and_ack(tmp_path):
    """Order journaled, sent, broker filled it — but we crashed before the ack.
    Restart must match the intent to the broker position and NOT double-fire."""
    fake = FakeMT5()
    orch = make_orch(tmp_path, fake=fake, dry_run=False, x=3.0)
    orch.startup()

    # simulate the crash: intent journaled + broker filled, no ack recorded
    sig = "EURUSD-20260706100000"
    orch.journal.record_order_intent(sig, sig, "EURUSD", "LONG", 0.05,
                                     1.0990, 1.1020, 1.1000, "live")
    fake.add_position(8001, "EURUSD", 1, 0.05, 1.1002,
                      magic=load_config().settings.broker.magic_number,
                      comment=f"danalit:{sig}")

    orch2 = make_orch(tmp_path, fake=fake, dry_run=False, x=3.0)
    orch2.startup()
    con = orch2.journal._con()
    order = con.execute("SELECT * FROM orders WHERE client_id=?", (sig,)).fetchone()
    con.close()
    assert order["status"] == "filled" and order["broker_ticket"] == 8001
    # the open position means the ENTRY path won't fire again for EURUSD
    # (management closes reference an existing broker position — those are fine)
    orch2.tick()
    entries = [c for c in fake.order_send_calls
               if c.get("action") == fake.TRADE_ACTION_DEAL and "position" not in c]
    assert entries == []  # no duplicate entry order


def test_unfilled_crash_intent_marked_failed(tmp_path):
    fake = FakeMT5()
    orch = make_orch(tmp_path, fake=fake)
    orch.journal.record_order_intent("ghost-sig", "ghost-sig", "EURUSD", "LONG",
                                     0.05, 1.0990, 1.1020, 1.1000, "live")
    orch.startup()
    con = orch.journal._con()
    order = con.execute("SELECT * FROM orders WHERE client_id='ghost-sig'").fetchone()
    con.close()
    assert order["status"] == "failed"


def test_exception_in_tick_notifies_and_halts(tmp_path):
    orch = make_orch(tmp_path)
    orch.startup()
    orch.features = lambda n: (_ for _ in ()).throw(RuntimeError("boom"))
    orch.tick()
    assert orch.state == HALTED
    assert any(level == "CRITICAL" for level, *_ in orch.notifier.sent)
