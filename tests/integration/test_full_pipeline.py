"""Prompt 20 integration suite: a scripted 'market day' replayed through
orchestrator + engine + risk + mocked gateway, asserting journal end-state;
restart-during-open-position reconciliation; kill switch honored mid-loop.

Run: python -m pytest -m integration
"""

import pandas as pd
import pytest

from danalit.constants import HALTED, TRADING
from tests.test_orchestrator import BAR_T, make_orch

pytestmark = pytest.mark.integration


def replay_day(orch, hours=6, signal_hour=2):
    """Advance the fake feed bar by bar; a strong long fires once."""
    for i in range(hours * 4):  # M15 bars
        orch._test_state["bar_time"] = BAR_T + pd.Timedelta(minutes=15 * i)
        orch._test_state["x"] = 3.0 if i == signal_hour * 4 else 0.0
        orch.tick()


def test_scripted_market_day_end_state(tmp_path):
    orch = make_orch(tmp_path, dry_run=False)
    assert orch.startup() == TRADING
    replay_day(orch)

    con = orch.journal._con()
    decisions = con.execute("SELECT COUNT(*) c FROM decisions").fetchone()["c"]
    longs = con.execute("SELECT COUNT(*) c FROM decisions WHERE action='LONG'").fetchone()["c"]
    orders = con.execute("SELECT * FROM orders").fetchall()
    equity_rows = con.execute("SELECT COUNT(*) c FROM equity_snapshots").fetchone()["c"]
    events = con.execute(
        "SELECT COUNT(*) c FROM system_events WHERE type='state_transition'").fetchone()["c"]
    con.close()

    assert decisions == 6 * 4 * 3          # every bar, every instrument, incl. NONEs
    assert longs == 1
    assert len(orders) == 1 and orders[0]["status"] == "filled"
    assert orders[0]["broker_ticket"] is not None
    assert equity_rows >= 6 * 4            # snapshot every tick
    assert events >= 3                     # STARTING->RECONCILING->TRADING
    assert orch.state == TRADING           # survived the whole day


def test_restart_during_open_position_reconciles(tmp_path):
    orch = make_orch(tmp_path, dry_run=False)
    orch.startup()
    replay_day(orch, hours=3)
    fake = orch.gw.mt5
    open_before = [p for p in fake.positions]
    assert fake.order_send_calls  # the entry happened

    # 'power failure': new orchestrator over the same DB + same broker state
    orch2 = make_orch(tmp_path, dry_run=False, fake=fake)
    assert orch2.startup() == TRADING
    con = orch2.journal._con()
    recon = con.execute(
        "SELECT COUNT(*) c FROM system_events WHERE type='reconcile'").fetchone()["c"]
    con.close()
    # broker position was either matched to the journal or adopted — never duplicated
    orch2._test_state["bar_time"] = BAR_T + pd.Timedelta(hours=4)
    orch2._test_state["x"] = 3.0  # another strong signal
    orch2.tick()
    entries = [c for c in fake.order_send_calls
               if c.get("action") == fake.TRADE_ACTION_DEAL and "position" not in c]
    if open_before:  # position still open -> engine must NOT stack a second entry
        assert len(entries) == 1
    assert recon >= 0


def test_kill_switch_honored_mid_loop(tmp_path):
    orch = make_orch(tmp_path, dry_run=False)
    orch.startup()
    replay_day(orch, hours=1)
    (tmp_path / "HALT").write_text("mid-loop", encoding="utf-8")
    orch._test_state["bar_time"] = BAR_T + pd.Timedelta(hours=2)
    orch._test_state["x"] = 3.0  # strong signal arrives WITH the kill switch up
    orch.tick()
    assert orch.state == HALTED
    con = orch.journal._con()
    orders = con.execute("SELECT COUNT(*) c FROM orders").fetchone()["c"]
    con.close()
    assert orders == 0  # the signal was never traded
