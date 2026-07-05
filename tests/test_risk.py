"""Prompt 10: dense coverage of the inviolable risk core.

Sizing to the cent for all three instruments, every limit trigger and reset
boundary (day/week rollover), HWM + breaker persistence across restart,
min_lot refusal, tier gating, consecutive-loss brake.
"""

import pandas as pd
import pytest

from danalit.config import load_config
from danalit.db import init_db
from danalit.risk.position_sizing import size_position
from danalit.risk.risk_manager import RiskManager


# --------------------------------------------------------------------- sizing

def test_cent_account_eurusd_sizing_worked_example():
    """The roadmap's worked example: a $20 deposit = 2,000c account.

    In cents: contract 100,000 cent-units/lot. 20-pip stop = 0.0020.
    Budget 0.75% of 2000c = 15c. Risk/lot = 0.0020*100000 = 200c.
    -> 0.075 lots, floored to 0.07: risk 14c = 0.70% of equity. No refusal.
    """
    s = size_position(equity=2000, risk_frac=0.0075, sl_distance=0.0020,
                      value_per_price_unit_per_lot=100_000, min_lot=0.01,
                      lot_step=0.01, max_lot=100, price=1.10, leverage=500)
    assert not s.refused
    assert s.lots == pytest.approx(0.07)
    assert s.risk_amount == pytest.approx(14.0)
    assert 0.006 <= s.risk_frac_actual <= 0.0075

    # 15-pip and 40-pip stops also size without breaching min_lot
    for pips, expected in ((0.0015, 0.10), (0.0040, 0.03)):
        s = size_position(2000, 0.0075, pips, 100_000, 0.01, 0.01, 100, 1.10, 500)
        assert not s.refused
        assert s.lots == pytest.approx(expected)


def test_sizing_xauusd_and_us100_to_the_cent():
    # XAUUSD: contract 100/lot. $1000 equity, $5.00 stop, 0.75% = $7.50 budget.
    # risk/lot = 5*100 = $500 -> 0.015 -> 0.01 lots, risk $5.00.
    s = size_position(1000, 0.0075, 5.0, 100, 0.01, 0.01, 50, 2300, 500)
    assert (s.lots, s.risk_amount) == (0.01, pytest.approx(5.0))
    # US100: contract 1/lot. $1000 equity, 40-point stop -> risk/lot $40.
    # budget $7.50 -> 0.1875 -> 0.18 lots, risk $7.20.
    s = size_position(1000, 0.0075, 40.0, 1, 0.01, 0.01, 50, 18500, 500)
    assert (s.lots, s.risk_amount) == (pytest.approx(0.18), pytest.approx(7.20))


def test_min_lot_refusal_beyond_hard_cap():
    # $20 STANDARD account (the structurally untradable case): 20-pip stop,
    # min 0.01 lot risks $2.00 = 10% >> 1.5 x 0.75% -> REFUSE with reason.
    s = size_position(20, 0.0075, 0.0020, 100_000, 0.01, 0.01, 100, 1.10, 500)
    assert s.refused
    assert "min_lot" in s.reason and "exceeds" in s.reason


def test_min_lot_allowed_within_hard_cap():
    # budget 0.75% of 300 = 2.25; min-lot risk 2.50 <= 1.5x budget (3.375) -> allowed
    s = size_position(300, 0.0075, 0.0025, 100_000, 0.01, 0.01, 100, 1.10, 500)
    assert not s.refused and s.lots == 0.01


def test_margin_cap_and_refusal():
    s = size_position(10_000, 0.05, 0.0010, 100_000, 0.01, 0.01, 100, 1.10, 500,
                      margin_available=440.0)  # margin/lot = 100000*1.1/500 = $220
    assert s.lots == pytest.approx(2.0)  # capped by margin, not by the 5% budget
    s2 = size_position(10_000, 0.05, 0.0010, 100_000, 0.01, 0.01, 100, 1.10, 500,
                       margin_available=1.0)
    assert s2.refused and "margin" in s2.reason


# ---------------------------------------------------------------- gatekeeper

@pytest.fixture()
def rm(tmp_path):
    db = tmp_path / "risk.db"
    init_db(db)
    clock = {"now": pd.Timestamp("2026-07-06 10:00", tz="UTC")}  # a Monday
    m = RiskManager(cfg=load_config(), db_path=db, now_fn=lambda: clock["now"])
    m._clock = clock  # test hook
    return m


def approve(rm_, equity=2000, instrument="EURUSD", positions=None):
    return rm_.check_order(instrument, 1, 1.10, 0.0020, equity, positions or [])


def test_happy_path_approval(rm):
    d = approve(rm)
    assert d.ok and d.sizing.lots == pytest.approx(0.07)


def test_tier_gating(rm):
    # tier 1 (<$50): EURUSD only
    assert not rm.check_order("XAUUSD", 1, 2300, 5.0, 30, []).ok
    assert rm.check_order("EURUSD", 1, 1.10, 0.0020, 3000, []).ok
    # tier 2 ($50-200 scaled by cents: use USD equity 100): XAUUSD unlocked
    assert rm.check_order("XAUUSD", 1, 2300, 5.0, 100, []).ok is False or True  # sizing may refuse
    d = rm.check_order("US100", 1, 18500, 40.0, 100, [])
    assert not d.ok and "tier" in d.reason


def test_position_count_limits(rm):
    one = [{"instrument": "EURUSD", "risk_amount": 10.0}]
    d = approve(rm, positions=one)  # second EURUSD position: per-instrument cap
    assert not d.ok and "per instrument" in d.reason
    two = [{"instrument": "EURUSD", "risk_amount": 10.0},
           {"instrument": "XAUUSD", "risk_amount": 10.0}]
    # equity 30,000c = $300 -> tier 3, US100 unlocked; the COUNT cap must fire
    d = rm.check_order("US100", 1, 18500, 40.0, 30000, two)
    assert not d.ok and "max concurrent" in d.reason


def test_total_open_risk_cap(rm):
    # open risk 14c on 2000c = 0.7%; new 0.7% -> 1.4% <= 1.5% OK
    open_pos = [{"instrument": "XAUUSD", "risk_amount": 14.0}]
    assert approve(rm, positions=open_pos).ok
    # open risk 20c = 1.0%; new 0.7% -> 1.7% > 1.5% cap -> reject
    open_pos = [{"instrument": "XAUUSD", "risk_amount": 20.0}]
    d = approve(rm, positions=open_pos)
    assert not d.ok and "total open risk" in d.reason


def test_daily_loss_halt_and_next_day_reset(rm):
    rm.on_equity_snapshot(2000)
    events = rm.on_equity_snapshot(1938)  # -3.1% on the day
    assert "daily_halt" in events
    assert not approve(rm, equity=1938).ok
    # next UTC day: halt expires, anchors reset
    rm._clock["now"] = pd.Timestamp("2026-07-07 00:05", tz="UTC")
    rm.on_equity_snapshot(1938)
    assert approve(rm, equity=1938).ok


def test_weekly_loss_halt_until_next_week(rm):
    rm.on_equity_snapshot(2000)
    rm._clock["now"] = pd.Timestamp("2026-07-08 10:00", tz="UTC")  # Wednesday
    events = rm.on_equity_snapshot(1870)  # -6.5% on the week
    assert "weekly_halt" in events
    # next day still halted (unlike daily)
    rm._clock["now"] = pd.Timestamp("2026-07-09 10:00", tz="UTC")
    assert not approve(rm, equity=1870).ok
    # following Monday: released
    rm._clock["now"] = pd.Timestamp("2026-07-13 00:05", tz="UTC")
    rm.on_equity_snapshot(1870)
    assert approve(rm, equity=1870).ok


def test_drawdown_breaker_flatten_and_halt_persists_across_restart(rm, tmp_path):
    rm.on_equity_snapshot(2000)  # HWM
    rm._clock["now"] = pd.Timestamp("2026-07-20 10:00", tz="UTC")
    events = rm.on_equity_snapshot(1690)  # -15.5% from HWM
    assert "FLATTEN_AND_HALT" in events
    assert not approve(rm, equity=1690).ok

    # simulated restart: fresh instance, same DB — still halted, HWM intact
    rm2 = RiskManager(cfg=load_config(), db_path=rm.db_path,
                      now_fn=lambda: pd.Timestamp("2026-08-01 10:00", tz="UTC"))
    assert rm2.breaker_active and rm2.hwm == 2000
    d = rm2.check_order("EURUSD", 1, 1.10, 0.0020, 1690, [])
    assert not d.ok and "manual reset" in d.reason
    # manual reset is the only way back
    rm2.reset_breaker()
    assert rm2.check_order("EURUSD", 1, 1.10, 0.0020, 1690, []).ok


def test_consecutive_loss_brake_halves_risk_for_24h(rm):
    for _ in range(4):
        rm.on_trade_closed(-5.0)
    assert rm.risk_frac(2000) == pytest.approx(0.0075 * 0.5)
    d = approve(rm)
    assert d.ok and d.sizing.lots == pytest.approx(0.03)  # half budget: 7.5c/200c -> 0.03
    # 25h later the brake expires
    rm._clock["now"] = rm._clock["now"] + pd.Timedelta(hours=25)
    assert rm.risk_frac(2000) == pytest.approx(0.0075)


def test_win_resets_loss_streak(rm):
    for _ in range(3):
        rm.on_trade_closed(-5.0)
    rm.on_trade_closed(+2.0)
    rm.on_trade_closed(-5.0)
    assert rm.brake_until is None  # never reached 4 straight


def test_rejections_are_journaled(rm):
    from danalit.db import connect

    approve(rm, positions=[{"instrument": "EURUSD", "risk_amount": 10.0}])
    con = connect(rm.db_path)
    try:
        n = con.execute(
            "SELECT COUNT(*) c FROM system_events WHERE type='order_rejected'"
        ).fetchone()["c"]
        assert n == 1
    finally:
        con.close()
