"""Prompt 11: each management rule on scripted states + interaction ordering."""

import pandas as pd
import pytest

from danalit.backtest.engine import Position
from danalit.trading.trade_manager import BarInfo, ManageParams, TradeManager

T0 = pd.Timestamp("2026-07-06 10:00", tz="UTC")  # Monday


def make_pos(side=1, entry=1.1000, sl=None, tp=None, lots=0.10, pid=1,
             entry_time=T0, best=None):
    return Position(id=pid, instrument="EURUSD", side=side, lots=lots,
                    entry_price=entry, entry_time=entry_time, sl=sl, tp=tp,
                    contract_size=100_000, initial_lots=lots,
                    best_price=best if best is not None else entry,
                    worst_price=entry)


def bar(minutes=60, close=1.1000, atr=0.0010, spread=0.0001, blackout=False, time=None):
    return BarInfo(time=time or (T0 + pd.Timedelta(minutes=minutes)), close=close,
                   atr=atr, spread=spread, blackout=blackout)


def test_breakeven_moves_sl_to_entry_plus_spread():
    tm = TradeManager()
    pos = make_pos(sl=1.0990)
    orders, logs = tm.manage(pos, bar(close=1.1010))  # +1.0 ATR
    mod = next(o for o in orders if o["type"] == "modify")
    assert mod["sl"] == pytest.approx(1.1001)  # entry + spread
    assert any(l["rule"] == "breakeven" for l in logs)


def test_no_breakeven_below_trigger():
    tm = TradeManager()
    pos = make_pos(sl=1.0990)
    orders, _ = tm.manage(pos, bar(close=1.1009))  # +0.9 ATR
    assert not [o for o in orders if o["type"] == "modify"]


def test_trailing_after_breakeven_monotonic_never_widens():
    tm = TradeManager()
    pos = make_pos(sl=1.0990, best=1.1000)
    # tick 1: breakeven fires at +2 ATR, best price 1.1020 -> trail = 1.1020-0.0015
    pos.best_price = 1.1020
    orders, _ = tm.manage(pos, bar(close=1.1020))
    sl1 = [o for o in orders if o["type"] == "modify"][-1]["sl"]
    assert sl1 == pytest.approx(1.1020 - 0.0015)  # trail beats breakeven level
    pos.sl = sl1
    # tick 2: price falls back — trailing must NOT widen
    orders, _ = tm.manage(pos, bar(minutes=75, close=1.1008))
    assert not [o for o in orders if o["type"] == "modify"]
    # tick 3: new high -> trail advances
    pos.best_price = 1.1040
    orders, _ = tm.manage(pos, bar(minutes=90, close=1.1035))
    sl3 = [o for o in orders if o["type"] == "modify"][-1]["sl"]
    assert sl3 == pytest.approx(1.1040 - 0.0015)
    assert sl3 > sl1


def test_partial_tp_and_min_lot_skip():
    tm = TradeManager(min_lot=0.01)
    pos = make_pos(lots=0.10)
    orders, logs = tm.manage(pos, bar(close=1.1010))
    partial = next(o for o in orders if o["type"] == "close")
    assert partial["fraction"] == 0.5 and partial["reason"] == "partial_tp"
    # second tick: no second partial
    pos.lots = 0.05
    orders, _ = tm.manage(pos, bar(minutes=75, close=1.1012))
    assert not [o for o in orders if o.get("reason") == "partial_tp"]

    tm2 = TradeManager(min_lot=0.01)
    tiny = make_pos(lots=0.01, pid=2)  # cannot split
    orders, logs = tm2.manage(tiny, bar(close=1.1010))
    assert not [o for o in orders if o.get("reason") == "partial_tp"]
    assert any(l["rule"] == "partial_skip_min_lot" for l in logs)


def test_time_exit_fires_last_at_horizon():
    tm = TradeManager(ManageParams(hold_bars=4, bar_minutes=15))
    pos = make_pos()
    orders, _ = tm.manage(pos, bar(minutes=59, close=1.1000))
    assert not orders
    orders, logs = tm.manage(pos, bar(minutes=60, close=1.1000))
    assert orders[-1]["reason"] == "time_exit"


def test_news_tighten_overrides_all_other_rules():
    tm = TradeManager()  # default: tighten to 0.5*ATR
    pos = make_pos(sl=1.0990, best=1.1040)
    # +4 ATR profit: breakeven+partial+trailing would all fire — but blackout
    orders, logs = tm.manage(pos, bar(close=1.1040, blackout=True))
    assert len(orders) == 1
    assert orders[0]["type"] == "modify"
    assert orders[0]["sl"] == pytest.approx(1.1040 - 0.0005)  # close - 0.5*ATR
    assert [l["rule"] for l in logs] == ["news_tighten"]


def test_news_flatten_mode():
    tm = TradeManager(ManageParams(news_action="flatten"))
    pos = make_pos()
    orders, logs = tm.manage(pos, bar(close=1.1000, blackout=True))
    assert orders[0] == {"type": "close", "position_id": 1, "fraction": 1.0,
                         "reason": "news_flatten"}


def test_news_tighten_is_monotonic_for_shorts():
    tm = TradeManager()
    # tighten target = close + 0.5*ATR = 1.1005: WIDER than current 1.1004 -> no order
    snug = make_pos(side=-1, entry=1.1000, sl=1.1004)
    orders, _ = tm.manage(snug, bar(close=1.1000, blackout=True))
    assert orders == []
    # current SL 1.1010 is looser -> tightened down to 1.1005
    loose = make_pos(side=-1, entry=1.1000, sl=1.1010, pid=2)
    orders, _ = tm.manage(loose, bar(close=1.1000, blackout=True))
    assert orders[0]["sl"] == pytest.approx(1.1005)


def test_weekend_flatten_friday_2030():
    tm = TradeManager()
    pos = make_pos(entry_time=pd.Timestamp("2026-07-10 08:00", tz="UTC"))
    friday_early = pd.Timestamp("2026-07-10 20:15", tz="UTC")
    orders, _ = tm.manage(pos, bar(time=friday_early, close=1.1000))
    assert not [o for o in orders if o.get("reason") == "weekend_flatten"]
    friday_late = pd.Timestamp("2026-07-10 20:30", tz="UTC")
    orders, _ = tm.manage(pos, bar(time=friday_late, close=1.1000))
    assert orders[0]["reason"] == "weekend_flatten"


def test_breakeven_applies_before_trailing_single_tick():
    """Both rules newly applicable in one tick: final SL must be the tighter of
    breakeven and trail (monotonic composition), not the looser."""
    tm = TradeManager()
    pos = make_pos(sl=1.0990, best=1.1012)
    orders, logs = tm.manage(pos, bar(close=1.1012))  # +1.2 ATR
    # breakeven = 1.1001; trail = 1.1012 - 0.0015 = 1.0997 -> breakeven wins
    mods = [o for o in orders if o["type"] == "modify"]
    assert mods[-1]["sl"] == pytest.approx(1.1001)
    rules = [l["rule"] for l in logs]
    assert rules.index("breakeven") < len(rules)  # breakeven logged; trail not tighter
    assert "trail" not in rules


def test_adopt_state_after_restart():
    tm = TradeManager()
    pos = make_pos(sl=1.1002, lots=0.05)  # SL beyond entry, half the size gone
    pos.initial_lots = 0.10
    tm.adopt_state(pos)
    # neither breakeven nor partial may fire again
    orders, logs = tm.manage(pos, bar(close=1.1015))
    assert not [l for l in logs if l["rule"] in ("breakeven", "partial_tp")]
