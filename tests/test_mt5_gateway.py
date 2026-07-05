"""Prompt 14: pure-logic gateway tests against a mocked MetaTrader5 module."""

import pytest

from danalit.config import load_config
from danalit.trading.mt5_gateway import (
    GatewayConfigError,
    MT5Gateway,
    RC_DONE,
    RC_LOCKED,
    RC_NO_MONEY,
    RC_REQUOTE,
)
from tests.fake_mt5 import FakeMT5


@pytest.fixture()
def gw():
    fake = FakeMT5()
    for name, inst in load_config().instruments.items():
        fake.add_symbol(inst.broker_symbol,
                        contract_size=inst.contract_size,
                        volume_min=inst.min_lot, volume_step=inst.lot_step)
    fake.set_tick("EURUSD", bid=1.1000, ask=1.1002)
    g = MT5Gateway(mt5=fake, sleep=lambda s: None)
    g.connect()
    g.validate_symbols()
    return g


def test_symbol_validation_caches_specs(gw):
    assert set(gw.specs) == {"EURUSD", "XAUUSD", "US100"}
    assert gw.specs["EURUSD"].contract_size == 100_000


def test_symbol_validation_refuses_on_config_mismatch():
    fake = FakeMT5()
    cfg = load_config()
    for name, inst in cfg.instruments.items():
        fake.add_symbol(inst.broker_symbol, contract_size=inst.contract_size,
                        volume_min=0.10,  # broker says 0.10, config says 0.01
                        volume_step=inst.lot_step)
    g = MT5Gateway(mt5=fake, sleep=lambda s: None)
    g.connect()
    with pytest.raises(GatewayConfigError, match="min_lot"):
        g.validate_symbols()


def test_market_order_success_first_try(gw):
    r = gw.market_order("EURUSD", 1, 0.05, sl=1.0950, tp=1.1100, comment="danalit:sig1")
    assert r.ok and r.retcode == RC_DONE and r.attempts == 1
    req = gw.mt5.order_send_calls[-1]
    assert req["volume"] == 0.05 and req["type"] == FakeMT5.ORDER_TYPE_BUY
    assert req["price"] == 1.1002  # long fills at ask
    assert req["comment"] == "danalit:sig1"
    assert req["magic"] == load_config().settings.broker.magic_number


def test_requote_retries_with_fresh_price_then_succeeds(gw):
    gw.mt5.retcode_queue = [RC_REQUOTE, RC_REQUOTE, RC_DONE]
    r = gw.market_order("EURUSD", 1, 0.01)
    assert r.ok and r.attempts == 3
    assert len(gw.mt5.order_send_calls) == 3


def test_no_money_fails_fast_no_retry(gw):
    gw.mt5.retcode_queue = [RC_NO_MONEY]
    r = gw.market_order("EURUSD", 1, 50.0)
    assert not r.ok and r.attempts == 1
    assert "money" in r.error
    assert len(gw.mt5.order_send_calls) == 1


def test_context_busy_backoff_retry(gw):
    gw.mt5.retcode_queue = [RC_LOCKED, RC_DONE]
    r = gw.market_order("EURUSD", -1, 0.01)
    assert r.ok and r.attempts == 2
    assert gw.mt5.order_send_calls[-1]["price"] == 1.1000  # short fills at bid


def test_gives_up_after_max_retries(gw):
    gw.mt5.retcode_queue = [RC_REQUOTE, RC_REQUOTE, RC_REQUOTE]
    r = gw.market_order("EURUSD", 1, 0.01)
    assert not r.ok and r.attempts == 3 and "gave up" in r.error


def test_stops_level_adjustment(gw):
    gw.specs["EURUSD"].stops_level = 100  # 100 points = 0.00100
    r = gw.market_order("EURUSD", 1, 0.01, sl=1.1001, tp=1.1003)  # both too close
    req = gw.mt5.order_send_calls[-1]
    assert req["sl"] == pytest.approx(1.1002 - 0.00100)
    assert req["tp"] == pytest.approx(1.1002 + 0.00100)


def test_positions_normalized_and_magic_filtered(gw):
    gw.mt5.add_position(7001, "EURUSD", 1, 0.05, 1.0990, sl=1.0950,
                        comment="danalit:EURUSD-20260706100000")
    gw.mt5.add_position(7002, "EURUSD", -1, 0.10, 1.1010, magic=999)  # foreign magic
    mine = gw.get_open_positions()
    assert [p.id for p in mine] == [7001]
    p = mine[0]
    assert p.side == 1 and p.instrument == "EURUSD"
    assert p.signal_id == "EURUSD-20260706100000"
    assert p.contract_size == 100_000


def test_partial_close_rounds_to_lot_step(gw):
    gw.mt5.add_position(7001, "EURUSD", 1, 0.05, 1.0990)
    r = gw.close_position(7001, fraction=0.5)
    assert r.ok
    req = gw.mt5.order_send_calls[-1]
    assert req["volume"] == pytest.approx(0.02)  # floor(0.025/0.01)*0.01
    assert req["type"] == FakeMT5.ORDER_TYPE_SELL and req["position"] == 7001


def test_modify_sltp(gw):
    gw.mt5.add_position(7001, "EURUSD", 1, 1.0990, 1.0990)
    r = gw.modify_position_sltp(7001, sl=1.1000, tp=None)
    assert r.ok
    req = gw.mt5.order_send_calls[-1]
    assert req["action"] == FakeMT5.TRADE_ACTION_SLTP and req["sl"] == 1.1000


def test_reconcile_orphans_and_ghosts(gw):
    gw.mt5.add_position(7001, "EURUSD", 1, 0.05, 1.0990, comment="danalit:sigA")
    gw.mt5.add_position(7002, "XAUUSD", -1, 0.01, 2300.0, comment="")  # orphan
    journal = [
        {"ticket": None, "signal_id": "sigA", "instrument": "EURUSD"},   # matches by signal
        {"ticket": 9999, "signal_id": "sigZ", "instrument": "US100"},    # ghost
    ]
    rep = gw.reconcile(journal)
    assert [p.id for p in rep.orphans] == [7002]
    assert [g["signal_id"] for g in rep.ghosts] == ["sigZ"]
    assert not rep.clean


def test_account_demo_detection(gw):
    a = gw.get_account()
    assert a.is_demo and a.balance == 2000.0
