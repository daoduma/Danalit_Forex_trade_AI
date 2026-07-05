"""Prompt 8: backtester — hand-computed P&L to the cent, pessimistic stops,
gap fills, swap accrual (triple Wednesday), partial closes, toy strategies."""

import numpy as np
import pandas as pd
import pytest

from danalit.backtest.costs import CostModel
from danalit.backtest.engine import Backtester, BuyAndHold, MACross

CONTRACT = {"EURUSD": 100_000.0}


def bars_frame(prices, start="2024-05-06 09:00", freq="15min"):
    """OHLC bars where each bar's open=prev close; prices are closes."""
    prices = np.asarray(prices, dtype=float)
    opens = np.concatenate([[prices[0]], prices[:-1]])
    return pd.DataFrame({
        "time_utc": pd.date_range(start, periods=len(prices), freq=freq, tz="UTC"),
        "open": opens,
        "high": np.maximum(opens, prices),
        "low": np.minimum(opens, prices),
        "close": prices,
        "tick_volume": 1,
        "spread": np.nan,  # force use of the config estimate
    })


class Scripted:
    """Emit given orders at given bar indices (counted per on_bar call)."""

    def __init__(self, script: dict[int, list[dict]]):
        self.script, self.i = script, 0

    def on_bar(self, ctx):
        orders = self.script.get(self.i, [])
        self.i += 1
        return orders


def make_bt(bars, cost, balance=20.0):
    return Backtester({"EURUSD": bars}, {"EURUSD": cost}, CONTRACT,
                      initial_balance=balance, leverage=500)


def test_hand_computed_pnl_to_the_cent():
    # open long 0.10 at bar1 open (1.1010 bid) + spread 0.0002 -> 1.1012
    # close at bar3 open (1.1050 bid). gross = 0.0038*0.1*100000 = $38.00
    # commission $7/lot -> $0.70. net = $37.30. balance = 100 + 37.30 = 137.30
    # (0.10 lots at 1:500 needs ~$22 margin, so a $100 account)
    bars = bars_frame([1.1010, 1.1030, 1.1050, 1.1050, 1.1050])
    cost = CostModel(spread=0.0002, commission_per_lot=7.0, slippage=0.0)
    bt = make_bt(bars, cost, balance=100.0)
    res = bt.run(Scripted({
        0: [{"type": "open", "instrument": "EURUSD", "side": 1, "lots": 0.10,
             "sl": None, "tp": None}],
        2: [{"type": "close", "position_id": 1, "fraction": 1.0}],
    }))
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t.entry_price == pytest.approx(1.1012)
    assert t.exit_price == pytest.approx(1.1050)
    assert t.gross_pnl == pytest.approx(38.00)
    assert t.commission == pytest.approx(0.70)
    assert t.net_pnl == pytest.approx(37.30)
    assert res["final_balance"] == pytest.approx(137.30)
    assert t.spread_cost == pytest.approx(0.0002 * 0.1 * 100_000)  # $2.00


def test_pessimistic_rule_sl_before_tp():
    bars = bars_frame([1.1000] * 6)
    # bar 2 has a huge range covering both SL (1.0980) and TP (1.1040)
    bars.loc[2, "high"] = 1.1100
    bars.loc[2, "low"] = 1.0900
    cost = CostModel(spread=0.0, commission_per_lot=0.0)
    bt = make_bt(bars, cost)
    res = bt.run(Scripted({
        0: [{"type": "open", "instrument": "EURUSD", "side": 1, "lots": 0.01,
             "sl": 1.0980, "tp": 1.1040}],
    }))
    t = res["trades"][0]
    assert t.exit_reason == "sl"
    assert t.exit_price == pytest.approx(1.0980)


def test_gap_fills_at_open_not_stop_price():
    bars = bars_frame([1.1000] * 6)
    bars.loc[3:, ["open", "high", "low", "close"]] = [1.0900, 1.0905, 1.0895, 1.0900]
    cost = CostModel(spread=0.0)
    bt = make_bt(bars, cost)
    res = bt.run(Scripted({
        0: [{"type": "open", "instrument": "EURUSD", "side": 1, "lots": 0.01,
             "sl": 1.0980, "tp": None}],
    }))
    t = res["trades"][0]
    assert t.exit_reason == "sl_gap"
    assert t.exit_price == pytest.approx(1.0900)  # gapped open, worse than the stop


def test_swap_accrual_with_triple_wednesday():
    # Hold Mon 12:00 -> Thu 12:00: crossings into Tue, Wed, Thu.
    # Leaving Wednesday (Wed->Thu) counts triple => 1 + 1 + 3 = 5 nights.
    times = pd.date_range("2024-05-06 09:00", "2024-05-09 15:00", freq="1h", tz="UTC")
    n = len(times)
    bars = pd.DataFrame({
        "time_utc": times, "open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1,
        "tick_volume": 1, "spread": np.nan,
    })
    swap_long = -0.0001
    cost = CostModel(spread=0.0, swap_long=swap_long)
    bt = make_bt(bars, cost, balance=100.0)
    res = bt.run(Scripted({
        0: [{"type": "open", "instrument": "EURUSD", "side": 1, "lots": 0.01,
             "sl": None, "tp": None}],
        n - 2: [{"type": "close", "position_id": 1, "fraction": 1.0}],
    }))
    t = res["trades"][0]
    assert t.swap == pytest.approx(5 * swap_long * 0.01 * 100_000)  # -$0.50


def test_partial_close_respects_remaining_lots():
    bars = bars_frame([1.1000, 1.1000, 1.1020, 1.1040, 1.1040, 1.1040])
    cost = CostModel(spread=0.0)
    bt = make_bt(bars, cost, balance=100.0)
    res = bt.run(Scripted({
        0: [{"type": "open", "instrument": "EURUSD", "side": 1, "lots": 0.10,
             "sl": None, "tp": None}],
        2: [{"type": "close", "position_id": 1, "fraction": 0.5}],
        3: [{"type": "close", "position_id": 1, "fraction": 1.0}],
    }))
    assert len(res["trades"]) == 2
    assert res["trades"][0].lots == pytest.approx(0.05)
    assert res["trades"][1].lots == pytest.approx(0.05)
    # first partial exits at bar3 open 1.1020? no: order queued at bar2 close fills bar3 open (1.1040 prev close 1.1020)
    assert res["trades"][0].exit_price == pytest.approx(1.1020)
    assert res["trades"][1].exit_price == pytest.approx(1.1040)


def test_short_pays_spread_on_exit():
    bars = bars_frame([1.1000, 1.1000, 1.0950, 1.0950])
    cost = CostModel(spread=0.0002)
    bt = make_bt(bars, cost, balance=100.0)
    res = bt.run(Scripted({
        0: [{"type": "open", "instrument": "EURUSD", "side": -1, "lots": 0.01,
             "sl": None, "tp": None}],
        2: [{"type": "close", "position_id": 1, "fraction": 1.0}],
    }))
    t = res["trades"][0]
    assert t.entry_price == pytest.approx(1.1000)  # sold at bid, no spread on entry
    assert t.exit_price == pytest.approx(1.0950 + 0.0002)  # bought back at ask
    assert t.net_pnl == pytest.approx((1.1000 - 1.0952) * 0.01 * 100_000)


def test_buy_and_hold_matches_hand_calc():
    from danalit.data.synthetic import generate_m1
    from danalit.data import price_store
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        m1 = generate_m1("2024-03-01", "2024-05-01", s0=1.10, seed=9)
        price_store.write_bars("EURUSD", "M1", m1, root=root)
        m15 = price_store.resample("EURUSD", "M1", "M15", root=root)

    cost = CostModel(spread=0.0002)  # no swap/commission for clean arithmetic
    bt = Backtester({"EURUSD": m15}, {"EURUSD": cost}, CONTRACT,
                    initial_balance=1000.0, leverage=500)
    res = bt.run(BuyAndHold("EURUSD", 0.01))
    t = res["trades"][0]
    # engine prefers the bar's RECORDED spread over the config estimate
    entry = m15["open"].iloc[1] + m15["spread"].iloc[1]
    exit_ = m15["close"].iloc[-1]              # liquidated at final close
    expected = (exit_ - entry) * 0.01 * 100_000
    assert t.net_pnl == pytest.approx(expected, abs=0.01)


def test_ma_cross_runs_and_reports(tmp_path):
    from danalit.backtest.metrics import summarize
    from danalit.backtest.report import build_report
    from danalit.data.synthetic import generate_m1
    from danalit.data import price_store

    root = tmp_path / "store"
    m1 = generate_m1("2024-01-01", "2024-06-01", s0=1.10, seed=11)
    price_store.write_bars("EURUSD", "M1", m1, root=root)
    m15 = price_store.resample("EURUSD", "M1", "M15", root=root)

    cost = CostModel(spread=0.00012, commission_per_lot=0.0, swap_long=-0.00006,
                     swap_short=0.00002)
    bt = Backtester({"EURUSD": m15}, {"EURUSD": cost}, CONTRACT,
                    initial_balance=1000.0, leverage=500)
    res = bt.run(MACross("EURUSD", 0.01))
    assert len(res["trades"]) > 5
    assert len(res["equity_curve"]) == len(m15)
    s = summarize(res["trades"], res["equity_curve"], 1000.0)
    assert s["n_trades"] == len(res["trades"])
    assert 0 <= s["max_drawdown_equity"] <= 1
    path = build_report(res, s, "MA-cross test", tmp_path / "report.html")
    html = path.read_text(encoding="utf-8")
    assert "Equity curve" in html and "Cost breakdown" in html


def test_metrics_known_values():
    from danalit.backtest.metrics import consecutive_losses, max_drawdown

    eq = pd.Series([100, 120, 90, 95, 130], dtype=float)
    assert max_drawdown(eq) == pytest.approx(0.25)  # 120 -> 90
    assert consecutive_losses([1, -1, -2, -3, 5, -1]) == 3


def test_bootstrap_drawdown_shapes():
    from danalit.backtest.metrics import bootstrap_drawdowns

    out = bootstrap_drawdowns([1.0, -0.5, 2.0, -1.0] * 25, 100.0, n_paths=200)
    assert 0 <= out["dd_p50"] <= out["dd_p95"] <= out["dd_p99"] <= 1
