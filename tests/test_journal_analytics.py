"""Prompt 16: lifecycle stitching (incl. partials), MAE/MFE, cost math, checklist."""

import pandas as pd
import pytest

from danalit.journal import analytics
from danalit.journal.analytics import (
    compute_mae_mfe,
    cost_comparison,
    evaluate_checklist,
    stitch_trades,
    wilson_interval,
)


def test_stitch_simple_trade():
    deals = [
        {"position_id": 1, "time_utc": "2026-07-01T10:00:00Z", "kind": "entry",
         "price": 1.1002, "volume": 0.05, "side": 1, "commission": 0.0, "swap": 0.0,
         "profit": 0.0, "instrument": "EURUSD", "reason": ""},
        {"position_id": 1, "time_utc": "2026-07-01T14:00:00Z", "kind": "exit",
         "price": 1.1052, "volume": 0.05, "side": 1, "commission": 0.35, "swap": -0.1,
         "profit": 25.0, "instrument": "EURUSD", "reason": "tp"},
    ]
    trades = stitch_trades(deals)
    assert len(trades) == 1
    t = trades[0]
    assert t["lots"] == 0.05 and t["exit_price"] == 1.1052
    assert t["net_pnl"] == pytest.approx(25.0 - 0.35 - 0.1)
    assert t["exit_reason"] == "tp" and t["n_partials"] == 0


def test_stitch_partial_closes_weighted():
    deals = [
        {"position_id": 2, "time_utc": "2026-07-01T10:00:00Z", "kind": "entry",
         "price": 2300.0, "volume": 0.10, "side": 1, "commission": 0, "swap": 0,
         "profit": 0, "instrument": "XAUUSD", "reason": ""},
        {"position_id": 2, "time_utc": "2026-07-01T12:00:00Z", "kind": "exit",
         "price": 2310.0, "volume": 0.05, "side": 1, "commission": 0, "swap": 0,
         "profit": 50.0, "instrument": "XAUUSD", "reason": "partial_tp"},
        {"position_id": 2, "time_utc": "2026-07-01T16:00:00Z", "kind": "exit",
         "price": 2320.0, "volume": 0.05, "side": 1, "commission": 0, "swap": 0,
         "profit": 100.0, "instrument": "XAUUSD", "reason": "tp"},
    ]
    t = stitch_trades(deals)[0]
    assert t["exit_price"] == pytest.approx(2315.0)  # volume-weighted
    assert t["net_pnl"] == pytest.approx(150.0)
    assert t["n_partials"] == 1 and t["closed_utc"] == "2026-07-01T16:00:00Z"


def test_stitch_open_position_has_no_close():
    deals = [{"position_id": 3, "time_utc": "2026-07-01T10:00:00Z", "kind": "entry",
              "price": 18500.0, "volume": 0.02, "side": -1, "commission": 0, "swap": 0,
              "profit": 0, "instrument": "US100", "reason": ""}]
    t = stitch_trades(deals)[0]
    assert t["closed_utc"] is None and t["net_pnl"] is None


def test_mae_mfe_long_and_short():
    bars = pd.DataFrame({
        "time_utc": pd.date_range("2026-07-01 10:00", periods=4, freq="15min", tz="UTC"),
        "high": [1.1010, 1.1030, 1.1020, 1.1015],
        "low": [1.0990, 1.1005, 1.0985, 1.1000],
    })
    o, c = bars["time_utc"].iloc[0], bars["time_utc"].iloc[-1]
    mae, mfe = compute_mae_mfe(bars, 1, 1.1000, o, c)
    assert mae == pytest.approx(1.0985 - 1.1000)
    assert mfe == pytest.approx(1.1030 - 1.1000)
    mae_s, mfe_s = compute_mae_mfe(bars, -1, 1.1000, o, c)
    assert mae_s == pytest.approx(1.1000 - 1.1030)
    assert mfe_s == pytest.approx(1.1000 - 1.0985)


def test_cost_comparison_flags_overrun():
    orders = pd.DataFrame({
        "intended_price": [1.1000, 1.1000, 1.1000],
        "filled_price": [1.10005, 1.10004, 1.10006],
    })
    ok = cost_comparison(orders, modeled_slippage=0.00005)
    assert ok["n"] == 3 and not ok["flag"]  # mean 0.00005 == modeled
    bad = cost_comparison(orders, modeled_slippage=0.00002)
    assert bad["flag"]  # realized 2.5x the model


def test_wilson_interval_sane():
    lo, hi = wilson_interval(55, 100)
    assert 0.44 < lo < 0.55 < hi < 0.65
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_checklist_honest_fails_and_passes():
    early = evaluate_checklist({"weeks": 2, "n_trades": 8, "profit_factor": None,
                                "max_drawdown": None, "cost_ratio": None,
                                "unapproved_orders": 0, "overridden_rejections": 0})
    by_item = {c["item"]: c["pass"] for c in early}
    assert not by_item["Forward-test duration >= 12 weeks"]
    assert not by_item["Trades >= 100"]
    assert not by_item["Max drawdown < 12%"]        # unknown DD counts as FAIL
    assert by_item["Discipline: zero unapproved orders"]

    ready = evaluate_checklist({"weeks": 13, "n_trades": 140, "profit_factor": 1.3,
                                "max_drawdown": 0.08, "cost_ratio": 1.1,
                                "unapproved_orders": 0, "overridden_rejections": 0})
    assert all(c["pass"] for c in ready)


def test_gather_and_report_end_to_end(tmp_path):
    from danalit.db import connect, init_db

    db = tmp_path / "j.db"
    init_db(db)
    con = connect(db)
    with con:
        con.execute("INSERT INTO decisions (ts_utc, instrument, action, veto_reason,"
                    " mode, signal_id) VALUES ('2026-07-01T10:00:00Z','EURUSD','LONG',"
                    " NULL,'demo','sigA')")
        con.execute("INSERT INTO decisions (ts_utc, instrument, action, veto_reason,"
                    " mode, signal_id) VALUES ('2026-07-01T10:15:00Z','EURUSD','NONE',"
                    " 'news blackout','demo','sigB')")
        con.execute("INSERT INTO orders (client_id, signal_id, ts_utc, instrument, side,"
                    " lots, status, intended_price, filled_price) VALUES"
                    " ('sigA','sigA','2026-07-01T10:00:05Z','EURUSD','LONG',0.05,"
                    " 'filled',1.1000,1.10004)")
        con.execute("INSERT INTO trades (signal_id, instrument, side, opened_utc,"
                    " closed_utc, entry_price, exit_price, lots, net_pnl, mode) VALUES"
                    " ('sigA','EURUSD','LONG','2026-07-01T10:00:05Z',"
                    " '2026-07-01T14:00:00Z',1.10004,1.105,0.05,24.5,'demo')")
        con.execute("INSERT INTO equity_snapshots (ts_utc, balance, equity, margin,"
                    " open_risk, mode) VALUES ('2026-07-01T10:00:00Z',2000,2000,0,0,'demo')")
        con.execute("INSERT INTO equity_snapshots (ts_utc, balance, equity, margin,"
                    " open_risk, mode) VALUES ('2026-07-01T14:00:00Z',2024.5,2024.5,0,0,'demo')")
    con.close()

    stats = analytics.gather(db, pd.Timestamp("2026-07-01", tz="UTC"),
                             pd.Timestamp("2026-07-02", tz="UTC"))
    assert stats["n_decisions"] == 2 and stats["n_trades"] == 1
    assert stats["veto_counts"]["news blackout"] == 1
    assert stats["unapproved_orders"] == 0
    assert stats["max_drawdown"] == 0.0

    path = analytics.write_report(stats, out_path=tmp_path / "ft.html")
    html = path.read_text(encoding="utf-8")
    assert "Go-live checklist" in html and "FAIL" in html  # honest early FAILs
    assert "Discipline audit" in html
