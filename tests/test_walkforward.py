"""Prompt 9: fold-isolation guard, ML strategy behavior, tuning resumability."""

import numpy as np
import pandas as pd
import pytest

from danalit.backtest.engine import Backtester
from danalit.backtest.costs import CostModel
from danalit.backtest.walkforward import (
    FoldIsolationGuard,
    FoldLeakError,
    MLSignalStrategy,
)


def test_guard_trips_on_test_timestamps():
    guard = FoldIsolationGuard(pd.Timestamp("2024-06-01", tz="UTC"),
                               pd.Timestamp("2024-12-31", tz="UTC"))
    clean = pd.DataFrame(index=pd.date_range("2024-01-01", "2024-05-01", freq="1D", tz="UTC"))
    assert guard.check(clean, "train") is clean
    dirty = pd.DataFrame(index=pd.date_range("2024-05-20", "2024-06-10", freq="1D", tz="UTC"))
    with pytest.raises(FoldLeakError, match="test-period timestamps"):
        guard.check(dirty, "train")


def _flat_bars(n=60, price=1.10, start="2024-05-06 09:00"):
    times = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "time_utc": times, "open": price, "high": price + 0.0002,
        "low": price - 0.0002, "close": price, "tick_volume": 1, "spread": np.nan,
    })


def test_ml_strategy_enters_with_label_barriers_and_time_exits():
    bars = _flat_bars(60)
    probs = pd.DataFrame(index=bars["time_utc"])
    probs["p_long"], probs["p_short"], probs["atr"] = 0.0, 0.0, 0.0010
    strong = bars["time_utc"].iloc[5]
    probs.loc[strong, "p_long"] = 0.80  # one strong long signal

    strat = MLSignalStrategy("EURUSD", probs, tau=0.55, k_tp=2.0, k_sl=1.0,
                             horizon_bars=8, contract_size=100_000,
                             risk_frac=0.0075, min_lot=0.01, lot_step=0.01)
    bt = Backtester({"EURUSD": bars}, {"EURUSD": CostModel(spread=0.0001)},
                    {"EURUSD": 100_000}, initial_balance=2000.0)
    res = bt.run(strat)
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t.side == 1
    assert t.exit_reason == "time_exit"  # flat market: barriers never hit
    # held the horizon (8 bars), then the close decided at that bar's close
    # fills at the NEXT bar's open — matching MT5 semantics: 9 bars total
    assert (t.exit_time - t.entry_time) == pd.Timedelta(minutes=9 * 15)
    # placeholder sizing: (2000 * 0.0075) / (0.0010 * 100000) = 0.15 lots
    assert t.lots == pytest.approx(0.15)


def test_ml_strategy_respects_tau_and_blackout():
    bars = _flat_bars(40)
    probs = pd.DataFrame(index=bars["time_utc"])
    probs["p_long"], probs["p_short"], probs["atr"] = 0.0, 0.0, 0.0010
    probs["blackout"] = 0.0
    probs.iloc[5, probs.columns.get_loc("p_long")] = 0.50      # below tau
    probs.iloc[10, probs.columns.get_loc("p_long")] = 0.80     # strong but...
    probs.iloc[10, probs.columns.get_loc("blackout")] = 1.0    # ...blacked out

    strat = MLSignalStrategy("EURUSD", probs, tau=0.55, k_tp=2.0, k_sl=1.0,
                             horizon_bars=8, contract_size=100_000)
    bt = Backtester({"EURUSD": bars}, {"EURUSD": CostModel(spread=0.0001)},
                    {"EURUSD": 100_000}, initial_balance=2000.0)
    res = bt.run(strat)
    assert len(res["trades"]) == 0


def test_ml_strategy_one_position_at_a_time():
    bars = _flat_bars(40)
    probs = pd.DataFrame(index=bars["time_utc"])
    probs["p_long"] = 0.9  # signal EVERY bar
    probs["p_short"], probs["atr"] = 0.0, 0.0010
    strat = MLSignalStrategy("EURUSD", probs, tau=0.55, k_tp=2.0, k_sl=1.0,
                             horizon_bars=100, contract_size=100_000)
    bt = Backtester({"EURUSD": bars}, {"EURUSD": CostModel(spread=0.0001)},
                    {"EURUSD": 100_000}, initial_balance=2000.0)
    res = bt.run(strat)
    # never more than one open position despite constant signals
    assert res["equity_curve"]["n_positions"].max() == 1


def test_tuning_resumable_and_logged(tmp_path):
    optuna = pytest.importorskip("optuna")  # noqa: F841
    from danalit.db import init_db
    from danalit.models.tuning import tune_fold

    rng = np.random.default_rng(5)
    n = 800
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    x0 = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "x0": x0, "x1": rng.normal(0, 1, n),
        "label": np.where(x0 > 0.8, 1, np.where(x0 < -0.8, 2, 0)),
        "ret_long": np.where(x0 > 0.8, 0.002, -0.001),
        "ret_short": np.where(x0 < -0.8, 0.002, -0.001),
        "atr": 0.001,
    }, index=idx)
    tr, va = df.iloc[:600], df.iloc[600:]
    db = tmp_path / "t.db"
    init_db(db)

    r1 = tune_fold("EURUSD", tr, va, ["x0", "x1"], n_trials=2,
                   study_name="unit_study", db_path=db, storage_dir=tmp_path)
    assert r1["n_trials"] == 2
    r2 = tune_fold("EURUSD", tr, va, ["x0", "x1"], n_trials=2,
                   study_name="unit_study", db_path=db, storage_dir=tmp_path)
    assert r2["n_trials"] == 4  # resumed, not restarted
    assert 0.45 <= r2["tau"] <= 0.70

    from danalit.db import connect
    con = connect(db)
    try:
        n_logged = con.execute("SELECT COUNT(*) c FROM optuna_trials WHERE study='unit_study'"
                               ).fetchone()["c"]
        assert n_logged == 4  # every trial audited
    finally:
        con.close()


def test_guard_wired_into_tuning(tmp_path):
    pytest.importorskip("optuna")
    from danalit.models.tuning import tune_fold

    idx = pd.date_range("2024-06-02", periods=100, freq="15min", tz="UTC")
    df = pd.DataFrame({"x0": 0.0, "label": 0, "ret_long": 0.0, "ret_short": 0.0,
                       "atr": 0.001}, index=idx)
    guard = FoldIsolationGuard(idx[50], idx[-1])  # "test" overlaps the data
    with pytest.raises(FoldLeakError):
        tune_fold("EURUSD", df, df, ["x0"], guard=guard, n_trials=1,
                  study_name="leak_study", storage_dir=tmp_path, db_path=tmp_path / "x.db")
