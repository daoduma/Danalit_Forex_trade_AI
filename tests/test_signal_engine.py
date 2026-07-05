"""Prompt 12: every veto stage, determinism, explanation format, golden sequence."""

import numpy as np
import pandas as pd
import pytest

from danalit.constants import LONG, NONE, SHORT
from danalit.trading.signal_engine import Decision, SignalEngine, SignalParams

NOW = pd.Timestamp("2026-07-06 10:00", tz="UTC")


class FakeModel:
    """Deterministic: p_long driven by feature 'x'."""

    def predict_proba(self, X):
        x = float(X["x"].iloc[0])
        p_long = 1 / (1 + np.exp(-4 * x))
        p_short = 1 / (1 + np.exp(-4 * -x))
        rest = max(1 - p_long - p_short, 0.0)
        s = p_long + p_short + rest
        return np.array([[rest / s, p_long / s, p_short / s]])


def row(**over):
    base = {
        "x": 2.0,                # strong long
        "adx14": 30.0,
        "atr_pctile_90d": 0.5,
        "blackout": 0.0,
        "atr": 0.0010,
        "h4_trend": 1.0,
        "sent_usd_4h": 0.5,
        "sent_eur_4h": 0.0,
        "mins_to_next_high": 252.0,
    }
    base.update(over)
    return pd.Series(base)


@pytest.fixture()
def eng():
    return SignalEngine(SignalParams(), k_tp=2.0, k_sl=1.0)


def test_happy_path_long_with_barriers_and_explanation(eng):
    d = eng.decide("EURUSD", NOW, row(), 1.1000, FakeModel(), tau=0.55)
    assert d.action == LONG
    assert d.sl_price == pytest.approx(1.1000 - 0.0010)
    assert d.tp_price == pytest.approx(1.1000 + 0.0020)
    assert d.veto_reason is None
    for part in ("LONG EURUSD p=", "tau=0.55", "H4 up", "sent +0.50", "ADX 30",
                 "next high-impact 4h12m"):
        assert part in d.explanation, d.explanation
    assert d.features_snapshot["adx14"] == 30.0  # full snapshot on every call


def test_stale_bar_and_stale_collector_veto(eng):
    d = eng.decide("EURUSD", NOW, row(), 1.1, FakeModel(), 0.55, bar_age_intervals=3)
    assert d.action == NONE and "stale data: bar" in d.veto_reason
    d = eng.decide("EURUSD", NOW, row(), 1.1, FakeModel(), 0.55, collector_age_sec=3600)
    assert d.action == NONE and "collector" in d.veto_reason
    assert d.features_snapshot  # snapshot present even on vetoes


def test_blackout_veto(eng):
    d = eng.decide("EURUSD", NOW, row(blackout=1.0), 1.1, FakeModel(), 0.55)
    assert d.action == NONE and d.veto_reason == "news blackout"


def test_regime_veto_and_pass(eng):
    # ADX low AND ATR percentile outside band -> veto
    d = eng.decide("EURUSD", NOW, row(adx14=10.0, atr_pctile_90d=0.99), 1.1, FakeModel(), 0.55)
    assert d.action == NONE and d.veto_reason.startswith("regime")
    # ADX low but ATR percentile in band -> passes
    d = eng.decide("EURUSD", NOW, row(adx14=10.0, atr_pctile_90d=0.5), 1.1, FakeModel(), 0.55)
    assert d.action == LONG


def test_tau_gate(eng):
    d = eng.decide("EURUSD", NOW, row(x=0.05), 1.1, FakeModel(), 0.55)  # p ~ 0.52
    assert d.action == NONE and "below tau" in d.veto_reason
    assert d.confidence > 0  # the near-miss confidence is journaled


def test_sentiment_veto_blocks_long_and_short(eng):
    d = eng.decide("EURUSD", NOW, row(sent_usd_4h=-1.5), 1.1, FakeModel(), 0.55)
    assert d.action == NONE and "sentiment veto" in d.veto_reason and "LONG" in d.veto_reason
    d = eng.decide("EURUSD", NOW, row(x=-2.0, sent_usd_4h=1.5), 1.1, FakeModel(), 0.55)
    assert d.action == NONE and "SHORT" in d.veto_reason
    # veto can be disabled by config
    off = SignalEngine(SignalParams(sentiment_veto_enabled=False))
    d = off.decide("EURUSD", NOW, row(sent_usd_4h=-1.5), 1.1, FakeModel(), 0.55)
    assert d.action == LONG


def test_confluence_bonus_affects_confidence_only(eng):
    # moderate signal so the +0.05 bonus isn't clipped at 1.0
    with_conf = eng.decide("EURUSD", NOW, row(x=0.5), 1.1, FakeModel(), 0.55)
    against = eng.decide("EURUSD", NOW, row(x=0.5, h4_trend=-1.0), 1.1, FakeModel(), 0.55)
    assert with_conf.action == against.action == LONG
    assert with_conf.confidence == pytest.approx(against.confidence + 0.05)
    assert with_conf.sl_price == against.sl_price  # sizing inputs unchanged


def test_determinism(eng):
    a = eng.decide("EURUSD", NOW, row(), 1.1000, FakeModel(), 0.55)
    b = eng.decide("EURUSD", NOW, row(), 1.1000, FakeModel(), 0.55)
    assert a == b and a.signal_id == b.signal_id


def test_golden_decision_sequence(eng):
    """Pinned behavior over a scripted 'week' — refactors cannot silently
    change decisions without touching this test."""
    script = [
        row(),                                    # LONG
        row(x=-2.0, sent_usd_4h=0.0),             # SHORT
        row(blackout=1.0),                        # NONE (blackout)
        row(x=0.02),                              # NONE (tau)
        row(adx14=5.0, atr_pctile_90d=0.99),      # NONE (regime)
        row(sent_usd_4h=-2.0),                    # NONE (sentiment)
        row(x=1.0),                               # LONG
    ]
    actions = [
        eng.decide("EURUSD", NOW + pd.Timedelta(minutes=15 * i), r, 1.1,
                   FakeModel(), 0.55).action
        for i, r in enumerate(script)
    ]
    assert actions == [LONG, SHORT, NONE, NONE, NONE, NONE, LONG]


def test_decision_strategy_parity_with_ml_strategy():
    """One code path everywhere: the engine-driven backtest strategy must make
    the same trades as Prompt 9's probability-table strategy on shared inputs."""
    from danalit.backtest.costs import CostModel
    from danalit.backtest.engine import Backtester
    from danalit.backtest.walkforward import MLSignalStrategy
    from danalit.trading.signal_engine import DecisionStrategy

    n = 50
    times = pd.date_range("2026-07-06 09:00", periods=n, freq="15min", tz="UTC")
    bars = pd.DataFrame({"time_utc": times, "open": 1.10, "high": 1.1002,
                         "low": 1.0998, "close": 1.10, "tick_volume": 1,
                         "spread": np.nan})
    feats = pd.DataFrame({
        "x": 0.0, "adx14": 30.0, "atr_pctile_90d": 0.5, "blackout": 0.0,
        "atr": 0.0010, "h4_trend": 0.0, "mins_to_next_high": 300.0,
    }, index=times)
    feats.iloc[7, feats.columns.get_loc("x")] = 3.0  # one strong long

    model = FakeModel()
    proba = np.vstack([model.predict_proba(feats.iloc[[i]]) [0] for i in range(n)])
    probs = pd.DataFrame({"p_long": proba[:, 1], "p_short": proba[:, 2],
                          "atr": feats["atr"], "blackout": feats["blackout"]}, index=times)

    def run(strategy):
        bt = Backtester({"EURUSD": bars.copy()}, {"EURUSD": CostModel(spread=0.0001)},
                        {"EURUSD": 100_000}, initial_balance=2000.0)
        return bt.run(strategy)

    r1 = run(MLSignalStrategy("EURUSD", probs, 0.55, 2.0, 1.0, 8, 100_000))
    eng = SignalEngine(SignalParams(sentiment_veto_enabled=False))
    r2 = run(DecisionStrategy("EURUSD", eng, feats, model, 0.55, 8, 100_000))
    assert len(r1["trades"]) == len(r2["trades"]) == 1
    t1, t2 = r1["trades"][0], r2["trades"][0]
    assert (t1.entry_time, t1.side, t1.lots) == (t2.entry_time, t2.side, t2.lots)
    assert t1.entry_price == pytest.approx(t2.entry_price)
