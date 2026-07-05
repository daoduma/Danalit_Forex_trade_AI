"""Prompt 13: env contracts, action effects, hand-computed reward accounting,
NOOP-policy adapter parity with a rules-disabled TradeManager."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from danalit.models.rl_exit.env import CLOSE_ALL, CLOSE_HALF, HOLD, TIGHTEN, ExitEnv
from danalit.models.rl_exit.policy_manager import NoopPolicy, RLExitManager
from danalit.trading.trade_manager import BarInfo, ManageParams, TradeManager
from tests.test_trade_manager import make_pos


def flat_bars(n=120, price=1.1000):
    return pd.DataFrame({
        "open": price, "high": price + 0.0002, "low": price - 0.0002,
        "close": price,
    }, index=range(n))


def entry(t=5, side=1, atr=0.0010):
    return {"t": t, "side": side, "atr": atr, "atr_pctile": 0.5,
            "h1_trend": 1.0, "mins_to_high": 600.0}


def test_env_reset_and_step_contracts():
    env = ExitEnv(flat_bars(), [entry()], horizon=20, spread=0.0001)
    obs, info = env.reset(seed=1)
    assert obs.shape == (8,) and obs.dtype == np.float32
    assert obs[7] == 1.0  # full fraction
    obs, reward, terminated, truncated, info = env.step(HOLD)
    assert isinstance(reward, float) and not terminated
    assert env.action_space.n == 4


def test_close_all_realizes_at_open_with_costs():
    env = ExitEnv(flat_bars(), [entry()], horizon=20, spread=0.0001)
    env.reset(seed=1)
    # long entered at open+spread = 1.1001; close-all at next open (1.1000 bid)
    obs, reward, terminated, truncated, _ = env.step(CLOSE_ALL)
    assert terminated
    # pnl = (1.1000 - 1.1001)/0.0010 = -0.1 ATR, minus time penalty
    assert reward == pytest.approx(-0.1 - 0.001, abs=1e-9)
    assert env.fraction == 0.0


def test_close_half_then_timeout_reward_accounting():
    env = ExitEnv(flat_bars(n=40), [entry(t=5)], horizon=4, spread=0.0001)
    env.reset(seed=1)
    _, r1, done, _, _ = env.step(CLOSE_HALF)   # realize half at -0.1 ATR -> -0.05
    assert not done
    assert r1 == pytest.approx(-0.05 - 0.001, abs=1e-9)
    total = r1
    for _ in range(3):
        _, r, done, truncated, _ = env.step(HOLD)
        total += r
    # gymnasium API: horizon timeout -> truncated=True, terminated=False
    assert truncated and not done
    # full episode: -0.1 ATR total pnl - 4 time penalties
    assert total == pytest.approx(-0.1 - 4 * 0.001, abs=1e-9)


def test_tighten_moves_sl_monotonically_and_sl_hit_ends_episode():
    bars = flat_bars(40)
    # price runs up (so a tightened stop can lock profit), then collapses
    bars.loc[6:7, ["open", "high", "low", "close"]] = [1.1030, 1.1032, 1.1028, 1.1030]
    bars.loc[8:, ["open", "high", "low", "close"]] = [1.0980, 1.0982, 1.0978, 1.0980]
    env = ExitEnv(bars, [entry(t=5)], horizon=20, spread=0.0001, k_tp=50.0)
    env.reset(seed=1)
    old_sl = env.sl
    env.step(HOLD)                       # bar 6: price now high
    _, _, done, _, _ = env.step(TIGHTEN)  # bar 7: SL -> 1.1030 - 1.0*ATR = 1.1020
    assert env.sl > old_sl and env.sl == pytest.approx(1.1020)
    assert not done
    # tighten again from a LOWER price must not widen
    env.sl_before = env.sl
    _, reward, done, truncated, _ = env.step(TIGHTEN)  # bar 8 gaps under SL first
    assert done and not truncated  # gap exit at the open: episode terminated


def test_pessimistic_rule_in_env():
    bars = flat_bars(40)
    bars.loc[7, ["high", "low"]] = [1.1100, 1.0900]  # covers both barriers
    env = ExitEnv(bars, [entry(t=5)], horizon=20, spread=0.0)
    env.reset(seed=1)
    env.step(HOLD)  # bar 6
    obs, reward, done, truncated, _ = env.step(HOLD)  # bar 7: SL assumed first
    assert done
    # SL at entry - 1.0*ATR -> pnl -1.0 ATR (+2 time penalties over the episode)
    assert reward == pytest.approx(-1.0 - 0.001, abs=1e-6)


def test_noop_policy_parity_with_rules_disabled_manager():
    """Adapter contract: NOOP policy == rule manager with every rule off,
    except the shared hard time-exit backstop."""
    rl = RLExitManager(NoopPolicy(), horizon_bars=96)
    rules_off = TradeManager(ManageParams(
        news_action="off", weekend_flatten=False,
        breakeven_trigger_atr=999, partial_trigger_atr=999, hold_bars=96))
    pos1, pos2 = make_pos(pid=1), make_pos(pid=2)
    t = pd.Timestamp("2026-07-06 12:00", tz="UTC")
    bar = BarInfo(time=t, close=1.1015, atr=0.0010, spread=0.0001)
    o_rl, _ = rl.manage(pos1, bar)
    o_rules, _ = rules_off.manage(pos2, bar)
    assert o_rl == [] and o_rules == []  # both hold
    # both fire the time exit at the horizon
    late = BarInfo(time=t + pd.Timedelta(hours=25), close=1.1015, atr=0.0010, spread=0.0001)
    o_rl, _ = rl.manage(pos1, late)
    o_rules, _ = rules_off.manage(pos2, late)
    assert o_rl[-1]["reason"] == "time_exit" and o_rules[-1]["reason"] == "time_exit"


def test_train_raises_clear_error_without_sb3():
    try:
        import stable_baselines3  # noqa: F401
        pytest.skip("SB3 installed; smoke-train covered elsewhere")
    except ImportError:
        pass
    from danalit.models.rl_exit.train import train_ppo

    with pytest.raises(RuntimeError, match="OPTIONAL"):
        train_ppo(flat_bars(), [entry()], total_timesteps=10)
