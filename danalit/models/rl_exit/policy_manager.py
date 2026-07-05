"""TradeManager-compatible adapter so the RL exit policy can be swapped in via
config — the backtester and live loop call manage() exactly as they would call
the rule-based manager. Selected by settings (trade management 'rl' vs 'rules').
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from danalit.models.rl_exit.env import CLOSE_ALL, CLOSE_HALF, HOLD, TIGHTEN, TIGHTEN_ATR
from danalit.trading.trade_manager import BarInfo


class NoopPolicy:
    """Always holds — used for the adapter-parity contract test."""

    def predict(self, obs, deterministic: bool = True):
        return HOLD, None


class SB3Policy:
    """Wraps a trained stable-baselines3 model (lazy import at load time)."""

    def __init__(self, model_path: str):
        from stable_baselines3 import PPO  # heavy; optional

        self.model = PPO.load(model_path, device="cpu")

    def predict(self, obs, deterministic: bool = True):
        action, state = self.model.predict(obs, deterministic=deterministic)
        return int(action), state


class RLExitManager:
    """Drives exit actions from a policy. Same call surface as TradeManager."""

    def __init__(self, policy, k_tp: float = 2.0, k_sl: float = 1.0,
                 horizon_bars: int = 96, bar_minutes: int = 15, min_lot: float = 0.01):
        self.policy = policy
        self.k_tp, self.k_sl = k_tp, k_sl
        self.horizon_bars, self.bar_minutes = horizon_bars, bar_minutes
        self.min_lot = min_lot
        self._partial_taken: set[int] = set()

    def adopt_state(self, pos) -> None:
        if pos.initial_lots and pos.lots < pos.initial_lots - 1e-9:
            self._partial_taken.add(pos.id)

    def _obs(self, pos, bar: BarInfo) -> np.ndarray:
        side, atr = pos.side, max(bar.atr, 1e-9)
        bars_in = (bar.time - pos.entry_time).total_seconds() / 60 / self.bar_minutes
        frac = pos.lots / pos.initial_lots if pos.initial_lots else 1.0
        sl = pos.sl if pos.sl is not None else pos.entry_price - side * self.k_sl * atr
        tp = pos.tp if pos.tp is not None else pos.entry_price + side * self.k_tp * atr
        return np.array([
            side * (bar.close - pos.entry_price) / atr,
            min(bars_in / self.horizon_bars, 1.0),
            0.5,
            side * (bar.close - sl) / atr,
            side * (tp - bar.close) / atr,
            0.0,
            1.0,
            frac,
        ], dtype=np.float32)

    def manage(self, pos, bar: BarInfo) -> tuple[list[dict], list[dict]]:
        orders: list[dict] = []
        logs: list[dict] = []
        action, _ = self.policy.predict(self._obs(pos, bar), deterministic=True)

        def log(rule, before, after):
            logs.append({"ts_utc": bar.time, "position_id": pos.id,
                         "instrument": pos.instrument, "rule": rule,
                         "before": before, "after": after})

        if action == CLOSE_ALL:
            orders.append({"type": "close", "position_id": pos.id, "fraction": 1.0,
                           "reason": "rl_close"})
            log("rl_close", {"lots": pos.lots}, {"lots": 0})
        elif action == CLOSE_HALF and pos.id not in self._partial_taken:
            if pos.lots >= 2 * self.min_lot:
                orders.append({"type": "close", "position_id": pos.id, "fraction": 0.5,
                               "reason": "rl_partial"})
                log("rl_partial", {"lots": pos.lots}, {"lots": pos.lots * 0.5})
            self._partial_taken.add(pos.id)
        elif action == TIGHTEN:
            tight = bar.close - pos.side * TIGHTEN_ATR * bar.atr
            if pos.sl is None or pos.side * (tight - pos.sl) > 0:
                orders.append({"type": "modify", "position_id": pos.id, "sl": tight})
                log("rl_tighten", {"sl": pos.sl}, {"sl": tight})
        # time exit is a hard backstop regardless of policy
        age = bar.time - pos.entry_time
        import pandas as pd
        if age >= pd.Timedelta(minutes=self.horizon_bars * self.bar_minutes) \
                and not any(o.get("reason") == "rl_close" for o in orders):
            orders.append({"type": "close", "position_id": pos.id, "fraction": 1.0,
                           "reason": "time_exit"})
            log("time_exit", {"age_min": age.total_seconds() / 60}, {"lots": 0})
        return orders, logs
