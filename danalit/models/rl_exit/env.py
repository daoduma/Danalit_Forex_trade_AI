"""Gymnasium environment for SINGLE open-position management (exit only).

Episode = one historical trade (entered by the Prompt 12 engine; sample entries
from walk-forward TRAINING folds only). The agent never chooses direction or
size — only {hold, tighten SL, close 50%, close all} — so the action space is
small enough to actually learn from limited data.

Reward: realized P&L change per step in ATR units, net of costs, minus a small
time penalty. Intrabar SL/TP uses the same pessimistic rule as the backtester.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise RuntimeError("gymnasium is required for the RL exit env "
                       "(pip install gymnasium)") from e

HOLD, TIGHTEN, CLOSE_HALF, CLOSE_ALL = 0, 1, 2, 3
TIME_PENALTY = 0.001
TIGHTEN_ATR = 1.0


class ExitEnv(gym.Env):
    """observation (8,): [unrealized_atr, progress, atr_pctile, dist_sl_atr,
    dist_tp_atr, h1_agree, mins_to_high_norm, fraction_remaining]"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: pd.DataFrame,          # open/high/low/close, positional index
        entries: list[dict],         # {'t': int bar idx, 'side': +-1, 'atr': float,
                                     #  'atr_pctile': float, 'h1_trend': float,
                                     #  'mins_to_high': float}
        k_tp: float = 2.0,
        k_sl: float = 1.0,
        horizon: int = 96,
        spread: float = 0.0001,
        sequential: bool = False,
    ):
        super().__init__()
        self.bars = bars.reset_index(drop=True)
        self.entries = entries
        self.k_tp, self.k_sl = k_tp, k_sl
        self.horizon, self.spread = horizon, spread
        self.sequential = sequential
        self._entry_cursor = 0
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    # ------------------------------------------------------------------ core
    def _price(self, i: int, col: str) -> float:
        return float(self.bars.iloc[self.e["t"] + 1 + i][col])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.sequential:
            self.e = self.entries[self._entry_cursor % len(self.entries)]
            self._entry_cursor += 1
        else:
            self.e = self.entries[self.np_random.integers(len(self.entries))]
        side, atr = self.e["side"], self.e["atr"]
        open0 = float(self.bars.iloc[self.e["t"] + 1]["open"])
        self.entry = open0 + self.spread if side > 0 else open0
        self.sl = self.entry - side * self.k_sl * atr
        self.tp = self.entry + side * self.k_tp * atr
        self.fraction = 1.0
        self.realized_atr = 0.0
        self.i = 0
        self.done = False
        return self._obs(self.entry), {}

    def _exit_price(self, raw: float) -> float:
        # long exits at bid (raw); short buys back at ask
        return raw if self.e["side"] > 0 else raw + self.spread

    def _realize(self, raw_price: float, frac: float) -> float:
        """Realize frac of the position at raw (bid) price; returns reward delta (ATR)."""
        side, atr = self.e["side"], self.e["atr"]
        pnl_atr = side * (self._exit_price(raw_price) - self.entry) / atr * frac
        self.realized_atr += pnl_atr
        self.fraction = max(self.fraction - frac, 0.0)
        return pnl_atr

    def step(self, action: int):
        assert not self.done, "call reset() first"
        side, atr = self.e["side"], self.e["atr"]
        reward = -TIME_PENALTY
        bar = self.bars.iloc[self.e["t"] + 1 + self.i]
        o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))

        # 1 — agent acts at this bar's open
        if action == CLOSE_ALL and self.fraction > 0:
            reward += self._realize(o, self.fraction)
            self.done = True
        elif action == CLOSE_HALF and self.fraction > 0:
            reward += self._realize(o, self.fraction * 0.5)
        elif action == TIGHTEN:
            tight = o - side * TIGHTEN_ATR * atr
            if side * (tight - self.sl) > 0:
                self.sl = tight

        # 2 — the bar plays out (pessimistic SL first), unless already flat
        if not self.done:
            hit_sl = (l <= self.sl) if side > 0 else (h + self.spread >= self.sl)
            hit_tp = (h >= self.tp) if side > 0 else (l + self.spread <= self.tp)
            gap_sl = (o <= self.sl) if side > 0 else (o + self.spread >= self.sl)
            if gap_sl:
                reward += self._realize(o, self.fraction)
                self.done = True
            elif hit_sl:
                reward += self._realize(self.sl if side > 0 else self.sl - self.spread,
                                        self.fraction)
                self.done = True
            elif hit_tp:
                reward += self._realize(self.tp if side > 0 else self.tp - self.spread,
                                        self.fraction)
                self.done = True

        self.i += 1
        truncated = False
        if not self.done and (self.i >= self.horizon
                              or self.e["t"] + 1 + self.i >= len(self.bars) - 1):
            reward += self._realize(c, self.fraction)
            self.done, truncated = True, True
        return self._obs(c), float(reward), self.done and not truncated, truncated, {}

    def _obs(self, price: float) -> np.ndarray:
        side, atr = self.e["side"], self.e["atr"]
        return np.array([
            side * (price - self.entry) / atr,
            self.i / self.horizon,
            self.e.get("atr_pctile", 0.5),
            side * (price - self.sl) / atr,
            side * (self.tp - price) / atr,
            1.0 if self.e.get("h1_trend", 0.0) == side else 0.0,
            min(self.e.get("mins_to_high", 2880.0), 2880.0) / 2880.0,
            self.fraction,
        ], dtype=np.float32)
