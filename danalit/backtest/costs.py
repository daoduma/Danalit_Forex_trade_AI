"""Cost model: spread, commission, slippage (news-aware), swap with triple-Wednesday.

Stored bar prices are BID. Longs buy at ask = bid + spread and exit at bid;
shorts sell at bid and buy back at ask. Fill in your broker's real values in
instruments.yaml — the demo forward test (Prompt 16) measures the true numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from danalit.config import InstrumentConfig


@dataclass
class CostModel:
    spread: float                      # price units (used when bar has no recorded spread)
    commission_per_lot: float = 0.0    # account ccy per lot, charged round-turn at close
    slippage: float = 0.0              # price units on market fills
    news_slippage_extra: float = 0.0   # added within news windows
    swap_long: float = 0.0             # price units per lot per night (sign as broker quotes)
    swap_short: float = 0.0
    triple_swap_prev_weekday: int = 2  # rollover leaving Wednesday counts 3 nights
    is_news_time: Optional[Callable[[pd.Timestamp], bool]] = field(default=None, repr=False)

    @classmethod
    def from_instrument(cls, inst: InstrumentConfig) -> "CostModel":
        pip = inst.pip_size
        return cls(
            spread=inst.spread_estimate_pips * pip,
            commission_per_lot=inst.commission_per_lot,
            slippage=0.2 * pip,
            news_slippage_extra=1.0 * pip,
            swap_long=inst.swap_long_pips * pip,
            swap_short=inst.swap_short_pips * pip,
        )

    def spread_at(self, bar_spread: Optional[float]) -> float:
        """Per-bar recorded spread if present and sane, else the config estimate."""
        if bar_spread is not None and bar_spread == bar_spread and bar_spread > 0:
            return float(bar_spread)
        return self.spread

    def slippage_at(self, t: pd.Timestamp) -> float:
        extra = self.news_slippage_extra if (self.is_news_time and self.is_news_time(t)) else 0.0
        return self.slippage + extra

    def swap_for_night(self, side: int, crossing_from_weekday: int) -> float:
        """Swap price adjustment for one midnight crossing (side +1 long / -1 short)."""
        mult = 3.0 if crossing_from_weekday == self.triple_swap_prev_weekday else 1.0
        per_night = self.swap_long if side > 0 else self.swap_short
        return per_night * mult
