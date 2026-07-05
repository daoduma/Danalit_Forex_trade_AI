"""Fixed-fractional position sizing — THE only place lot sizes are computed.

Unit-agnostic: pass equity and value_per_price_unit_per_lot in the SAME
account-currency unit and the math holds. On a cent account, balances are in
cents and 1 cent-lot moves 100,000 cent-units, so with equity in cents and
contract_size 100,000 this is numerically identical to a USD account — which
is exactly why the cent account makes a $20 (2,000c) deposit tradable at
professional risk fractions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class SizingResult:
    lots: float
    risk_amount: float          # account ccy actually at risk (entry -> SL)
    risk_frac_actual: float     # risk_amount / equity
    refused: bool = False
    reason: str = ""


def size_position(
    equity: float,
    risk_frac: float,
    sl_distance: float,               # price units, entry -> stop
    value_per_price_unit_per_lot: float,  # = contract_size for USD-quoted instruments
    min_lot: float,
    lot_step: float,
    max_lot: float,
    price: float,
    leverage: float,
    margin_available: Optional[float] = None,
    hard_cap_mult: float = 1.5,
) -> SizingResult:
    """Derive lots from the risk budget. Position size is NEVER chosen by a model."""
    if sl_distance <= 0:
        return SizingResult(0, 0, 0, refused=True, reason="non-positive SL distance")
    if equity <= 0:
        return SizingResult(0, 0, 0, refused=True, reason="non-positive equity")

    risk_budget = equity * risk_frac
    risk_per_lot = sl_distance * value_per_price_unit_per_lot
    lots = risk_budget / risk_per_lot
    lots = math.floor(lots / lot_step + 1e-9) * lot_step

    if lots < min_lot:
        # min_lot would exceed the budget: allow only within the hard cap
        min_risk = min_lot * risk_per_lot
        if min_risk <= equity * risk_frac * hard_cap_mult:
            lots = min_lot
        else:
            return SizingResult(
                0, min_risk, min_risk / equity, refused=True,
                reason=(f"min_lot {min_lot} risks {min_risk:.2f} "
                        f"({100 * min_risk / equity:.2f}% of equity) — exceeds "
                        f"{hard_cap_mult}x the {100 * risk_frac:.2f}% budget"),
            )

    lots = min(lots, max_lot)

    # margin headroom at configured leverage
    if margin_available is not None and price > 0:
        margin_per_lot = value_per_price_unit_per_lot * price / leverage
        if margin_per_lot > 0:
            max_by_margin = math.floor(margin_available / margin_per_lot / lot_step + 1e-9) * lot_step
            if max_by_margin < min_lot:
                return SizingResult(0, 0, 0, refused=True, reason="insufficient margin")
            lots = min(lots, max_by_margin)

    lots = round(lots, 8)
    risk_amount = lots * risk_per_lot
    return SizingResult(lots, risk_amount, risk_amount / equity)
