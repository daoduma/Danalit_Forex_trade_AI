"""Triple-barrier labeling — the trade definition the whole system shares.

For each bar T (decision made at T's close, entry at the NEXT bar's open):
  long framing : entry at ask = open[T+1] + spread; TP at entry + k_tp*ATR(T),
                 SL at entry - k_sl*ATR(T); barrier checks against BID prices.
  short framing: entry at bid = open[T+1]; buy-back at ask = price + spread.
Walk forward up to `horizon` bars. Pessimistic intrabar rule: if SL and TP both
lie inside one bar's range, SL is assumed hit first. Gaps: if a bar OPENS past
a barrier, the exit price is the gapped open, not the barrier price.
Timeout: sign of the net return with a +/- dead_zone*ATR dead zone, else 0.

Costs are inside the labels: the spread is paid on entry (long) / exit (short),
exactly as the backtester and live execution will pay it.

3-class output: 0 = no-trade, 1 = profitable-long, 2 = profitable-short.
If both framings hit TP, the earlier hit wins; a tie is no-trade.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from danalit.features.technical import atr as _atr

LABEL_NONE, LABEL_LONG, LABEL_SHORT = 0, 1, 2

LABEL_COLUMNS = [
    "label", "label_long", "label_short", "atr",
    "entry_long", "entry_short", "tp_long", "sl_long", "tp_short", "sl_short",
    "hit_bar_long", "hit_bar_short", "ret_long", "ret_short", "ret_atr",
]


def triple_barrier(
    bars: pd.DataFrame,
    spread: float,
    k_tp: float = 2.0,
    k_sl: float = 1.0,
    horizon: int = 96,
    dead_zone_atr: float = 0.25,
) -> pd.DataFrame:
    """Label every bar with enough forward history. Returns a frame indexed by
    the DECISION bar's open time (features at T align 1:1 with labels at T)."""
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    t = pd.to_datetime(bars["time_utc"]).to_numpy()
    atr = _atr(bars, 14).to_numpy()
    n = len(bars)
    m = n - horizon - 1  # decision bars with a full label window
    if m <= 0:
        return pd.DataFrame(columns=LABEL_COLUMNS)

    T = np.arange(m)
    a = atr[T]
    entry_long = o[T + 1] + spread   # buy at ask
    entry_short = o[T + 1]           # sell at bid
    tp_L, sl_L = entry_long + k_tp * a, entry_long - k_sl * a
    sl_S, tp_S = entry_short + k_sl * a, entry_short - k_tp * a

    INF = np.iinfo(np.int32).max
    tL = np.full(m, INF, dtype=np.int64)   # bar offset of long exit (TP or SL)
    winL = np.zeros(m, dtype=np.int8)      # +1 TP, -1 SL, 0 open
    exitL = np.full(m, np.nan)
    tS = np.full(m, INF, dtype=np.int64)
    winS = np.zeros(m, dtype=np.int8)
    exitS = np.full(m, np.nan)

    for j in range(1, horizon + 1):
        oj, hj, lj = o[T + j], h[T + j], l[T + j]
        # ---- long: bid barriers ------------------------------------------
        open_ = winL == 0
        gap_sl = open_ & (oj <= sl_L)
        gap_tp = open_ & ~gap_sl & (oj >= tp_L)
        hit_sl = open_ & ~gap_sl & ~gap_tp & (lj <= sl_L)          # pessimistic first
        hit_tp = open_ & ~gap_sl & ~gap_tp & ~hit_sl & (hj >= tp_L)
        for mask, win, price in (
            (gap_sl, -1, oj), (gap_tp, 1, oj), (hit_sl, -1, sl_L), (hit_tp, 1, tp_L),
        ):
            winL[mask] = win
            tL[mask] = j
            exitL[mask] = price[mask] if isinstance(price, np.ndarray) else price
        # ---- short: ask barriers (price + spread) -------------------------
        oja, hja, lja = oj + spread, hj + spread, lj + spread
        open_ = winS == 0
        gap_sl = open_ & (oja >= sl_S)
        gap_tp = open_ & ~gap_sl & (oja <= tp_S)
        hit_sl = open_ & ~gap_sl & ~gap_tp & (hja >= sl_S)
        hit_tp = open_ & ~gap_sl & ~gap_tp & ~hit_sl & (lja <= tp_S)
        for mask, win, price in (
            (gap_sl, -1, oja), (gap_tp, 1, oja), (hit_sl, -1, sl_S), (hit_tp, 1, tp_S),
        ):
            winS[mask] = win
            tS[mask] = j
            exitS[mask] = price[mask] if isinstance(price, np.ndarray) else price

    # ---- timeouts ----------------------------------------------------------
    to_L = winL == 0
    exitL[to_L] = c[T + horizon][to_L]
    tL[to_L] = horizon
    to_S = winS == 0
    exitS[to_S] = c[T + horizon][to_S] + spread
    tS[to_S] = horizon

    ret_long = exitL - entry_long
    ret_short = entry_short - exitS
    dz = dead_zone_atr * a

    lab_L = np.where(winL == 1, 1, np.where(winL == -1, -1,
             np.where(ret_long > dz, 1, np.where(ret_long < -dz, -1, 0))))
    lab_S = np.where(winS == 1, 1, np.where(winS == -1, -1,
             np.where(ret_short > dz, 1, np.where(ret_short < -dz, -1, 0))))

    long_wins = (lab_L == 1) & ((lab_S != 1) | (tL < tS))
    short_wins = (lab_S == 1) & ((lab_L != 1) | (tS < tL))
    label = np.where(long_wins, LABEL_LONG, np.where(short_wins, LABEL_SHORT, LABEL_NONE))
    ret_chosen = np.where(label == LABEL_LONG, ret_long,
                          np.where(label == LABEL_SHORT, ret_short, 0.0))

    out = pd.DataFrame(
        {
            "label": label.astype(np.int8),
            "label_long": lab_L.astype(np.int8),
            "label_short": lab_S.astype(np.int8),
            "atr": a,
            "entry_long": entry_long, "entry_short": entry_short,
            "tp_long": tp_L, "sl_long": sl_L, "tp_short": tp_S, "sl_short": sl_S,
            "hit_bar_long": tL, "hit_bar_short": tS,
            "ret_long": ret_long, "ret_short": ret_short,
            "ret_atr": np.where(a > 0, ret_chosen / a, 0.0),
        },
        index=pd.DatetimeIndex(t[:m], name="time_utc"),
    )
    return out.dropna(subset=["atr"])


def label_span(horizon: int, bar_minutes: int = 15) -> pd.Timedelta:
    """Total wall-clock span a label at T can peek into: entry bar + horizon bars."""
    return pd.Timedelta(minutes=(horizon + 1) * bar_minutes)
