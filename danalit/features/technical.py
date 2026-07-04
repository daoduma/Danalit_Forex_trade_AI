"""Leakage-safe technical feature engine.

CORRECTNESS RULE (overrides everything): a feature value at bar T may use ONLY
data with time_utc <= T's bar CLOSE time. Features are consumed at the NEXT
bar's open. All computations are strictly backward-looking rolling operations;
higher-timeframe context joins AS-OF the last CLOSED higher-TF bar.

Indicators are hand-rolled in pure pandas/numpy. (The roadmap suggested
pandas-ta; it is unmaintained and breaks on numpy>=2 / Python 3.13, so the
~20 indicators used here are implemented directly — deterministic, tested,
no compiled dependencies.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from danalit.config import load_config
from danalit.constants import SESSIONS_UTC, TIMEFRAMES
from danalit.data import price_store

# ---------------------------------------------------------------------------
# Feature registry: name -> {description, params, group}
# ---------------------------------------------------------------------------
FEATURE_REGISTRY: dict[str, dict] = {}


def _reg(name: str, description: str, group: str, **params) -> str:
    FEATURE_REGISTRY[name] = {"description": description, "group": group, "params": params}
    return name


# ---------------------------------------------------------------------------
# Indicator primitives (all backward-looking)
# ---------------------------------------------------------------------------

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    denom = gain + loss
    out = pd.Series(np.where(denom > 0, 100 * gain / denom, 50.0), index=close.index)
    return out.where(gain.notna() & loss.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = ema(close, fast) - ema(close, slow)
    return line - line.ewm(span=signal, adjust=False, min_periods=signal).mean()


def bollinger(close: pd.Series, period: int = 20, ndev: float = 2.0) -> tuple[pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    width = (2 * ndev * std) / mid
    pctb = (close - (mid - ndev * std)) / (2 * ndev * std).replace(0, np.nan)
    return width, pctb


def rolling_percentile(s: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    return s.rolling(window, min_periods=min_periods or window // 4).rank(pct=True)


# ---------------------------------------------------------------------------
# Candlestick pattern scores (simplified, hand-rolled)
# ---------------------------------------------------------------------------

def pattern_scores(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l

    bull_engulf = (c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))
    bear_engulf = (c < o) & (c.shift(1) > o.shift(1)) & (c <= o.shift(1)) & (o >= c.shift(1))
    hammer = (lower > 2 * body) & (upper < body) & (body / rng < 0.35)
    shooting = (upper > 2 * body) & (lower < body) & (body / rng < 0.35)
    doji = body / rng < 0.1

    bull = bull_engulf.astype(float) + hammer.astype(float) + 0.5 * doji.astype(float)
    bear = bear_engulf.astype(float) + shooting.astype(float) + 0.5 * doji.astype(float)
    return bull.fillna(0), bear.fillna(0)


# ---------------------------------------------------------------------------
# Higher-timeframe context (as-of the last CLOSED higher-TF bar)
# ---------------------------------------------------------------------------

def _trend_stack(df: pd.DataFrame) -> pd.Series:
    e8, e21, e50 = ema(df["close"], 8), ema(df["close"], 21), ema(df["close"], 50)
    up = (e8 > e21) & (e21 > e50)
    down = (e8 < e21) & (e21 < e50)
    return pd.Series(np.where(up, 1.0, np.where(down, -1.0, 0.0)), index=df.index)


def higher_tf_features(htf_bars: pd.DataFrame, tf: str, prefix: str) -> pd.DataFrame:
    """Features on a higher timeframe, keyed by that bar's CLOSE time for as-of joins."""
    out = pd.DataFrame({"close_time": htf_bars["time_utc"] + pd.Timedelta(minutes=TIMEFRAMES[tf])})
    out[f"{prefix}_trend"] = _trend_stack(htf_bars)
    if prefix == "h4":
        out[f"{prefix}_rsi"] = rsi(htf_bars["close"], 14)
    if prefix == "d1":
        a = atr(htf_bars, 14)
        out[f"{prefix}_atr_pctile"] = rolling_percentile(a, 90, min_periods=30)
    return out


def asof_join(base: pd.DataFrame, htf: pd.DataFrame) -> pd.DataFrame:
    """Join higher-TF features onto base rows using only bars CLOSED by base open time.

    A higher-TF bar whose close time == the M15 open time is fully known before
    the M15 bar completes, so exact matches are allowed; a forming bar never is.
    """
    return pd.merge_asof(
        base.sort_values("time_utc"),
        htf.sort_values("close_time"),
        left_on="time_utc",
        right_on="close_time",
        direction="backward",
        allow_exact_matches=True,
    ).drop(columns="close_time")


# ---------------------------------------------------------------------------
# Daily / weekly reference levels (previous COMPLETED period only)
# ---------------------------------------------------------------------------

def _prev_period_levels(bars: pd.DataFrame, freq: str, prefix: str) -> pd.DataFrame:
    g = bars.set_index("time_utc").resample(freq)
    levels = pd.DataFrame({"p_high": g["high"].max(), "p_low": g["low"].min()}).dropna()
    # Key by period END: a period's levels become usable once the period completes.
    if freq == "1D":
        period_end = levels.index + pd.Timedelta(days=1)
    else:  # weekly W-FRI: index is the Friday label; usable from Friday 24:00
        period_end = levels.index + pd.Timedelta(days=1)
    return pd.DataFrame(
        {"close_time": period_end,
         f"{prefix}_high": levels["p_high"].values,
         f"{prefix}_low": levels["p_low"].values}
    )


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def build_features(
    instrument: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    root: Optional[Path] = None,
    include_news: bool = False,
    db_path: Optional[Path] = None,
    primary_tf: str = "M15",
) -> pd.DataFrame:
    """Build the technical (and optionally news/calendar) feature matrix.

    Returns a DataFrame indexed by time_utc (bar OPEN time); the feature row at
    T is fully computable at T's close and is consumed at the next bar's open.
    Warmup rows with NaNs are dropped; nothing is forward-filled across gaps.
    """
    cfg = load_config()
    inst = cfg.instruments[instrument]
    bars = price_store.read_bars(instrument, primary_tf, start=None, end=end, root=root)
    if bars.empty:
        raise ValueError(f"no {primary_tf} bars stored for {instrument}")
    bars = bars.reset_index(drop=True)
    f = pd.DataFrame({"time_utc": bars["time_utc"]})
    c, h, l = bars["close"], bars["high"], bars["low"]
    a14 = atr(bars, 14)
    safe_atr = a14.replace(0, np.nan)

    # --- trend / momentum -------------------------------------------------
    for span in (8, 21, 50, 200):
        e = ema(c, span)
        f[_reg(f"ema{span}_dist", f"(close-EMA{span})/ATR", "trend", span=span)] = (c - e) / safe_atr
        f[_reg(f"ema{span}_slope", f"EMA{span} 4-bar slope in ATR", "trend", span=span)] = (
            e - e.shift(4)
        ) / safe_atr
    mh = macd_hist(c)
    f[_reg("macd_hist", "MACD(12,26,9) histogram / ATR", "trend")] = mh / safe_atr
    f[_reg("macd_hist_slope", "MACD hist 3-bar slope / ATR", "trend")] = (mh - mh.shift(3)) / safe_atr
    r = rsi(c, 14)
    f[_reg("rsi14", "RSI(14)", "momentum")] = r
    f[_reg("rsi_bull_div", "price 20-bar low but RSI is not", "momentum")] = (
        (l <= l.rolling(20).min()) & (r > r.rolling(20).min() + 2)
    ).astype(float)
    f[_reg("rsi_bear_div", "price 20-bar high but RSI is not", "momentum")] = (
        (h >= h.rolling(20).max()) & (r < r.rolling(20).max() - 2)
    ).astype(float)
    f[_reg("adx14", "ADX(14)", "trend")] = adx(bars, 14)
    for n in (4, 16, 96):
        f[_reg(f"roc_{n}", f"{n}-bar rate of change", "momentum", n=n)] = c.pct_change(n)

    # --- volatility ---------------------------------------------------------
    f[_reg("atr_norm", "ATR(14)/close", "volatility")] = a14 / c
    f[_reg("atr_pctile_90d", "ATR percentile over ~90 days", "volatility")] = rolling_percentile(
        a14, 90 * 96, min_periods=96 * 20
    )
    bw, pctb = bollinger(c, 20, 2.0)
    f[_reg("bb_width", "Bollinger(20,2) width / mid", "volatility")] = bw
    f[_reg("bb_pctb", "Bollinger %B", "volatility")] = pctb
    f[_reg("range_regime", "BB width above its 96-bar median", "volatility")] = (
        bw > bw.rolling(96).median()
    ).astype(float)

    # --- structure ----------------------------------------------------------
    for n in (20, 100, 500):
        f[_reg(f"swing_high_dist_{n}", f"(rolling {n}-bar high - close)/ATR", "structure", n=n)] = (
            h.rolling(n).max() - c
        ) / safe_atr
        f[_reg(f"swing_low_dist_{n}", f"(close - rolling {n}-bar low)/ATR", "structure", n=n)] = (
            c - l.rolling(n).min()
        ) / safe_atr
    grid = inst.round_number_grid
    f[_reg("round_dist", "distance to nearest round-number level / ATR", "structure", grid=grid)] = (
        (c - (c / grid).round() * grid).abs() / safe_atr
    )

    base = f  # keep reference; merge_asof returns new frames below
    daily = _prev_period_levels(bars, "1D", "yest")
    weekly = _prev_period_levels(bars, "W-FRI", "lastweek")
    merged = asof_join(pd.concat([base, bars[["open", "high", "low", "close"]]], axis=1), daily)
    merged = asof_join(merged, weekly)
    for prefix in ("yest", "lastweek"):
        merged[_reg(f"{prefix}_high_dist", f"({prefix} high - close)/ATR", "structure")] = (
            merged[f"{prefix}_high"] - merged["close"]
        ) / safe_atr.values
        merged[_reg(f"{prefix}_low_dist", f"(close - {prefix} low)/ATR", "structure")] = (
            merged["close"] - merged[f"{prefix}_low"]
        ) / safe_atr.values
    merged = merged.drop(columns=["yest_high", "yest_low", "lastweek_high", "lastweek_low"])

    # --- candlestick patterns ------------------------------------------------
    bull, bear = pattern_scores(bars)
    merged[_reg("pattern_bull", "bullish candlestick score", "pattern")] = bull.values
    merged[_reg("pattern_bear", "bearish candlestick score", "pattern")] = bear.values

    # --- multi-timeframe context (as-of last CLOSED higher-TF bar) -----------
    for tf, prefix in (("H1", "h1"), ("H4", "h4"), ("D1", "d1")):
        htf_bars = price_store.read_bars(instrument, tf, root=root)
        if htf_bars.empty:
            raise ValueError(f"no {tf} bars for {instrument}; run build_all_timeframes first")
        merged = asof_join(merged, higher_tf_features(htf_bars, tf, prefix))
    _reg("h1_trend", "H1 EMA-stack trend (-1/0/+1)", "mtf")
    _reg("h4_trend", "H4 EMA-stack trend (-1/0/+1)", "mtf")
    _reg("h4_rsi", "H4 RSI(14)", "mtf")
    _reg("d1_trend", "D1 EMA-stack trend (-1/0/+1)", "mtf")
    _reg("d1_atr_pctile", "D1 ATR percentile (~90 days)", "mtf")

    # --- time encodings -------------------------------------------------------
    ts = merged["time_utc"].dt
    hod = ts.hour + ts.minute / 60.0
    merged[_reg("hod_sin", "sin(hour of day)", "time")] = np.sin(2 * np.pi * hod / 24)
    merged[_reg("hod_cos", "cos(hour of day)", "time")] = np.cos(2 * np.pi * hod / 24)
    merged[_reg("dow_sin", "sin(day of week)", "time")] = np.sin(2 * np.pi * ts.weekday / 7)
    merged[_reg("dow_cos", "cos(day of week)", "time")] = np.cos(2 * np.pi * ts.weekday / 7)
    for name, (s0, s1) in SESSIONS_UTC.items():
        merged[_reg(f"sess_{name}", f"{name} session active", "time")] = (
            (ts.hour >= s0) & (ts.hour < s1)
        ).astype(float)
    merged[_reg("sess_overlap", "London/NY overlap", "time")] = (
        (merged["sess_london"] > 0) & (merged["sess_newyork"] > 0)
    ).astype(float)
    lon_open_min = SESSIONS_UTC["london"][0] * 60
    mins = ts.hour * 60 + ts.minute
    merged[_reg("mins_since_london_open", "minutes since London open (capped 12h)", "time")] = (
        ((mins - lon_open_min) % (24 * 60)).clip(upper=720) / 720.0
    )
    # bars to weekend: minutes until Friday 21:00 UTC
    dow, minute_of_week = ts.weekday, ts.weekday * 1440 + mins
    fri_close = 4 * 1440 + 21 * 60
    to_weekend = (fri_close - minute_of_week) % (7 * 1440)
    merged[_reg("bars_to_weekend", "M15 bars until Friday 21:00 UTC (capped 5d)", "time")] = (
        to_weekend.clip(upper=5 * 1440) / 15.0
    )

    merged = merged.drop(columns=["open", "high", "low", "close"])

    # --- optional news/calendar features (Prompt 5) ---------------------------
    if include_news:
        from danalit.features import fundamental, sentiment

        merged = sentiment.add_sentiment_features(merged, instrument, db_path=db_path)
        merged = fundamental.add_calendar_features(merged, instrument, db_path=db_path)

    merged = merged.set_index("time_utc")
    merged = merged.dropna()  # warmup rows only; later rows are complete by construction
    if start is not None:
        s = pd.Timestamp(start)
        s = s.tz_localize("UTC") if s.tz is None else s.tz_convert("UTC")
        merged = merged[merged.index >= s]
    return merged
