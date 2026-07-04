"""Calendar/event features per bar per instrument, from calendar_events.

Rules, not ML: proximity to scheduled events (the schedule is known in advance,
so 'minutes_to_next' is not leakage), blackout flags, and post-release surprise
z-scores computed against each canonical event's OWN history (expanding —
only releases strictly before the current one contribute to its std).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from danalit.config import load_config
from danalit.db import connect

CAP_MIN = 2880  # cap proximity features at 48h
SURPRISE_WINDOW_MIN = 480  # surprise decays over 8h
MIN_HISTORY_FOR_Z = 3


def load_events(db_path: Optional[Path], currencies: list[str]) -> pd.DataFrame:
    cfg = load_config()
    con = connect(db_path or cfg.settings.paths.db_path)
    try:
        q = ",".join("?" for _ in currencies)
        rows = con.execute(
            f"""SELECT event_utc, currency, canonical_name, impact, actual, forecast, previous
                FROM calendar_events WHERE currency IN ({q}) ORDER BY event_utc""",
            [c.upper() for c in currencies],
        ).fetchall()
    finally:
        con.close()
    df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame(
        columns=["event_utc", "currency", "canonical_name", "impact", "actual", "forecast", "previous"])
    if not df.empty:
        df["t"] = pd.to_datetime(df["event_utc"], utc=True)
        for col in ("actual", "forecast", "previous"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def surprise_z(events: pd.DataFrame) -> pd.DataFrame:
    """Expanding, leakage-safe z-score of (actual - forecast|previous) per canonical event."""
    ev = events.copy()
    baseline = ev["forecast"].where(ev["forecast"].notna(), ev["previous"])
    ev["surprise"] = ev["actual"] - baseline
    ev["z"] = 0.0
    for _, idx in ev.groupby("canonical_name").groups.items():
        s = ev.loc[idx].sort_values("t")["surprise"]
        # std of releases strictly BEFORE each one
        prior_std = s.expanding().std().shift(1)
        prior_n = s.expanding().count().shift(1).fillna(0)
        z = np.where(
            (prior_n >= MIN_HISTORY_FOR_Z) & (prior_std > 0) & s.notna(),
            s / prior_std, 0.0)
        ev.loc[s.index, "z"] = pd.Series(np.clip(z, -5, 5), index=s.index).astype(float)
    return ev


def add_calendar_features(
    df: pd.DataFrame,
    instrument: str,
    db_path: Optional[Path] = None,
    bar_minutes: int = 15,
    blackout_minutes: Optional[int] = None,
) -> pd.DataFrame:
    """Append calendar features to a frame with a time_utc column."""
    from danalit.features.technical import FEATURE_REGISTRY

    cfg = load_config()
    inst = cfg.instruments[instrument]
    blackout_minutes = blackout_minutes or cfg.settings.news.blackout_minutes
    events = load_events(db_path, inst.news_currencies)

    df = df.copy()
    closes = pd.to_datetime(df["time_utc"]) + pd.Timedelta(minutes=bar_minutes)
    c_ns = closes.values.astype("datetime64[ns]").astype("int64")
    n = len(df)

    def _times(sub: pd.DataFrame) -> np.ndarray:
        return sub["t"].values.astype("datetime64[ns]").astype("int64") if not sub.empty else np.array([], dtype="int64")

    high = events[events["impact"] == "high"] if not events.empty else events
    med = events[events["impact"] == "medium"] if not events.empty else events
    ht, mt = _times(high), _times(med)
    NS_MIN = 60_000_000_000

    def reg(name, desc):
        FEATURE_REGISTRY.setdefault(name, {"description": desc, "group": "calendar", "params": {}})
        return name

    if len(ht):
        nxt = np.searchsorted(ht, c_ns, side="left")
        to_next = np.where(nxt < len(ht), (ht[np.minimum(nxt, len(ht) - 1)] - c_ns) / NS_MIN, CAP_MIN)
        to_next = np.clip(to_next, 0, CAP_MIN)
        prv = np.searchsorted(ht, c_ns, side="right") - 1
        since_last = np.where(prv >= 0, (c_ns - ht[np.maximum(prv, 0)]) / NS_MIN, CAP_MIN)
        since_last = np.clip(since_last, 0, CAP_MIN)
    else:
        to_next = np.full(n, CAP_MIN, dtype=float)
        since_last = np.full(n, CAP_MIN, dtype=float)

    df[reg("mins_to_next_high", "minutes to next high-impact event (capped 48h)")] = to_next
    df[reg("mins_since_high", "minutes since last high-impact event (capped 48h)")] = since_last
    df[reg("blackout", f"within +/-{blackout_minutes}min of a high-impact event")] = (
        (to_next <= blackout_minutes) | (since_last <= blackout_minutes)
    ).astype(float)

    day_ns = 1440 * NS_MIN
    for name, t_arr in (("high", ht), ("medium", mt)):
        if len(t_arr):
            cnt = np.searchsorted(t_arr, c_ns + day_ns, side="right") - np.searchsorted(
                t_arr, c_ns, side="left")
        else:
            cnt = np.zeros(n)
        df[reg(f"cal_{name}_next24h", f"count of {name}-impact events in next 24h")] = cnt.astype(float)

    # Post-release surprise, decayed over 8h — only events already released (t <= close)
    surprise = np.zeros(n)
    if not events.empty:
        ev = surprise_z(events)
        ev = ev[(ev["z"] != 0) & ev["z"].notna()]
        for _, row in ev.iterrows():
            e_ns = row["t"].value
            # strictly-after: a bar closing exactly AT the release cannot know it yet
            lo = np.searchsorted(c_ns, e_ns, side="right")
            hi = np.searchsorted(c_ns, e_ns + SURPRISE_WINDOW_MIN * NS_MIN, side="right")
            if hi > lo:
                age_min = (c_ns[lo:hi] - e_ns) / NS_MIN
                surprise[lo:hi] += row["z"] * np.exp(-age_min / (SURPRISE_WINDOW_MIN / 3))
    df[reg("surprise_8h", "sum of decayed surprise z-scores over last 8h")] = surprise

    df[reg("cal_avail", "availability mask: any calendar history in prior 30 days")] = (
        _windowed_any(ht, c_ns, 30 * 1440) if len(ht) else np.zeros(n)
    )
    fill_cols = ["mins_to_next_high", "mins_since_high", "blackout",
                 "cal_high_next24h", "cal_medium_next24h", "surprise_8h", "cal_avail"]
    df[fill_cols] = df[fill_cols].fillna(0.0)
    return df


def _windowed_any(event_times: np.ndarray, closes: np.ndarray, window_min: int) -> np.ndarray:
    lo = np.searchsorted(event_times, closes - window_min * 60_000_000_000, side="right")
    hi = np.searchsorted(event_times, closes, side="right")
    return (hi > lo).astype(float)
