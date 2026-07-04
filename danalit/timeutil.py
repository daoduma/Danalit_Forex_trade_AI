"""UTC time helpers — the single place time-string conventions live."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts) -> str:
    """Render any timestamp-like as ISO-8601 UTC with second precision."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now_iso() -> str:
    return iso(utc_now())


def parse_iso(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
