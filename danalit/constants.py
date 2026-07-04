"""Shared constants: timeframes, sessions, canonical enums."""

from __future__ import annotations

# Timeframe name -> minutes per bar
TIMEFRAMES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

# Timeframe name -> pandas resample frequency string
PANDAS_FREQ: dict[str, str] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

# Trading sessions in UTC hours [start, end). London/NY shift with DST; these
# fixed UTC windows are a deliberate simplification — the session *feature* only
# needs to be consistent between training and live, not civil-time exact.
SESSIONS_UTC: dict[str, tuple[int, int]] = {
    "tokyo": (0, 9),
    "london": (7, 16),
    "newyork": (12, 21),
}

# Decision actions
LONG = "LONG"
SHORT = "SHORT"
NONE = "NONE"

# Orchestrator states
STARTING = "STARTING"
RECONCILING = "RECONCILING"
TRADING = "TRADING"
HALTED = "HALTED"
STOPPED = "STOPPED"

# Journal modes
MODE_DRY_RUN = "dry_run"
MODE_DEMO = "demo"
MODE_LIVE = "live"

# Calendar impact levels (canonical)
IMPACT_HIGH = "high"
IMPACT_MEDIUM = "medium"
IMPACT_LOW = "low"
