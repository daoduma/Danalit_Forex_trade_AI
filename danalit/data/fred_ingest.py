"""FRED backfill: authoritative US/EU macro series -> calendar_events (source='fred').

Free API key from https://fred.stlouisfed.org (env var FRED_API_KEY).
FRED gives observation values but not consensus forecasts; the fundamental
feature layer z-scores 'actual' against each event's own history, so forecast
stays NULL here. Release timestamps are approximated at the typical US release
time (13:30 UTC) — good enough for daily-scale surprise features.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional

import requests

from danalit.data.calendar_ingest import upsert_events
from danalit.logging_setup import setup_logging

log = setup_logging("fred_ingest")

API_URL = "https://api.stlouisfed.org/fred/series/observations"

# series_id -> (canonical event name, currency, impact)
SERIES: dict[str, tuple[str, str, str]] = {
    "CPIAUCSL": ("CPI", "USD", "high"),
    "CPILFESL": ("CORE_CPI", "USD", "high"),
    "PAYEMS": ("NFP", "USD", "high"),
    "UNRATE": ("UNEMPLOYMENT", "USD", "high"),
    "FEDFUNDS": ("FOMC_RATE", "USD", "high"),
    "PCEPI": ("PCE", "USD", "medium"),
    "GDP": ("GDP", "USD", "high"),
    "RSAFS": ("RETAIL_SALES", "USD", "medium"),
    "ECBDFR": ("ECB_RATE", "EUR", "high"),
}


def fetch_series(series_id: str, api_key: str, start: str = "2010-01-01", fetcher=None) -> list[dict]:
    if fetcher is None:
        def fetcher(url, params):
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
    data = fetcher(API_URL, {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "observation_start": start,
    })
    return data.get("observations", [])


def observations_to_events(series_id: str, observations: list[dict]) -> list[dict]:
    name, currency, impact = SERIES[series_id]
    events, prev = [], None
    for obs in observations:
        raw = obs.get("value")
        if raw in (None, "", "."):
            continue
        value = float(raw)
        events.append({
            "source": "fred",
            "event_utc": f"{obs['date']}T13:30:00Z",
            "currency": currency,
            "name": name,
            "canonical_name": name,
            "impact": impact,
            "actual": value,
            "forecast": None,
            "previous": prev,
            "revised": None,
        })
        prev = value
    return events


def backfill(con: sqlite3.Connection, start: str = "2010-01-01",
             api_key: Optional[str] = None, fetcher=None) -> int:
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY env var not set (free key: fred.stlouisfed.org)")
    total = 0
    for series_id in SERIES:
        try:
            obs = fetch_series(series_id, api_key, start=start, fetcher=fetcher)
            n = upsert_events(con, observations_to_events(series_id, obs))
            log.info("fred %s: %d events", series_id, n)
            total += n
        except Exception as e:
            log.error("fred %s failed: %s", series_id, e)
    return total
