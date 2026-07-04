"""GDELT DOC 2.0 backfill: daily article counts + average tone per keyword set.

One-time, resumable, politely rate-limited. Monthly API windows are exploded
into daily rows in the gdelt_daily table. Aggregates only — single articles
from GDELT are too noisy to use directly.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import date, timedelta
from typing import Optional

import requests

from danalit.logging_setup import setup_logging

log = setup_logging("gdelt_backfill")

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

KEYWORD_SETS: dict[str, str] = {
    "fed": '"federal reserve" OR FOMC',
    "ecb": '"european central bank" OR ECB',
    "gold": '"gold price" OR "gold market"',
    "nasdaq": 'nasdaq OR "tech stocks"',
}


def fetch_timeline(query: str, start: date, end: date, mode: str, fetcher=None) -> list[dict]:
    """One GDELT timeline call; returns [{'date': 'YYYY-MM-DD', 'value': float}, ...]."""
    if fetcher is None:
        def fetcher(url, params):
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
    data = fetcher(API_URL, {
        "query": query, "mode": mode, "format": "json",
        "STARTDATETIME": start.strftime("%Y%m%d000000"),
        "ENDDATETIME": end.strftime("%Y%m%d235959"),
    })
    out = []
    for series in data.get("timeline", []):
        for pt in series.get("data", []):
            out.append({"date": str(pt.get("date", ""))[:10].replace("/", "-"),
                        "value": float(pt.get("value", 0.0))})
    return out


def existing_days(con: sqlite3.Connection, keyword_set: str) -> set[str]:
    rows = con.execute("SELECT date FROM gdelt_daily WHERE keyword_set=?", (keyword_set,)).fetchall()
    return {r["date"] for r in rows}


def backfill(
    con: sqlite3.Connection,
    start: date = date(2015, 1, 1),
    end: Optional[date] = None,
    rate_limit_sec: float = 6.0,
    fetcher=None,
) -> int:
    """Fill gdelt_daily from start..end, skipping days already present (resumable)."""
    end = end or date.today()
    written = 0
    for name, query in KEYWORD_SETS.items():
        have = existing_days(con, name)
        cursor = start
        while cursor < end:
            month_end = min(cursor + timedelta(days=30), end)
            window_days = {
                (cursor + timedelta(days=i)).isoformat()
                for i in range((month_end - cursor).days)
            }
            if window_days - have:  # anything missing in this window?
                try:
                    vol = {p["date"]: p["value"]
                           for p in fetch_timeline(query, cursor, month_end, "timelinevolraw", fetcher)}
                    tone = {p["date"]: p["value"]
                            for p in fetch_timeline(query, cursor, month_end, "timelinetone", fetcher)}
                    with con:
                        for day in sorted(window_days - have):
                            if day in vol or day in tone:
                                con.execute(
                                    "INSERT OR REPLACE INTO gdelt_daily"
                                    " (date, keyword_set, article_count, avg_tone) VALUES (?,?,?,?)",
                                    (day, name, int(vol.get(day, 0)), tone.get(day)),
                                )
                                written += 1
                    if fetcher is None:
                        time.sleep(rate_limit_sec)
                except KeyboardInterrupt:
                    log.info("interrupted — resumable, %d rows written so far", written)
                    raise
                except Exception as e:
                    log.error("gdelt %s %s: %s", name, cursor, e)
            cursor = month_end
    log.info("gdelt backfill wrote %d day-rows", written)
    return written
