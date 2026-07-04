"""Economic calendar ingestion (ForexFactory weekly JSON feed + backfill).

Live feed: the free weekly JSON at
    https://nfs.faireconomy.media/ff_calendar_thisweek.json
gives event, country/currency, impact, actual/forecast/previous. We re-fetch on a
schedule; upserts are idempotent and update 'actual' after releases.

Deep backfill: ForexFactory's site is JS-rendered and scraping it reliably is
fragile; the supported paths are (a) `ingest_json_files()` for weekly JSON files
in the same schema obtained from community archives, fetched politely with the
rate-limited `polite_fetch()` helper, and (b) the FRED backfill
(danalit/data/fred_ingest.py), which provides the authoritative deep history for
the *surprise* features. Respect robots/ToS; default rate limit 1 request / 2 s.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

import requests

from danalit.logging_setup import setup_logging
from danalit.timeutil import iso

log = setup_logging("calendar_ingest")

FF_WEEKLY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

_IMPACT_MAP = {
    "high": "high", "red": "high",
    "medium": "medium", "orange": "medium", "yellow": "medium",
    "low": "low", "gray": "low", "grey": "low", "holiday": "low", "non-economic": "low",
}

_NUM_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([%kmbKMB])?")


def parse_value(raw) -> Optional[float]:
    """'3.4%' -> 3.4, '225K' -> 225000, '1.2M' -> 1200000, '' -> None."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    m = _NUM_RE.match(str(raw).replace(",", ""))
    if not m:
        return None
    val = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    mult = {"k": 1e3, "m": 1e6, "b": 1e9}.get(suffix, 1.0)
    return val * mult  # '%' keeps the bare number


def normalize_event(raw: dict, source: str = "forexfactory") -> Optional[dict]:
    """Map one feed entry to a calendar_events row dict; None if unusable."""
    date = raw.get("date") or raw.get("dateline")
    title = raw.get("title") or raw.get("name")
    currency = raw.get("country") or raw.get("currency")
    if not (date and title and currency):
        return None
    try:
        event_utc = iso(date)
    except (ValueError, TypeError):
        return None
    impact = _IMPACT_MAP.get(str(raw.get("impact", "")).lower(), "low")
    return {
        "source": source,
        "event_utc": event_utc,
        "currency": str(currency).upper(),
        "name": str(title).strip(),
        "canonical_name": canonical_event_name(title),
        "impact": impact,
        "actual": parse_value(raw.get("actual")),
        "forecast": parse_value(raw.get("forecast")),
        "previous": parse_value(raw.get("previous")),
        "revised": parse_value(raw.get("revised")),
    }


# Event-name normalization so 'Non-Farm Employment Change' variants map together.
_CANONICAL_PATTERNS = [
    (re.compile(r"non.?farm", re.I), "NFP"),
    (re.compile(r"\bCPI\b|consumer price", re.I), "CPI"),
    (re.compile(r"\bFOMC\b|federal funds rate|fed interest", re.I), "FOMC_RATE"),
    (re.compile(r"main refinancing|ECB.*rate|deposit facility", re.I), "ECB_RATE"),
    (re.compile(r"unemployment rate", re.I), "UNEMPLOYMENT"),
    (re.compile(r"\bGDP\b", re.I), "GDP"),
    (re.compile(r"retail sales", re.I), "RETAIL_SALES"),
    (re.compile(r"\bPCE\b|personal consumption", re.I), "PCE"),
    (re.compile(r"\bPMI\b|purchasing managers", re.I), "PMI"),
]


def canonical_event_name(title: str) -> str:
    for pat, canon in _CANONICAL_PATTERNS:
        if pat.search(title):
            return canon
    return re.sub(r"[^A-Z0-9]+", "_", title.upper()).strip("_")[:64]


def upsert_events(con: sqlite3.Connection, events: Iterable[dict]) -> int:
    """Idempotent upsert; re-fetching updates actual/forecast/previous/revised."""
    n = 0
    with con:
        for ev in events:
            if ev is None:
                continue
            con.execute(
                """INSERT INTO calendar_events
                   (source, event_utc, currency, name, canonical_name, impact,
                    actual, forecast, previous, revised)
                   VALUES (:source,:event_utc,:currency,:name,:canonical_name,:impact,
                           :actual,:forecast,:previous,:revised)
                   ON CONFLICT (source, event_utc, currency, name) DO UPDATE SET
                     actual=excluded.actual, forecast=excluded.forecast,
                     previous=excluded.previous, revised=excluded.revised,
                     impact=excluded.impact, canonical_name=excluded.canonical_name""",
                ev,
            )
            n += 1
    return n


def fetch_weekly(con: sqlite3.Connection, url: str = FF_WEEKLY_URL, fetcher=None) -> int:
    """Fetch the current weekly feed and upsert. Returns rows processed."""
    if fetcher is None:
        fetcher = lambda u: requests.get(u, timeout=30, headers={"User-Agent": "danalit/0.1"}).text  # noqa: E731
    try:
        raw = json.loads(fetcher(url))
    except Exception as e:
        log.error("calendar fetch failed: %s", e)
        return 0
    events = [normalize_event(r) for r in raw if isinstance(r, dict)]
    n = upsert_events(con, events)
    log.info("calendar: upserted %d events", n)
    return n


class polite_fetch:  # noqa: N801 — small callable utility
    """Rate-limited, on-disk-cached HTTP GET for backfill jobs (resumable)."""

    def __init__(self, cache_dir: Path, min_interval_sec: float = 2.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval_sec
        self._last = 0.0

    def __call__(self, url: str) -> str:
        key = re.sub(r"[^A-Za-z0-9._-]+", "_", url)[-120:]
        cached = self.cache_dir / key
        if cached.exists():
            return cached.read_text(encoding="utf-8")
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        resp = requests.get(url, timeout=30, headers={"User-Agent": "danalit/0.1"})
        self._last = time.monotonic()
        resp.raise_for_status()
        cached.write_text(resp.text, encoding="utf-8")
        return resp.text


def ingest_json_files(con: sqlite3.Connection, json_dir: Path) -> int:
    """Backfill from weekly-JSON files (ff_calendar schema) saved on disk."""
    total = 0
    for path in sorted(Path(json_dir).glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("skipping unparseable %s", path.name)
            continue
        total += upsert_events(con, (normalize_event(r) for r in raw if isinstance(r, dict)))
    return total
