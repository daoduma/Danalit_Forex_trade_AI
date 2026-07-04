"""RSS news collector: polls configurable feeds, dedups by content hash.

Both published_utc (feed's claim) and ingested_utc (when WE received it) are
stored — features may only ever use ingested_utc, so the archive is aligned
with what the live system could actually have known.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from typing import Optional

import feedparser
import requests

from danalit.logging_setup import setup_logging
from danalit.timeutil import iso, utc_now_iso

log = setup_logging("news_ingest")

DEFAULT_FEEDS: dict[str, str] = {
    "fxstreet": "https://www.fxstreet.com/rss/news",
    "investing": "https://www.investing.com/rss/news_25.rss",
    "dailyfx": "https://www.dailyfx.com/feeds/market-news",
    "gnews_fed": "https://news.google.com/rss/search?q=%22federal+reserve%22&hl=en-US&gl=US&ceid=US:en",
    "gnews_ecb": "https://news.google.com/rss/search?q=ECB+euro&hl=en-US&gl=US&ceid=US:en",
    "gnews_gold": "https://news.google.com/rss/search?q=%22gold+price%22&hl=en-US&gl=US&ceid=US:en",
    "gnews_nasdaq": "https://news.google.com/rss/search?q=nasdaq&hl=en-US&gl=US&ceid=US:en",
}


def content_hash(title: str, summary: str = "") -> str:
    norm = " ".join((title or "").lower().split()) + "|" + " ".join((summary or "").lower().split())[:400]
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def store_item(
    con: sqlite3.Connection,
    source: str,
    title: str,
    summary: str = "",
    url: str = "",
    published_utc: Optional[str] = None,
) -> bool:
    """Insert one headline; returns True if new (False on duplicate)."""
    try:
        with con:
            con.execute(
                "INSERT INTO news (source, published_utc, ingested_utc, title, body, url, content_hash)"
                " VALUES (?,?,?,?,?,?,?)",
                (source, published_utc, utc_now_iso(), title.strip(), summary, url,
                 content_hash(title, summary)),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def poll_feed(con: sqlite3.Connection, source: str, url: str, fetcher=None) -> int:
    """Fetch one RSS feed and store new items. Returns count of new rows."""
    if fetcher is None:
        fetcher = lambda u: requests.get(  # noqa: E731
            u, timeout=30, headers={"User-Agent": "danalit/0.1"}
        ).content
    parsed = feedparser.parse(fetcher(url))
    new = 0
    for entry in parsed.entries:
        title = getattr(entry, "title", "") or ""
        if not title.strip():
            continue
        published = None
        pp = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
        if pp:
            published = iso(time.strftime("%Y-%m-%dT%H:%M:%SZ", pp))
        if store_item(
            con, source, title,
            summary=getattr(entry, "summary", "") or "",
            url=getattr(entry, "link", "") or "",
            published_utc=published,
        ):
            new += 1
    return new


def poll_all(con: sqlite3.Connection, feeds: Optional[dict[str, str]] = None) -> dict[str, int]:
    """Poll every feed with per-source error isolation."""
    feeds = feeds or DEFAULT_FEEDS
    results = {}
    for source, url in feeds.items():
        try:
            results[source] = poll_feed(con, source, url)
        except Exception as e:  # one dead feed must not stop the others
            log.error("feed %s failed: %s", source, e)
            results[source] = -1
    log.info("rss poll: %s", results)
    return results
