"""Prompt 3: news dedup, calendar upserts, RSS fixture parse, FRED/GDELT transforms."""

from datetime import date

import pytest

from danalit.data import calendar_ingest, fred_ingest, gdelt_backfill, news_ingest
from danalit.data.collector_daemon import heartbeat_age_seconds, write_heartbeat
from danalit.db import connect, init_db


@pytest.fixture()
def con(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    c = connect(db)
    yield c
    c.close()


def test_news_dedup_identical_content(con):
    assert news_ingest.store_item(con, "fxstreet", "ECB holds rates", "summary text") is True
    assert news_ingest.store_item(con, "other_src", "ECB  holds   rates", "Summary TEXT") is False
    assert con.execute("SELECT COUNT(*) c FROM news").fetchone()["c"] == 1


RSS_FIXTURE = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Test Feed</title>
<item><title>Gold surges on Fed cut bets</title>
<link>https://example.com/a1</link>
<description>Gold rallied two percent.</description>
<pubDate>Thu, 02 Jul 2026 14:30:00 GMT</pubDate></item>
<item><title>Nasdaq slips ahead of CPI</title>
<link>https://example.com/a2</link>
<description>Tech stocks eased.</description>
<pubDate>Thu, 02 Jul 2026 15:00:00 GMT</pubDate></item>
</channel></rss>"""


def test_rss_poll_from_fixture(con):
    n = news_ingest.poll_feed(con, "testfeed", "http://ignored", fetcher=lambda u: RSS_FIXTURE)
    assert n == 2
    # Second poll of identical feed inserts nothing
    assert news_ingest.poll_feed(con, "testfeed", "http://ignored", fetcher=lambda u: RSS_FIXTURE) == 0
    row = con.execute("SELECT * FROM news WHERE title LIKE 'Gold%'").fetchone()
    assert row["published_utc"] == "2026-07-02T14:30:00Z"
    assert row["ingested_utc"] is not None
    assert row["url"] == "https://example.com/a1"


def test_calendar_upsert_updates_actual_without_duplicating(con):
    ev = {
        "date": "2026-07-03T12:30:00-04:00",
        "title": "Non-Farm Employment Change",
        "country": "USD",
        "impact": "High",
        "forecast": "190K",
        "previous": "180K",
        "actual": "",
    }
    calendar_ingest.upsert_events(con, [calendar_ingest.normalize_event(ev)])
    ev["actual"] = "225K"  # release happened; feed re-fetched
    calendar_ingest.upsert_events(con, [calendar_ingest.normalize_event(ev)])

    rows = con.execute("SELECT * FROM calendar_events").fetchall()
    assert len(rows) == 1
    r = rows[0]
    assert r["actual"] == 225000.0
    assert r["forecast"] == 190000.0
    assert r["event_utc"] == "2026-07-03T16:30:00Z"  # -04:00 converted to UTC
    assert r["impact"] == "high"
    assert r["canonical_name"] == "NFP"


def test_parse_value_variants():
    pv = calendar_ingest.parse_value
    assert pv("3.4%") == 3.4
    assert pv("-0.1%") == -0.1
    assert pv("225K") == 225000
    assert pv("1.2M") == 1200000
    assert pv("2,350K") == 2350000
    assert pv("") is None
    assert pv(None) is None
    assert pv(2.5) == 2.5


def test_canonical_event_names():
    c = calendar_ingest.canonical_event_name
    assert c("Non-Farm Employment Change") == "NFP"
    assert c("Nonfarm Payrolls") == "NFP"
    assert c("CPI m/m") == "CPI"
    assert c("Federal Funds Rate") == "FOMC_RATE"


def test_fred_observations_to_events(con):
    obs = [
        {"date": "2026-05-01", "value": "3.1"},
        {"date": "2026-06-01", "value": "."},  # FRED missing marker
        {"date": "2026-06-15", "value": "3.3"},
    ]
    events = fred_ingest.observations_to_events("CPIAUCSL", obs)
    assert len(events) == 2
    assert events[0]["actual"] == 3.1 and events[0]["previous"] is None
    assert events[1]["previous"] == 3.1
    assert events[1]["currency"] == "USD" and events[1]["impact"] == "high"
    assert calendar_ingest.upsert_events(con, events) == 2


def test_gdelt_backfill_resumable(con):
    calls = []

    def fake_fetcher(url, params):
        calls.append(params["mode"])
        return {"timeline": [{"data": [
            {"date": "2015-01-01", "value": 10},
            {"date": "2015-01-02", "value": 20},
        ]}]}

    n1 = gdelt_backfill.backfill(con, start=date(2015, 1, 1), end=date(2015, 1, 3), fetcher=fake_fetcher)
    assert n1 == 2 * len(gdelt_backfill.KEYWORD_SETS)
    calls.clear()
    # All days present now -> resumable run makes zero API calls
    n2 = gdelt_backfill.backfill(con, start=date(2015, 1, 1), end=date(2015, 1, 3), fetcher=fake_fetcher)
    assert n2 == 0
    assert calls == []


def test_heartbeat_roundtrip(tmp_path):
    hb = tmp_path / "collector.heartbeat"
    assert heartbeat_age_seconds(hb) is None
    write_heartbeat(hb)
    age = heartbeat_age_seconds(hb)
    assert age is not None and 0 <= age < 10
