"""Prompt 5: sentiment/calendar features — ingest-time alignment, surprise z, blackout edges."""

import numpy as np
import pandas as pd
import pytest

from danalit.data import news_ingest
from danalit.db import connect, init_db
from danalit.features import fundamental, sentiment


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "t.db"
    init_db(p)
    return p


def bars_frame(start="2026-07-01 11:00", periods=12):
    times = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    return pd.DataFrame({"time_utc": times, "dummy": np.arange(periods, dtype=float)})


def _insert_scored(db, title, ingested_utc, p_pos, p_neg, entities):
    con = connect(db)
    try:
        with con:
            cur = con.execute(
                "INSERT INTO news (source, ingested_utc, title, content_hash) VALUES (?,?,?,?)",
                ("test", ingested_utc, title, news_ingest.content_hash(title + ingested_utc)),
            )
            con.execute(
                "INSERT INTO news_scores (news_id, model_version, p_pos, p_neg, p_neu, entities, scored_utc)"
                " VALUES (?,?,?,?,?,?,?)",
                (cur.lastrowid, "test-v1", p_pos, p_neg, 1 - p_pos - p_neg, entities, ingested_utc),
            )
    finally:
        con.close()


def test_alignment_headline_at_1207_first_affects_bar_closing_1215(db):
    _insert_scored(db, "Dollar surges on strong data", "2026-07-01T12:07:00Z", 0.9, 0.05, "usd")
    df = sentiment.add_sentiment_features(bars_frame(), "EURUSD", db_path=db)
    by_open = df.set_index("time_utc")
    # bar open 11:45 closes 12:00 — before ingestion: must NOT see the headline
    assert by_open.loc["2026-07-01 11:45:00+00:00", "sent_usd_1h"] == 0.0
    # bar open 12:00 closes 12:15 — first bar that may know it
    assert by_open.loc["2026-07-01 12:00:00+00:00", "sent_usd_1h"] == pytest.approx(0.85)
    assert by_open.loc["2026-07-01 12:00:00+00:00", "news_avail"] == 1.0


def test_neutral_degradation_when_no_news(db):
    df = sentiment.add_sentiment_features(bars_frame(), "XAUUSD", db_path=db)
    sent_cols = [c for c in df.columns if c.startswith(("sent_", "maxneg_", "news_"))]
    assert df[sent_cols].isna().sum().sum() == 0
    assert (df["news_avail"] == 0).all()
    assert (df[[c for c in sent_cols if c != "news_avail"]] == 0).all().all()


def test_score_news_incremental_with_lexicon(db):
    con = connect(db)
    try:
        news_ingest.store_item(con, "t", "Gold rallies to record high", url="u1")
        news_ingest.store_item(con, "t", "Nasdaq plunges as recession fears grow", url="u2")
        n1 = sentiment.score_news(con, scorer=sentiment.lexicon_score, model_version="lexicon-v1")
        assert n1 == 2
        n2 = sentiment.score_news(con, scorer=sentiment.lexicon_score, model_version="lexicon-v1")
        assert n2 == 0  # incremental: nothing left to score
        rows = con.execute(
            "SELECT n.title, s.p_pos, s.p_neg, s.entities FROM news n JOIN news_scores s ON s.news_id=n.id"
        ).fetchall()
        gold = next(r for r in rows if "Gold" in r["title"])
        nasdaq = next(r for r in rows if "Nasdaq" in r["title"])
        assert gold["p_pos"] > gold["p_neg"] and "gold" in gold["entities"]
        assert nasdaq["p_neg"] > nasdaq["p_pos"] and "nasdaq" in nasdaq["entities"]
    finally:
        con.close()


def _insert_event(db, event_utc, name, impact="high", currency="USD",
                  actual=None, forecast=None, previous=None):
    con = connect(db)
    try:
        with con:
            con.execute(
                "INSERT INTO calendar_events (source, event_utc, currency, name, canonical_name,"
                " impact, actual, forecast, previous) VALUES (?,?,?,?,?,?,?,?,?)",
                ("test", event_utc, currency, name, name, impact, actual, forecast, previous),
            )
    finally:
        con.close()


def test_blackout_flag_edges(db):
    _insert_event(db, "2026-07-01T12:00:00Z", "NFP")
    df = fundamental.add_calendar_features(bars_frame(), "XAUUSD", db_path=db)
    by_open = df.set_index("time_utc")
    assert by_open.loc["2026-07-01 11:30:00+00:00", "blackout"] == 1.0  # close 11:45, 15min before
    assert by_open.loc["2026-07-01 11:45:00+00:00", "blackout"] == 1.0  # close 12:00 = event
    assert by_open.loc["2026-07-01 12:00:00+00:00", "blackout"] == 1.0  # close 12:15, 15min after
    assert by_open.loc["2026-07-01 12:15:00+00:00", "blackout"] == 0.0  # close 12:30, 30min after
    assert by_open.loc["2026-07-01 11:00:00+00:00", "mins_to_next_high"] == 45.0


def test_surprise_z_on_synthetic_history(db):
    # 6 monthly releases with surprise (actual-forecast) = [1,-1,1,-1,1] then a +2 shock
    base = pd.Timestamp("2026-01-03 13:30", tz="UTC")
    surprises = [1, -1, 1, -1, 1]
    for i, s in enumerate(surprises):
        t = base + pd.Timedelta(days=30 * i)
        _insert_event(db, t.strftime("%Y-%m-%dT%H:%M:%SZ"), "NFP", actual=200 + s, forecast=200)
    shock_t = base + pd.Timedelta(days=150)
    _insert_event(db, shock_t.strftime("%Y-%m-%dT%H:%M:%SZ"), "NFP", actual=202, forecast=200)

    bars = pd.DataFrame({"time_utc": pd.date_range(shock_t - pd.Timedelta(minutes=15),
                                                   periods=8, freq="15min")})
    df = fundamental.add_calendar_features(bars, "XAUUSD", db_path=db).set_index("time_utc")
    # prior surprises std = std([1,-1,1,-1,1]) ~ 1.095; z = 2/1.095 ~ 1.83,
    # decayed by exp(-15/160) at the first close 15 min after release -> ~1.66
    first_after = df.loc[shock_t]  # bar opening at release closes 15min later
    assert first_after["surprise_8h"] == pytest.approx(1.83 * np.exp(-15 / 160), abs=0.1)
    # decays: 4 bars later the contribution is smaller but positive
    later = df.iloc[5]
    assert 0 < later["surprise_8h"] < first_after["surprise_8h"]
    # bar closing exactly AT the release must not see the actual yet... its close == event time
    # (allowed: z applies from the first close >= event time, which is this one) — check the one before
    before = df.iloc[0]
    assert before["surprise_8h"] == 0.0


def test_expanding_z_requires_history(db):
    _insert_event(db, "2026-07-01T12:00:00Z", "CPI", actual=5, forecast=3)  # first ever release
    bars = bars_frame()
    df = fundamental.add_calendar_features(bars, "XAUUSD", db_path=db)
    assert (df["surprise_8h"] == 0).all()  # <3 prior samples -> no z
