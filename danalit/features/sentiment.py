"""News sentiment scoring + leakage-safe alignment onto the price timeline.

Scoring: ProsusAI/finbert via HuggingFace transformers when installed (CPU,
batched, cached into news_scores keyed by news_id+model_version so re-runs are
incremental). Falls back to a keyword lexicon scorer — clearly versioned
'lexicon-v1' — so the pipeline runs end-to-end in light environments.

LEAKAGE RULE FOR NEWS: alignment uses ingested_utc, NEVER published_utc — we
can only act on news once we received it. A headline ingested at 12:07 first
affects the bar closing at 12:15 (the M15 bar whose open is 12:00).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from danalit.config import load_config
from danalit.db import connect
from danalit.logging_setup import setup_logging
from danalit.timeutil import utc_now_iso

log = setup_logging("sentiment")

FINBERT_VERSION = "finbert-v1"
LEXICON_VERSION = "lexicon-v1"

# keyword -> entity tagger (lists are overlappable; extend in one place)
ENTITY_KEYWORDS: dict[str, list[str]] = {
    "usd": ["fed", "federal reserve", "fomc", "dollar", "powell", "treasury",
            "us economy", "u.s. economy", "nonfarm", "us cpi", "us inflation"],
    "eur": ["ecb", "euro", "eurozone", "lagarde", "bundesbank", "eur"],
    "gold": ["gold", "bullion", "xau", "precious metal"],
    "nasdaq": ["nasdaq", "tech stock", "us100", "equities", "s&p", "wall street"],
}

# instrument -> entities whose sentiment plausibly moves it
INSTRUMENT_ENTITIES: dict[str, list[str]] = {
    "EURUSD": ["eur", "usd"],
    "XAUUSD": ["gold", "usd"],
    "US100": ["nasdaq", "usd"],
}

WINDOWS = {"1h": 60, "4h": 240, "24h": 1440}

_POS_WORDS = ["surge", "rally", "rallies", "gain", "beat", "beats", "strong", "bullish",
              "rise", "rises", "soar", "record high", "upbeat", "optimis", "boost", "growth"]
_NEG_WORDS = ["fall", "falls", "drop", "drops", "plunge", "weak", "bearish", "miss",
              "misses", "fear", "crash", "recession", "slump", "decline", "warn", "cuts jobs",
              "selloff", "sell-off", "tumble"]


def tag_entities(text: str) -> list[str]:
    t = (text or "").lower()
    return [e for e, kws in ENTITY_KEYWORDS.items() if any(k in t for k in kws)]


def lexicon_score(texts: list[str]) -> list[dict]:
    out = []
    for text in texts:
        t = (text or "").lower()
        pos = sum(t.count(w) for w in _POS_WORDS)
        neg = sum(t.count(w) for w in _NEG_WORDS)
        total = pos + neg + 1.0
        out.append({"p_pos": pos / total, "p_neg": neg / total, "p_neu": 1.0 / total})
    return out


def get_scorer() -> tuple[Callable[[list[str]], list[dict]], str]:
    """Return (scoring function, model_version). Prefers FinBERT, degrades to lexicon."""
    try:
        from transformers import pipeline  # heavy; optional

        pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, device=-1)

        def finbert(texts: list[str]) -> list[dict]:
            results = pipe([t[:512] for t in texts], batch_size=16, truncation=True)
            out = []
            for scores in results:
                d = {s["label"].lower(): s["score"] for s in scores}
                out.append({"p_pos": d.get("positive", 0.0),
                            "p_neg": d.get("negative", 0.0),
                            "p_neu": d.get("neutral", 0.0)})
            return out

        return finbert, FINBERT_VERSION
    except Exception as e:  # ImportError or model download failure
        log.warning("FinBERT unavailable (%s) — using lexicon scorer", type(e).__name__)
        return lexicon_score, LEXICON_VERSION


def score_news(
    con: sqlite3.Connection,
    scorer: Optional[Callable] = None,
    model_version: Optional[str] = None,
    batch_size: int = 64,
) -> int:
    """Score every unscored news row; incremental and resumable. Returns rows scored."""
    if scorer is None:
        scorer, model_version = get_scorer()
    assert model_version, "model_version required when passing a custom scorer"
    rows = con.execute(
        """SELECT n.id, n.title, COALESCE(n.body,'') AS body FROM news n
           LEFT JOIN news_scores s ON s.news_id = n.id AND s.model_version = ?
           WHERE s.news_id IS NULL ORDER BY n.id""",
        (model_version,),
    ).fetchall()
    scored = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [f"{r['title']}. {r['body'][:300]}" for r in batch]
        results = scorer(texts)
        with con:
            for r, s in zip(batch, results):
                entities = ",".join(tag_entities(r["title"] + " " + r["body"]))
                con.execute(
                    "INSERT OR REPLACE INTO news_scores"
                    " (news_id, model_version, p_pos, p_neg, p_neu, entities, scored_utc)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (r["id"], model_version, s["p_pos"], s["p_neg"], s["p_neu"],
                     entities, utc_now_iso()),
                )
        scored += len(batch)
    if scored:
        log.info("scored %d news rows with %s", scored, model_version)
    return scored


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------

def _windowed_sum(event_times: np.ndarray, values: np.ndarray,
                  closes: np.ndarray, window_min: int) -> np.ndarray:
    """For each close time, sum of values with event time in (close - window, close]."""
    if len(event_times) == 0:
        return np.zeros(len(closes))
    order = np.argsort(event_times)
    et, v = event_times[order], values[order]
    cum = np.concatenate([[0.0], np.cumsum(v)])
    hi = np.searchsorted(et, closes, side="right")
    lo = np.searchsorted(et, closes - window_min * 60_000_000_000, side="right")
    return cum[hi] - cum[lo]


def add_sentiment_features(
    df: pd.DataFrame,
    instrument: str,
    db_path: Optional[Path] = None,
    bar_minutes: int = 15,
) -> pd.DataFrame:
    """Append rolling sentiment features to a frame with a time_utc column."""
    from danalit.features.technical import FEATURE_REGISTRY

    cfg = load_config()
    con = connect(db_path or cfg.settings.paths.db_path)
    try:
        rows = con.execute(
            """SELECT n.ingested_utc, s.p_pos, s.p_neg, s.entities
               FROM news n JOIN news_scores s ON s.news_id = n.id"""
        ).fetchall()
    finally:
        con.close()

    df = df.copy()
    closes = (pd.to_datetime(df["time_utc"]) + pd.Timedelta(minutes=bar_minutes)).values.astype("datetime64[ns]").astype("int64")
    entities = INSTRUMENT_ENTITIES.get(instrument, ["usd"])

    news = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame(
        columns=["ingested_utc", "p_pos", "p_neg", "entities"])
    if not news.empty:
        news["t"] = pd.to_datetime(news["ingested_utc"], utc=True).values.astype("datetime64[ns]").astype("int64")
        news["net"] = news["p_pos"].astype(float) - news["p_neg"].astype(float)
        news["conf"] = 1.0 - news["p_neu"].astype(float) if "p_neu" in news else 1.0

    cols: dict[str, np.ndarray] = {}
    any_news = np.zeros(len(df))
    for ent in entities:
        sub = news[news["entities"].fillna("").str.contains(ent)] if not news.empty else news
        et = sub["t"].to_numpy() if not sub.empty else np.array([], dtype="int64")
        net = (sub["net"] * 1.0).to_numpy() if not sub.empty else np.array([])
        for wname, wmin in WINDOWS.items():
            name = f"sent_{ent}_{wname}"
            FEATURE_REGISTRY.setdefault(name, {
                "description": f"net {ent} sentiment (pos-neg) over {wname}, by ingest time",
                "group": "news", "params": {"window_min": wmin}})
            cols[name] = _windowed_sum(et, net, closes, wmin)
        # max single-article negativity in last 4h
        name = f"maxneg_{ent}_4h"
        FEATURE_REGISTRY.setdefault(name, {
            "description": f"max single-article P(neg) for {ent} in last 4h",
            "group": "news", "params": {}})
        cols[name] = _windowed_max(et, sub["p_neg"].to_numpy() if not sub.empty else np.array([]),
                                   closes, 240)
        any_news += _windowed_sum(et, np.ones(len(et)), closes, 7 * 1440)

    # news intensity: articles in 24h vs 30-day average (all relevant entities pooled)
    pooled = news[news["entities"].fillna("").apply(
        lambda e: any(x in e for x in entities))] if not news.empty else news
    et_all = pooled["t"].to_numpy() if not pooled.empty else np.array([], dtype="int64")
    ones = np.ones(len(et_all))
    day_count = _windowed_sum(et_all, ones, closes, 1440)
    month_avg = _windowed_sum(et_all, ones, closes, 30 * 1440) / 30.0
    FEATURE_REGISTRY.setdefault("news_intensity", {
        "description": "24h article count vs 30-day daily average", "group": "news", "params": {}})
    cols["news_intensity"] = np.divide(
        day_count, month_avg, out=np.zeros(len(closes)), where=month_avg > 0
    )

    FEATURE_REGISTRY.setdefault("news_avail", {
        "description": "availability mask: any relevant news in prior 7 days", "group": "news",
        "params": {}})
    cols["news_avail"] = (any_news > 0).astype(float)

    out = pd.DataFrame(cols, index=df.index)
    # neutral degradation: rows with no news history get exactly-neutral values
    out.loc[out["news_avail"] == 0, [c for c in out.columns if c != "news_avail"]] = 0.0
    return pd.concat([df, out.fillna(0.0)], axis=1)


def _windowed_max(event_times: np.ndarray, values: np.ndarray,
                  closes: np.ndarray, window_min: int) -> np.ndarray:
    """Rolling max of values in (close - window, close]; 0 when empty."""
    out = np.zeros(len(closes))
    if len(event_times) == 0:
        return out
    order = np.argsort(event_times)
    et, v = event_times[order], values[order]
    lo = np.searchsorted(et, closes - window_min * 60_000_000_000, side="right")
    hi = np.searchsorted(et, closes, side="right")
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = v[a:b].max()
    return out
