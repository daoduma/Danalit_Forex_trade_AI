# Danalit — project context for Claude Code sessions

## Purpose

Danalit is an autonomous AI trading system for MetaTrader 5 trading **EURUSD, XAUUSD, US100**
using ML signals from technical + news features. Zero-budget open-source stack, Windows host,
demo first → cent account second. Full design: [DESIGN.md](DESIGN.md); source spec:
`Danalit_AI_Trading_System_Roadmap.pdf` (20 sequential build prompts, Chapter 13).

## Architecture (six layers, two loops)

1. **Data sources** — Dukascopy/HistData, broker MT5 history, RSS/GDELT news, ForexFactory calendar, FRED.
2. **Data layer** — Parquet price store (M1 base, resampled to M5..D1); SQLite (WAL) for news/events/journal.
3. **Intelligence** — leakage-safe features → FinBERT sentiment → triple-barrier labels → LightGBM per instrument.
4. **Decision & risk** — signal engine (fuse + veto) → risk manager (sizing/limits/breakers) → trade manager.
5. **Execution** — MT5 gateway (orders/retries/reconciliation) → orchestrator (bar-close state machine, kill switch).
6. **Operations** — journal everything, Telegram alerts, Streamlit dashboard, monthly champion/challenger retraining, capital policy.

Fast loop = every completed M15 bar. Slow loop = journal → scheduled retraining with promotion gates.

## Layout

```
danalit/            config.py db.py logging_setup.py constants.py
  data/             price_store, dukascopy_ingest, mt5_history_ingest, news_ingest,
                    calendar_ingest, fred_ingest, gdelt_backfill, collector_daemon, quality
  features/         technical, sentiment, fundamental, labeling, dataset
  models/           train, calibrate, evaluate, registry, tuning, retrain, rl_exit/
  backtest/         engine, costs, metrics, report, walkforward
  risk/             risk_manager, position_sizing, capital        <- INVIOLABLE CORE
  trading/          signal_engine, trade_manager, mt5_gateway, orchestrator
  monitor/          notifier, telegram_bot, dashboard
  journal/          journal, analytics
config/             settings.yaml, instruments.yaml
scripts/            CLIs; scripts/deploy/ for Task Scheduler + watchdog
tests/              pytest; tests/integration/ marked "integration"
data_store/ models_store/ reports/ logs/   (artifacts, gitignored)
```

## Conventions (inviolable)

- **ALL timestamps are timezone-aware UTC.** SQLite stores ISO-8601 strings; DataFrames use tz-aware indexes.
- **No lookahead in any feature/label/backtest.** A feature at bar T may use only data with time ≤ T's close;
  news aligns by `ingested_utc`, never `published_utc`. Every feature module ships a truncation test.
- **Risk limits are inviolable and live ONLY in `danalit/risk/`.** No other module computes position size.
  Models never output size.
- **Only `danalit/trading/mt5_gateway.py` may import `MetaTrader5`.** Everything else uses its normalized types.
  Heavy optional deps (torch/transformers/SB3/MetaTrader5/streamlit/telegram) are lazy-imported so the test
  suite runs without them.
- **Config-driven instruments** — no instrument-specific logic in code; add a block to `instruments.yaml`.
- **Everything journaled** — every decision (including NONE + vetoes) with its feature snapshot.
- **Every module gets pytest tests.** Risk manager and backtester get the densest coverage.
- Secrets only via env vars: `DANALIT_MT5_LOGIN/SERVER/PASSWORD`, `FRED_API_KEY`, `DANALIT_TG_TOKEN/CHAT_ID`.

## Running tests

```
python -m pytest                      # unit tests
python -m pytest -m integration      # integration suite (Prompt 20)
```

Initialize DB: `python -m danalit.db --init`. Build price data: `python scripts/build_price_data.py`.
