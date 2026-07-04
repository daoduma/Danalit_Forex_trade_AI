# Danalit

Autonomous AI trading system for MetaTrader 5 — EURUSD · XAUUSD · US100.
Zero-budget, open-source stack. Demo first, cent account second.

> Trading leveraged products involves substantial risk of loss. Nothing here is financial advice.

- **Design:** [DESIGN.md](DESIGN.md)
- **Roadmap / spec:** `Danalit_AI_Trading_System_Roadmap.pdf`
- **Operations manual:** `RUNBOOK.md` (written in Prompt 20)

## Quickstart (development)

```powershell
# Python 3.11+ on Windows
pip install -r requirements-core.txt     # light env for dev + tests
# pip install -r requirements.txt        # full env for live trading

python -m danalit.db --init              # create SQLite schema
python -m pytest                         # run the test suite
```

## Secrets (environment variables — never files)

| Variable | Purpose |
|---|---|
| `DANALIT_MT5_LOGIN` / `DANALIT_MT5_SERVER` / `DANALIT_MT5_PASSWORD` | MT5 account |
| `FRED_API_KEY` | FRED macro backfill (free key) |
| `DANALIT_TG_TOKEN` / `DANALIT_TG_CHAT_ID` | Telegram alerts + remote control |

## Key commands

| Command | Purpose |
|---|---|
| `python -m danalit.db --init` | Create/upgrade the database |
| `python scripts/build_price_data.py` | Ingest Dukascopy CSVs + MT5 history → Parquet + quality report |
| `python scripts/run_collector.py` | 24/7 news + calendar collector daemon |
| `python scripts/build_dataset.py` | Features + labels → versioned training datasets |
| `python scripts/train_models.py` | Train + calibrate + evaluate LightGBM models |
| `python scripts/run_walkforward.py` | Honest walk-forward backtest report |
| `python scripts/run_trading.py` | Live orchestrator (dry-run by default) |
| `python scripts/run_dashboard.py` | Streamlit dashboard (localhost) |
| `python scripts/journal_report.py` | Forward-test analytics + go-live checklist |
