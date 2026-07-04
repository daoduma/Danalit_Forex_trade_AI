"""SQLite database: schema, migrations, connection factory.

Usage:  python -m danalit.db --init [--db PATH]

WAL mode; all timestamps stored as ISO-8601 UTC strings ("...Z" or "+00:00").
Schema changes are applied as forward-only migrations keyed by schema_version.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1

# All CREATE statements are idempotent; the journal tables are append-only.
_TABLES: dict[str, str] = {
    "schema_version": """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        )""",
    "news": """
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            published_utc TEXT,
            ingested_utc TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT,
            url TEXT,
            content_hash TEXT NOT NULL UNIQUE
        )""",
    "news_scores": """
        CREATE TABLE IF NOT EXISTS news_scores (
            news_id INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            p_pos REAL, p_neg REAL, p_neu REAL,
            entities TEXT,
            scored_utc TEXT,
            PRIMARY KEY (news_id, model_version)
        )""",
    "calendar_events": """
        CREATE TABLE IF NOT EXISTS calendar_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            event_utc TEXT NOT NULL,
            currency TEXT NOT NULL,
            name TEXT NOT NULL,
            canonical_name TEXT,
            impact TEXT,
            actual REAL, forecast REAL, previous REAL, revised REAL,
            UNIQUE (source, event_utc, currency, name)
        )""",
    "gdelt_daily": """
        CREATE TABLE IF NOT EXISTS gdelt_daily (
            date TEXT NOT NULL,
            keyword_set TEXT NOT NULL,
            article_count INTEGER,
            avg_tone REAL,
            PRIMARY KEY (date, keyword_set)
        )""",
    "decisions": """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            instrument TEXT NOT NULL,
            action TEXT NOT NULL,
            confidence REAL,
            sl_price REAL, tp_price REAL,
            explanation TEXT,
            veto_reason TEXT,
            features_snapshot TEXT,
            mode TEXT NOT NULL DEFAULT 'dry_run',
            signal_id TEXT UNIQUE
        )""",
    "orders": """
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL UNIQUE,
            signal_id TEXT,
            ts_utc TEXT NOT NULL,
            instrument TEXT NOT NULL,
            side TEXT,
            lots REAL,
            sl REAL, tp REAL,
            status TEXT NOT NULL,
            retcode INTEGER,
            intended_price REAL,
            filled_price REAL,
            broker_ticket INTEGER,
            error TEXT
        )""",
    "trades": """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT,
            instrument TEXT NOT NULL,
            side TEXT NOT NULL,
            opened_utc TEXT,
            closed_utc TEXT,
            entry_price REAL,
            exit_price REAL,
            lots REAL,
            sl REAL, tp REAL,
            gross_pnl REAL,
            costs REAL,
            net_pnl REAL,
            mae REAL, mfe REAL,
            exit_reason TEXT,
            mode TEXT
        )""",
    "managed_actions": """
        CREATE TABLE IF NOT EXISTS managed_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            trade_id INTEGER,
            instrument TEXT,
            rule TEXT NOT NULL,
            before_state TEXT,
            after_state TEXT
        )""",
    "equity_snapshots": """
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            balance REAL,
            equity REAL,
            margin REAL,
            open_risk REAL,
            mode TEXT
        )""",
    "system_events": """
        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            type TEXT NOT NULL,
            detail TEXT
        )""",
    "model_registry": """
        CREATE TABLE IF NOT EXISTS model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument TEXT NOT NULL,
            version TEXT NOT NULL,
            dataset_version TEXT,
            path TEXT,
            metrics TEXT,
            git_commit TEXT,
            created_utc TEXT,
            is_champion INTEGER NOT NULL DEFAULT 0,
            UNIQUE (instrument, version)
        )""",
    "risk_state": """
        CREATE TABLE IF NOT EXISTS risk_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_utc TEXT
        )""",
    "set_aside_ledger": """
        CREATE TABLE IF NOT EXISTS set_aside_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            month TEXT NOT NULL,
            net_pnl REAL,
            credit REAL,
            withdrawal REAL,
            balance REAL NOT NULL,
            note TEXT
        )""",
    "optuna_trials": """
        CREATE TABLE IF NOT EXISTS optuna_trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study TEXT NOT NULL,
            trial INTEGER,
            params TEXT,
            value REAL,
            state TEXT,
            ts_utc TEXT
        )""",
}

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_news_ingested ON news (ingested_utc)",
    "CREATE INDEX IF NOT EXISTS idx_calendar_event_utc ON calendar_events (event_utc)",
    "CREATE INDEX IF NOT EXISTS idx_calendar_currency ON calendar_events (currency, impact)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions (ts_utc, instrument)",
    "CREATE INDEX IF NOT EXISTS idx_trades_opened ON trades (opened_utc, instrument)",
    "CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots (ts_utc)",
    "CREATE INDEX IF NOT EXISTS idx_sysevents_ts ON system_events (ts_utc, type)",
]

EXPECTED_TABLES = sorted(_TABLES)


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a connection with WAL mode and sane defaults."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.row_factory = sqlite3.Row
    return con


def existing_tables(con: sqlite3.Connection) -> set[str]:
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r["name"] for r in rows}


def init_db(db_path: Path | str) -> list[str]:
    """Create schema; idempotent. Returns list of newly created tables."""
    con = connect(db_path)
    try:
        before = existing_tables(con)
        with con:
            for ddl in _TABLES.values():
                con.execute(ddl)
            for idx in _INDEXES:
                con.execute(idx)
            row = con.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                con.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            elif row["version"] < SCHEMA_VERSION:
                _migrate(con, row["version"])
        return sorted(existing_tables(con) - before)
    finally:
        con.close()


def _migrate(con: sqlite3.Connection, from_version: int) -> None:
    """Forward-only migrations. Add steps as SCHEMA_VERSION grows."""
    # v1 is the baseline; nothing to migrate yet.
    con.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))


def main(argv: Optional[list[str]] = None) -> int:
    from danalit.config import load_config

    ap = argparse.ArgumentParser(description="Danalit database admin")
    ap.add_argument("--init", action="store_true", help="create schema (idempotent)")
    ap.add_argument("--db", type=Path, default=None, help="database path override")
    args = ap.parse_args(argv)

    db_path = args.db or load_config().settings.paths.db_path
    if args.init:
        created = init_db(db_path)
        if created:
            print(f"Initialized {db_path} — created tables: {', '.join(created)}")
        else:
            print(f"{db_path} already initialized — no changes.")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
