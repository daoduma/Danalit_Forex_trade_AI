"""Prompt 1: db --init creates all tables; a second --init is a no-op."""

from danalit.db import EXPECTED_TABLES, connect, existing_tables, init_db


def test_init_creates_all_tables(tmp_path):
    db = tmp_path / "danalit.db"
    created = init_db(db)
    assert sorted(created) == EXPECTED_TABLES
    con = connect(db)
    try:
        assert existing_tables(con) >= set(EXPECTED_TABLES)
        row = con.execute("SELECT version FROM schema_version").fetchone()
        assert row["version"] == 1
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        con.close()


def test_second_init_is_noop(tmp_path):
    db = tmp_path / "danalit.db"
    init_db(db)
    con = connect(db)
    try:
        con.execute(
            "INSERT INTO news (source, ingested_utc, title, content_hash) VALUES (?,?,?,?)",
            ("test", "2026-07-04T00:00:00Z", "hello", "h1"),
        )
        con.commit()
    finally:
        con.close()

    created = init_db(db)  # must not recreate or wipe anything
    assert created == []
    con = connect(db)
    try:
        n = con.execute("SELECT COUNT(*) c FROM news").fetchone()["c"]
        assert n == 1
        versions = con.execute("SELECT COUNT(*) c FROM schema_version").fetchone()["c"]
        assert versions == 1
    finally:
        con.close()


def test_content_hash_unique(tmp_path):
    import sqlite3

    db = tmp_path / "danalit.db"
    init_db(db)
    con = connect(db)
    try:
        con.execute(
            "INSERT INTO news (source, ingested_utc, title, content_hash) VALUES (?,?,?,?)",
            ("test", "2026-07-04T00:00:00Z", "a", "same"),
        )
        try:
            con.execute(
                "INSERT INTO news (source, ingested_utc, title, content_hash) VALUES (?,?,?,?)",
                ("test", "2026-07-04T00:00:01Z", "b", "same"),
            )
            raised = False
        except sqlite3.IntegrityError:
            raised = True
        assert raised
    finally:
        con.close()
