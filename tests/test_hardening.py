"""Prompt 20 failure injection: broken gateway mid-sequence, journal write
failure (orders fail CLOSED), corrupted config, clock skew, stale collector."""

import pandas as pd
import pytest

from danalit.constants import HALTED, TRADING
from danalit.preflight import check_clock_skew, run_preflight
from tests.test_orchestrator import BAR_T, make_orch


def test_gateway_disconnect_mid_sequence_fails_order_not_process(tmp_path):
    orch = make_orch(tmp_path, dry_run=False, x=3.0)
    orch.startup()
    orch.gw.mt5.order_send = lambda req: None  # hard disconnect at send time
    orch.tick()
    # order failed, but the loop survives (no unhandled exception -> TRADING)
    assert orch.state == TRADING
    con = orch.journal._con()
    order = con.execute("SELECT * FROM orders").fetchone()
    con.close()
    assert order["status"] == "failed"


def test_journal_failure_means_no_order_fail_closed(tmp_path):
    """THE fail-closed rule: no journal, no order."""
    orch = make_orch(tmp_path, dry_run=False, x=3.0)
    orch.startup()

    def broken_intent(*a, **k):
        raise RuntimeError("disk full")

    orch.journal.record_order_intent = broken_intent
    orch.tick()
    assert orch.state == HALTED                      # loud, not silent
    entries = [c for c in orch.gw.mt5.order_send_calls
               if "position" not in c]
    assert entries == []                             # nothing was ever sent


def test_corrupted_config_refuses_to_start(tmp_path):
    from danalit.config import load_config

    bad = tmp_path / "settings.yaml"
    bad.write_text("risk: {risk_per_trade: 5.0}\ncapital: {tiers: []}", encoding="utf-8")
    with pytest.raises(Exception):
        load_config(settings_path=bad)


def test_clock_skew_thresholds():
    now = 1_750_000_000.0
    assert check_clock_skew(now + 1, now)[0] == "PASS"
    assert check_clock_skew(now + 30, now)[0] == "WARN"
    assert check_clock_skew(now + 120, now)[0] == "FAIL"
    # a known +2h server offset is removed before judging skew
    assert check_clock_skew(now + 7200 + 2, now, server_offset_s=7200)[0] == "PASS"


def test_stale_collector_vetoes_entries_during_trading(tmp_path):
    orch = make_orch(tmp_path, dry_run=False, x=3.0)
    orch.startup()
    real = orch.features
    orch.features = lambda n: {**real(n), "collector_age_sec": 3600.0}  # dead 1h
    orch.tick()
    con = orch.journal._con()
    d = con.execute("SELECT * FROM decisions WHERE instrument='EURUSD'").fetchone()
    n_orders = con.execute("SELECT COUNT(*) c FROM orders").fetchone()["c"]
    con.close()
    assert d["action"] == "NONE" and "collector" in d["veto_reason"]
    assert n_orders == 0


def test_preflight_passes_on_healthy_repo(tmp_path):
    from danalit.db import init_db
    from danalit.models import registry
    from danalit.config import load_config

    db = tmp_path / "pf.db"
    init_db(db)
    # register champions so the registry check passes
    from danalit.db import connect

    con = connect(db)
    with con:
        for name in load_config().enabled_instruments():
            con.execute("INSERT INTO model_registry (instrument, version, created_utc,"
                        " is_champion) VALUES (?, 'v1', '2026-07-01T00:00:00Z', 1)", (name,))
    con.close()
    passed, rows = run_preflight(db_path=db)
    assert passed, rows
    checks = {r["check"]: r["status"] for r in rows}
    assert checks["db integrity"] == "PASS"
    assert checks["model registry"] == "PASS"


def test_preflight_fails_without_champions(tmp_path):
    from danalit.db import init_db

    db = tmp_path / "pf2.db"
    init_db(db)
    passed, rows = run_preflight(db_path=db)
    assert not passed
    assert any(r["check"] == "model registry" and r["status"] == "FAIL" for r in rows)
