"""Prompt 18: promotion-gate branches, rollback trigger, PSI vs hand-computed."""

import json

import numpy as np
import pandas as pd
import pytest

from danalit.models.retrain import (
    GateResult,
    check_probation,
    compute_psi,
    promotion_gate,
)


def m(exp, dd, ll):
    return {"expectancy": exp, "max_drawdown": dd, "log_loss": ll}


def test_gate_all_pass_promotes():
    g = promotion_gate(m(0.02, 0.10, 1.00), m(0.03, 0.10, 0.98))
    assert g.promote and len(g.reasons) == 3


def test_gate_expectancy_fail_blocks():
    g = promotion_gate(m(0.03, 0.10, 1.00), m(0.02, 0.05, 0.90))
    assert not g.promote and any("expectancy" in r and "FAIL" in r for r in g.reasons)


def test_gate_drawdown_tolerance_boundary():
    # challenger dd exactly at champion * 1.1 -> allowed
    assert promotion_gate(m(0.02, 0.10, 1.0), m(0.02, 0.11, 1.0)).promote
    assert not promotion_gate(m(0.02, 0.10, 1.0), m(0.02, 0.12, 1.0)).promote


def test_gate_calibration_fail_blocks():
    g = promotion_gate(m(0.02, 0.10, 1.00), m(0.03, 0.10, 1.10))
    assert not g.promote and any("calibration" in r and "FAIL" in r for r in g.reasons)


def test_gate_missing_metrics_keeps_champion():
    g = promotion_gate({"expectancy": None}, m(0.03, 0.1, 1.0))
    assert not g.promote


def test_psi_identical_distributions_near_zero():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    assert compute_psi(a, b) < 0.02


def test_psi_shifted_distribution_flags():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(1.0, 1, 5000)  # full sigma shift
    assert compute_psi(a, b) > 0.2


def test_psi_hand_computed_two_bins():
    # train: 50/50 across the median split; live: 90/10
    train = np.array([0.0] * 50 + [1.0] * 50)
    live = np.array([0.0] * 90 + [1.0] * 10)
    psi = compute_psi(train, live, bins=2)
    expected = (0.9 - 0.5) * np.log(0.9 / 0.5) + (0.1 - 0.5) * np.log(0.1 / 0.5)
    assert psi == pytest.approx(expected, rel=1e-3)


class SpyNotifier:
    def __init__(self):
        self.sent = []

    def notify(self, level, title, body=""):
        self.sent.append((level, title, body))


@pytest.fixture()
def promo_db(tmp_path):
    """DB with a registered champion+challenger and a recorded promotion."""
    from danalit.db import connect, init_db

    db = tmp_path / "p.db"
    init_db(db)
    con = connect(db)
    with con:
        for version, champ in (("m_old", 0), ("m_new", 1)):
            con.execute(
                "INSERT INTO model_registry (instrument, version, created_utc, is_champion)"
                " VALUES ('EURUSD', ?, '2026-06-01T00:00:00Z', ?)", (version, champ))
        promo = {"instrument": "EURUSD", "champion": "m_old", "challenger": "m_new",
                 "challenger_metrics": {"expectancy": 0.05}, "promote": True}
        con.execute("INSERT INTO system_events (ts_utc, type, detail) VALUES (?,?,?)",
                    ("2026-07-01T00:00:00Z", "promotion", json.dumps(promo)))
    con.close()
    return db


def test_probation_rollback_on_degradation(promo_db):
    from danalit.db import connect
    from danalit.models import registry

    con = connect(promo_db)  # live trades losing money after promotion
    with con:
        for i in range(6):
            con.execute(
                "INSERT INTO trades (signal_id, instrument, side, closed_utc, net_pnl)"
                " VALUES (?, 'EURUSD', 'LONG', '2026-07-03T10:00:00Z', -2.0)", (f"s{i}",))
    con.close()

    spy = SpyNotifier()
    result = check_probation("EURUSD", db_path=promo_db, notifier=spy,
                             now=pd.Timestamp("2026-07-05", tz="UTC"))
    assert result == "rolled_back"
    assert registry.champion_version("EURUSD", db_path=promo_db) == "m_old"
    assert any(level == "CRITICAL" for level, *_ in spy.sent)


def test_probation_ok_when_profitable(promo_db):
    from danalit.db import connect
    from danalit.models import registry

    con = connect(promo_db)
    with con:
        for i in range(6):
            con.execute(
                "INSERT INTO trades (signal_id, instrument, side, closed_utc, net_pnl)"
                " VALUES (?, 'EURUSD', 'LONG', '2026-07-03T10:00:00Z', +1.5)", (f"s{i}",))
    con.close()
    result = check_probation("EURUSD", db_path=promo_db,
                             now=pd.Timestamp("2026-07-05", tz="UTC"))
    assert result == "ok"
    assert registry.champion_version("EURUSD", db_path=promo_db) == "m_new"


def test_probation_expires_after_window(promo_db):
    result = check_probation("EURUSD", db_path=promo_db,
                             now=pd.Timestamp("2026-08-01", tz="UTC"))
    assert result is None
