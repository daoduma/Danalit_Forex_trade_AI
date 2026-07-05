"""Prompt 19: tier hysteresis, set-aside math, HWM pause, working equity, withdrawals."""

import pytest

from danalit.config import load_config
from danalit.db import init_db
from danalit.risk import capital
from danalit.risk.capital import TierManager, month_end_close, record_withdrawal
from danalit.risk.risk_manager import RiskManager


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "cap.db"
    init_db(p)
    return p


class SpyNotifier:
    def __init__(self):
        self.sent = []

    def notify(self, level, title, body=""):
        self.sent.append((level, title, body))


def test_tier_hysteresis_requires_n_consecutive_days(db):
    spy = SpyNotifier()
    tm = TierManager(db_path=db, notifier=spy)
    assert tm.tier_index == 0
    # 4 days above the $50 boundary (cents: 5000+) — not enough (N=5)
    for i, day in enumerate(["07-06", "07-07", "07-08", "07-09"]):
        assert tm.daily_update(6000, f"2026-{day}") is None
    assert tm.tier_index == 0 and tm.streak == 4
    # 5th consecutive day -> change applied at rollover + notified
    assert tm.daily_update(6000, "2026-07-10") == 1
    assert tm.tier_index == 1
    assert any("tier change" in t for _, t, _ in spy.sent)


def test_tier_flapping_resets_streak(db):
    tm = TierManager(db_path=db)
    tm.daily_update(6000, "2026-07-06")
    tm.daily_update(6000, "2026-07-07")
    tm.daily_update(4000, "2026-07-08")   # dips back under $50: streak resets
    tm.daily_update(6000, "2026-07-09")
    assert tm.tier_index == 0 and tm.streak == 1


def test_tier_update_idempotent_within_day_and_persistent(db):
    tm = TierManager(db_path=db)
    for _ in range(10):  # same day repeated: only one streak increment
        tm.daily_update(6000, "2026-07-06")
    assert tm.streak == 1
    tm2 = TierManager(db_path=db)  # restart: state persisted
    assert tm2.streak == 1 and tm2.candidate == 1


def test_month_end_set_aside_math(db):
    # tier 2 (10%): profitable month at/above HWM credits 10% of net
    r = month_end_close("2026-07", net_pnl=40.0, equity=2100, hwm=2100,
                        set_aside_pct=0.10, db_path=db)
    assert r["credit"] == pytest.approx(4.0)
    assert capital.set_aside_balance(db) == pytest.approx(4.0)
    # losing month: no credit, balance unchanged
    r = month_end_close("2026-08", net_pnl=-30.0, equity=2070, hwm=2100,
                        set_aside_pct=0.10, db_path=db)
    assert r["credit"] == 0.0
    assert capital.set_aside_balance(db) == pytest.approx(4.0)


def test_hwm_pause_rule(db):
    # profitable month but equity below HWM: recover first, skim later
    r = month_end_close("2026-07", net_pnl=50.0, equity=1900, hwm=2000,
                        set_aside_pct=0.25, db_path=db)
    assert r["paused"] and r["credit"] == 0.0


def test_withdrawal_recommendation_and_ledger_consistency(db):
    # account_units=cents: min_withdrawal $25 = 2500 cents
    month_end_close("2026-07", net_pnl=30000.0, equity=40000, hwm=40000,
                    set_aside_pct=0.10, db_path=db)
    assert capital.set_aside_balance(db) == pytest.approx(3000.0)
    r = month_end_close("2026-08", net_pnl=0.0, equity=40000, hwm=40000,
                        set_aside_pct=0.10, db_path=db)
    assert r["withdrawal_recommended"]  # 3000c = $30 >= $25

    new_balance = record_withdrawal(2500.0, db_path=db)
    assert new_balance == pytest.approx(500.0)
    with pytest.raises(ValueError, match="exceeds"):
        record_withdrawal(10_000.0, db_path=db)


def test_risk_manager_sizes_from_working_equity(db):
    """Provably: reserved profits are excluded from the sizing basis."""
    rm = RiskManager(cfg=load_config(), db_path=db)
    full = rm.check_order("EURUSD", 1, 1.10, 0.0020, 2000, [])
    assert full.ok and full.sizing.lots == pytest.approx(0.07)
    # reserve 1000c: working equity 1000 -> budget 7.5c -> 0.03 lots
    month_end_close("2026-07", net_pnl=10000.0, equity=2000, hwm=2000,
                    set_aside_pct=0.10, db_path=db)
    assert capital.set_aside_balance(db) == pytest.approx(1000.0)
    reduced = rm.check_order("EURUSD", 1, 1.10, 0.0020, 2000, [])
    assert reduced.ok and reduced.sizing.lots == pytest.approx(0.03)


def test_months_to_next_tier_and_projections():
    cfg = load_config()
    # equity $30 (3000c), 5%/mo -> log(50/30)/log(1.05) ~ 10.5 months
    m = capital.months_to_next_tier(3000, 0.05, cfg)
    assert m == pytest.approx(10.5, abs=0.2)
    assert capital.months_to_next_tier(3000, 0.0, cfg) is None
    rows = capital.projection_table(2000.0)
    assert rows[-1]["months"] == 12
    assert rows[-1]["optimistic"] > rows[-1]["conservative"] > 2000


def test_monthly_report_renders(db, tmp_path):
    month_end_close("2026-07", net_pnl=40.0, equity=2100, hwm=2100,
                    set_aside_pct=0.10, db_path=db)
    path = capital.monthly_report(2100, 2100, db_path=db, out_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "working equity" in text and "projections, not promises" in text
