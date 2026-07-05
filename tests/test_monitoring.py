"""Prompt 17: notifier batching/levels, command auth, status/digest assembly."""

import pandas as pd
import pytest

from danalit.db import connect, init_db
from danalit.monitor.notifier import TelegramNotifier
from danalit.monitor.telegram_bot import (
    build_digest,
    build_positions,
    build_status,
    cmd_halt,
    cmd_resume,
    is_authorized,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture()
def notifier():
    sent = []
    clock = FakeClock()
    n = TelegramNotifier(token="t", chat_id="c", transport=sent.append,
                         clock=clock, batch_seconds=60)
    n._sent, n._clock_ref = sent, clock
    return n


def test_info_messages_are_batched(notifier):
    sent, clock = notifier._sent, notifier._clock_ref
    clock.t = 100.0
    notifier.notify("INFO", "trade opened", "EURUSD long")   # first flush window
    assert len(sent) == 1
    notifier.notify("INFO", "sl moved")
    notifier.notify("INFO", "partial closed")
    assert len(sent) == 1                                    # buffered
    clock.t = 161.0
    notifier.notify("INFO", "trade closed", "+2.31")
    assert len(sent) == 2                                    # window elapsed -> flush
    assert "sl moved" in sent[1] and "trade closed" in sent[1]


def test_critical_is_immediate_and_flushes_buffer(notifier):
    sent, clock = notifier._sent, notifier._clock_ref
    clock.t = 100.0
    notifier.notify("INFO", "a")
    notifier.notify("INFO", "b")          # buffered (window just flushed 'a')
    notifier.notify("CRITICAL", "drawdown breaker")
    assert any("drawdown breaker" in s and "🚨" in s for s in sent)
    idx_crit = next(i for i, s in enumerate(sent) if "drawdown" in s)
    assert any("b" in s for s in sent[:idx_crit])  # buffer flushed before critical


def test_unconfigured_notifier_is_a_noop():
    n = TelegramNotifier(token=None, chat_id=None, transport=lambda t: 1 / 0)
    n.notify("CRITICAL", "should not raise")  # transport never called


def test_command_auth_whitelist():
    assert is_authorized("12345", allowed="12345")
    assert not is_authorized("99999", allowed="12345")
    assert not is_authorized("12345", allowed="")  # unset -> reject everyone


def test_halt_resume_files(tmp_path):
    msg = cmd_halt(tmp_path, flat=False)
    assert (tmp_path / "HALT").exists() and "HALT" in msg
    cmd_halt(tmp_path, flat=True)
    assert (tmp_path / "HALT_FLAT").exists()
    msg = cmd_resume(tmp_path)
    assert not (tmp_path / "HALT").exists()
    assert not (tmp_path / "HALT_FLAT").exists()
    assert (tmp_path / "RESUME").exists() and "re-reconcile" in msg


@pytest.fixture()
def journal_db(tmp_path):
    db = tmp_path / "j.db"
    init_db(db)
    con = connect(db)
    with con:
        con.execute("INSERT INTO system_events (ts_utc, type, detail) VALUES"
                    " ('2026-07-05T09:00:00Z','state_transition','RECONCILING -> TRADING')")
        con.execute("INSERT INTO equity_snapshots (ts_utc, balance, equity, margin,"
                    " open_risk, mode) VALUES ('2026-07-05T09:00:00Z',2000,2010,22,14,'demo')")
        con.execute("INSERT INTO orders (client_id, signal_id, ts_utc, instrument, side,"
                    " lots, sl, tp, status, filled_price) VALUES ('s1','s1',"
                    " '2026-07-05T08:00:00Z','EURUSD','LONG',0.07,1.095,1.105,'filled',1.1001)")
    con.close()
    return db


def test_status_text_assembly(journal_db, tmp_path):
    text = build_status(journal_db, kill_dir=tmp_path,
                        orchestrator_heartbeat=tmp_path / "none.hb")
    assert "RECONCILING -> TRADING" in text
    assert "equity: 2010.00" in text
    assert "kill switch: none" in text
    (tmp_path / "HALT").write_text("x", encoding="utf-8")
    text = build_status(journal_db, kill_dir=tmp_path,
                        orchestrator_heartbeat=tmp_path / "none.hb")
    assert "HALT" in text


def test_positions_and_digest_from_fixtures(journal_db):
    pos = build_positions(journal_db)
    assert "EURUSD LONG 0.07" in pos and "SL 1.095" in pos
    digest = build_digest(journal_db, date="2026-07-05")
    assert "daily digest" in digest and "equity 2010.00" in digest
    assert "collector heartbeat" in digest
