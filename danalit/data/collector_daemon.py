"""Always-on collector: calendar every 30 min, RSS every 5 min, forever.

Per-source error isolation (one dead feed must not stop the others), rotating
logs, and a heartbeat file the trading loop uses as a data-freshness signal.
Run via scripts/run_collector.py; see README for the Windows Task Scheduler
setup so it starts at logon and survives reboots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from danalit.config import load_config
from danalit.data import calendar_ingest, news_ingest
from danalit.db import connect, init_db
from danalit.logging_setup import setup_logging
from danalit.timeutil import parse_iso, utc_now, utc_now_iso

log = setup_logging("collector")

HEARTBEAT_NAME = "collector.heartbeat"


def heartbeat_path(root: Optional[Path] = None) -> Path:
    root = root or load_config().settings.paths.absolute("logs")
    return root / HEARTBEAT_NAME


def write_heartbeat(path: Optional[Path] = None) -> None:
    p = path or heartbeat_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(utc_now_iso(), encoding="utf-8")


def heartbeat_age_seconds(path: Optional[Path] = None) -> Optional[float]:
    """Seconds since last heartbeat; None if no heartbeat exists."""
    p = path or heartbeat_path()
    if not p.exists():
        return None
    ts = parse_iso(p.read_text(encoding="utf-8").strip())
    if ts is None:
        return None
    return (utc_now() - ts.to_pydatetime()).total_seconds()


def calendar_job(db_path=None) -> None:
    con = connect(db_path or load_config().settings.paths.db_path)
    try:
        calendar_ingest.fetch_weekly(con)
        write_heartbeat()
    except Exception as e:
        log.error("calendar job failed: %s", e)
    finally:
        con.close()


def rss_job(db_path=None) -> None:
    con = connect(db_path or load_config().settings.paths.db_path)
    try:
        news_ingest.poll_all(con)
        write_heartbeat()
    except Exception as e:
        log.error("rss job failed: %s", e)
    finally:
        con.close()


def run_forever() -> None:  # pragma: no cover — exercised manually / in ops
    from apscheduler.schedulers.blocking import BlockingScheduler

    cfg = load_config()
    init_db(cfg.settings.paths.db_path)
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(rss_job, "interval", minutes=cfg.settings.news.poll_minutes,
                  next_run_time=utc_now(), max_instances=1, coalesce=True)
    sched.add_job(calendar_job, "interval", minutes=cfg.settings.news.calendar_poll_minutes,
                  next_run_time=utc_now(), max_instances=1, coalesce=True)
    log.info("collector starting: rss every %dm, calendar every %dm",
             cfg.settings.news.poll_minutes, cfg.settings.news.calendar_poll_minutes)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("collector stopped")
