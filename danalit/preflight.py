"""Pre-flight checks: `python -m danalit.preflight`

Config validation, DB integrity, disk space, clock sync (vs broker when a
terminal is reachable), model-registry sanity, pending migrations, writable
heartbeat paths. Prints a PASS/FAIL table; the orchestrator refuses the
TRADING state if preflight fails (wired in scripts/run_trading.py).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

from danalit.db import SCHEMA_VERSION, connect
from danalit.timeutil import utc_now

CLOCK_WARN_S = 5.0
CLOCK_HALT_S = 60.0
MIN_DISK_GB = 2.0


def check_clock_skew(broker_epoch: float, local_epoch: Optional[float] = None,
                     server_offset_s: float = 0.0) -> tuple[str, float]:
    """Returns ('PASS'|'WARN'|'FAIL', skew_seconds) after removing the known
    server timezone offset."""
    local_epoch = local_epoch if local_epoch is not None else time.time()
    skew = abs((broker_epoch - server_offset_s) - local_epoch)
    if skew >= CLOCK_HALT_S:
        return "FAIL", skew
    if skew >= CLOCK_WARN_S:
        return "WARN", skew
    return "PASS", skew


def run_preflight(gateway=None, db_path: Optional[Path] = None,
                  cfg=None) -> tuple[bool, list[dict]]:
    rows: list[dict] = []

    def add(name: str, ok: bool, detail: str = "", warn: bool = False):
        rows.append({"check": name, "status": "WARN" if (ok and warn) else
                     ("PASS" if ok else "FAIL"), "detail": detail})

    # 1 — config loads and validates
    try:
        from danalit.config import load_config

        cfg = cfg or load_config()
        add("config", True, f"{len(cfg.instruments)} instruments, "
            f"dry_run={cfg.settings.trading.dry_run}")
    except Exception as e:
        add("config", False, str(e))
        return False, rows  # nothing else is meaningful

    db = db_path or cfg.settings.paths.db_path

    # 2 — DB integrity + migrations
    try:
        con = connect(db)
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        add("db integrity", integrity == "ok", integrity)
        row = con.execute("SELECT version FROM schema_version").fetchone()
        version = row["version"] if row else None
        add("db migrations", version == SCHEMA_VERSION,
            f"schema v{version} (code expects v{SCHEMA_VERSION})")
        con.close()
    except Exception as e:
        add("db", False, str(e))

    # 3 — disk space
    free_gb = shutil.disk_usage(str(db.parent)).free / 1e9
    add("disk space", free_gb >= MIN_DISK_GB, f"{free_gb:.1f} GB free")

    # 4 — model registry: a champion for every enabled instrument
    try:
        from danalit.models import registry

        missing = [n for n in cfg.enabled_instruments()
                   if registry.champion_version(n, db_path=db) is None]
        add("model registry", not missing,
            "champions set" if not missing else f"no champion for {missing}")
    except Exception as e:
        add("model registry", False, str(e))

    # 5 — heartbeat paths writable
    try:
        hb = cfg.settings.paths.absolute("logs") / ".preflight_probe"
        hb.parent.mkdir(parents=True, exist_ok=True)
        hb.write_text("ok", encoding="utf-8")
        hb.unlink()
        add("logs writable", True)
    except Exception as e:
        add("logs writable", False, str(e))

    # 6 — clock sync vs broker (only when a gateway is provided and connected)
    if gateway is not None:
        try:
            gateway.ensure_connected()
            tick = gateway.mt5.symbol_info_tick(
                cfg.instruments[cfg.enabled_instruments()[0]].broker_symbol)
            if tick is not None:
                status, skew = check_clock_skew(float(tick.time),
                                                utc_now().timestamp())
                add("clock sync", status != "FAIL",
                    f"apparent skew {skew:.1f}s (incl. server offset — verify once)",
                    warn=status == "WARN")
            else:
                add("clock sync", True, "no tick (market closed?) — skipped", warn=True)
        except Exception as e:
            add("clock sync", False, str(e))

    passed = all(r["status"] != "FAIL" for r in rows)
    return passed, rows


def print_table(rows: list[dict]) -> None:
    width = max(len(r["check"]) for r in rows) + 2
    for r in rows:
        print(f"  {r['check']:<{width}} [{r['status']:^4}] {r['detail']}")


def main() -> int:
    passed, rows = run_preflight()
    print("Danalit preflight:")
    print_table(rows)
    print("\nRESULT:", "PASS — clear to trade" if passed
          else "FAIL — the orchestrator will refuse TRADING")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
