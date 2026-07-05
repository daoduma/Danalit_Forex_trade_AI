"""Watchdog (Task Scheduler runs this every 5 minutes).

Checks orchestrator + collector heartbeat files; restarts the corresponding
scheduled task on staleness — with restart-storm protection: more than 3
restarts of the same task within an hour stops restarting and notifies
CRITICAL instead (something structural is wrong; a human must look).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from danalit.config import load_config  # noqa: E402
from danalit.data.collector_daemon import heartbeat_age_seconds, heartbeat_path  # noqa: E402
from danalit.monitor.notifier import make_notifier  # noqa: E402

STALE_SECONDS = {"Danalit Orchestrator": 300, "Danalit Collector": 1200}
MAX_RESTARTS_PER_HOUR = 3
STATE_FILE = Path(__file__).with_name("watchdog_state.json")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def restart_allowed(state: dict, task: str, now: float) -> bool:
    history = [t for t in state.get(task, []) if now - t < 3600]
    state[task] = history
    return len(history) < MAX_RESTARTS_PER_HOUR


def restart_task(task: str) -> None:
    subprocess.run(["schtasks", "/End", "/TN", task], capture_output=True)
    subprocess.run(["schtasks", "/Run", "/TN", task], capture_output=True)


def main() -> int:
    cfg = load_config()
    notifier = make_notifier()
    state = load_state()
    now = time.time()

    orch_hb = cfg.settings.paths.absolute("logs") / "orchestrator.heartbeat"
    ages = {
        "Danalit Orchestrator": _age(orch_hb),
        "Danalit Collector": heartbeat_age_seconds(heartbeat_path()),
    }
    for task, age in ages.items():
        limit = STALE_SECONDS[task]
        if age is None or age > limit:
            if restart_allowed(state, task, now):
                state.setdefault(task, []).append(now)
                restart_task(task)
                notifier.notify("WARNING", f"Watchdog restarted {task}",
                                f"heartbeat age {age if age is not None else 'missing'}s "
                                f"(limit {limit}s)")
            else:
                notifier.notify("CRITICAL", f"{task} restart storm",
                                f">{MAX_RESTARTS_PER_HOUR} restarts/hour — watchdog "
                                "stopped restarting; manual intervention required")
    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    return 0


def _age(path: Path):
    from danalit.data.collector_daemon import heartbeat_age_seconds as hb_age

    return hb_age(path)


if __name__ == "__main__":
    raise SystemExit(main())
