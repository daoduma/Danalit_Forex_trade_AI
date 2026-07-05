"""Run the Telegram control bot as a sidecar process (see telegram_bot.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danalit.monitor.telegram_bot import run_bot  # noqa: E402

if __name__ == "__main__":
    run_bot()
