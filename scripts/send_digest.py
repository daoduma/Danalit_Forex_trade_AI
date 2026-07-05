"""Build + send the daily digest (scheduled 21:00 UTC by Task Scheduler)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danalit.monitor.notifier import make_notifier  # noqa: E402
from danalit.monitor.telegram_bot import build_digest, save_digest  # noqa: E402

if __name__ == "__main__":
    text = build_digest()
    path = save_digest(text)
    make_notifier().notify("WARNING", "Daily digest", text)  # WARNING = immediate
    print(f"digest saved to {path}")
