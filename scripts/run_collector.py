"""Run the 24/7 news + calendar collector. Start this the week you begin building —
the archive it accumulates becomes the project's proprietary training data.

Windows Task Scheduler setup (run once, as your user):
  schtasks /Create /TN "Danalit Collector" /SC ONLOGON /TR "py -m scripts.run_collector" /RL LIMITED
(or use scripts/deploy/install_tasks.ps1 from Prompt 20, which sets everything up.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danalit.data.collector_daemon import run_forever  # noqa: E402

if __name__ == "__main__":
    run_forever()
