"""Launch the Streamlit dashboard on localhost."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app = Path(__file__).resolve().parents[1] / "danalit" / "monitor" / "dashboard.py"
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app),
         "--server.address", "localhost", "--server.headless", "true"]))
