"""Start the local Streamlit service when needed and open Agent Workspace."""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


BASE_URL = "http://127.0.0.1:8501"
ROOT = Path(__file__).resolve().parents[1]
HTTP = build_opener(ProxyHandler({}))


def ready() -> bool:
    try:
        with HTTP.open(f"{BASE_URL}/_stcore/health", timeout=0.5) as response:
            return response.status == 200 and response.read(16).strip() == b"ok"
    except (URLError, TimeoutError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not ready():
        log = ROOT / "artifacts" / "streamlit.log"
        log.parent.mkdir(exist_ok=True)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with log.open("ab") as output:
            child = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                 "--server.address", "127.0.0.1", "--server.port", "8501"], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, creationflags=flags,
            )
        deadline = monotonic() + 20
        while not ready():
            if child.poll() is not None or monotonic() >= deadline:
                print(f"Streamlit did not start. See {log}", file=sys.stderr)
                return 1
            sleep(0.25)
    url = f"{BASE_URL}/workspace"
    print(url)
    if not args.no_browser:
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
