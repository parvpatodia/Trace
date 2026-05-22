"""
Trace — startup script.

Usage:
    python run.py

Environment:
    Copy .env.example to .env and fill in your values before running.

    Minimum required:
        ANTHROPIC_API_KEY=sk-ant-...
        BROWSER_HISTORY_PATH=/path/to/BrowserHistory.json

    Or upload a file via the web UI at http://localhost:8000/
"""
import logging
import sys
from pathlib import Path


def _check_env() -> bool:
    """Warn loudly if critical config is missing, but don't crash — let FastAPI handle it."""
    import os

    ok = True
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "\n[TRACE] WARNING: ANTHROPIC_API_KEY is not set.\n"
            "  Set it in .env or export it before running:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n",
            file=sys.stderr,
        )
        ok = False

    history_path = os.getenv("BROWSER_HISTORY_PATH", "./BrowserHistory.json")
    if not Path(history_path).exists():
        print(
            f"\n[TRACE] INFO: BrowserHistory.json not found at {history_path}.\n"
            "  You can upload your Google Takeout history file via the web UI.\n"
            "  Export guide: takeout.google.com → Chrome → export → extract ZIP\n",
            file=sys.stderr,
        )

    return ok


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    _check_env()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    import os
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    print(f"\n  Trace is running at http://{host}:{port}/\n")

    uvicorn.run(
        "trace.delivery.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
