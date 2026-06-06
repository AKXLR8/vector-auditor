"""Dev entry point — starts uvicorn bound to 0.0.0.0 on PORT (default 8000).

Single-instance lock via .run/backend.pid — kills any previous instance
of THIS project before starting, so you never get the "port already in use"
error from running `python run.py` twice.

Usage:
    python run.py                # dev mode, auto-reload
    python run.py --no-reload    # prod-like, no reload
    python run.py --port 9000    # custom port
    python run.py --force        # kill any python on the port and start
"""
import argparse
import os
import signal
import socket
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# ── Load .env BEFORE importing the app so os.getenv() sees all keys ──────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=False)

# ── Single-instance lock — never bind 8000 twice ──────────────────────────────
_LOCK_DIR = _PROJECT_ROOT / ".run"
_LOCK_DIR.mkdir(exist_ok=True)
_PID_FILE = _LOCK_DIR / "backend.pid"


def _kill_previous_instance() -> None:
    """Kill any previous instance of THIS project recorded in backend.pid.
    Also kills any python still holding the requested port (orphan)."""
    # 1) PID-file lock: the canonical signal
    if _PID_FILE.exists():
        try:
            old_pid = int(_PID_FILE.read_text().strip())
            if old_pid and old_pid != os.getpid():
                # Windows-compatible: taskkill the whole tree
                import subprocess
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(old_pid), "/T"],
                    capture_output=True,
                )
        except (ValueError, OSError):
            pass
    _PID_FILE.write_text(str(os.getpid()))

    # 2) Defense in depth: if anything else is still on the port, kill it.
    #    We do this AFTER writing our PID so a recursive call doesn't loop.


def _free_port_aggressively(port: int) -> None:
    """Find any process listening on `port` and taskkill it. Windows-only."""
    import subprocess
    try:
        out = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return
    pids = set()
    for line in out.splitlines():
        line = line.strip()
        if f":{port} " not in line or "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                pids.add(int(parts[-1]))
            except ValueError:
                pass
    for pid in pids:
        if pid == os.getpid():
            continue
        subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"], capture_output=True)


def _port_is_free(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        s.close()
        return True
    except OSError:
        s.close()
        return False


# ── Cleanup on exit ───────────────────────────────────────────────────────────
def _cleanup() -> None:
    try:
        if _PID_FILE.exists():
            txt = _PID_FILE.read_text().strip()
            if txt == str(os.getpid()):
                _PID_FILE.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--no-reload", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Kill any process on --port and bind anyway")
    args = parser.parse_args()

    # 1) Kill any previous instance of THIS project
    _kill_previous_instance()

    # 2) Optional: free the port by killing whoever's holding it
    if args.force and not _port_is_free(args.host, args.port):
        print(f"[run.py] --force: killing any process on port {args.port}...")
        _free_port_aggressively(args.port)
        # Give Windows a moment to release the socket
        import time
        for _ in range(10):
            if _port_is_free(args.host, args.port):
                break
            time.sleep(0.3)

    # 3) Refuse early with a helpful message if the port is still in use
    if not _port_is_free(args.host, args.port):
        print(
            f"[run.py] ERROR: port {args.port} is still in use.\n"
            f"         Re-run with --force to kill the holder, or use --port <other>.",
            file=sys.stderr,
        )
        sys.exit(1)

    import atexit
    atexit.register(_cleanup)
    try:
        signal.signal(signal.SIGINT, lambda *_: (_cleanup(), sys.exit(0)))
        signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(0)))
    except (ValueError, OSError):
        pass

    print(f"[run.py] Starting on http://{args.host}:{args.port} (pid {os.getpid()})")
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        reload_dirs=["src", "alembic", "scripts"] if not args.no_reload else None,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )