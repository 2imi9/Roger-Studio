"""Dev launcher for the FastAPI backend.

Runs uvicorn with ``reload=True`` so code edits hot-reload. Two quirks this
script guards against, both specific to Windows dev:

1. **Zombie multiprocessing children.** uvicorn's reload mode spawns a
   subprocess per worker. When the parent is killed ungracefully (e.g. a
   wrapper tool's stop command doesn't walk the process tree), the child
   keeps the port bound for several minutes — subsequent `python run.py`
   calls appear to succeed but the HTTP server you hit is the *old* one.
   That's how we ended up shipping a backend with the old inference
   pipeline for an entire debugging session.

   Guard: before binding, probe port 8000. If something else is holding it,
   try to hunt down + kill any `python.exe` whose command line references
   this backend dir, then bail out loudly rather than silently piling
   another zombie on top.

2. **Stale .pyc caches.** Python bytecode compiled against an older source
   tree can shadow live source if __pycache__ dirs aren't cleared after a
   revert. Nuke them on every launch — cheap, deterministic.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import uvicorn

BACKEND_DIR = Path(__file__).resolve().parent
PORT = 8000


def _port_in_use(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        return True
    finally:
        s.close()
    return False


def _kill_backend_pythons() -> int:
    """Find any `python.exe` child whose command line references this
    backend dir and stop it. Returns the number killed."""
    if os.name != "nt":
        return 0
    # Use WMI via PowerShell — no extra Python deps needed.
    cmd = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -match 'run\\.py|uvicorn' -and "
        f"$_.CommandLine -match '{BACKEND_DIR.name}' }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force "
        "-ErrorAction SilentlyContinue; $_.ProcessId }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, timeout=10,
        )
        killed = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        return len(killed)
    except Exception:
        return 0


def _nuke_pycache() -> None:
    for pc in BACKEND_DIR.rglob("__pycache__"):
        shutil.rmtree(pc, ignore_errors=True)


if __name__ == "__main__":
    if _port_in_use(PORT):
        killed = _kill_backend_pythons()
        if killed:
            print(f"[run.py] Killed {killed} orphan backend process(es) holding port {PORT}.")
            # Socket release on Windows can lag even after the process dies.
            import time
            for _ in range(20):
                time.sleep(0.5)
                if not _port_in_use(PORT):
                    break
        if _port_in_use(PORT):
            print(
                f"[run.py] Port {PORT} is still in use and nothing we killed "
                f"owned it. Something else (another backend? another app?) is "
                f"holding it. Close that process manually, then retry.",
                file=sys.stderr,
            )
            sys.exit(1)

    _nuke_pycache()

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=PORT,
        reload=True,
        reload_dirs=[str(BACKEND_DIR)],
    )
