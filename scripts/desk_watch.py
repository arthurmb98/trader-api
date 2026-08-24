"""Keep local API + Vite alive across VPN SIGHUP / shell death."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
VENV_PY = ROOT / ".venv" / "bin" / "python"
NPM = Path.home() / ".nvm" / "versions" / "node" / "v20.20.0" / "bin" / "npm"
API_LOG = Path("/tmp/trader-desk-api.log")
WEB_LOG = Path("/tmp/trader-desk-web.log")


def _daemonize() -> None:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if os.fork() > 0:
        os._exit(0)
    os.chdir("/")
    fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(fd, 0)


def _listening(port: int) -> bool:
    return (
        subprocess.call(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def _spawn(cmd: list[str], cwd: Path, log: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["PATH"] = f"{NPM.parent}:{ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(ROOT / "src")
    out = log.open("a", encoding="utf-8")
    out.write(f"\n--- spawn {' '.join(cmd)} ---\n")
    out.flush()
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=out,
        stderr=out,
        start_new_session=True,
    )


def main() -> None:
    _daemonize()
    api: subprocess.Popen | None = None
    web: subprocess.Popen | None = None
    while True:
        if not _listening(8000) and (api is None or api.poll() is not None):
            api = _spawn(
                [str(VENV_PY), "-m", "trader", "serve", "--host", "127.0.0.1", "--port", "8000"],
                ROOT,
                API_LOG,
            )
        if not _listening(5173) and (web is None or web.poll() is not None):
            web = _spawn([str(NPM), "run", "dev"], WEB, WEB_LOG)
        time.sleep(3)


if __name__ == "__main__":
    main()
