from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trader.fnhttp import JsonHandler  # noqa: E402

from _common import send_cloud  # noqa: E402


class handler(JsonHandler):
    def do_GET(self) -> None:
        send_cloud(self)
