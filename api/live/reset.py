from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trader.fnhttp import JsonHandler  # noqa: E402
from trader.live import empty_live_snapshot  # noqa: E402


class handler(JsonHandler):
    def do_POST(self) -> None:
        self.send_json(200, empty_live_snapshot())
