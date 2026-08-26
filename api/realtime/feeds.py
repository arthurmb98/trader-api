from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trader.fnhttp import JsonHandler  # noqa: E402
from trader.realtime import cloud_feeds  # noqa: E402


class handler(JsonHandler):
    def do_GET(self) -> None:
        try:
            self.send_json(200, cloud_feeds())
        except Exception as exc:  # noqa: BLE001
            self.send_json(400, {"detail": str(exc)})
