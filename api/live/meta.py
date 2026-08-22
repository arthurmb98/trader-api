from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trader.fnhttp import JsonHandler  # noqa: E402
from trader.live import live_meta  # noqa: E402


class handler(JsonHandler):
    def do_GET(self) -> None:
        timeframe = self.query().get("timeframe") or "m5"
        try:
            self.send_json(200, live_meta(timeframe))
        except Exception as exc:  # noqa: BLE001
            self.send_json(400, {"detail": str(exc)})
