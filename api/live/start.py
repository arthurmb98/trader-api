from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trader.fnhttp import JsonHandler  # noqa: E402
from trader.live import paper_batch_snapshot  # noqa: E402


class handler(JsonHandler):
    def do_POST(self) -> None:
        body = self.read_json()
        try:
            snap = paper_batch_snapshot(
                case=str(body.get("case") or "last_candles"),
                timeframe=str(body.get("timeframe") or "m5"),
                initial_bank=float(body.get("initial_bank") or 1000),
                start=body.get("start"),
                end=body.get("end"),
                interval_sec=float(body.get("interval_sec") or 0.001),
            )
            self.send_json(200, snap)
        except Exception as exc:  # noqa: BLE001
            self.send_json(400, {"detail": str(exc)})
