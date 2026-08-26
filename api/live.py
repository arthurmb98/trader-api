from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trader.fnhttp import JsonHandler  # noqa: E402
from trader.http_dispatch import dispatch_live  # noqa: E402


class handler(JsonHandler):
    def do_GET(self) -> None:
        status, payload = dispatch_live("GET", self.op(), query=self.query())
        self.send_json(status, payload)

    def do_POST(self) -> None:
        status, payload = dispatch_live("POST", self.op(), self.read_json(), self.query())
        self.send_json(status, payload)
