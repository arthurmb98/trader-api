from __future__ import annotations

import os
import tempfile

os.environ["WIN_DISABLE_YAHOO"] = "1"

from trader.http_dispatch import dispatch_live, dispatch_realtime, op_from_request


def test_op_from_path_and_query() -> None:
    assert op_from_request("/api/realtime/start") == "start"
    assert op_from_request("/api/realtime?op=stop") == "stop"
    assert op_from_request("/api/live/meta", {"op": "meta", "timeframe": "m5"}) == "meta"
    assert op_from_request("/api/realtime") == ""


def test_dispatch_realtime_start_stop_reset() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        path = handle.name
    os.environ["WIN_CLOUD_STATE"] = path
    try:
        status, idle = dispatch_realtime("GET", "")
        assert status == 200
        assert idle["running"] is False
        status, armed = dispatch_realtime("POST", "start")
        assert status == 200
        assert armed["running"] is True
        assert armed["order_mode"] == "paper"
        assert armed["n_trades"] == 0
        status, paused = dispatch_realtime("POST", "stop")
        assert status == 200
        assert paused["running"] is False
        status, cleared = dispatch_realtime("POST", "reset")
        assert status == 200
        assert cleared["n_trades"] == 0
        status, missing = dispatch_realtime("GET", "nope")
        assert status == 404
    finally:
        os.environ.pop("WIN_CLOUD_STATE", None)
        try:
            os.unlink(path)
        except OSError:
            pass


def test_dispatch_live_get_and_pause() -> None:
    status, snap = dispatch_live("GET", "")
    assert status == 200
    assert snap["running"] is False
    status, meta = dispatch_live("GET", "meta", query={"timeframe": "m5"})
    assert status == 200
    assert "min_date" in meta
    status, stopped = dispatch_live("POST", "stop")
    assert status == 200
    status, reset = dispatch_live("POST", "reset")
    assert status == 200
    status, missing = dispatch_live("POST", "nope")
    assert status == 404
