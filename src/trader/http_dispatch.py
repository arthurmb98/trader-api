from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse


def op_from_request(path: str, query: dict[str, str] | None = None) -> str:
    """Action from ?op= or the last /api/<group>/<op> segment."""
    if query and query.get("op"):
        return str(query["op"]).strip().lower()
    parsed = urlparse(path)
    extra = parse_qs(parsed.query)
    if extra.get("op"):
        return str(extra["op"][-1]).strip().lower()
    parts = [item for item in parsed.path.strip("/").split("/") if item]
    if len(parts) >= 3:
        return parts[-1].lower()
    return ""


def dispatch_realtime(method: str, op: str, _body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    from trader.realtime import cloud_feeds, cloud_snapshot

    verb = method.upper()
    key = (op or "").strip().lower()
    try:
        if verb == "GET" and key in {"", "status"}:
            return 200, cloud_snapshot()
        if verb == "GET" and key == "feeds":
            return 200, cloud_feeds()
        if verb == "POST" and key in {"", "start", "source", "candles"}:
            return 200, cloud_snapshot(arm=True)
        if verb == "POST" and key == "stop":
            return 200, cloud_snapshot(pause=True)
        if verb == "POST" and key == "reset":
            return 200, cloud_snapshot(reset=True)
    except Exception as exc:  # noqa: BLE001
        return 400, {"detail": str(exc)}
    return 404, {"detail": "rota ao vivo desconhecida"}


def dispatch_live(
    method: str,
    op: str,
    body: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    from trader.live import empty_live_snapshot, live_meta, paper_batch_snapshot

    verb = method.upper()
    key = (op or "").strip().lower()
    payload = body or {}
    qs = query or {}
    try:
        if verb == "GET" and key in {"", "index"}:
            return 200, empty_live_snapshot()
        if verb == "GET" and key == "meta":
            return 200, live_meta(qs.get("timeframe") or "m5")
        if verb == "POST" and key == "start":
            snap = paper_batch_snapshot(
                case=str(payload.get("case") or "last_candles"),
                timeframe=str(payload.get("timeframe") or "m5"),
                initial_bank=float(payload.get("initial_bank") or 1000),
                start=payload.get("start"),
                end=payload.get("end"),
                interval_sec=float(payload.get("interval_sec") or 0.001),
                lot=str(payload.get("lot") or "fixed"),
            )
            return 200, snap
        if verb == "POST" and key == "stop":
            snap = empty_live_snapshot()
            snap["error"] = None
            return 200, snap
        if verb == "POST" and key == "reset":
            return 200, empty_live_snapshot()
    except Exception as exc:  # noqa: BLE001
        return 400, {"detail": str(exc)}
    return 404, {"detail": "rota replay desconhecida"}
