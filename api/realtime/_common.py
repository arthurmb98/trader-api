from __future__ import annotations

from typing import Any


def send_cloud(handler: Any, **opts: Any) -> None:
    from trader.realtime import cloud_snapshot

    try:
        handler.send_json(200, cloud_snapshot(**opts))
    except Exception as exc:  # noqa: BLE001
        handler.send_json(400, {"detail": str(exc)})
