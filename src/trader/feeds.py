from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from trader.domain import Candle
from trader.mt5_session import _load_dotenv
from trader.paths import DATASETS_DIR, ROOT
from trader.ports import MarketFeed

DEFAULT_FILE = DATASETS_DIR / "win_stream.json"
FALLBACK_FILES = (
    DATASETS_DIR / "win_stream.json",
    DATASETS_DIR / "mt5_m5_week.csv",
    DATASETS_DIR / "WIN_5min_test.csv",
)


def _ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    else:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def parse_candle_row(row: dict[str, Any], symbol: str = "WIN$") -> Candle:
    lower = {str(key).lower(): value for key, value in row.items()}
    stamp = lower.get("t") or lower.get("timestamp") or lower.get("time") or lower.get("datetime")
    if stamp is None:
        raise ValueError("Candle sem timestamp (t).")
    high = lower.get("high")
    if high is None:
        high = lower.get("maximo", lower.get("máximo"))
    low = lower.get("low")
    if low is None:
        low = lower.get("minimo", lower.get("mínimo"))
    close = lower.get("close")
    if close is None:
        close = lower.get("fechamento")
    open_ = lower.get("open")
    if open_ is None:
        open_ = lower.get("abertura")
    return Candle(
        symbol=str(lower.get("symbol") or lower.get("ativo") or symbol),
        timestamp=_ts(stamp),
        open=float(open_),
        high=float(high),
        low=float(low),
        close=float(close),
        volume=float(lower.get("volume") or 0),
    )


def parse_stream_payload(raw: Any, symbol: str = "WIN$") -> tuple[str, list[Candle]]:
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(payload, list):
        rows, name = payload, symbol
    elif isinstance(payload, dict):
        name = str(payload.get("symbol") or payload.get("ativo") or symbol)
        rows = payload.get("candles") or payload.get("data") or payload.get("bars") or []
        if isinstance(rows, dict):
            rows = [rows]
        keys = {str(k).lower() for k in payload}
        if not rows and ({"open", "close", "abertura", "fechamento"} & keys):
            rows = [payload]
    else:
        raise ValueError("Payload de stream inválido.")
    candles = [parse_candle_row(row, name) for row in rows]
    candles.sort(key=lambda item: item.timestamp)
    return name, candles


class StreamFeed(MarketFeed):
    """Candles from ingest memory, HTTP URL, or a local JSON/CSV file."""

    def __init__(self) -> None:
        self.symbol = "WIN$"
        self.origin = "none"
        self._ingested: list[Candle] = []
        self._error: str | None = None
        self._detail = "sem candles"
        self._file_cache: list[Candle] | None = None
        self._file_cache_key: tuple[str, float] | None = None

    def ingest(self, raw: Any) -> list[Candle]:
        name, candles = parse_stream_payload(raw, self.symbol)
        if name:
            self.symbol = name
        by_ts = {item.timestamp: item for item in self._ingested}
        for candle in candles:
            by_ts[candle.timestamp] = candle
        self._ingested = sorted(by_ts.values(), key=lambda item: item.timestamp)
        self.origin = "ingest"
        self._detail = f"ingest {len(self._ingested)} candles"
        self._error = None
        return self._ingested

    def clear(self) -> None:
        self._ingested = []
        self.origin = "none"
        self._error = None
        self._detail = "sem candles"

    def _url(self) -> str:
        _load_dotenv()
        return (os.environ.get("WIN_STREAM_URL") or "").strip()

    def _file(self) -> Path | None:
        _load_dotenv()
        raw = (os.environ.get("WIN_STREAM_FILE") or "").strip()
        paths: list[Path] = []
        if raw:
            path = Path(raw)
            if not path.is_absolute():
                path = ROOT / path
            paths.append(path)
        paths.extend(FALLBACK_FILES)
        seen: set[str] = set()
        for path in paths:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                return path
        return None

    def _token(self) -> str:
        _load_dotenv()
        return (os.environ.get("WIN_STREAM_TOKEN") or "").strip()

    def _from_http(self) -> list[Candle] | None:
        url = self._url()
        if not url:
            return None
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        token = self._token()
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = resp.read().decode("utf-8")
        name, candles = parse_stream_payload(body, self.symbol)
        if name:
            self.symbol = name
        self.origin = "http"
        self._detail = f"http {len(candles)} candles"
        return candles

    def _from_file(self) -> list[Candle] | None:
        path = self._file()
        if path is None:
            return None
        mtime = path.stat().st_mtime
        cache_key = (str(path), mtime)
        if self._file_cache is not None and self._file_cache_key == cache_key:
            self.origin = "file"
            self._detail = f"file {path.name} {len(self._file_cache)} candles"
            return self._file_cache
        if path.suffix.lower() == ".csv":
            from trader.data import load_candles

            frame = load_candles(path)
            candles = [
                Candle(
                    symbol=str(row.get("Ativo") or self.symbol),
                    timestamp=_ts(row["timestamp"]),
                    open=float(row["Abertura"]),
                    high=float(row["Máximo"]),
                    low=float(row["Mínimo"]),
                    close=float(row["Fechamento"]),
                    volume=float(row.get("Volume") or 0),
                )
                for _, row in frame.iterrows()
            ]
        else:
            name, candles = parse_stream_payload(path.read_text(encoding="utf-8"), self.symbol)
            if name:
                self.symbol = name
        self._file_cache = candles
        self._file_cache_key = cache_key
        self.origin = "file"
        self._detail = f"file {path.name} {len(candles)} candles"
        return candles

    def last_closed_candles(self, symbol: str, timeframe: str, count: int, *, allow_file: bool = True) -> list[Candle]:
        del timeframe
        self.symbol = symbol or self.symbol
        self._error = None
        self.origin = "none"
        candles: list[Candle] = []
        if self._ingested:
            candles = list(self._ingested)
            self.origin = "ingest"
            self._detail = f"ingest {len(candles)} candles"
        else:
            try:
                http = self._from_http()
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                self._error = str(exc)
                http = None
            if http:
                candles = http
            elif allow_file:
                try:
                    file_rows = self._from_file()
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    self._error = str(exc)
                    file_rows = None
                if file_rows:
                    candles = file_rows
        if not candles:
            self._detail = "sem candles"
            return []
        return candles[-max(count, 1) :]

    def ready(self, *, live_only: bool = False) -> bool:
        if self._ingested:
            return True
        if self._url():
            return True
        if live_only:
            return False
        return self._file() is not None

    def status(self) -> dict[str, Any]:
        path = self._file()
        return {
            "ready": self.ready(),
            "symbol": self.symbol,
            "detail": self._detail,
            "error": self._error,
            "origin": self.origin,
            "ingested": len(self._ingested),
            "url": bool(self._url()),
            "file": path.name if path is not None else None,
        }
