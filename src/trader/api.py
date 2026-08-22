from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trader.broker import Mt5Broker
from trader.config import AppConfig, load_config
from trader.data import load_candles
from trader.domain import Side
from trader.ml import CandleRegressor
from trader.paths import CONFIGS_DIR, RESULTS_DIR, WEB_DIST
from trader.signals import SignalPolicy

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


class CandleIn(BaseModel):
    abertura: float
    maximo: float
    minimo: float
    fechamento: float
    timestamp: str | None = None
    ativo: str = "WIN$"


class SignalRequest(BaseModel):
    config: str = "live"
    timeframe: str = "m1"
    candles: list[CandleIn] | None = None
    send_order: bool = False


class OrderRequest(BaseModel):
    config: str = "live"
    side: str
    entry: float | None = None
    stop: float
    take: float
    volume: float = Field(default=1.0, gt=0)


class LiveStart(BaseModel):
    case: str = "last_candles"
    timeframe: str = "m5"
    initial_bank: float = 1000
    start: str | None = None
    end: str | None = None
    source: str = "paper"
    interval_sec: float = Field(default=0.001, ge=0.0, le=600.0)


def _load_named_config(name: str) -> AppConfig:
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        path = CONFIGS_DIR / name
    if not path.exists():
        raise HTTPException(404, f"Config {name} não encontrada")
    return load_config(path)


def _load_model(timeframe: str) -> CandleRegressor:
    if joblib is None:
        raise HTTPException(500, "joblib não instalado")
    path = RESULTS_DIR / f"model_{timeframe}.joblib"
    if not path.exists():
        raise HTTPException(404, "Modelo ainda não treinado. Rode python -m trader study")
    return joblib.load(path)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        from trader.live import ENGINE

        if ENGINE is not None:
            ENGINE.stop()

    app = FastAPI(
        title="Trader API",
        description="Sinais de day trade no mini índice (WIN) + estudo de eficácia.",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/studies")
    def studies() -> Any:
        path = RESULTS_DIR / "studies.json"
        if not path.exists():
            raise HTTPException(404, "Estudo ainda não gerado. Rode python -m trader study")
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/configs")
    def configs() -> list[str]:
        return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))

    @app.post("/api/signal")
    def signal(body: SignalRequest) -> dict[str, Any]:
        cfg = _load_named_config(body.config)
        model = _load_model(body.timeframe)
        if body.candles:
            import pandas as pd

            rows = []
            for i, c in enumerate(body.candles):
                rows.append(
                    {
                        "Ativo": c.ativo,
                        "Abertura": c.abertura,
                        "Máximo": c.maximo,
                        "Mínimo": c.minimo,
                        "Fechamento": c.fechamento,
                        "timestamp": pd.Timestamp(c.timestamp) if c.timestamp else pd.Timestamp.now(),
                    }
                )
            frame = pd.DataFrame(rows)
        elif cfg.mt5.enabled:
            broker = Mt5Broker(
                cfg.mt5.symbol, cfg.mt5.magic, cfg.mt5.deviation, cfg.mt5.filling, cfg.mt5.comment
            )
            broker.connect()
            try:
                candles = broker.last_closed_candles(cfg.mt5.symbol, body.timeframe, 30)
            finally:
                broker.shutdown()
            import pandas as pd

            frame = pd.DataFrame(
                [
                    {
                        "Ativo": c.symbol,
                        "Abertura": c.open,
                        "Máximo": c.high,
                        "Mínimo": c.low,
                        "Fechamento": c.close,
                        "timestamp": c.timestamp,
                    }
                    for c in candles
                ]
            )
        else:
            # Demo fallback: last train candles never used. Last TEST file is allowed only as
            # a dry illustration and labelled as such — we still do not fit.
            test = load_candles(cfg.resolve_csv(cfg.data.test_csv)).tail(40)
            frame = test
            demo = True
        policy = SignalPolicy(cfg, model)
        sig = policy.from_candles(frame)
        payload = {
            "side": sig.side.value,
            "entry": sig.entry,
            "stop": sig.stop,
            "take": sig.take,
            "reason": sig.reason,
            "predicted": None
            if sig.predicted is None
            else {
                "abertura": sig.predicted.open,
                "maximo": sig.predicted.high,
                "minimo": sig.predicted.low,
                "fechamento": sig.predicted.close,
            },
            "demo_without_mt5": "demo" in locals(),
            "config": cfg.name,
        }
        if body.send_order and sig.side is not Side.FLAT:
            if not cfg.mt5.enabled:
                raise HTTPException(400, "MT5 desligado na config. Ligue mt5.enabled para enviar ordem.")
            order = _send(cfg, sig)
            payload["order"] = order
        return payload

    @app.post("/api/orders")
    def orders(body: OrderRequest) -> dict[str, Any]:
        cfg = _load_named_config(body.config)
        if not cfg.mt5.enabled:
            raise HTTPException(400, "MT5 desligado. Edite configs/live.yaml (mt5.enabled: true).")
        from trader.domain import Signal

        side = Side.BUY if body.side.upper() == "BUY" else Side.SELL
        sig = Signal(side=side, entry=body.entry or 0, stop=body.stop, take=body.take, reason="manual")
        return _send(cfg, sig)

    @app.get("/api/live/meta")
    def live_meta(timeframe: str = "m5") -> dict[str, Any]:
        from trader.live import live_meta as _live_meta

        try:
            return _live_meta(timeframe)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/live")
    def live_status() -> dict[str, Any]:
        from trader.live import get_engine

        return get_engine().snapshot()

    @app.post("/api/live/start")
    async def live_start(body: LiveStart) -> dict[str, Any]:
        from trader.live import get_engine

        try:
            return await get_engine().start(
                case=body.case,
                timeframe=body.timeframe,
                initial_bank=body.initial_bank,
                source=body.source,
                interval_sec=body.interval_sec,
                start=body.start,
                end=body.end,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/live/stop")
    def live_stop() -> dict[str, Any]:
        from trader.live import get_engine

        get_engine().stop()
        return get_engine().snapshot()

    @app.post("/api/live/reset")
    def live_reset() -> dict[str, Any]:
        from trader.live import get_engine

        get_engine().reset()
        return get_engine().snapshot()

    def _send(cfg: AppConfig, sig) -> dict[str, Any]:
        broker = Mt5Broker(
            cfg.mt5.symbol, cfg.mt5.magic, cfg.mt5.deviation, cfg.mt5.filling, cfg.mt5.comment
        )
        broker.connect()
        try:
            return broker.send(sig, float(cfg.account.contracts))
        finally:
            broker.shutdown()

    if WEB_DIST.exists():
        fonts = WEB_DIST / "fonts"
        if fonts.exists():
            app.mount("/fonts", StaticFiles(directory=fonts), name="fonts")
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/studies.json")
        def studies_file() -> FileResponse:
            path = RESULTS_DIR / "studies.json"
            if not path.exists():
                raise HTTPException(404, "Estudo ainda não gerado.")
            return FileResponse(path)

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(WEB_DIST / "index.html")

        @app.get("/live")
        def live_page() -> FileResponse:
            return FileResponse(WEB_DIST / "index.html")

    else:

        @app.get("/")
        def root() -> dict[str, str]:
            return {"message": "Trader API. Rode o front em web/ ou gere studies.json."}

    return app


app = create_app()
