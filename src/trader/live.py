from __future__ import annotations

import asyncio
import calendar
import json
from datetime import date, datetime
from typing import Any

import pandas as pd

from trader.backtest import BacktestEngine, SessionFilter, contracts_for_bank
from trader.broker import Mt5Broker
from trader.config import AppConfig
from trader.data import frame_from_candles, load_candles
from trader.domain import Side, Signal, Trade
from trader.paths import DATASETS_DIR, RESULTS_DIR
from trader.replay import ensure_model, load_named_config
from trader.signals import SignalPolicy, overlay_config

SESSION_PATH = RESULTS_DIR / "live_session.json"
DEFAULT_CONFIG = "best_candles_m5_1000_a"
DEFAULT_CASE = "last_candles"
DEFAULT_TIMEFRAME = "m5"
DEFAULT_BANK = 1000.0
WARMUP_DAYS = 5
LOOKBACK_BARS = 80
DEFAULT_INTERVAL_SEC = 0.001
WALL_DATE = date(2025, 1, 1)
LIVE_BANKS = (500, 1000, 2000, 3000, 5000, 10000, 15000)
LIVE_CASES = ("last_candle", "last_candles")
LIVE_TIMEFRAMES = ("m1", "m5")
CASE_SLUG = {"last_candle": "lc", "last_candles": "candles"}
CASE_LABEL = {"last_candle": "Último candle", "last_candles": "Últimos candles"}
TF_LABEL = {"m1": "1 min", "m5": "5 min"}
_BOUNDS_CACHE: dict[str, tuple[date, date]] = {}


def _iso(value: datetime | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return value.isoformat()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return pd.Timestamp(value).to_pydatetime()


def add_months(day: date, months: int) -> date:
    month = day.month - 1 + months
    year = day.year + month // 12
    month = month % 12 + 1
    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last))


def parse_day(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Data inválida: {value}") from exc


def study_bank_for(initial_bank: float) -> int:
    return 500 if float(initial_bank) < 1000 else 1000


def live_config_name(case: str, timeframe: str, initial_bank: float) -> str:
    slug = CASE_SLUG[case]
    family = study_bank_for(initial_bank)
    return f"best_{slug}_{timeframe}_{family}_a"


def resolve_live_config(case: str, timeframe: str, initial_bank: float) -> tuple[str, AppConfig]:
    name = live_config_name(case, timeframe, initial_bank)
    cfg = load_named_config(name)
    bank = float(initial_bank)
    if abs(float(cfg.account.initial_bank) - bank) > 1e-9:
        cfg = overlay_config(cfg, account__initial_bank=bank)
    return name, cfg


def test_csv_for(timeframe: str):
    key = "1min" if timeframe == "m1" else "5min"
    return DATASETS_DIR / f"WIN_{key}_test.csv"


def dataset_bounds(timeframe: str) -> tuple[date, date]:
    tf = timeframe if timeframe in LIVE_TIMEFRAMES else DEFAULT_TIMEFRAME
    cached = _BOUNDS_CACHE.get(tf)
    if cached is not None:
        return cached
    path = test_csv_for(tf)
    if not path.exists():
        raise FileNotFoundError(f"CSV de teste não encontrado: {path.name}")
    frame = load_candles(path)
    if frame.empty:
        raise RuntimeError(f"CSV de teste vazio: {path.name}")
    ts = pd.to_datetime(frame["timestamp"])
    first = ts.min().date()
    last = ts.max().date()
    bounds = (max(WALL_DATE, first), last)
    _BOUNDS_CACHE[tf] = bounds
    return bounds


def default_window(min_date: date, max_date: date) -> tuple[date, date]:
    start = add_months(max_date, -1)
    if start < min_date:
        start = min_date
    return start, max_date


def validate_live_params(
    case: str,
    timeframe: str,
    initial_bank: float,
    source: str,
    start: date | None,
    end: date | None,
) -> tuple[str, str, float, date | None, date | None]:
    if case not in LIVE_CASES:
        raise ValueError("Caso deve ser last_candle ou last_candles")
    if timeframe not in LIVE_TIMEFRAMES:
        raise ValueError("Timeframe deve ser m1 ou m5")
    bank = float(initial_bank)
    if int(bank) not in LIVE_BANKS:
        allowed = ", ".join(str(v) for v in LIVE_BANKS)
        raise ValueError(f"Banca deve ser uma de: {allowed}")
    if source not in {"paper", "mt5"}:
        raise ValueError("source deve ser paper ou mt5")
    window_start, window_end = start, end
    if source == "paper":
        min_date, max_date = dataset_bounds(timeframe)
        if window_start is None or window_end is None:
            window_start, window_end = default_window(min_date, max_date)
        if window_end < window_start:
            raise ValueError("Data final deve ser maior ou igual à data inicial")
        if window_start < min_date:
            raise ValueError(f"Data inicial mínima: {min_date.isoformat()}")
        if window_end > max_date:
            raise ValueError(f"Data final máxima: {max_date.isoformat()}")
        if window_end > add_months(window_start, 3):
            raise ValueError("Janela limitada a 3 meses (treino trimestral)")
    return case, timeframe, bank, window_start, window_end


def live_meta(timeframe: str = DEFAULT_TIMEFRAME) -> dict[str, Any]:
    tf = timeframe if timeframe in LIVE_TIMEFRAMES else DEFAULT_TIMEFRAME
    min_date, max_date = dataset_bounds(tf)
    start, end = default_window(min_date, max_date)
    return {
        "banks": list(LIVE_BANKS),
        "cases": [{"key": key, "label": CASE_LABEL[key]} for key in LIVE_CASES],
        "timeframes": [{"key": key, "label": TF_LABEL[key]} for key in LIVE_TIMEFRAMES],
        "timeframe": tf,
        "min_date": min_date.isoformat(),
        "max_date": max_date.isoformat(),
        "default_start": start.isoformat(),
        "default_end": end.isoformat(),
        "max_span_months": 3,
    }


def slice_paper_frame(frame: pd.DataFrame, start: date, end: date) -> tuple[pd.DataFrame, int]:
    ts = pd.to_datetime(frame["timestamp"])
    warmup_from = pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)
    end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
    sliced = frame.loc[(ts >= warmup_from) & (ts < end_exclusive)].copy().reset_index(drop=True)
    if sliced.empty:
        raise RuntimeError("Sem candles na janela escolhida.")
    sliced_ts = pd.to_datetime(sliced["timestamp"])
    trade = sliced_ts.dt.normalize() >= pd.Timestamp(start)
    if not bool(trade.any()):
        raise RuntimeError("Sem candles operáveis na janela escolhida.")
    first_trade = int(trade.to_numpy().nonzero()[0][0])
    cursor = max(0, first_trade - 1)
    if len(sliced) >= 2:
        cursor = min(cursor, len(sliced) - 2)
    return sliced, cursor


def _signal_dict(sig: Signal | None) -> dict[str, Any] | None:
    if sig is None:
        return None
    pred = None
    if sig.predicted is not None:
        pred = {
            "abertura": sig.predicted.open,
            "maximo": sig.predicted.high,
            "minimo": sig.predicted.low,
            "fechamento": sig.predicted.close,
        }
    return {
        "side": sig.side.value,
        "entry": sig.entry,
        "stop": sig.stop,
        "take": sig.take,
        "reason": sig.reason,
        "predicted": pred,
    }


def _signal_from_dict(data: dict[str, Any] | None) -> Signal | None:
    if not data:
        return None
    from trader.domain import PredictedCandle

    pred = None
    raw = data.get("predicted")
    if raw:
        pred = PredictedCandle(
            open=float(raw["abertura"]),
            high=float(raw["maximo"]),
            low=float(raw["minimo"]),
            close=float(raw["fechamento"]),
        )
    return Signal(
        side=Side(data["side"]),
        entry=float(data.get("entry") or 0),
        stop=float(data.get("stop") or 0),
        take=float(data.get("take") or 0),
        reason=str(data.get("reason") or ""),
        predicted=pred,
    )


def _position_dict(position: dict | None) -> dict[str, Any] | None:
    if not position:
        return None
    side = position["side"]
    return {
        "side": side.value if isinstance(side, Side) else str(side),
        "entry": float(position["entry"]),
        "stop": float(position["stop"]),
        "take": float(position["take"]),
        "time": _iso(position["time"]),
        "hour": int(position["hour"]),
        "extreme": float(position["extreme"]),
        "contracts": int(position.get("contracts") or 1),
        "reason": str(position.get("reason") or ""),
    }


def _position_from_dict(data: dict[str, Any] | None) -> dict | None:
    if not data:
        return None
    ts = _parse_ts(data.get("time"))
    return {
        "side": Side(data["side"]),
        "entry": float(data["entry"]),
        "stop": float(data["stop"]),
        "take": float(data["take"]),
        "time": ts or datetime.now(),
        "hour": int(data.get("hour") or 0),
        "extreme": float(data.get("extreme") or data["entry"]),
        "contracts": int(data.get("contracts") or 1),
        "reason": str(data.get("reason") or ""),
    }


def window_calendar_days(start: date | None, end: date | None) -> int:
    if start is None or end is None:
        return 0
    return max(0, (end - start).days + 1)


def period_levels_for(n_days: int) -> list[str]:
    levels = ["daily"]
    if n_days >= 7:
        levels.append("weekly")
    if n_days >= 30:
        levels.append("monthly")
    if n_days >= 90:
        levels.append("quarterly")
    return levels


def _trade_exit(trade: dict[str, Any]) -> datetime | None:
    raw = str(trade.get("exit_time") or "")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def _period_key(trade: dict[str, Any], kind: str) -> str | None:
    ts = _trade_exit(trade)
    if ts is None:
        return None
    if kind == "daily":
        return ts.date().isoformat()
    if kind == "weekly":
        iso = ts.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if kind == "monthly":
        return ts.strftime("%Y-%m")
    quarter = (ts.month - 1) // 3 + 1
    return f"{ts.year}-Q{quarter}"


def _bucket_pnl(trades: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    buckets: dict[str, float] = {}
    for trade in trades:
        key = _period_key(trade, kind)
        if not key:
            continue
        buckets[key] = round(buckets.get(key, 0.0) + float(trade.get("pnl") or 0), 2)
    return [{"t": key, "pnl": value} for key, value in sorted(buckets.items())]


def _gain_loss_avg(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    pnls = [float(row["pnl"]) for row in rows]
    gains = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    return {
        "avg": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
        "avg_gain": round(sum(gains) / len(gains), 2) if gains else 0.0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "n": len(rows),
        "n_gain": len(gains),
        "n_loss": len(losses),
    }


def live_period_stats(
    trades: list[dict[str, Any]],
    start: date | None,
    end: date | None,
) -> dict[str, Any]:
    n_days = window_calendar_days(start, end)
    levels = period_levels_for(n_days)
    series = {kind: _bucket_pnl(trades, kind) for kind in ("daily", "weekly", "monthly", "quarterly")}
    return {
        "window_days": n_days,
        "levels": levels,
        "series": {kind: series[kind] for kind in levels},
        "avg": {kind: _gain_loss_avg(series[kind]) for kind in levels},
    }


class LiveEngine:
    """Paper (or MT5-data) walk-forward session. Never sends orders."""

    def __init__(self) -> None:
        self.config_name = DEFAULT_CONFIG
        self.case = DEFAULT_CASE
        self.timeframe = DEFAULT_TIMEFRAME
        self.source = "paper"
        self.interval_sec = DEFAULT_INTERVAL_SEC
        self.window_start: date | None = None
        self.window_end: date | None = None
        self.running = False
        self.done = False
        self.error: str | None = None
        self.cursor = 0
        self.last_tick: str | None = None
        self.last_bar_time: str | None = None
        self.bank = DEFAULT_BANK
        self.initial_bank = DEFAULT_BANK
        self.peak = DEFAULT_BANK
        self.max_dd = 0.0
        self.max_contracts = 1
        self._ticks_since_save = 0
        self.day_key: date | None = None
        self.day_pnl = 0.0
        self.trades_today = 0
        self.position: dict | None = None
        self.last_signal: Signal | None = None
        self.trades: list[dict[str, Any]] = []
        self.equity: list[dict[str, Any]] = []
        self.signals: list[dict[str, Any]] = []
        self.candles_tail: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._frame: pd.DataFrame | None = None
        self._cfg = None
        self._policy: SignalPolicy | None = None
        self._bt: BacktestEngine | None = None
        self._session: SessionFilter | None = None

    def snapshot(self) -> dict[str, Any]:
        wins = [t for t in self.trades if t.get("result") == "win"]
        n = len(self.trades)
        days = {str(t.get("exit_time", ""))[:10] for t in self.trades if t.get("exit_time")}
        n_days = max(len(days), 1) if n else 0
        net = round(self.bank - self.initial_bank, 2)
        today_key = str(self.day_key) if self.day_key else (self.last_bar_time or "")[:10]
        today_closed = sum(
            float(t.get("pnl") or 0)
            for t in self.trades
            if str(t.get("exit_time", ""))[:10] == today_key
        )
        daily: dict[str, float] = {}
        for t in self.trades:
            key = str(t.get("exit_time", ""))[:10]
            if not key:
                continue
            daily[key] = round(daily.get(key, 0.0) + float(t.get("pnl") or 0), 2)
        daily_rows = [{"t": k, "pnl": v} for k, v in sorted(daily.items())]
        periods = live_period_stats(self.trades, self.window_start, self.window_end)
        avg_daily = float((periods.get("avg") or {}).get("daily", {}).get("avg") or 0.0)
        return {
            "running": self.running,
            "done": self.done,
            "error": self.error,
            "config": self.config_name,
            "case": self.case,
            "timeframe": self.timeframe,
            "source": self.source,
            "interval_sec": self.interval_sec,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "start": self.window_start.isoformat() if self.window_start else None,
            "end": self.window_end.isoformat() if self.window_end else None,
            "last_tick": self.last_tick,
            "last_bar_time": self.last_bar_time,
            "cursor": self.cursor,
            "n_bars": 0 if self._frame is None else int(len(self._frame)),
            "initial_bank": round(self.initial_bank, 2),
            "bank": round(self.bank, 2),
            "net_pnl": net,
            "today_pnl": round(today_closed, 2),
            "avg_daily": avg_daily,
            "n_days": n_days,
            "n_trades": n,
            "n_wins": len(wins),
            "win_rate": round(100.0 * len(wins) / n, 1) if n else 0.0,
            "max_drawdown": round(self.max_dd, 2),
            "max_drawdown_pct": round(100.0 * self.max_dd / self.initial_bank, 1) if self.initial_bank else 0.0,
            "contracts": contracts_for_bank(self.bank, self.initial_bank) if self.initial_bank else 1,
            "max_contracts": int(self.max_contracts),
            "signal": _signal_dict(self.last_signal),
            "position": _position_dict(self.position),
            "trades": list(reversed(self.trades[-80:])),
            "equity": self.equity[-400:],
            "daily": daily_rows,
            "periods": periods,
            "signals": list(reversed(self.signals[-40:])),
            "candles": self.candles_tail,
        }

    def _persist(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_name": self.config_name,
            "case": self.case,
            "timeframe": self.timeframe,
            "source": self.source,
            "interval_sec": self.interval_sec,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "running": False,
            "done": self.done,
            "cursor": self.cursor,
            "last_bar_time": self.last_bar_time,
            "bank": self.bank,
            "initial_bank": self.initial_bank,
            "peak": self.peak,
            "max_dd": self.max_dd,
            "max_contracts": self.max_contracts,
            "day_key": self.day_key.isoformat() if self.day_key else None,
            "day_pnl": self.day_pnl,
            "trades_today": self.trades_today,
            "position": _position_dict(self.position),
            "last_signal": _signal_dict(self.last_signal),
            "trades": self.trades,
            "equity": self.equity,
            "signals": self.signals,
        }
        SESSION_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _restore_state(self, raw: dict[str, Any]) -> None:
        self.config_name = str(raw.get("config_name") or DEFAULT_CONFIG)
        self.case = str(raw.get("case") or DEFAULT_CASE)
        self.timeframe = str(raw.get("timeframe") or DEFAULT_TIMEFRAME)
        self.source = str(raw.get("source") or "paper")
        self.interval_sec = float(raw.get("interval_sec") or DEFAULT_INTERVAL_SEC)
        self.window_start = parse_day(raw.get("window_start"))
        self.window_end = parse_day(raw.get("window_end"))
        self.done = bool(raw.get("done"))
        self.cursor = int(raw.get("cursor") or 0)
        self.last_bar_time = raw.get("last_bar_time")
        self.bank = float(raw.get("bank") or 1000)
        self.initial_bank = float(raw.get("initial_bank") or 1000)
        self.peak = float(raw.get("peak") or self.initial_bank)
        self.max_dd = float(raw.get("max_dd") or 0)
        self.max_contracts = int(raw.get("max_contracts") or 1)
        day = raw.get("day_key")
        self.day_key = date.fromisoformat(day) if day else None
        self.day_pnl = float(raw.get("day_pnl") or 0)
        self.trades_today = int(raw.get("trades_today") or 0)
        self.position = _position_from_dict(raw.get("position"))
        self.last_signal = _signal_from_dict(raw.get("last_signal"))
        self.trades = list(raw.get("trades") or [])
        self.equity = list(raw.get("equity") or [])
        self.signals = list(raw.get("signals") or [])

    def load_saved(self) -> None:
        if not SESSION_PATH.exists():
            return
        try:
            raw = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._restore_state(raw)
        try:
            self._prepare_runtime(reset_cursor=False)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)

    def reset(self) -> None:
        self.stop()
        self.done = False
        self.error = None
        self.cursor = 0
        self.last_tick = None
        self.last_bar_time = None
        self.position = None
        self.last_signal = None
        self.trades = []
        self.equity = []
        self.signals = []
        self.candles_tail = []
        self.day_key = None
        self.day_pnl = 0.0
        self.trades_today = 0
        self.max_dd = 0.0
        self.max_contracts = 1
        self._ticks_since_save = 0
        self._frame = None
        self._cfg = None
        self._policy = None
        self._bt = None
        self._session = None
        if SESSION_PATH.exists():
            SESSION_PATH.unlink()

    def stop(self) -> None:
        self.running = False
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
        self._persist()

    def _prepare_runtime(self, reset_cursor: bool) -> None:
        name, cfg = resolve_live_config(self.case, self.timeframe, self.initial_bank)
        self.config_name = name
        self._cfg = cfg
        self.initial_bank = float(cfg.account.initial_bank)
        self.timeframe = str(cfg.data.timeframe)
        model = ensure_model(cfg.data.timeframe, cfg.resolve_csv(cfg.data.train_csv))
        self._policy = SignalPolicy(cfg, model)
        self._bt = BacktestEngine(cfg)
        self._session = SessionFilter(cfg)
        cursor = 0
        if self.source == "mt5":
            self._frame = self._load_mt5()
        else:
            path = test_csv_for(self.timeframe)
            if not path.exists():
                path = cfg.resolve_csv(cfg.data.test_csv)
            full = load_candles(path)
            start = self.window_start
            end = self.window_end
            if start is None or end is None:
                min_date, max_date = dataset_bounds(self.timeframe)
                start, end = default_window(min_date, max_date)
                self.window_start, self.window_end = start, end
            self._frame, cursor = slice_paper_frame(full, start, end)
        if self._frame is None or self._frame.empty:
            raise RuntimeError("Sem candles para a sessão ao vivo.")
        if reset_cursor:
            self.bank = self.initial_bank
            self.peak = self.bank
            self.max_dd = 0.0
            self.max_contracts = 1
            if self.source == "paper":
                self.cursor = cursor
            else:
                self.cursor = max(1, len(self._frame) - 1)
            self.equity = [
                {
                    "t": pd.Timestamp(self._frame.iloc[self.cursor]["timestamp"]).isoformat(),
                    "bank": round(self.bank, 2),
                }
            ]
        elif self._frame is not None and len(self._frame):
            self.cursor = min(max(0, self.cursor), len(self._frame) - 1)
        self._refresh_candles_tail()

    def _load_mt5(self) -> pd.DataFrame:
        assert self._cfg is not None
        mt5 = self._cfg.mt5
        broker = Mt5Broker(mt5.symbol, mt5.magic, mt5.deviation, mt5.filling, mt5.comment)
        broker.connect()
        try:
            candles = broker.last_closed_candles(mt5.symbol, self._cfg.data.timeframe, LOOKBACK_BARS)
        finally:
            broker.shutdown()
        return frame_from_candles(candles, source_file="mt5")

    def _refresh_candles_tail(self) -> None:
        if self._frame is None or self._frame.empty:
            self.candles_tail = []
            return
        end = min(self.cursor + 1, len(self._frame))
        start = max(0, end - 60)
        slice_ = self._frame.iloc[start:end]
        self.candles_tail = [
            {
                "t": pd.Timestamp(row["timestamp"]).isoformat(),
                "open": float(row["Abertura"]),
                "high": float(row["Máximo"]),
                "low": float(row["Mínimo"]),
                "close": float(row["Fechamento"]),
            }
            for _, row in slice_.iterrows()
        ]

    def _close_position(self, ts: datetime, price: float, reason: str) -> None:
        assert self._bt is not None and self.position is not None
        trade: Trade = self._bt._close(self.position, ts, price, reason)
        packed = trade.to_dict()
        self.trades.append(packed)
        self.bank += trade.pnl
        self.day_pnl += trade.pnl
        self.peak = max(self.peak, self.bank)
        self.max_dd = max(self.max_dd, self.peak - self.bank)
        self.equity.append({"t": ts.isoformat(), "bank": round(self.bank, 2)})
        self.position = None

    def _process_index(self, i: int) -> None:
        assert self._frame is not None and self._policy is not None
        assert self._bt is not None and self._session is not None
        row = self._frame.iloc[i]
        ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.replace(tzinfo=None)
        o, h, l, c = float(row["Abertura"]), float(row["Máximo"]), float(row["Mínimo"]), float(row["Fechamento"])
        prev = self._frame.iloc[i - 1]
        prev_ts = pd.Timestamp(prev["timestamp"]).to_pydatetime()
        prev_close = float(prev["Fechamento"])
        self.last_bar_time = ts.isoformat()

        key = ts.date()
        if key != self.day_key:
            if self.position is not None:
                self._close_position(prev_ts, prev_close, "fim_do_dia")
            self.day_key = key
            self.day_pnl = 0.0
            self.trades_today = 0

        if self.position is not None:
            exit_price, reason = self._bt._manage(self.position, o, h, l, c)
            if exit_price is None and self._session.flatten_day(ts):
                exit_price, reason = c, "fim_da_sessao"
            if exit_price is not None:
                self._close_position(ts, float(exit_price), reason)

        if self.position is not None:
            last_eq = self.equity[-1]["t"] if self.equity else None
            if last_eq != ts.isoformat():
                self.equity.append({"t": ts.isoformat(), "bank": round(self.bank, 2)})
            if self.cursor % 8 == 0:
                self._refresh_candles_tail()
            return

        history = self._frame.iloc[:i]
        sig = self._policy.from_candles(history)
        self.last_signal = sig
        self.signals.append(
            {
                "t": ts.isoformat(),
                **(_signal_dict(sig) or {}),
            }
        )
        self.signals = self.signals[-200:]

        if self._session.flatten_day(ts) or not self._session.allows(ts):
            return
        if sig.side is Side.FLAT:
            return
        acc = self._cfg.account
        n_contracts = contracts_for_bank(self.bank, self.initial_bank)
        self.max_contracts = max(self.max_contracts, n_contracts)
        if self.bank < max(50.0, acc.contract_cost * n_contracts + 20.0):
            return
        daily_loss_money = float(self._cfg.risk.daily_loss_points) * acc.point_value * n_contracts
        if self._cfg.risk.daily_loss_points > 0 and self.day_pnl <= -abs(daily_loss_money):
            return
        if self.trades_today >= self._cfg.risk.max_trades_per_day:
            return

        entry = o
        stop, take = self._bt.risk.levels(sig.side, entry, None)
        self.position = {
            "side": sig.side,
            "entry": float(entry),
            "stop": float(stop),
            "take": float(take),
            "time": ts,
            "hour": ts.hour,
            "extreme": float(entry),
            "contracts": n_contracts,
            "reason": sig.reason,
        }
        self.trades_today += 1
        exit_price, reason = self._bt._manage(self.position, o, h, l, c)
        if exit_price is not None:
            self._close_position(ts, float(exit_price), reason)

        last_eq = self.equity[-1]["t"] if self.equity else None
        if last_eq != ts.isoformat():
            self.equity.append({"t": ts.isoformat(), "bank": round(self.bank, 2)})
        if self.cursor % 8 == 0:
            self._refresh_candles_tail()

    def tick(self) -> None:
        self.last_tick = datetime.now().isoformat(timespec="seconds")
        self.error = None
        if self._frame is None or self._policy is None:
            self._prepare_runtime(reset_cursor=False)
        assert self._frame is not None
        if self.source == "mt5":
            try:
                fresh = self._load_mt5()
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                self.running = False
                return
            last_ts = pd.Timestamp(fresh.iloc[-1]["timestamp"]).isoformat()
            if last_ts == self.last_bar_time:
                return
            self._frame = fresh
            i = len(fresh) - 1
            if i < 1:
                return
            self.cursor = i
            self._process_index(i)
        else:
            nxt = self.cursor + 1
            if nxt >= len(self._frame):
                self.done = True
                self.running = False
                self._persist()
                return
            self.cursor = nxt
            self._process_index(self.cursor)
        self._ticks_since_save += 1
        if self.done or self._ticks_since_save >= 40:
            self._persist()
            self._ticks_since_save = 0

    async def run_loop(self) -> None:
        try:
            while self.running:
                async with self._lock:
                    self.tick()
                await asyncio.sleep(max(0.0, float(self.interval_sec)))
        except asyncio.CancelledError:
            self.running = False
            raise

    async def start(
        self,
        case: str,
        timeframe: str,
        initial_bank: float,
        source: str,
        interval_sec: float,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            self.stop()
            case, timeframe, bank, window_start, window_end = validate_live_params(
                case,
                timeframe,
                initial_bank,
                source,
                parse_day(start),
                parse_day(end),
            )
            self.case = case
            self.timeframe = timeframe
            self.initial_bank = bank
            self.source = source
            self.interval_sec = float(interval_sec)
            self.window_start = window_start
            self.window_end = window_end
            self.done = False
            self.error = None
            self._prepare_runtime(reset_cursor=True)
            self.running = True
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self.run_loop())
            return self.snapshot()


ENGINE = LiveEngine()
ENGINE.load_saved()
