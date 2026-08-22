from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from typing import Any

import pandas as pd

from trader.backtest import BacktestEngine, SessionFilter, contracts_for_bank
from trader.broker import Mt5Broker
from trader.data import frame_from_candles, load_candles
from trader.domain import Side, Signal, Trade
from trader.paths import DATASETS_DIR, RESULTS_DIR
from trader.replay import ensure_model, load_named_config
from trader.signals import SignalPolicy

SESSION_PATH = RESULTS_DIR / "live_session.json"
WEEK_CSV = DATASETS_DIR / "mt5_m5_week.csv"
DEFAULT_CONFIG = "best_candles_m5_1000_a"
WARMUP = 40
WEEK_DAYS = 5
LOOKBACK_BARS = 80
DEFAULT_INTERVAL_SEC = 0.001


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


class LiveEngine:
    """Paper (or MT5-data) walk-forward session. Never sends orders."""

    def __init__(self) -> None:
        self.config_name = DEFAULT_CONFIG
        self.source = "paper"
        self.interval_sec = DEFAULT_INTERVAL_SEC
        self.running = False
        self.done = False
        self.error: str | None = None
        self.cursor = 0
        self.last_tick: str | None = None
        self.last_bar_time: str | None = None
        self.bank = 1000.0
        self.initial_bank = 1000.0
        self.peak = 1000.0
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
        return {
            "running": self.running,
            "done": self.done,
            "error": self.error,
            "config": self.config_name,
            "source": self.source,
            "interval_sec": self.interval_sec,
            "last_tick": self.last_tick,
            "last_bar_time": self.last_bar_time,
            "cursor": self.cursor,
            "n_bars": 0 if self._frame is None else int(len(self._frame)),
            "initial_bank": round(self.initial_bank, 2),
            "bank": round(self.bank, 2),
            "net_pnl": net,
            "today_pnl": round(today_closed, 2),
            "avg_daily": round(net / n_days, 2) if n_days else 0.0,
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
            "signals": list(reversed(self.signals[-40:])),
            "candles": self.candles_tail,
        }

    def _persist(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_name": self.config_name,
            "source": self.source,
            "interval_sec": self.interval_sec,
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
        self.source = str(raw.get("source") or "paper")
        self.interval_sec = float(raw.get("interval_sec") or DEFAULT_INTERVAL_SEC)
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
        cfg = load_named_config(self.config_name)
        self._cfg = cfg
        self.initial_bank = float(cfg.account.initial_bank)
        model = ensure_model(cfg.data.timeframe, cfg.resolve_csv(cfg.data.train_csv))
        self._policy = SignalPolicy(cfg, model)
        self._bt = BacktestEngine(cfg)
        self._session = SessionFilter(cfg)
        if self.source == "mt5":
            self._frame = self._load_mt5()
        else:
            if WEEK_CSV.exists():
                path = WEEK_CSV
            else:
                path = cfg.resolve_csv(cfg.data.test_csv)
                if not path.exists():
                    path = DATASETS_DIR / "WIN_5min_test.csv"
            self._frame = load_candles(path)
        if self._frame is None or self._frame.empty:
            raise RuntimeError("Sem candles para a sessão ao vivo.")
        if reset_cursor:
            self.bank = self.initial_bank
            self.peak = self.bank
            self.max_dd = 0.0
            self.max_contracts = 1
            self.cursor = self._paper_start_index()
            self.equity = [
                {
                    "t": pd.Timestamp(self._frame.iloc[self.cursor]["timestamp"]).isoformat(),
                    "bank": round(self.bank, 2),
                }
            ]
        self._refresh_candles_tail()

    def _paper_start_index(self) -> int:
        """First bar of the last 5 pregões (semana de estudo), with ATR warmup."""
        assert self._frame is not None
        ts = pd.to_datetime(self._frame["timestamp"])
        days = ts.dt.normalize().drop_duplicates().sort_values()
        week = days.tail(WEEK_DAYS)
        start_day = week.iloc[0]
        idx = int((ts.dt.normalize() == start_day).to_numpy().nonzero()[0][0])
        return max(WARMUP, idx - 1)

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

    async def start(self, config: str, source: str, interval_sec: float) -> dict[str, Any]:
        async with self._lock:
            self.stop()
            self.config_name = config or DEFAULT_CONFIG
            self.source = source if source in {"paper", "mt5"} else "paper"
            self.interval_sec = float(interval_sec)
            self.running = True
            self.done = False
            self.error = None
            self._prepare_runtime(reset_cursor=True)
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self.run_loop())
            return self.snapshot()


ENGINE = LiveEngine()
ENGINE.load_saved()
