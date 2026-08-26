from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from typing import Any

import pandas as pd

from trader.backtest import LOT_SCALED, BacktestEngine, SessionFilter, size_contracts
from trader.broker import Mt5Broker
from trader.config import AppConfig
from trader.data import frame_from_candles
from trader.domain import Side, Signal
from trader.feeds import StreamFeed
from trader.live import (
    _position_dict,
    _position_from_dict,
    _signal_dict,
    _signal_from_dict,
    live_period_stats,
)
from trader.mt5_session import (
    DEFAULT_MAGIC,
    DEMO_PLAYBOOK,
    LOOKBACK_BARS,
    bar_is_live,
    env_credentials,
    next_gold_window,
    planned_order,
    resolve_symbol,
    session_wait_reason,
)
from trader.paths import RESULTS_DIR
from trader.replay import ensure_model, load_named_config
from trader.execution import limit_fill_from_mark, limit_hits_mark, planned_limit_entry
from trader.risk import RiskCalculator, protect_levels

SESSION_PATH = RESULTS_DIR / "realtime_session.json"
CONFIG_NAME = "best_candles_m5_1000_a"
DEFAULT_BANK = 1000.0
MIN_LIVE_BANK = 500.0
ORDER_MODES = {"paper", "mt5", "prd"}
ORDER_MODE_ALIASES = {
    "prod": "prd",
    "production": "prd",
    "producao": "prd",
    "produção": "prd",
}


def normalize_order_mode(mode: str | None) -> str:
    key = str(mode or "paper").strip().lower()
    key = ORDER_MODE_ALIASES.get(key, key)
    if key not in ORDER_MODES:
        raise ValueError("order_mode deve ser paper, mt5 ou prd")
    return key
POLL_SEC = 0.5


def load_live_config() -> AppConfig:
    cfg = load_named_config(CONFIG_NAME)
    data = cfg.to_dict()
    data["mt5"]["enabled"] = True
    return AppConfig.from_dict(data)


def empty_realtime_snapshot() -> dict[str, Any]:
    return {
        "running": False,
        "done": False,
        "error": None,
        "config": CONFIG_NAME,
        "case": "last_candles",
        "timeframe": "m5",
        "source": "mt5",
        "order_mode": "mt5",
        "interval_sec": POLL_SEC,
        "window_start": None,
        "window_end": None,
        "start": None,
        "end": None,
        "last_tick": None,
        "last_bar_time": None,
        "cursor": 0,
        "n_bars": 0,
        "initial_bank": DEFAULT_BANK,
        "bank": DEFAULT_BANK,
        "net_pnl": 0.0,
        "today_pnl": 0.0,
        "avg_daily": 0.0,
        "n_days": 0,
        "n_trades": 0,
        "n_wins": 0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "lot": LOT_SCALED,
        "contracts": 1,
        "max_contracts": 1,
        "signal": None,
        "position": None,
        "pending": None,
        "trades": [],
        "equity": [],
        "daily": [],
        "periods": live_period_stats([], None, None),
        "signals": [],
        "candles": [],
        "quote": None,
        "open_pnl": 0.0,
        "skip_reason": None,
        "wait_reason": "mercado_fechado",
        "next_gold": next_gold_window(),
        "playbook": None,
        "mode": "mt5",
        "feed": {"ready": False, "symbol": None, "detail": None, "error": None},
        "mt5": {
            "ready": False,
            "demo": None,
            "login": None,
            "server": None,
            "symbol": None,
            "filling": None,
            "trade_allowed": False,
            "balance": None,
            "equity": None,
            "credit": None,
            "profit": None,
            "margin_free": None,
            "bank": None,
        },
    }


class RealtimeEngine:
    """Live MT5 session: reads closed M5 bars and sends demo orders."""

    def __init__(self) -> None:
        self.config_name = CONFIG_NAME
        self.case = "last_candles"
        self.timeframe = "m5"
        self.source = "mt5"
        self.order_mode = "mt5"
        self.interval_sec = POLL_SEC
        self.running = False
        self.done = False
        self.error: str | None = None
        self.wait_reason = "mercado_fechado"
        self.cursor = 0
        self.last_tick: str | None = None
        self.last_bar_time: str | None = None
        self.processed_bar: str | None = None
        self.bank = DEFAULT_BANK
        self.initial_bank = DEFAULT_BANK
        self.lot = LOT_SCALED
        self.peak = DEFAULT_BANK
        self.max_dd = 0.0
        self.max_contracts = 1
        self.day_key: date | None = None
        self.day_pnl = 0.0
        self.trades_today = 0
        self.position: dict | None = None
        self.pending: dict | None = None
        self.last_signal: Signal | None = None
        self.trades: list[dict[str, Any]] = []
        self.equity: list[dict[str, Any]] = []
        self.signals: list[dict[str, Any]] = []
        self.candles_tail: list[dict[str, Any]] = []
        self.mt5_info: dict[str, Any] = empty_realtime_snapshot()["mt5"]
        self.feed_info: dict[str, Any] = empty_realtime_snapshot()["feed"]
        self._stream = StreamFeed()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._frame: pd.DataFrame | None = None
        self._cfg: AppConfig | None = None
        self._policy = None
        self._bt: BacktestEngine | None = None
        self._session: SessionFilter | None = None
        self._broker: Mt5Broker | None = None
        self._ticks_since_save = 0
        self._ticks_mt5 = 0
        self.quote: dict[str, Any] | None = None
        self._enter_now = False
        self._mt5_profit: float | None = None
        self._demo_tried = False
        self.demo_probe: dict[str, Any] | None = None
        self.ui_logs: list[str] = []

    def snapshot(self) -> dict[str, Any]:
        wins = [t for t in self.trades if t.get("result") == "win"]
        n = len(self.trades)
        days = {str(t.get("exit_time", ""))[:10] for t in self.trades if t.get("exit_time")}
        n_days = max(len(days), 1) if n else 0
        today_key = str(self.day_key) if self.day_key else (self.last_bar_time or "")[:10]
        today_closed = sum(
            float(t.get("pnl") or 0)
            for t in self.trades
            if str(t.get("exit_time", ""))[:10] == today_key
        )
        daily: dict[str, float] = {}
        for trade in self.trades:
            key = str(trade.get("exit_time", ""))[:10]
            if not key:
                continue
            daily[key] = round(daily.get(key, 0.0) + float(trade.get("pnl") or 0), 2)
        daily_rows = [{"t": key, "pnl": value} for key, value in sorted(daily.items())]
        start = self.trades[0]["exit_time"][:10] if self.trades else None
        end = today_key or None
        window_start = date.fromisoformat(start) if start else None
        window_end = date.fromisoformat(end) if end and len(end) == 10 else None
        try:
            periods = live_period_stats(self.trades, window_start, window_end)
        except Exception:  # noqa: BLE001
            periods = live_period_stats(self.trades, None, None)
        avg_daily = float((periods.get("avg") or {}).get("daily", {}).get("avg") or 0.0)
        open_meta = self._open_mark()
        open_pnl = float(open_meta.get("pnl") or 0.0)
        pos = _position_dict(self.position)
        if pos is not None:
            pos.update(open_meta)
        candles = list(self.candles_tail)
        if self.quote and self.last_tick:
            candles = [
                *candles,
                {
                    "t": self.last_tick,
                    "open": float(self.quote["last"]),
                    "high": float(self.quote["ask"] or self.quote["last"]),
                    "low": float(self.quote["bid"] or self.quote["last"]),
                    "close": float(self.quote["last"]),
                },
            ]
        equity = list(self.equity[-400:])
        if self.last_tick:
            equity = [*equity, {"t": self.last_tick, "bank": round(self.bank + open_pnl, 2)}]
        open_row = self._open_trade_row(pos, open_meta)
        return {
            "running": self.running,
            "done": self.done,
            "error": self.error,
            "config": self.config_name,
            "case": self.case,
            "timeframe": self.timeframe,
            "source": self.source,
            "order_mode": self.order_mode,
            "interval_sec": self.interval_sec,
            "window_start": None,
            "window_end": None,
            "start": None,
            "end": None,
            "last_tick": self.last_tick,
            "last_bar_time": self.last_bar_time,
            "cursor": self.cursor,
            "n_bars": 0 if self._frame is None else int(len(self._frame)),
            "initial_bank": round(self.initial_bank, 2),
            "bank": round(self.bank + open_pnl, 2),
            "net_pnl": round(self.bank - self.initial_bank + open_pnl, 2),
            "today_pnl": round(today_closed + open_pnl, 2),
            "avg_daily": avg_daily,
            "n_days": n_days,
            "n_trades": n,
            "n_wins": len(wins),
            "win_rate": round(100.0 * len(wins) / n, 1) if n else 0.0,
            "max_drawdown": round(self.max_dd, 2),
            "max_drawdown_pct": round(100.0 * self.max_dd / self.initial_bank, 1) if self.initial_bank else 0.0,
            "lot": self.lot,
            "contracts": self._n_contracts(),
            "max_contracts": self.max_contracts,
            "signal": _signal_dict(self.last_signal),
            "position": pos,
            "pending": _position_dict(self.pending),
            "trades": ([open_row] if open_row else []) + list(reversed(self.trades[-80:])),
            "equity": equity,
            "daily": daily_rows,
            "periods": periods,
            "signals": list(reversed(self.signals[-40:])),
            "candles": candles,
            "quote": dict(self.quote) if self.quote else None,
            "open_pnl": round(open_pnl, 2),
            "skip_reason": self._skip_reason(),
            "wait_reason": self.wait_reason,
            "next_gold": next_gold_window(),
            "playbook": DEMO_PLAYBOOK if self.wait_reason == "aguardando_login" else None,
            "mode": self.order_mode,
            "feed": dict(self.feed_info),
            "mt5": dict(self.mt5_info),
            "armed": bool(self.running and self._task is not None and not self._task.done()),
            "can_send": bool(
                (self.order_mode == "mt5" and self.mt5_info.get("demo") is True)
                or self.order_mode == "prd"
            ),
            "demo_probe": dict(self.demo_probe) if self.demo_probe else None,
        }

    def status(self) -> dict[str, Any]:
        snap = self.snapshot()
        return {
            "running": snap["running"],
            "wait_reason": snap["wait_reason"],
            "next_gold": snap["next_gold"],
            "error": snap["error"],
            "config": snap["config"],
            "last_bar_time": snap["last_bar_time"],
            "last_tick": snap["last_tick"],
            "source": snap["source"],
            "order_mode": snap["order_mode"],
            "mode": snap["mode"],
            "feed": snap["feed"],
            "mt5": snap["mt5"],
            "playbook": snap["playbook"],
        }

    def list_feeds(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "order_mode": self.order_mode,
            "feeds": [
                {
                    "key": "mt5",
                    "label": f"MT5 {self.mt5_info.get('symbol') or 'WIN'}",
                    "ready": bool(self.mt5_info.get("symbol")),
                    "mode": self.order_mode,
                },
                {
                    "key": "stream",
                    "label": "Stream (URL/POST)",
                    "ready": self._stream.ready(live_only=True),
                    "mode": "paper",
                },
            ],
        }

    def ingest_candles(self, raw: Any) -> dict[str, Any]:
        candles = self._stream.ingest(raw)
        self.feed_info = self._stream.status()
        return {"ok": True, "n": len(candles), "symbol": self._stream.symbol, "feed": self.feed_info}

    def set_order_mode(self, mode: str) -> None:
        key = normalize_order_mode(mode)
        self.order_mode = key
        self._enter_now = True
        if key == "paper":
            self._cancel_pending(self._broker if self.pending and self.pending.get("ticket") else None)
            if self.position is not None and self.position.get("ticket"):
                self.position = None
                self._mt5_profit = None
            if self.source != "stream":
                self.source = "mt5"
            if not self.mt5_info.get("login"):
                self.wait_reason = "aguardando_login"
            self.error = None
            return
        if self.position is not None and not self.position.get("ticket"):
            self.position = None
        self.source = "mt5"
        if not self.mt5_info.get("login"):
            self.wait_reason = "aguardando_login"

    def _will_send(self) -> bool:
        if self.order_mode == "prd":
            return True
        return self.order_mode == "mt5" and self.mt5_info.get("demo") is True

    def set_source(self, source: str) -> None:
        key = str(source or "mt5").strip().lower()
        if key not in {"mt5", "stream"}:
            raise ValueError("source deve ser mt5 ou stream")
        if key == "stream":
            self.source = "stream"
            self.order_mode = "paper"
            self.error = None
            self._stream.last_closed_candles(self._stream.symbol, "m5", LOOKBACK_BARS, allow_file=False)
            self.feed_info = self._stream.status()
            self.wait_reason = "aguardando_candle"
            return
        self.source = "mt5"
        if self.order_mode not in ORDER_MODES:
            self.order_mode = "mt5"
        self.wait_reason = "aguardando_login"

    def _persist(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_name": self.config_name,
            "source": self.source,
            "order_mode": self.order_mode,
            "running": False,
            "processed_bar": self.processed_bar,
            "last_bar_time": self.last_bar_time,
            "bank": self.bank,
            "initial_bank": self.initial_bank,
            "peak": self.peak,
            "max_dd": self.max_dd,
            "lot": self.lot,
            "max_contracts": self.max_contracts,
            "day_key": self.day_key.isoformat() if self.day_key else None,
            "day_pnl": self.day_pnl,
            "trades_today": self.trades_today,
            "position": _position_dict(self.position),
            "pending": _position_dict(self.pending),
            "last_signal": _signal_dict(self.last_signal),
            "trades": self.trades,
            "equity": self.equity,
            "signals": self.signals,
        }
        SESSION_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load_saved(self) -> None:
        if not SESSION_PATH.exists():
            return
        try:
            raw = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if raw.get("order_mode") in ORDER_MODES:
            self.order_mode = str(raw["order_mode"])
        elif raw.get("source") == "stream":
            self.order_mode = "paper"
        if raw.get("source") in {"mt5", "stream"}:
            self.source = str(raw["source"])
        if self.order_mode in {"mt5", "prd"}:
            self.source = "mt5"
        elif self.source not in {"mt5", "stream"}:
            self.source = "mt5"
        self.processed_bar = raw.get("processed_bar")
        self.last_bar_time = raw.get("last_bar_time")
        self.bank = float(raw.get("bank") or DEFAULT_BANK)
        self.initial_bank = float(raw.get("initial_bank") or DEFAULT_BANK)
        self.peak = float(raw.get("peak") or self.initial_bank)
        self.max_dd = float(raw.get("max_dd") or 0)
        self.max_contracts = int(raw.get("max_contracts") or 1)
        if raw.get("lot"):
            self.lot = str(raw["lot"])
        day = raw.get("day_key")
        self.day_key = date.fromisoformat(day) if day else None
        self.day_pnl = float(raw.get("day_pnl") or 0)
        self.trades_today = int(raw.get("trades_today") or 0)
        self.position = _position_from_dict(raw.get("position"))
        self.pending = _position_from_dict(raw.get("pending"))
        self.last_signal = _signal_from_dict(raw.get("last_signal"))
        self.trades = list(raw.get("trades") or [])
        self.equity = list(raw.get("equity") or [])
        self.signals = list(raw.get("signals") or [])
        self._keep_today_only(date.today())

    def _keep_today_only(self, today: date) -> None:
        day = today.isoformat()

        def _on_day(stamp: Any) -> bool:
            return str(stamp or "")[:10] == day

        kept = [t for t in self.trades if _on_day(t.get("exit_time") or t.get("entry_time"))]
        pos_ok = self.position is None or _on_day(self.position.get("time"))
        pend_ok = self.pending is None or _on_day(self.pending.get("time"))
        if len(kept) == len(self.trades) and pos_ok and pend_ok:
            return
        self.trades = kept
        self.signals = [item for item in self.signals if _on_day(item.get("t"))]
        self.equity = [item for item in self.equity if _on_day(item.get("t"))]
        self.bank = self.initial_bank + sum(float(item.get("pnl") or 0) for item in kept)
        self.peak = max([self.initial_bank, self.bank, *[float(item.get("bank") or self.bank) for item in self.equity]])
        self.max_dd = 0.0
        self.day_pnl = sum(float(item.get("pnl") or 0) for item in kept)
        self.trades_today = len(kept)
        if self.position is not None and not _on_day(self.position.get("time")):
            self.position = None
        if self.pending is not None and not _on_day(self.pending.get("time")):
            self.pending = None
        self.processed_bar = None

    def _prepare_policy(self) -> None:
        cfg = load_live_config()
        self._cfg = cfg
        self.initial_bank = float(cfg.account.initial_bank)
        model = ensure_model(cfg.data.timeframe, cfg.resolve_csv(cfg.data.train_csv))
        from trader.signals import SignalPolicy

        self._policy = SignalPolicy(cfg, model)
        self._bt = BacktestEngine(cfg)
        self._session = SessionFilter(cfg)

    def _ensure_broker(self) -> Mt5Broker | None:
        creds = env_credentials()
        if self._broker is None:
            cfg = self._cfg or load_live_config()
            mt5 = cfg.mt5
            self._broker = Mt5Broker(mt5.symbol, mt5.magic or DEFAULT_MAGIC, mt5.deviation, mt5.filling, mt5.comment)
        try:
            self._broker.connect(select_symbol=False, **creds)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.wait_reason = "aguardando_login"
            self.mt5_info = empty_realtime_snapshot()["mt5"]
            return None
        acc = self._broker.account_payload()
        term = self._broker.terminal_payload()
        symbol = resolve_symbol(self._broker)
        if symbol:
            try:
                self._broker.use_symbol(symbol)
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                symbol = None
        if acc.get("demo") is False:
            self.error = None
            if self.position is not None and self.position.get("ticket") and self.order_mode == "paper":
                self.position = None
                self._mt5_profit = None
        if not term.get("trade_allowed"):
            try:
                nudged = self._broker.ensure_algo_trading()
                if nudged.get("trade_allowed"):
                    term = self._broker.terminal_payload()
            except Exception:  # noqa: BLE001
                pass
        self.mt5_info = {
            "ready": bool(acc.get("account") and symbol),
            "demo": acc.get("demo"),
            "login": acc.get("login"),
            "server": acc.get("server"),
            "symbol": symbol,
            "filling": self._broker.filling if symbol else None,
            "trade_allowed": bool(term.get("trade_allowed")),
            "balance": acc.get("balance"),
            "equity": acc.get("equity"),
            "credit": acc.get("credit"),
            "profit": acc.get("profit"),
            "margin_free": acc.get("margin_free"),
            "bank": acc.get("bank"),
        }
        if not acc.get("account"):
            self.wait_reason = "aguardando_login"
            return None
        return self._broker

    def _refresh_candles_tail(self) -> None:
        if self._frame is None or self._frame.empty:
            self.candles_tail = []
            return
        tail = self._frame.tail(60)
        self.candles_tail = [
            {
                "t": pd.Timestamp(row["timestamp"]).isoformat(),
                "open": float(row["Abertura"]),
                "high": float(row["Máximo"]),
                "low": float(row["Mínimo"]),
                "close": float(row["Fechamento"]),
            }
            for _, row in tail.iterrows()
        ]

    def _record_close(self, ts: datetime, price: float, reason: str, profit: float | None = None) -> None:
        assert self._bt is not None and self.position is not None
        trade = self._bt._close(self.position, ts, price, reason)
        packed = trade.to_dict()
        if profit is not None:
            packed["pnl"] = round(float(profit), 2)
            packed["result"] = "win" if float(profit) > 0 else "loss"
        pnl = float(packed["pnl"])
        self.trades.append(packed)
        self.bank += pnl
        self.day_pnl += pnl
        self.peak = max(self.peak, self.bank)
        self.max_dd = max(self.max_dd, self.peak - self.bank)
        self.equity.append({"t": ts.isoformat(), "bank": round(self.bank, 2)})
        self.position = None

    def _sync_position(self, broker: Mt5Broker, now: datetime) -> None:
        live = broker.open_positions()
        if live:
            pos = live[0]
            had = self.position is not None
            from_pending = self.pending is not None
            if self.position is None:
                src = self.pending or {}
                self.position = {
                    "side": pos["side"],
                    "entry": pos["entry"],
                    "stop": pos["stop"] or src.get("stop") or pos["entry"],
                    "take": pos["take"] or src.get("take") or pos["entry"],
                    "time": pos["time"],
                    "hour": pos["time"].hour,
                    "extreme": pos["entry"],
                    "contracts": float(pos.get("volume") or src.get("contracts") or self._n_contracts()),
                    "reason": str(src.get("reason") or "mt5"),
                    "ticket": pos["ticket"],
                    "entry_bar": src.get("entry_bar") or self.last_bar_time,
                    "orig_stop": float(src.get("orig_stop") or pos["stop"] or pos["entry"]),
                    "orig_take": float(src.get("orig_take") or pos["take"] or pos["entry"]),
                }
                if from_pending and not had:
                    self.trades_today += 1
            else:
                self.position["ticket"] = pos["ticket"]
                self.position["contracts"] = float(
                    pos.get("volume") or self.position.get("contracts") or self._n_contracts()
                )
                if pos["stop"]:
                    self.position["stop"] = pos["stop"]
                if pos["take"]:
                    self.position["take"] = pos["take"]
                if pos.get("entry"):
                    self.position["entry"] = pos["entry"]
            self._mt5_profit = float(pos.get("profit") or 0)
            self.pending = None
            return
        self._mt5_profit = None
        self._drop_stale_pending(broker)
        if self.position is not None:
            ticket = self.position.get("ticket")
            deal = broker.closing_deal(int(ticket)) if ticket else None
            price = float(deal["price"]) if deal else float(self.position["entry"])
            ts = deal["time"] if deal else now
            reason = "mt5_sltp" if deal else "mt5_fechou"
            profit = None
            if deal is not None and deal.get("profit") is not None:
                profit = float(deal["profit"])
            self._record_close(ts, price, reason, profit=profit)

    def _can_enter(self, now: datetime, *, live_mt5: bool) -> bool:
        if self._session is None:
            return False
        if self.position is not None:
            return False
        if self.pending is not None:
            return False
        if not self._session.allows(now):
            return False
        if self._cfg and self.trades_today >= self._cfg.risk.max_trades_per_day:
            return False
        if live_mt5:
            if self.order_mode != "prd" and self.mt5_info.get("demo") is False:
                return False
            if not self.mt5_info.get("trade_allowed"):
                return False
            if not self.mt5_info.get("symbol"):
                return False
            live = self._live_bank()
            if live is None or live < MIN_LIVE_BANK:
                return False
        return True

    def _live_bank(self) -> float | None:
        info = self.mt5_info or {}
        for key in ("bank", "equity", "balance", "margin_free"):
            raw = info.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    def _n_contracts(self) -> float:
        bank = self.bank
        if self._will_send():
            live = self._live_bank()
            if live is None or live < MIN_LIVE_BANK:
                return 0.0
            bank = live
        n = float(size_contracts(bank, self.lot))
        self.max_contracts = max(float(self.max_contracts), n)
        return n

    def _is_buy(self, side: Any) -> bool:
        return side is Side.BUY or str(side) == "BUY"

    def _mark_price(self, side: Any | None = None) -> float | None:
        if not self.quote:
            return None
        if side is not None:
            return float(self.quote["bid"] if self._is_buy(side) else self.quote["ask"])
        return float(self.quote["last"])

    def _open_mark(self) -> dict[str, Any]:
        if self.position is None:
            return {}
        side = self.position["side"]
        entry = float(self.position["entry"])
        stop = float(self.position["stop"])
        take = float(self.position["take"])
        n = float(self.position.get("contracts") or self._n_contracts())
        mark = self._mark_price(side)
        if mark is None and self._frame is not None and not self._frame.empty:
            mark = float(self._frame.iloc[-1]["Fechamento"])
        if mark is None:
            return {}
        points = (mark - entry) if self._is_buy(side) else (entry - mark)
        if self._mt5_profit is not None and self.order_mode in {"mt5", "prd"}:
            pnl = float(self._mt5_profit)
        else:
            pv = float(self._cfg.account.point_value) if self._cfg else 0.2
            pnl = points * pv * n
        to_stop = (mark - stop) if self._is_buy(side) else (stop - mark)
        to_take = (take - mark) if self._is_buy(side) else (mark - take)
        return {
            "mark": round(mark, 4),
            "pnl": round(pnl, 2),
            "points": round(points, 4),
            "to_stop": round(to_stop, 4),
            "to_take": round(to_take, 4),
        }

    def _skip_reason(self) -> str | None:
        sig = self.last_signal
        if self.position is not None:
            held = self.position["side"]
            held_s = held.value if isinstance(held, Side) else str(held)
            if sig is not None and sig.side is not Side.FLAT:
                return f"Sinal {sig.side.value} não virou ordem: já tem {held_s} aberta até stop/alvo."
            return f"Já tem {held_s} aberta. Novos sinais não abrem outra ordem até stop/alvo."
        if self.pending is not None:
            held = self.pending["side"]
            held_s = held.value if isinstance(held, Side) else str(held)
            px = float(self.pending["entry"])
            return f"Pendente LIMIT {held_s} @ {px:.0f}. Espera o fill no preço previsto; não persegue o tick."
        if sig is None or sig.side is Side.FLAT:
            if sig is not None and sig.reason == "mercado_estranho":
                return "FLAT mercado_estranho: o filtro ml_guard recusou o candle. Isso não vira ordem."
            return None
        if self.wait_reason and self.wait_reason not in {"pronto"}:
            return f"Sinal {sig.side.value} sem ordem: {self.wait_reason}."
        if self._will_send():
            live = self._live_bank()
            if live is None or live < MIN_LIVE_BANK:
                return f"Sinal {sig.side.value} sem ordem: banca MT5 abaixo de R$ {MIN_LIVE_BANK:.0f}."
        return None

    def _open_trade_row(self, pos: dict[str, Any] | None, open_meta: dict[str, Any]) -> dict[str, Any] | None:
        if not pos:
            return None
        entry = float(pos["entry"])
        mark = float(open_meta.get("mark") or entry)
        return {
            "side": pos["side"],
            "entry_time": pos.get("time"),
            "exit_time": None,
            "entry": entry,
            "exit": mark,
            "points": float(open_meta.get("points") or 0),
            "pnl": float(open_meta.get("pnl") or 0),
            "result": "open",
            "reason": str(pos.get("reason") or "aberta"),
            "contracts": pos.get("contracts"),
        }

    def _remember_signal(self, sig: Signal, ts: datetime) -> None:
        packed = {"t": ts.isoformat(), **(_signal_dict(sig) or {})}
        last = self.signals[-1] if self.signals else None
        if (
            last
            and last.get("t") == packed["t"]
            and last.get("side") == packed.get("side")
            and last.get("reason") == packed.get("reason")
        ):
            self.signals[-1] = packed
            return
        self.signals.append(packed)
        self.signals = self.signals[-200:]

    def _planned_entry(self, sig: Signal) -> float:
        tick = float(self._cfg.instrument.tick_size) if self._cfg else 5.0
        pred = float(sig.predicted.open) if sig.predicted is not None else None
        delay_points = float(getattr(self._cfg.execution, "entry_delay_points", 0.0) or 0.0) if self._cfg else 0.0
        return planned_limit_entry(pred, float(sig.entry), tick, side=sig.side, delay_points=delay_points)

    def _pending_row(self, sig: Signal, entry: float, stop: float, take: float, ts: datetime, n: float, ticket: Any) -> dict[str, Any]:
        return {
            "side": sig.side,
            "entry": float(entry),
            "stop": float(stop),
            "take": float(take),
            "time": ts,
            "hour": ts.hour,
            "extreme": float(entry),
            "contracts": n,
            "reason": sig.reason,
            "ticket": ticket,
            "entry_bar": self.last_bar_time,
            "orig_stop": float(stop),
            "orig_take": float(take),
        }

    def _cancel_pending(self, broker: Mt5Broker | None) -> None:
        pending = self.pending
        self.pending = None
        if pending is None:
            return
        ticket = pending.get("ticket")
        if broker is None or not ticket:
            return
        try:
            broker.cancel_order(int(ticket))
        except Exception as exc:  # noqa: BLE001
            self.error = f"cancel: {exc}"

    def _drop_stale_pending(self, broker: Mt5Broker) -> None:
        if self.pending is None or self.position is not None:
            return
        ticket = self.pending.get("ticket")
        if not ticket:
            return
        try:
            orders = broker.open_orders()
        except Exception as exc:  # noqa: BLE001
            self.error = f"orders: {exc}"
            return
        if any(int(row["ticket"]) == int(ticket) for row in orders):
            return
        self.pending = None

    def _expire_pending(self, broker: Mt5Broker | None, sig: Signal | None, now: datetime) -> bool:
        """Cancel working LIMIT. True = do not place a new entry this tick (FLAT, opposite, or 17:00)."""
        if self.pending is None:
            return bool(self._session and self._session.flatten_day(now))
        flatten = bool(self._session and self._session.flatten_day(now))
        new_closed = bool(
            self.last_bar_time and self.pending.get("entry_bar") and self.last_bar_time != self.pending.get("entry_bar")
        )
        against = self._signal_invalidates(sig, self.pending["side"])
        if flatten or new_closed or against:
            self._cancel_pending(broker)
            return flatten or against
        return False

    def _fill_pending_paper(self, mark: float | None, ts: datetime) -> None:
        if self.pending is None or mark is None:
            return
        if self.pending.get("ticket"):
            return
        limit = float(self.pending["entry"])
        side = self.pending["side"]
        if not limit_hits_mark(side, limit, mark):
            return
        fill = limit_fill_from_mark(side, limit, mark)
        row = dict(self.pending)
        row["entry"] = float(fill)
        row["extreme"] = float(fill)
        row["ticket"] = None
        row["time"] = ts
        row["hour"] = ts.hour
        self.position = row
        self.pending = None
        self.trades_today += 1

    def _send_signal(self, broker: Mt5Broker, sig: Signal, entry: float) -> None:
        assert self._cfg is not None and self._bt is not None
        if self.order_mode == "paper":
            raise RuntimeError("Modo simular não envia ordem.")
        if self.order_mode == "mt5" and self.mt5_info.get("demo") is False:
            raise RuntimeError("Recusado: conta real. Use Produção para enviar no PRD.")
        if self.order_mode not in {"mt5", "prd"}:
            raise RuntimeError("Modo atual não envia ordem.")
        n = self._n_contracts()
        if n < 1:
            raise RuntimeError(f"Banca MT5 abaixo de R$ {MIN_LIVE_BANK:.0f}.")
        self._cancel_pending(broker)
        stop, take = self._bt.risk.levels(sig.side, entry, None)
        order_sig = Signal(side=sig.side, entry=entry, stop=stop, take=take, reason=sig.reason, predicted=sig.predicted)
        result = broker.send(order_sig, float(n))
        retcode = int(result.get("retcode") or 0)
        if retcode not in {0, 10009, 10008}:
            raise RuntimeError(f"order_send retcode={retcode} {result.get('comment')}")
        ticket = result.get("order") or result.get("deal")
        now = datetime.fromisoformat(self.last_tick) if self.last_tick else datetime.now()
        self.pending = self._pending_row(sig, entry, stop, take, now, n, ticket)
        self._sync_position(broker, now)

    def _open_paper(self, sig: Signal, entry: float, ts: datetime) -> None:
        assert self._bt is not None
        if self.source == "stream" and getattr(self._stream, "origin", None) == "file":
            return
        n = self._n_contracts()
        stop, take = self._bt.risk.levels(sig.side, entry, None)
        self.pending = self._pending_row(sig, entry, stop, take, ts, n, None)

    def _signal_invalidates(self, sig: Signal | None, pos_side: Any) -> bool:
        if sig is None:
            return False
        if sig.side is Side.FLAT:
            return True
        return self._is_buy(sig.side) != self._is_buy(pos_side)

    def _protect_open(
        self,
        broker: Mt5Broker | None,
        mark: float | None,
        closed: pd.Series | None,
        sig: Signal | None,
    ) -> bool:
        """Move SL/TP on the open ticket: break-even lock, next-bar invalidation, trail. Never close_position."""
        if self.position is None or self._bt is None:
            return False
        pos = self.position
        if pos.get("orig_stop") is None:
            pos["orig_stop"] = float(pos["stop"])
        if pos.get("orig_take") is None:
            pos["orig_take"] = float(pos["take"])
        buy = self._is_buy(pos["side"])
        entry = float(pos["entry"])
        old_stop = float(pos["stop"])
        old_take = float(pos["take"])
        risk = self._bt.config.risk
        tick = float(self._bt.config.instrument.tick_size)
        new_bar = bool(self.last_bar_time and pos.get("entry_bar") and self.last_bar_time != pos.get("entry_bar"))
        invalidate = new_bar and self._signal_invalidates(sig, pos["side"])
        bar_high = float(closed["Máximo"]) if closed is not None and "Máximo" in closed.index else None
        bar_low = float(closed["Mínimo"]) if closed is not None and "Mínimo" in closed.index else None
        new_stop, new_take, new_extreme = protect_levels(
            buy=buy,
            entry=entry,
            stop=old_stop,
            take=old_take,
            orig_stop=float(pos["orig_stop"]),
            orig_take=float(pos["orig_take"]),
            mark=mark,
            extreme=float(pos.get("extreme") or entry),
            tick=tick,
            be_trigger=float(risk.be_trigger_points),
            be_lock=float(risk.be_lock_points),
            invalidate=invalidate,
            bar_high=bar_high,
            bar_low=bar_low,
            invalidate_tp=float(risk.invalidate_tp_points),
            trail_enabled=bool(risk.trailing_enabled),
            trail_trigger=float(risk.trailing_trigger_points),
            trail_distance=float(risk.trailing_distance_points),
        )
        pos["extreme"] = new_extreme
        pos["stop"] = new_stop
        pos["take"] = new_take
        changed = abs(new_stop - old_stop) >= 1 or abs(new_take - old_take) >= 1
        ticket = pos.get("ticket")
        if changed and broker is not None and ticket:
            try:
                broker.modify_sltp(int(ticket), new_stop, new_take)
            except Exception as exc:  # noqa: BLE001
                self.error = f"protect: {exc}"
        return changed

    def _trail_live(self, broker: Mt5Broker, mark: float | None) -> None:
        self._protect_open(broker, mark, None, self.last_signal)

    def _flatten_live(self, broker: Mt5Broker, now: datetime) -> None:
        if self.position is None:
            return
        ticket = self.position.get("ticket")
        if not ticket:
            return
        try:
            broker.close_position(
                int(ticket), self.position["side"], float(self.position.get("contracts") or self._n_contracts())
            )
        except Exception as exc:  # noqa: BLE001
            self.error = f"close: {exc}"
        self._sync_position(broker, now)

    def _manage_open(self, broker: Mt5Broker | None, row: pd.Series, ts: datetime, now: datetime) -> None:
        assert self._bt is not None and self._session is not None and self.position is not None
        ticket = self.position.get("ticket")
        if broker is not None and ticket:
            if self._session.flatten_day(ts) or self._session.flatten_day(now):
                self._flatten_live(broker, now)
            return
        o = float(row["Abertura"])
        h = float(row["Máximo"])
        l = float(row["Mínimo"])
        c = float(row["Fechamento"])
        exit_price, reason = self._bt._manage(self.position, o, h, l, c)
        if exit_price is None and (self._session.flatten_day(ts) or self._session.flatten_day(now)):
            exit_price, reason = c, "fim_da_sessao"
        if exit_price is None:
            return
        self._record_close(ts, float(exit_price), reason)

    def _mark_only_row(self, last: pd.Series, mark: float | None) -> pd.Series:
        px = float(mark if mark is not None else last["Fechamento"])
        row = last.copy()
        row["Abertura"] = px
        row["Máximo"] = px
        row["Mínimo"] = px
        row["Fechamento"] = px
        return row

    def _paper_manage_row(self, last: pd.Series, mark: float | None) -> pd.Series:
        """Do not use the entry bar's historical high/low — that is lookahead and instant-closes paper fills."""
        if self.position is None:
            return last
        if self.position.get("ticket"):
            return last
        if self.position.get("entry_bar") != self.last_bar_time:
            return last
        return self._mark_only_row(last, mark)

    def tick(self, now: datetime | None = None) -> None:
        now = now or datetime.now()
        self.last_tick = now.isoformat(timespec="seconds")
        if self._policy is None or self._session is None:
            try:
                self._prepare_policy()
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                self.wait_reason = "aguardando_login" if self.source == "mt5" else "aguardando_candle"
                return
        assert self._session is not None and self._policy is not None
        self._keep_today_only(now.date())
        if self.source == "stream":
            self._tick_stream(now)
            return
        self._tick_mt5(now)

    def _tick_mt5(self, now: datetime) -> None:
        assert self._session is not None and self._policy is not None
        broker = self._ensure_broker()
        if broker is None:
            return
        key = now.date()
        if key != self.day_key:
            self.day_key = key
            self.day_pnl = 0.0
            self.trades_today = 0
        send = self._will_send()
        if not self.mt5_info.get("symbol") and not broker.symbol:
            self.wait_reason = "sem_simbolo"
            return
        if send:
            try:
                self._sync_position(broker, now)
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
        try:
            self.quote = broker.quote() if hasattr(broker, "quote") else None
        except Exception:  # noqa: BLE001
            self.quote = None
        self._ticks_mt5 += 1
        need_bars = self._frame is None or self._frame.empty or self._ticks_mt5 == 1 or self._ticks_mt5 % 4 == 0
        if need_bars:
            try:
                candles = broker.last_closed_candles(broker.symbol, "m5", LOOKBACK_BARS)
                self._frame = frame_from_candles(candles, source_file="mt5")
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                self.wait_reason = session_wait_reason(
                    connected=True,
                    account=bool(self.mt5_info.get("login")),
                    demo=True if (not send or self.order_mode == "prd") else self.mt5_info.get("demo"),
                    symbol=self.mt5_info.get("symbol"),
                    trade_allowed=True if not send else bool(self.mt5_info.get("trade_allowed")),
                    now=now,
                    last_bar=None,
                    in_position=self.position is not None,
                    session=self._session,
                )
                if self._frame is None or self._frame.empty:
                    return
        if self._frame is None or self._frame.empty:
            self.wait_reason = "aguardando_candle"
            return
        self.cursor = len(self._frame) - 1
        closed = self._frame.iloc[-1]
        last = closed
        ts = pd.Timestamp(last["timestamp"]).to_pydatetime()
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.replace(tzinfo=None)
        self.last_bar_time = ts.isoformat()
        self._refresh_candles_tail()
        mark = self._mark_price()
        if mark is not None:
            last = last.copy()
            last["Máximo"] = max(float(last["Máximo"]), mark)
            last["Mínimo"] = min(float(last["Mínimo"]), mark)
            last["Fechamento"] = mark
        self.feed_info = {
            "ready": True,
            "symbol": broker.symbol,
            "detail": (
                f"mt5 {broker.symbol} {len(self._frame)} M5"
                + (f" · {mark:.0f}" if mark is not None else "")
            ),
            "error": None,
            "origin": "mt5",
        }
        self.wait_reason = session_wait_reason(
            connected=True,
            account=True,
            demo=True if (not send or self.order_mode == "prd") else self.mt5_info.get("demo"),
            symbol=self.mt5_info.get("symbol") or broker.symbol,
            trade_allowed=True if not send else bool(self.mt5_info.get("trade_allowed")),
            now=now,
            last_bar=ts if bar_is_live(ts, now) else None,
            in_position=self.position is not None,
            session=self._session,
        )
        live_bar = bar_is_live(ts, now)
        sig = self._policy.from_candles(self._frame)
        self.last_signal = sig
        self._remember_signal(sig, ts)
        skip_enter = self._expire_pending(broker if send else None, sig, now)
        if self.pending is not None and not send:
            self._fill_pending_paper(mark, now)
        if self.position is not None:
            ticket = self.position.get("ticket")
            live_ticket = bool(send and ticket)
            changed = self._protect_open(broker if live_ticket else None, mark, closed, sig)
            if live_ticket:
                if self._session.flatten_day(now):
                    self._flatten_live(broker, now)
            else:
                row = self._paper_manage_row(last, mark)
                if changed:
                    row = self._mark_only_row(last, mark)
                self._manage_open(broker if send else None, row, now, now)
        new_bar = self.last_bar_time != self.processed_bar
        if not self._session.allows(now):
            self._enter_now = True
        try_enter = (new_bar or self._enter_now) and not skip_enter
        if try_enter and self._can_enter(now, live_mt5=send) and live_bar and sig.side is not Side.FLAT:
            try:
                entry = self._planned_entry(sig)
                plan = planned_order(sig, entry, *RiskCalculator(self._cfg).levels(sig.side, entry, None))
                if plan:
                    if send:
                        self._send_signal(broker, sig, entry)
                    else:
                        self._open_paper(sig, entry, now)
                        self._fill_pending_paper(mark, now)
                    self.error = None
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
        if self._session.allows(now):
            self._enter_now = False
        if new_bar:
            self.processed_bar = self.last_bar_time
        self._ticks_since_save += 1
        if self._ticks_since_save >= 3:
            self._persist()
            self._ticks_since_save = 0

    def _tick_stream(self, now: datetime) -> None:
        assert self._session is not None and self._policy is not None
        try:
            candles = self._stream.last_closed_candles(
                self._stream.symbol, "m5", LOOKBACK_BARS, allow_file=False
            )
        except Exception as exc:  # noqa: BLE001
            self.feed_info = self._stream.status()
            self.wait_reason = "aguardando_candle"
            if self._stream.origin != "none":
                self.error = str(exc)
            else:
                self.error = None
            return
        self.feed_info = self._stream.status()
        if not candles:
            self.error = None
            self.last_bar_time = None
            self.candles_tail = []
            self.wait_reason = session_wait_reason(
                connected=True,
                account=True,
                demo=True,
                symbol=self._stream.symbol,
                trade_allowed=True,
                now=now,
                last_bar=None,
                in_position=self.position is not None,
                session=self._session,
            )
            return
        self._frame = frame_from_candles(candles, source_file="stream")
        self.error = None
        if self._frame is None or self._frame.empty:
            self.wait_reason = "aguardando_candle"
            return
        key = now.date()
        if key != self.day_key:
            self.day_key = key
            self.day_pnl = 0.0
            self.trades_today = 0
        self.cursor = len(self._frame) - 1
        last = self._frame.iloc[-1]
        ts = pd.Timestamp(last["timestamp"]).to_pydatetime()
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.replace(tzinfo=None)
        self.last_bar_time = ts.isoformat()
        self._refresh_candles_tail()
        live_bar = self._stream.origin in {"ingest", "http"} and bar_is_live(ts, now)
        self.wait_reason = session_wait_reason(
            connected=True,
            account=True,
            demo=True,
            symbol=self._stream.symbol,
            trade_allowed=True,
            now=now,
            last_bar=ts if live_bar else None,
            in_position=self.position is not None,
            session=self._session,
        )
        if not live_bar:
            if not self._session.allows(now) and now.time().hour < 9:
                self.wait_reason = "mercado_fechado"
            self.last_bar_time = None
            self.candles_tail = []
            return
        sig = self._policy.from_candles(self._frame)
        self.last_signal = sig
        self._remember_signal(sig, ts)
        mark = float(last["Fechamento"])
        skip_enter = self._expire_pending(None, sig, now)
        if self.pending is not None:
            self._fill_pending_paper(mark, ts)
        if self.position is not None:
            changed = self._protect_open(None, mark, last, sig)
            row = self._paper_manage_row(last, mark)
            if changed:
                row = self._mark_only_row(last, mark)
            self._manage_open(None, row, ts, now)
        new_bar = self.last_bar_time != self.processed_bar
        if new_bar and (not skip_enter) and self._can_enter(now, live_mt5=False) and live_bar:
            if sig.side is not Side.FLAT:
                entry = self._planned_entry(sig)
                stop, take = RiskCalculator(self._cfg).levels(sig.side, entry, None)
                if planned_order(sig, entry, stop, take):
                    self._open_paper(sig, entry, ts)
                    self._fill_pending_paper(mark, ts)
        if new_bar:
            self.processed_bar = self.last_bar_time
        self._ticks_since_save += 1
        if self._ticks_since_save >= 3:
            self._persist()
            self._ticks_since_save = 0

    async def run_loop(self) -> None:
        try:
            while self.running:
                try:
                    async with self._lock:
                        await asyncio.to_thread(self.tick)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.error = str(exc)
                await asyncio.sleep(max(0.5, float(self.interval_sec)))
        except asyncio.CancelledError:
            self.running = False
            raise

    async def arm(self) -> dict[str, Any]:
        if self._policy is None:
            try:
                await asyncio.to_thread(self._prepare_policy)
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
        if self.running and self._task and not self._task.done():
            self._enter_now = True
            return self.snapshot()
        self.running = True
        self.done = False
        self._enter_now = True
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self.run_loop())
        return self.snapshot()

    def note_ui(self, line: str) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        self.ui_logs.append(f"{stamp} {line}")
        self.ui_logs = self.ui_logs[-80:]

    async def start(self, source: str | None = None, order_mode: str | None = None) -> dict[str, Any]:
        if order_mode is not None:
            self.set_order_mode(order_mode)
        elif source is not None:
            self.set_source(source)
        self._persist()
        return await self.arm()

    def stop(self) -> None:
        self.running = False
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
        self._persist()

    def reset(self) -> None:
        self.done = False
        self.error = None
        self.processed_bar = None
        self.last_bar_time = None
        self.position = None
        self.pending = None
        self.last_signal = None
        self.trades = []
        self.equity = []
        self.signals = []
        self.candles_tail = []
        self.day_key = None
        self.day_pnl = 0.0
        self.trades_today = 0
        self.max_dd = 0.0
        self.bank = self.initial_bank
        self.peak = self.initial_bank
        if SESSION_PATH.exists():
            SESSION_PATH.unlink()

    def disconnect(self) -> None:
        if self._broker is not None:
            self._broker.shutdown()
            self._broker = None


ENGINE: RealtimeEngine | None = None


def get_realtime_engine() -> RealtimeEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = RealtimeEngine()
        ENGINE.load_saved()
    return ENGINE
