from __future__ import annotations

import asyncio
import json
import math
import time
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
from trader.risk import RiskCalculator

SESSION_PATH = RESULTS_DIR / "realtime_session.json"
CONFIG_NAME = "best_candles_m5_1000_a"
DEFAULT_BANK = 1000.0
POLL_SEC = 10.0
ALIGN_SLACK_MS = 250.0
CATCHUP_SEC = 1.0


def seconds_until_aligned(
    now_ts: float,
    interval: float = POLL_SEC,
    slack_ms: float = ALIGN_SLACK_MS,
) -> float:
    """Seconds until the next unix time divisible by *interval*, plus slack."""
    step = max(1.0, float(interval))
    slack = max(0.0, float(slack_ms)) / 1000.0
    slot = math.floor(now_ts / step) * step
    target = slot + slack
    if now_ts < target:
        return target - now_ts
    return slot + step + slack - now_ts


def is_m5_close_slot(now: datetime) -> bool:
    """True on the :00 ten-second slot of a 5-minute close."""
    return now.minute % 5 == 0 and now.second < 10


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
        "source": "stream",
        "order_mode": "paper",
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
        "trades": [],
        "equity": [],
        "daily": [],
        "periods": live_period_stats([], None, None),
        "signals": [],
        "candles": [],
        "wait_reason": "mercado_fechado",
        "next_gold": next_gold_window(),
        "playbook": None,
        "mode": "paper",
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
        },
    }


class RealtimeEngine:
    """Paper live session on Mac: closed M5 bars from Yahoo/demo, simulated fills."""

    def __init__(self) -> None:
        self.config_name = CONFIG_NAME
        self.case = "last_candles"
        self.timeframe = "m5"
        self.source = "stream"
        self.order_mode = "paper"
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
        live_last = False
        if self.last_bar_time:
            try:
                live_last = bar_is_live(datetime.fromisoformat(self.last_bar_time), datetime.now())
            except ValueError:
                live_last = False
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
            "last_tick": self.last_tick if live_last else None,
            "last_bar_time": self.last_bar_time if live_last else None,
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
            "lot": self.lot,
            "contracts": size_contracts(self.bank, self.lot),
            "max_contracts": int(self.max_contracts),
            "signal": _signal_dict(self.last_signal),
            "position": _position_dict(self.position),
            "trades": list(reversed(self.trades[-80:])),
            "equity": self.equity[-400:],
            "daily": daily_rows,
            "periods": periods,
            "signals": list(reversed(self.signals[-40:])),
            "candles": self.candles_tail if live_last else [],
            "wait_reason": self.wait_reason,
            "next_gold": next_gold_window(),
            "playbook": DEMO_PLAYBOOK if self.order_mode == "mt5" and self.wait_reason == "aguardando_login" else None,
            "mode": self.order_mode,
            "feed": dict(self.feed_info),
            "mt5": dict(self.mt5_info),
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
                    "label": "Enviar (MT5 demo)",
                    "ready": bool(self.mt5_info.get("ready")),
                    "mode": "mt5",
                },
                {
                    "key": "stream",
                    "label": "Simular (Yahoo / demo)",
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
        key = str(mode or "paper").strip().lower()
        if key not in {"paper", "mt5"}:
            raise ValueError("order_mode deve ser paper ou mt5")
        self.order_mode = key
        if key == "paper":
            self.source = "stream"
            self.error = None
            self._stream.last_closed_candles(self._stream.symbol, "m5", LOOKBACK_BARS, allow_file=False)
            self.feed_info = self._stream.status()
            self.wait_reason = "aguardando_candle"
        else:
            self.source = "mt5"
            self.wait_reason = "aguardando_login"

    def set_source(self, source: str) -> None:
        key = str(source or "stream").strip().lower()
        if key not in {"mt5", "stream"}:
            raise ValueError("source deve ser mt5 ou stream")
        if key == "stream":
            self.set_order_mode("paper")
            return
        self.set_order_mode("mt5")

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
        if raw.get("order_mode") in {"paper", "mt5"}:
            self.order_mode = str(raw["order_mode"])
        elif raw.get("source") == "stream":
            self.order_mode = "paper"
        if raw.get("source") in {"mt5", "stream"}:
            self.source = str(raw["source"])
        if self.order_mode == "paper":
            self.source = "stream"
        else:
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
        if len(kept) == len(self.trades) and (self.position is None or _on_day(self.position.get("time"))):
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
        self.mt5_info = {
            "ready": bool(acc.get("account") and acc.get("demo") and symbol),
            "demo": acc.get("demo"),
            "login": acc.get("login"),
            "server": acc.get("server"),
            "symbol": symbol,
            "filling": self._broker.filling if symbol else None,
            "trade_allowed": bool(term.get("trade_allowed")),
            "balance": acc.get("balance"),
            "equity": acc.get("equity"),
        }
        if not acc.get("account"):
            self.wait_reason = "aguardando_login"
            return None
        if acc.get("demo") is False:
            self.wait_reason = "conta_real"
            self.error = "Conta real no MT5. Troque para a demo para o robô enviar ordem."
            return self._broker
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

    def _record_close(self, ts: datetime, price: float, reason: str) -> None:
        assert self._bt is not None and self.position is not None
        trade = self._bt._close(self.position, ts, price, reason)
        packed = trade.to_dict()
        self.trades.append(packed)
        self.bank += trade.pnl
        self.day_pnl += trade.pnl
        self.peak = max(self.peak, self.bank)
        self.max_dd = max(self.max_dd, self.peak - self.bank)
        self.equity.append({"t": ts.isoformat(), "bank": round(self.bank, 2)})
        self.position = None

    def _sync_position(self, broker: Mt5Broker, now: datetime) -> None:
        live = broker.open_positions()
        if live:
            pos = live[0]
            if self.position is None:
                self.position = {
                    "side": pos["side"],
                    "entry": pos["entry"],
                    "stop": pos["stop"] or pos["entry"],
                    "take": pos["take"] or pos["entry"],
                    "time": pos["time"],
                    "hour": pos["time"].hour,
                    "extreme": pos["entry"],
                    "contracts": int(pos.get("volume") or self._n_contracts()),
                    "reason": "mt5",
                    "ticket": pos["ticket"],
                }
            else:
                self.position["ticket"] = pos["ticket"]
                if pos["stop"]:
                    self.position["stop"] = pos["stop"]
                if pos["take"]:
                    self.position["take"] = pos["take"]
            return
        if self.position is not None:
            ticket = self.position.get("ticket")
            deal = broker.closing_deal(int(ticket)) if ticket else None
            price = float(deal["price"]) if deal else float(self.position["entry"])
            ts = deal["time"] if deal else now
            reason = "mt5_sltp" if deal else "mt5_fechou"
            self._record_close(ts, price, reason)

    def _can_enter(self, now: datetime, *, live_mt5: bool) -> bool:
        if self._session is None:
            return False
        if self.position is not None:
            return False
        if not self._session.allows(now):
            return False
        if self._cfg and self.trades_today >= self._cfg.risk.max_trades_per_day:
            return False
        if live_mt5:
            if self.mt5_info.get("demo") is False:
                return False
            if not self.mt5_info.get("trade_allowed"):
                return False
            if not self.mt5_info.get("symbol"):
                return False
        return True

    def _n_contracts(self) -> int:
        n = size_contracts(self.bank, self.lot)
        self.max_contracts = max(int(self.max_contracts), n)
        return n

    def _send_signal(self, broker: Mt5Broker, sig: Signal, entry: float) -> None:
        assert self._cfg is not None and self._bt is not None
        if self.order_mode == "paper":
            raise RuntimeError("Modo simular não envia ordem.")
        if self.mt5_info.get("demo") is False:
            raise RuntimeError("Recusado: conta real.")
        n = self._n_contracts()
        stop, take = self._bt.risk.levels(sig.side, entry, None)
        order_sig = Signal(side=sig.side, entry=entry, stop=stop, take=take, reason=sig.reason, predicted=sig.predicted)
        result = broker.send(order_sig, float(n))
        retcode = int(result.get("retcode") or 0)
        if retcode not in {0, 10009, 10008}:
            raise RuntimeError(f"order_send retcode={retcode} {result.get('comment')}")
        ticket = result.get("order") or result.get("deal")
        now = datetime.now()
        self.position = {
            "side": sig.side,
            "entry": float(entry),
            "stop": float(stop),
            "take": float(take),
            "time": now,
            "hour": now.hour,
            "extreme": float(entry),
            "contracts": n,
            "reason": sig.reason,
            "ticket": ticket,
        }
        self.trades_today += 1

    def _open_paper(self, sig: Signal, entry: float, ts: datetime) -> None:
        assert self._bt is not None
        if self.source == "stream" and getattr(self._stream, "origin", None) == "file":
            return
        n = self._n_contracts()
        stop, take = self._bt.risk.levels(sig.side, entry, None)
        self.position = {
            "side": sig.side,
            "entry": float(entry),
            "stop": float(stop),
            "take": float(take),
            "time": ts,
            "hour": ts.hour,
            "extreme": float(entry),
            "contracts": n,
            "reason": sig.reason,
            "ticket": None,
        }
        self.trades_today += 1

    def _manage_open(self, broker: Mt5Broker | None, row: pd.Series, ts: datetime, now: datetime) -> None:
        assert self._bt is not None and self._session is not None and self.position is not None
        o = float(row["Abertura"])
        h = float(row["Máximo"])
        l = float(row["Mínimo"])
        c = float(row["Fechamento"])
        old_stop = float(self.position["stop"])
        exit_price, reason = self._bt._manage(self.position, o, h, l, c)
        new_stop = float(self.position["stop"])
        ticket = self.position.get("ticket")
        if broker is not None and ticket and abs(new_stop - old_stop) >= 1:
            try:
                broker.modify_sltp(int(ticket), new_stop, float(self.position["take"]))
            except Exception as exc:  # noqa: BLE001
                self.error = f"trailing: {exc}"
        if exit_price is None and (self._session.flatten_day(ts) or self._session.flatten_day(now)):
            exit_price, reason = c, "fim_da_sessao"
        if exit_price is None:
            return
        if broker is not None and ticket:
            try:
                broker.close_position(
                    int(ticket), self.position["side"], float(self.position.get("contracts") or self._n_contracts())
                )
            except Exception as exc:  # noqa: BLE001
                self.error = f"close: {exc}"
            self._sync_position(broker, now)
            if self.position is not None:
                self._record_close(ts, float(exit_price), reason)
            return
        self._record_close(ts, float(exit_price), reason)

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
        if self.order_mode == "paper" or self.source == "stream":
            self._tick_stream(now)
            return
        broker = self._ensure_broker()
        if broker is None:
            return
        key = now.date()
        if key != self.day_key:
            self.day_key = key
            self.day_pnl = 0.0
            self.trades_today = 0
        try:
            self._sync_position(broker, now)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
        try:
            candles = broker.last_closed_candles(broker.symbol, "m5", LOOKBACK_BARS)
            self._frame = frame_from_candles(candles, source_file="mt5")
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.wait_reason = session_wait_reason(
                connected=True,
                account=bool(self.mt5_info.get("login")),
                demo=self.mt5_info.get("demo"),
                symbol=self.mt5_info.get("symbol"),
                trade_allowed=bool(self.mt5_info.get("trade_allowed")),
                now=now,
                last_bar=None,
                in_position=self.position is not None,
                session=self._session,
            )
            return
        if self._frame is None or self._frame.empty:
            self.wait_reason = "aguardando_candle"
            return
        self.cursor = len(self._frame) - 1
        last = self._frame.iloc[-1]
        ts = pd.Timestamp(last["timestamp"]).to_pydatetime()
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.replace(tzinfo=None)
        self.last_bar_time = ts.isoformat()
        self._refresh_candles_tail()
        self.wait_reason = session_wait_reason(
            connected=True,
            account=True,
            demo=self.mt5_info.get("demo"),
            symbol=self.mt5_info.get("symbol"),
            trade_allowed=bool(self.mt5_info.get("trade_allowed")),
            now=now,
            last_bar=ts,
            in_position=self.position is not None,
            session=self._session,
        )
        if self.position is not None:
            self._manage_open(broker, last, ts, now)
        new_bar = self.last_bar_time != self.processed_bar
        if new_bar and self._can_enter(now, live_mt5=True) and bar_is_live(ts, now):
            history = self._frame
            sig = self._policy.from_candles(history)
            self.last_signal = sig
            self.signals.append({"t": ts.isoformat(), **(_signal_dict(sig) or {})})
            self.signals = self.signals[-200:]
            if sig.side is not Side.FLAT:
                try:
                    tick = broker._require().symbol_info_tick(broker.symbol)
                    entry = float(tick.ask if sig.side is Side.BUY else tick.bid) if tick else float(last["Fechamento"])
                    plan = planned_order(sig, entry, *RiskCalculator(self._cfg).levels(sig.side, entry, None))
                    if plan:
                        self._send_signal(broker, sig, entry)
                        self.error = None
                except Exception as exc:  # noqa: BLE001
                    self.error = str(exc)
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
                self._stream.symbol, "m5", LOOKBACK_BARS, allow_file=False, now=now
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
        live_bar = self._stream.origin in {"ingest", "http", "yahoo", "demo"} and bar_is_live(ts, now)
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
        if self.position is not None:
            self._manage_open(None, last, ts, now)
        new_bar = self.last_bar_time != self.processed_bar
        if new_bar and self._can_enter(now, live_mt5=False) and live_bar:
            sig = self._policy.from_candles(self._frame)
            self.last_signal = sig
            self.signals.append({"t": ts.isoformat(), **(_signal_dict(sig) or {})})
            self.signals = self.signals[-200:]
            if sig.side is not Side.FLAT:
                entry = float(last["Fechamento"])
                stop, take = RiskCalculator(self._cfg).levels(sig.side, entry, None)
                if planned_order(sig, entry, stop, take):
                    self._open_paper(sig, entry, ts)
                    self._manage_open(None, last, ts, now)
        if new_bar:
            self.processed_bar = self.last_bar_time
        self._ticks_since_save += 1
        if self._ticks_since_save >= 3:
            self._persist()
            self._ticks_since_save = 0

    async def run_loop(self) -> None:
        interval = max(1.0, float(self.interval_sec))
        try:
            while self.running:
                await asyncio.sleep(seconds_until_aligned(time.time(), interval))
                if not self.running:
                    break
                clock = datetime.now()
                before = self.processed_bar
                async with self._lock:
                    await asyncio.to_thread(self.tick)
                if self.running and is_m5_close_slot(clock) and self.processed_bar == before:
                    await asyncio.sleep(CATCHUP_SEC)
                    if not self.running:
                        break
                    async with self._lock:
                        await asyncio.to_thread(self.tick)
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
            return self.snapshot()
        self.running = True
        self.done = False
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self.run_loop())
        return self.snapshot()

    async def start(self, source: str | None = None, order_mode: str | None = None) -> dict[str, Any]:
        del source, order_mode
        self.set_order_mode("paper")
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
