from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

from trader.backtest import SessionFilter
from trader.domain import Side, Signal

WIN_MONTH_CODE = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}
WIN_NAME_RE = re.compile(r"^WIN(\$|FUT|[FGHJKMNQUVXZ]\d{2})$")
TRADE_MODE_FULL = 4
ACCOUNT_TRADE_MODE_DEMO = 0
LOOKBACK_BARS = 80
DEFAULT_MAGIC = 20260818
MT5_WMCMD_EXPERTS = 32851
WM_COMMAND = 0x0111
DEMO_SERVERS = (
    "GENIAL-DEMO",
    "Genial Investimentos-Demo",
    "GenialInvestimentos-Demo",
    "GenialInvestimentos-DEMO",
    "CLEAR-DEMO",
    "CLEAR CTVM-DEMO",
    "Clear CTVM-Demo",
)

DEMO_PLAYBOOK = """WIN na B3: deixe o MT5 da Genial aberto e já logado na DEMO.

  1. Terminal Genial (ou Clear) logado — canto inferior direito com o número da conta demo.
  2. Ferramentas -> Opções -> Expert Advisors: permitir Algo Trading. AutoTrading verde.
  3. Market Watch: vencimento da frente (WINV26 em ago/2026) e WIN$ se existir.
     Abra um gráfico M5 desse contrato.
  4. Python e MT5 no mesmo usuário Windows (os dois sem 'Executar como administrador').
  5. Recarregue http://127.0.0.1:5173/ao-vivo. Enviar manda ordem só na demo.

Login e senha no .env são opcionais se o terminal já está logado. Não commite. Conta real é recusada.
"""


@dataclass(frozen=True)
class SymbolCandidate:
    name: str
    trade_mode: int
    volume: float = 0.0


def front_win_contract(today: date | None = None) -> str:
    """B3 WIN expires around the 15th of even months. After that, roll to the next even month."""
    day = today or date.today()
    year, month, dom = day.year, day.month, day.day
    if month % 2 == 0 and dom <= 15:
        code = WIN_MONTH_CODE[month]
    else:
        month += 2 - (month % 2)
        if month > 12:
            month -= 12
            year += 1
        code = WIN_MONTH_CODE[month]
    return f"WIN{code}{year % 100:02d}"


def server_looks_demo(name: str) -> bool:
    key = (name or "").upper()
    if not key or "PRD" in key or "REAL" in key:
        return False
    return "DEMO" in key


def preferred_win_symbols(today: date | None = None) -> list[str]:
    front = front_win_contract(today)
    extras = ["WINFUT", "WIN$"]
    seen = {front}
    ordered = [front]
    for name in extras:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def is_win_symbol(name: str) -> bool:
    return bool(WIN_NAME_RE.match((name or "").upper()))


def pick_win_symbol(candidates: list[SymbolCandidate], today: date | None = None) -> str | None:
    tradeable = [c for c in candidates if c.trade_mode == TRADE_MODE_FULL and is_win_symbol(c.name)]
    if not tradeable:
        tradeable = [c for c in candidates if is_win_symbol(c.name)]
    if not tradeable:
        return None
    rank = {name: i for i, name in enumerate(preferred_win_symbols(today))}
    tradeable.sort(key=lambda c: (rank.get(c.name.upper(), 99), -c.volume, c.name))
    return tradeable[0].name


def _load_dotenv() -> None:
    from trader.paths import ROOT

    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and value and not (os.environ.get(key) or "").strip():
            os.environ[key] = value


def env_credentials() -> dict[str, str]:
    _load_dotenv()
    login = (os.environ.get("MT5_LOGIN") or "").strip()
    password = (os.environ.get("MT5_PASSWORD") or "").strip()
    server = (os.environ.get("MT5_SERVER") or "").strip()
    path = (os.environ.get("MT5_PATH") or "").strip()
    out: dict[str, str] = {}
    if login:
        out["login"] = login
    if password:
        out["password"] = password
    if server:
        out["server"] = server
    if path:
        out["path"] = path
    return out


def next_gold_window(now: datetime | None = None) -> str | None:
    clock = now or datetime.now()
    windows = [(time(9, 15), time(11, 0)), (time(14, 30), time(17, 0))]
    weekday = clock.weekday()
    if weekday >= 5:
        days = 7 - weekday
        nxt = datetime.combine(clock.date() + timedelta(days=days), time(9, 15))
        return nxt.isoformat(timespec="minutes")
    for start, end in windows:
        if start <= clock.time() <= end:
            return None
        if clock.time() < start:
            return datetime.combine(clock.date(), start).isoformat(timespec="minutes")
    nxt = datetime.combine(clock.date() + timedelta(days=1), time(9, 15))
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.isoformat(timespec="minutes")


def session_wait_reason(
    *,
    connected: bool,
    account: bool,
    demo: bool | None,
    symbol: str | None,
    trade_allowed: bool,
    now: datetime,
    last_bar: datetime | None,
    in_position: bool,
    session: SessionFilter,
) -> str:
    if not connected:
        return "aguardando_login"
    if not account:
        return "aguardando_login"
    if demo is False:
        return "conta_real"
    if not symbol:
        return "sem_simbolo"
    if not trade_allowed:
        return "autotrading_desligado"
    if now.weekday() >= 5:
        return "mercado_fechado"
    if now.time() < time(9, 0) or now.time() >= time(18, 25):
        return "mercado_fechado"
    if session.flatten_day(now):
        return "fim_da_sessao"
    if in_position:
        return "em_posicao"
    if not session.allows(now):
        return "fora_do_ouro"
    if last_bar is None:
        return "aguardando_candle"
    if not bar_is_fresh(last_bar, now):
        return "aguardando_candle"
    return "pronto"


def bar_is_fresh(bar_ts: datetime, now: datetime | None = None, max_age_sec: int = 20 * 60) -> bool:
    clock = now or datetime.now()
    if getattr(bar_ts, "tzinfo", None) is not None:
        bar_ts = bar_ts.replace(tzinfo=None)
    if getattr(clock, "tzinfo", None) is not None:
        clock = clock.replace(tzinfo=None)
    age = (clock - bar_ts).total_seconds()
    return 0 <= age <= max_age_sec


def bar_is_today(bar_ts: datetime, now: datetime | None = None) -> bool:
    clock = now or datetime.now()
    if getattr(bar_ts, "tzinfo", None) is not None:
        bar_ts = bar_ts.replace(tzinfo=None)
    if getattr(clock, "tzinfo", None) is not None:
        clock = clock.replace(tzinfo=None)
    return bar_ts.date() == clock.date()


def bar_is_live(bar_ts: datetime, now: datetime | None = None, max_age_sec: int = 20 * 60) -> bool:
    return bar_is_today(bar_ts, now) and bar_is_fresh(bar_ts, now, max_age_sec)


def enable_algo_trading() -> dict[str, Any]:
    """Turn AutoTrading on via the MT5 toolbar command. No-op if the window is missing."""
    if sys.platform != "win32":
        return {"ok": False, "reason": "windows_only", "n": 0}
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_cb(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        title = ctypes.create_unicode_buffer(512)
        klass = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, title, 512)
        user32.GetClassNameW(hwnd, klass, 256)
        text = title.value or ""
        cls = klass.value or ""
        hit = (
            "MetaQuotes::MetaTrader" in cls
            or "MetaTrader 5" in text
            or "MetaTrader5" in text
            or ("Genial" in text and "Trader" in text)
            or "GenialInvestimentos" in text
        )
        if hit:
            found.append(int(hwnd))
        return True

    user32.EnumWindows(enum_cb, 0)
    if not found:
        return {"ok": False, "reason": "janela_mt5_nao_encontrada", "n": 0}
    for hwnd in found:
        user32.PostMessageW(hwnd, WM_COMMAND, MT5_WMCMD_EXPERTS, 0)
    return {"ok": True, "reason": "nudge", "n": len(found)}


def planned_order(signal: Signal, entry: float, stop: float, take: float, volume: float = 1.0) -> dict[str, Any] | None:
    if signal.side is Side.FLAT:
        return None
    return {
        "side": signal.side.value,
        "entry": float(entry),
        "stop": float(stop),
        "take": float(take),
        "volume": float(volume),
        "reason": signal.reason,
    }


def probe() -> dict[str, Any]:
    from trader.broker import Mt5Broker

    creds = env_credentials()
    broker = Mt5Broker("WIN$", DEFAULT_MAGIC, 20, "IOC", "trader-api")
    payload: dict[str, Any] = {
        "ok": False,
        "connected": False,
        "account": False,
        "demo": None,
        "login": None,
        "server": None,
        "name": None,
        "balance": None,
        "equity": None,
        "symbol": None,
        "filling": None,
        "trade_allowed": False,
        "last_bar": None,
        "wait_reason": "aguardando_login",
        "next_gold": next_gold_window(),
        "playbook": DEMO_PLAYBOOK,
        "error": None,
        "order_check": None,
    }
    try:
        broker.connect(select_symbol=False, **creds)
    except Exception as exc:  # noqa: BLE001
        payload["error"] = str(exc)
        payload["wait_reason"] = "aguardando_login"
        return payload
    payload["connected"] = True
    try:
        acc = broker.account_payload()
        term = broker.terminal_payload()
        payload.update(acc)
        payload["trade_allowed"] = bool(term.get("trade_allowed"))
        symbol = resolve_symbol(broker)
        payload["symbol"] = symbol
        if symbol:
            broker.use_symbol(symbol)
            payload["filling"] = broker.filling
            try:
                candles = broker.last_closed_candles(symbol, "m5", 3)
                if candles:
                    payload["last_bar"] = candles[-1].timestamp.isoformat()
            except Exception as exc:  # noqa: BLE001
                payload["error"] = str(exc)
            try:
                payload["order_check"] = broker.check_order(
                    Signal(side=Side.BUY, entry=0, stop=0, take=0, reason="mt5-check"),
                    1.0,
                    skip_levels=True,
                )
            except Exception as exc:  # noqa: BLE001
                payload["order_check"] = {"ok": False, "error": str(exc)}
        from trader.replay import load_named_config

        cfg = load_named_config("best_candles_m5_1000_a")
        last_bar = None
        if payload["last_bar"]:
            last_bar = datetime.fromisoformat(payload["last_bar"])
        payload["wait_reason"] = session_wait_reason(
            connected=True,
            account=bool(payload["account"]),
            demo=payload["demo"],
            symbol=symbol,
            trade_allowed=payload["trade_allowed"],
            now=datetime.now(),
            last_bar=last_bar,
            in_position=False,
            session=SessionFilter(cfg),
        )
        payload["ok"] = bool(payload["account"] and payload["demo"] and symbol)
        payload["next_gold"] = next_gold_window()
    finally:
        broker.shutdown()
    return payload


def resolve_symbol(broker: Any) -> str | None:
    raw = broker.list_win_symbols()
    picked = pick_win_symbol(raw)
    if picked:
        return picked
    for name in preferred_win_symbols():
        if broker.has_symbol(name):
            return name
    return None


def _safe_print(text: str) -> None:
    stream = getattr(sys.stdout, "buffer", None)
    payload = (text + "\n").encode("utf-8", "replace")
    if stream is not None:
        stream.write(payload)
        stream.flush()
        return
    print(text.encode(sys.stdout.encoding or "utf-8", "replace").decode(sys.stdout.encoding or "utf-8", "replace"))


def mt5_check() -> int:
    report = probe()
    _safe_print(json.dumps({k: v for k, v in report.items() if k != "playbook"}, ensure_ascii=True, indent=2))
    _safe_print("")
    if not report.get("ok"):
        _safe_print(DEMO_PLAYBOOK)
        return 1
    _safe_print("MT5 demo pronto. O motor ao vivo pode anexar a este terminal.")
    return 0
