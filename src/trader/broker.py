from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trader.domain import Candle, Side, Signal
from trader.mt5_session import DEMO_SERVERS, SymbolCandidate, enable_algo_trading, server_looks_demo
from trader.ports import Broker


def _mt5_time(stamp: Any) -> datetime:
    """MT5 unix times are server clock encoded as UTC, not the Windows local zone."""
    return datetime.fromtimestamp(int(stamp), tz=timezone.utc).replace(tzinfo=None)


class BrokerFactory:
    @staticmethod
    def create(config) -> Mt5Broker:
        mt5 = config.mt5
        return Mt5Broker(mt5.symbol, mt5.magic, mt5.deviation, mt5.filling, mt5.comment)


class Mt5Broker(Broker):
    """Live MetaTrader 5 adapter. Optional: the package only exists on Windows with MT5 open."""

    def __init__(self, symbol: str, magic: int, deviation: int, filling: str, comment: str) -> None:
        self.symbol = symbol
        self.magic = magic
        self.deviation = deviation
        self.filling = filling.upper()
        self.comment = comment
        self._mt5 = None
        self._last_at_nudge = 0.0

    def connect(
        self,
        select_symbol: bool = True,
        login: str | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
    ) -> None:
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise RuntimeError(
                "O pacote MetaTrader5 não está disponível neste sistema. "
                "A API e os estudos funcionam sem ele. Para enviar ordens, use Windows com o terminal MT5 aberto."
            ) from exc
        if self._mt5 is not None and self._mt5.account_info() is not None:
            if select_symbol:
                self._select_symbol()
            return
        kwargs: dict[str, Any] = {"timeout": 20_000}
        if path:
            kwargs["path"] = path
        else:
            for candidate in (
                r"C:\Program Files\MetaTrader 5\terminal64.exe",
                r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            ):
                if Path(candidate).exists():
                    kwargs["path"] = candidate
                    break
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize falhou: {mt5.last_error()}")
        self._mt5 = mt5
        acc = mt5.account_info()
        if acc is None and login and password:
            servers: list[str] = []
            hint = (server or "").strip()
            if hint and server_looks_demo(hint):
                servers.append(hint)
            servers.extend(DEMO_SERVERS)
            seen: set[str] = set()
            last_err = None
            for name in servers:
                key = (name or "").strip()
                if not key or key in seen or not server_looks_demo(key):
                    continue
                seen.add(key)
                if mt5.login(int(login), password=password, server=key):
                    last_err = None
                    break
                last_err = mt5.last_error()
            if last_err is not None and mt5.account_info() is None:
                raise RuntimeError(
                    f"MT5 login demo falhou: {last_err}. "
                    "Conta real no terminal: o ao vivo simula paper e não envia ordem."
                )
        if mt5.account_info() is None:
            raise RuntimeError(
                f"MT5 sem conta logada: {mt5.last_error()}. "
                "Abra o MetaTrader da Genial e deixe a janela aberta. Conta real só simula."
            )
        if select_symbol:
            self._select_symbol()

    def try_demo_login(self, login: str, password: str) -> dict[str, Any]:
        """Attempt demo servers only. Never logs into PRD/real. May leave the current session as-is on failure."""
        mt5 = self._require()
        tried: list[dict[str, Any]] = []
        for server in DEMO_SERVERS:
            ok = bool(mt5.login(int(login), password=password, server=server))
            acc = mt5.account_info()
            err = mt5.last_error() if not ok else None
            demo = acc is not None and int(getattr(acc, "trade_mode", -1)) == 0
            tried.append(
                {
                    "server": server,
                    "ok": ok,
                    "demo": demo,
                    "login": int(acc.login) if acc is not None else None,
                    "trade_mode": int(getattr(acc, "trade_mode", -1)) if acc is not None else None,
                    "err": str(err) if err else None,
                }
            )
            if demo:
                return {"switched": True, "server": server, "tried": tried}
            if acc is None:
                return {"switched": False, "lost_session": True, "tried": tried}
        acc = mt5.account_info()
        return {
            "switched": False,
            "lost_session": acc is None,
            "server": str(acc.server) if acc is not None else None,
            "tried": tried,
        }

    def _require(self):
        if self._mt5 is None:
            raise RuntimeError("MT5 desconectado. Chame connect().")
        return self._mt5

    def _select_symbol(self) -> None:
        mt5 = self._require()
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"Símbolo {self.symbol} não encontrado no Market Watch.")
        self.apply_symbol_filling()

    def use_symbol(self, symbol: str) -> None:
        self.symbol = symbol
        self._select_symbol()

    def has_symbol(self, symbol: str) -> bool:
        mt5 = self._require()
        info = mt5.symbol_info(symbol)
        if info is not None:
            return True
        if not mt5.symbol_select(symbol, True):
            return False
        return mt5.symbol_info(symbol) is not None

    def apply_symbol_filling(self) -> None:
        mt5 = self._require()
        info = mt5.symbol_info(self.symbol)
        if info is None:
            return
        mode = int(getattr(info, "filling_mode", 0) or 0)
        has_fok = bool(mode & int(getattr(mt5, "SYMBOL_FILLING_FOK", 1)))
        has_ioc = bool(mode & int(getattr(mt5, "SYMBOL_FILLING_IOC", 2)))
        has_return = bool(mode & int(getattr(mt5, "SYMBOL_FILLING_RETURN", 4)))
        self.filling = filling_from_mode(mode, has_return, has_ioc, has_fok)

    def list_win_symbols(self) -> list[SymbolCandidate]:
        mt5 = self._require()
        rows = mt5.symbols_get()
        if not rows:
            return []
        out: list[SymbolCandidate] = []
        for row in rows:
            name = str(row.name)
            if not name.upper().startswith("WIN"):
                continue
            out.append(
                SymbolCandidate(
                    name=name,
                    trade_mode=int(getattr(row, "trade_mode", 0) or 0),
                    volume=float(getattr(row, "volume", 0) or getattr(row, "session_volume", 0) or 0),
                )
            )
        return out

    def account_payload(self) -> dict[str, Any]:
        mt5 = self._require()
        acc = mt5.account_info()
        if acc is None:
            return {
                "account": False,
                "demo": None,
                "login": None,
                "server": None,
                "name": None,
                "balance": None,
                "equity": None,
                "credit": None,
                "profit": None,
                "margin_free": None,
                "bank": None,
            }
        trade_mode = int(getattr(acc, "trade_mode", -1))
        equity = float(getattr(acc, "equity", 0) or 0)
        balance = float(getattr(acc, "balance", 0) or 0)
        credit = float(getattr(acc, "credit", 0) or 0)
        profit = float(getattr(acc, "profit", 0) or 0)
        margin_free = float(getattr(acc, "margin_free", 0) or 0)
        bank = equity if equity else balance
        if not bank:
            bank = margin_free if margin_free else credit
        return {
            "account": True,
            "demo": trade_mode == 0,
            "login": int(acc.login),
            "server": str(acc.server),
            "name": str(acc.name),
            "balance": balance,
            "equity": equity,
            "credit": credit,
            "profit": profit,
            "margin_free": margin_free,
            "bank": float(bank),
            "trade_mode": trade_mode,
            "leverage": int(getattr(acc, "leverage", 0) or 0),
        }

    def ensure_algo_trading(self) -> dict[str, Any]:
        term = self.terminal_payload()
        if term.get("trade_allowed"):
            return {"ok": True, "already": True, "trade_allowed": True}
        import time

        now = time.monotonic()
        if now - self._last_at_nudge < 5:
            return {"ok": False, "reason": "cooldown", "trade_allowed": False}
        self._last_at_nudge = now
        result = enable_algo_trading()
        result["trade_allowed"] = bool(self.terminal_payload().get("trade_allowed"))
        return result

    def terminal_payload(self) -> dict[str, Any]:
        mt5 = self._require()
        term = mt5.terminal_info()
        if term is None:
            return {"trade_allowed": False, "connected": False}
        return {
            "trade_allowed": bool(term.trade_allowed),
            "connected": bool(term.connected),
            "name": str(getattr(term, "name", "") or ""),
        }

    def is_demo(self) -> bool:
        return bool(self.account_payload().get("demo"))

    def _timeframe(self, timeframe: str) -> int:
        mt5 = self._require()
        return mt5.TIMEFRAME_M5 if timeframe.lower() in {"m5", "5", "5min"} else mt5.TIMEFRAME_M1

    def _rows_to_candles(self, symbol: str, rates) -> list[Candle]:
        candles: list[Candle] = []
        for row in rates:
            candles.append(
                Candle(
                    symbol=symbol,
                    timestamp=_mt5_time(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        return candles

    def quote(self) -> dict[str, Any] | None:
        mt5 = self._require()
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        last = float(getattr(tick, "last", 0) or 0)
        bid = float(getattr(tick, "bid", 0) or 0)
        ask = float(getattr(tick, "ask", 0) or 0)
        mark = last or bid or ask
        if mark <= 0:
            return None
        stamp = getattr(tick, "time", None)
        msc = getattr(tick, "time_msc", None)
        return {
            "bid": bid or mark,
            "ask": ask or mark,
            "last": mark,
            "time": _mt5_time(stamp).isoformat() if stamp else None,
            "time_msc": int(msc) if msc else None,
        }

    def last_closed_candles(self, symbol: str, timeframe: str, count: int) -> list[Candle]:
        import time

        mt5 = self._require()
        mt5.symbol_select(symbol, True)
        tf = self._timeframe(timeframe)
        need = max(count, 1)
        rates = None
        last_err: Any = None
        for start in (1, 0):
            for _ in range(4):
                rates = mt5.copy_rates_from_pos(symbol, tf, start, need)
                if rates is not None and len(rates) > 0:
                    return self._rows_to_candles(symbol, rates)
                last_err = mt5.last_error()
                time.sleep(0.4)
        raise RuntimeError(f"copy_rates falhou: {last_err}")

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> list[Candle]:
        mt5 = self._require()
        rates = mt5.copy_rates_range(symbol, self._timeframe(timeframe), date_from, date_to)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_range falhou: {mt5.last_error()}")
        return self._rows_to_candles(symbol, rates)

    def copy_ticks_range(
        self,
        date_from: datetime,
        date_to: datetime,
        symbol: str | None = None,
    ) -> list[tuple[datetime, float]]:
        mt5 = self._require()
        name = symbol or self.symbol
        flags = int(getattr(mt5, "COPY_TICKS_ALL", 0))
        rows = mt5.copy_ticks_range(name, date_from, date_to, flags)
        if rows is None or len(rows) == 0:
            return []
        out: list[tuple[datetime, float]] = []
        for row in rows:
            last = float(row["last"] or 0)
            bid = float(row["bid"] or 0)
            ask = float(row["ask"] or 0)
            px = last or bid or ask
            if px <= 0:
                continue
            msc = int(row["time_msc"] or 0)
            if msc > 0:
                ts = datetime.fromtimestamp(msc / 1000.0)
            else:
                ts = _mt5_time(row["time"])
            out.append((ts, px))
        return out

    def ticks_since(self, from_msc: int, count: int = 256) -> list[tuple[datetime, float]]:
        mt5 = self._require()
        start = datetime.fromtimestamp(max(0, int(from_msc)) / 1000.0)
        flags = int(getattr(mt5, "COPY_TICKS_ALL", 0))
        rows = mt5.copy_ticks_from(self.symbol, start, max(1, int(count)), flags)
        if rows is None or len(rows) == 0:
            return []
        out: list[tuple[datetime, float]] = []
        for row in rows:
            msc = int(row["time_msc"] or 0)
            if msc <= from_msc:
                continue
            last = float(row["last"] or 0)
            bid = float(row["bid"] or 0)
            ask = float(row["ask"] or 0)
            px = last or bid or ask
            if px <= 0:
                continue
            ts = datetime.fromtimestamp(msc / 1000.0) if msc else _mt5_time(row["time"])
            out.append((ts, px))
        return out

    def _filling_const(self) -> int:
        mt5 = self._require()
        mapping = {
            "FOK": mt5.ORDER_FILLING_FOK,
            "IOC": mt5.ORDER_FILLING_IOC,
            "RETURN": mt5.ORDER_FILLING_RETURN,
        }
        return mapping.get(self.filling, mt5.ORDER_FILLING_IOC)

    def _pending_filling_const(self) -> int:
        mt5 = self._require()
        return int(getattr(mt5, "ORDER_FILLING_RETURN", 2))

    def _pending_request(self, signal: Signal, volume: float) -> dict[str, Any]:
        mt5 = self._require()
        is_buy = signal.side is Side.BUY
        price = float(signal.entry)
        stop = float(signal.stop) if signal.stop else 0.0
        take = float(signal.take) if signal.take else 0.0
        if stop <= 0 or take <= 0:
            dist = 100.0
            stop = price - dist if is_buy else price + dist
            take = price + 200.0 if is_buy else price - 200.0
        return {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY_LIMIT if is_buy else mt5.ORDER_TYPE_SELL_LIMIT,
            "price": price,
            "sl": float(stop),
            "tp": float(take),
            "magic": int(self.magic),
            "comment": f"{self.comment} limit",
            "type_time": int(getattr(mt5, "ORDER_TIME_DAY", 1)),
            "type_filling": self._pending_filling_const(),
        }

    def check_order(self, signal: Signal, volume: float, skip_levels: bool = False) -> dict[str, Any]:
        del skip_levels
        mt5 = self._require()
        request = self._pending_request(signal, volume)
        check = mt5.order_check(request)
        if check is None:
            return {"ok": False, "error": str(mt5.last_error()), "request": _safe_request(request)}
        packed = check._asdict() if hasattr(check, "_asdict") else {"retcode": getattr(check, "retcode", None)}
        retcode = packed.get("retcode")
        return {
            "ok": retcode in {0, mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED} or int(retcode or -1) in {0, 10008, 10009},
            "retcode": retcode,
            "comment": packed.get("comment"),
            "request": _safe_request(request),
        }

    def send(self, signal: Signal, volume: float) -> dict[str, Any]:
        mt5 = self._require()
        request = self._pending_request(signal, volume)
        check = mt5.order_check(request)
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"order_send falhou: {mt5.last_error()}")
        payload = result._asdict()
        payload["check"] = None if check is None else str(check)
        payload["request"] = _safe_request(request)
        return payload

    def open_orders(self) -> list[dict[str, Any]]:
        mt5 = self._require()
        rows = mt5.orders_get(symbol=self.symbol)
        if not rows:
            return []
        buy_limit = int(mt5.ORDER_TYPE_BUY_LIMIT)
        sell_limit = int(mt5.ORDER_TYPE_SELL_LIMIT)
        out: list[dict[str, Any]] = []
        for row in rows:
            if int(row.magic) != int(self.magic):
                continue
            kind = int(row.type)
            if kind == buy_limit:
                side = Side.BUY
            elif kind == sell_limit:
                side = Side.SELL
            else:
                continue
            out.append(
                {
                    "ticket": int(row.ticket),
                    "side": side,
                    "entry": float(row.price_open),
                    "stop": float(getattr(row, "sl", 0) or 0),
                    "take": float(getattr(row, "tp", 0) or 0),
                    "volume": float(row.volume_current or row.volume_initial),
                    "time": _mt5_time(row.time_setup),
                }
            )
        return out

    def cancel_order(self, ticket: int) -> dict[str, Any]:
        mt5 = self._require()
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(ticket),
            "magic": int(self.magic),
        }
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"cancel falhou: {mt5.last_error()}")
        return result._asdict()

    def modify_sltp(self, ticket: int, sl: float, tp: float) -> dict[str, Any]:
        mt5 = self._require()
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": int(ticket),
            "sl": float(sl),
            "tp": float(tp),
            "magic": int(self.magic),
        }
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"modify SL/TP falhou: {mt5.last_error()}")
        return result._asdict()

    def open_positions(self) -> list[dict[str, Any]]:
        mt5 = self._require()
        rows = mt5.positions_get(symbol=self.symbol)
        if not rows:
            return []
        out: list[dict[str, Any]] = []
        for pos in rows:
            if int(pos.magic) != int(self.magic):
                continue
            side = Side.BUY if pos.type == mt5.POSITION_TYPE_BUY else Side.SELL
            out.append(
                {
                    "ticket": int(pos.ticket),
                    "side": side,
                    "entry": float(pos.price_open),
                    "stop": float(pos.sl),
                    "take": float(pos.tp),
                    "volume": float(pos.volume),
                    "time": _mt5_time(pos.time),
                    "profit": float(getattr(pos, "profit", 0) or 0),
                }
            )
        return out

    def closing_deal(self, ticket: int) -> dict[str, Any] | None:
        mt5 = self._require()
        now = datetime.now()
        deals = mt5.history_deals_get(now - timedelta(days=7), now + timedelta(hours=2))
        if not deals:
            return None
        closes = [
            deal
            for deal in deals
            if int(deal.position_id) == int(ticket) and int(deal.entry) == int(mt5.DEAL_ENTRY_OUT)
        ]
        if not closes:
            return None
        deal = closes[-1]
        return {
            "price": float(deal.price),
            "time": _mt5_time(deal.time),
            "profit": float(deal.profit),
        }

    def close_position(self, ticket: int, side: Side, volume: float) -> dict[str, Any]:
        mt5 = self._require()
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError("Sem tick do símbolo.")
        is_buy_close = side is Side.SELL
        price = float(tick.ask if is_buy_close else tick.bid)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if is_buy_close else mt5.ORDER_TYPE_SELL,
            "position": int(ticket),
            "price": price,
            "deviation": int(self.deviation),
            "magic": int(self.magic),
            "comment": f"{self.comment} close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_const(),
        }
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"close falhou: {mt5.last_error()}")
        return result._asdict()

    def shutdown(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None


def _safe_request(request: dict[str, Any]) -> dict[str, Any]:
    return {key: request[key] for key in request}


def filling_from_mode(mode: int, has_return: bool, has_ioc: bool, has_fok: bool) -> str:
    del mode
    if has_return:
        return "RETURN"
    if has_ioc:
        return "IOC"
    if has_fok:
        return "FOK"
    return "IOC"
