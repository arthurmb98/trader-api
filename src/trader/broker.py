from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from trader.domain import Candle, Side, Signal
from trader.mt5_session import SymbolCandidate
from trader.ports import Broker


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

    def connect(
        self,
        select_symbol: bool = True,
        login: str | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
    ) -> None:
        if self._mt5 is not None and self._mt5.account_info() is not None:
            if select_symbol:
                self._select_symbol()
            return
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise RuntimeError(
                "O pacote MetaTrader5 não está disponível neste sistema. "
                "A API e os estudos funcionam sem ele. Para enviar ordens, use Windows com o terminal MT5 aberto."
            ) from exc
        kwargs: dict[str, Any] = {"timeout": 10_000}
        if path:
            kwargs["path"] = path
        elif not login:
            for candidate in (
                r"C:\Program Files\MetaTrader 5\terminal64.exe",
                r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            ):
                if Path(candidate).exists():
                    kwargs["path"] = candidate
                    break
        if login and password and server:
            kwargs["login"] = int(login)
            kwargs["password"] = password
            kwargs["server"] = server
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize falhou: {mt5.last_error()}")
        self._mt5 = mt5
        if select_symbol:
            self._select_symbol()

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
        return info is not None

    def apply_symbol_filling(self) -> None:
        mt5 = self._require()
        info = mt5.symbol_info(self.symbol)
        if info is None:
            return
        mode = int(getattr(info, "filling_mode", 0) or 0)
        if mode & mt5.SYMBOL_FILLING_RETURN:
            self.filling = "RETURN"
        elif mode & mt5.SYMBOL_FILLING_IOC:
            self.filling = "IOC"
        elif mode & mt5.SYMBOL_FILLING_FOK:
            self.filling = "FOK"

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
            }
        trade_mode = int(getattr(acc, "trade_mode", -1))
        return {
            "account": True,
            "demo": trade_mode == 0,
            "login": int(acc.login),
            "server": str(acc.server),
            "name": str(acc.name),
            "balance": float(acc.balance),
            "equity": float(acc.equity),
            "trade_mode": trade_mode,
            "leverage": int(getattr(acc, "leverage", 0) or 0),
        }

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
                    timestamp=datetime.fromtimestamp(int(row[0])),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        return candles

    def last_closed_candles(self, symbol: str, timeframe: str, count: int) -> list[Candle]:
        mt5 = self._require()
        rates = mt5.copy_rates_from_pos(symbol, self._timeframe(timeframe), 1, max(count, 1))
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates falhou: {mt5.last_error()}")
        return self._rows_to_candles(symbol, rates)

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

    def _filling_const(self) -> int:
        mt5 = self._require()
        mapping = {
            "FOK": mt5.ORDER_FILLING_FOK,
            "IOC": mt5.ORDER_FILLING_IOC,
            "RETURN": mt5.ORDER_FILLING_RETURN,
        }
        return mapping.get(self.filling, mt5.ORDER_FILLING_IOC)

    def _deal_request(self, signal: Signal, volume: float) -> dict[str, Any]:
        mt5 = self._require()
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError("Sem tick do símbolo.")
        is_buy = signal.side is Side.BUY
        price = float(tick.ask if is_buy else tick.bid)
        stop = float(signal.stop) if signal.stop else 0.0
        take = float(signal.take) if signal.take else 0.0
        if stop <= 0 or take <= 0:
            dist = 100.0
            stop = price - dist if is_buy else price + dist
            take = price + 200.0 if is_buy else price - 200.0
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": float(stop),
            "tp": float(take),
            "deviation": int(self.deviation),
            "magic": int(self.magic),
            "comment": self.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_const(),
        }

    def check_order(self, signal: Signal, volume: float, skip_levels: bool = False) -> dict[str, Any]:
        del skip_levels
        mt5 = self._require()
        request = self._deal_request(signal, volume)
        check = mt5.order_check(request)
        if check is None:
            return {"ok": False, "error": str(mt5.last_error()), "request": _safe_request(request)}
        packed = check._asdict() if hasattr(check, "_asdict") else {"retcode": getattr(check, "retcode", None)}
        retcode = packed.get("retcode")
        return {
            "ok": retcode in {0, mt5.TRADE_RETCODE_DONE} or int(retcode or -1) == 0,
            "retcode": retcode,
            "comment": packed.get("comment"),
            "request": _safe_request(request),
        }

    def send(self, signal: Signal, volume: float) -> dict[str, Any]:
        mt5 = self._require()
        request = self._deal_request(signal, volume)
        check = mt5.order_check(request)
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"order_send falhou: {mt5.last_error()}")
        payload = result._asdict()
        payload["check"] = None if check is None else str(check)
        payload["request"] = _safe_request(request)
        return payload

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
                    "time": datetime.fromtimestamp(int(pos.time)),
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
            "time": datetime.fromtimestamp(int(deal.time)),
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
