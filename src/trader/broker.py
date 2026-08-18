from __future__ import annotations

from datetime import datetime
from typing import Any

from trader.domain import Candle, Side, Signal
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

    def connect(self) -> None:
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise RuntimeError(
                "O pacote MetaTrader5 não está disponível neste sistema. "
                "A API e os estudos funcionam sem ele. Para enviar ordens, use Windows com o terminal MT5 aberto."
            ) from exc
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize falhou: {mt5.last_error()}")
        if not mt5.symbol_select(self.symbol, True):
            mt5.shutdown()
            raise RuntimeError(f"Símbolo {self.symbol} não encontrado no Market Watch.")
        self._mt5 = mt5

    def _require(self):
        if self._mt5 is None:
            raise RuntimeError("MT5 desconectado. Chame connect().")
        return self._mt5

    def _timeframe(self, timeframe: str) -> int:
        mt5 = self._require()
        return mt5.TIMEFRAME_M5 if timeframe.lower() in {"m5", "5", "5min"} else mt5.TIMEFRAME_M1

    def last_closed_candles(self, symbol: str, timeframe: str, count: int) -> list[Candle]:
        mt5 = self._require()
        rates = mt5.copy_rates_from_pos(symbol, self._timeframe(timeframe), 1, max(count, 1))
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates falhou: {mt5.last_error()}")
        candles: list[Candle] = []
        for row in rates:
            # MT5: time, open, high, low, close, tick_volume, spread, real_volume
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

    def _filling_const(self) -> int:
        mt5 = self._require()
        mapping = {
            "FOK": mt5.ORDER_FILLING_FOK,
            "IOC": mt5.ORDER_FILLING_IOC,
            "RETURN": mt5.ORDER_FILLING_RETURN,
        }
        return mapping.get(self.filling, mt5.ORDER_FILLING_IOC)

    def send(self, signal: Signal, volume: float) -> dict[str, Any]:
        mt5 = self._require()
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError("Sem tick do símbolo.")
        is_buy = signal.side is Side.BUY
        price = float(tick.ask if is_buy else tick.bid)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": float(signal.stop),
            "tp": float(signal.take),
            "deviation": int(self.deviation),
            "magic": int(self.magic),
            "comment": self.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_const(),
        }
        check = mt5.order_check(request)
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"order_send falhou: {mt5.last_error()}")
        payload = result._asdict()
        payload["check"] = None if check is None else str(check)
        payload["request"] = {k: request[k] for k in request}
        return payload

    def close_position(self, ticket: int, side: Side, volume: float) -> dict[str, Any]:
        mt5 = self._require()
        tick = mt5.symbol_info_tick(self.symbol)
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
