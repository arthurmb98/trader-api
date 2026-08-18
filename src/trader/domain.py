from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"

    def opposite(self) -> "Side":
        if self is Side.BUY:
            return Side.SELL
        if self is Side.SELL:
            return Side.BUY
        return Side.FLAT


@dataclass(frozen=True)
class Candle:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body(self) -> float:
        return self.close - self.open

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass
class PredictedCandle:
    open: float
    high: float
    low: float
    close: float

    @property
    def bullish(self) -> bool:
        return self.close >= self.open

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass
class Signal:
    side: Side
    entry: float
    stop: float
    take: float
    reason: str = ""
    predicted: PredictedCandle | None = None


@dataclass
class Trade:
    side: Side
    entry_time: datetime
    exit_time: datetime
    entry: float
    exit: float
    points: float
    pnl: float
    result: str
    reason: str
    hour: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side.value,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry": self.entry,
            "exit": self.exit,
            "points": self.points,
            "pnl": self.pnl,
            "result": self.result,
            "reason": self.reason,
            "hour": self.hour,
        }


@dataclass
class LeakageReport:
    n_train: int
    n_test_original: int
    n_removed: int
    n_test_clean: int
    removed_by_key: int
    removed_by_ohlc: int
    train_file: str
    test_file: str
    train_start: str | None
    train_end: str | None
    test_start: str | None
    test_end: str | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class StudyMetrics:
    n_candles: int
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    net_pnl: float
    final_bank: float
    initial_bank: float
    avg_win: float
    avg_loss: float
    median_win: float
    median_loss: float
    avg_points_win: float
    avg_points_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    expectancy: float
    trades_per_candle_pct: float
    hourly: dict[str, Any] = field(default_factory=dict)
    equity: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
