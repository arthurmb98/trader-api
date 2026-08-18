from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from trader.domain import Candle, Side, Signal


class Predictor(ABC):
    """Fits only on train candles; predict never refits."""

    @abstractmethod
    def fit(self, frame: pd.DataFrame) -> dict[str, float]:
        """Train on the given frame. Returns diagnostic scores on train residuals."""

    @abstractmethod
    def predict_next_ohlc(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict next candle OHLC for each row. Does not fit."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        ...


class Broker(ABC):
    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def last_closed_candles(self, symbol: str, timeframe: str, count: int) -> list[Candle]:
        ...

    @abstractmethod
    def send(self, signal: Signal, volume: float) -> dict[str, Any]:
        ...

    @abstractmethod
    def close_position(self, ticket: int, side: Side, volume: float) -> dict[str, Any]:
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...
