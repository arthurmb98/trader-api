from __future__ import annotations

from copy import deepcopy

import pandas as pd

from trader.config import AppConfig
from trader.domain import Side, Signal, PredictedCandle
from trader.ml import CandleRegressor, add_candle_features, add_true_range
from trader.price_action import last_bar_is_strange, lookback_for_timeframe, side_from_last_candles
from trader.risk import RiskCalculator


class SignalPolicy:
    """Turns a frozen regressor + last closed candles into a buy/sell/flat signal."""

    def __init__(self, config: AppConfig, predictor: CandleRegressor) -> None:
        if not predictor.is_fitted:
            raise RuntimeError("SignalPolicy exige modelo já treinado.")
        self.config = config
        self.predictor = predictor
        self.risk = RiskCalculator(config)

    def from_candles(self, frame: pd.DataFrame) -> Signal:
        if frame.empty:
            return Signal(Side.FLAT, 0, 0, 0, "sem_candles")
        featured = add_true_range(add_candle_features(frame), self.config.risk.atr_period)
        predicted = self.predictor.predict_next_ohlc(featured)
        last = featured.iloc[-1]
        pred = predicted.iloc[-1]
        pred_c = PredictedCandle(
            open=float(pred["pred_open"]),
            high=float(pred["pred_high"]),
            low=float(pred["pred_low"]),
            close=float(pred["pred_close"]),
        )
        flt = self.config.filters
        gap = abs(pred_c.open - float(last["Fechamento"]))
        if flt.max_gap_points is not None and gap > flt.max_gap_points:
            return Signal(Side.FLAT, 0, 0, 0, "gap_acima_do_limite", pred_c)
        if pred_c.range < flt.min_predicted_range:
            return Signal(Side.FLAT, 0, 0, 0, "range_previsto_curto", pred_c)
        if pred_c.body < flt.min_predicted_body:
            return Signal(Side.FLAT, 0, 0, 0, "corpo_previsto_curto", pred_c)

        last_close = float(last["Fechamento"])
        ml_buy = pred_c.close >= last_close
        decision = str(getattr(self.config.execution, "decision", "ml") or "ml")
        lookback = lookback_for_timeframe(self.config.data.timeframe)
        if decision == "ml_guard":
            if last_bar_is_strange(featured, lookback=lookback):
                return Signal(Side.FLAT, 0, 0, 0, "mercado_estranho", pred_c)
        if decision == "price_action_ml":
            pa = side_from_last_candles(featured, lookback=lookback)
            if pa is Side.FLAT:
                return Signal(Side.FLAT, 0, 0, 0, "sem_padrao_price_action", pred_c)
            if pa is Side.BUY and not ml_buy:
                return Signal(Side.FLAT, 0, 0, 0, "ml_nao_confirma_compra", pred_c)
            if pa is Side.SELL and ml_buy:
                return Signal(Side.FLAT, 0, 0, 0, "ml_nao_confirma_venda", pred_c)
            side = pa
            reason = "price_action_ml_proxima_abertura"
        else:
            side = Side.BUY if ml_buy else Side.SELL
            if self.config.execution.direction == "fade":
                side = side.opposite()
            reason = (
                "seguir_previsao_proxima_abertura"
                if self.config.execution.direction == "follow"
                else "fade_proxima_abertura"
            )
        # Fill is the next bar's open; stop/gain are planned from the signal close.
        entry = last_close
        atr = float(last["atr"]) if pd.notna(last.get("atr")) else None
        stop, take = self.risk.levels(side, entry, atr)
        return Signal(
            side=side,
            entry=entry,
            stop=stop,
            take=take,
            reason=reason,
            predicted=pred_c,
        )


def overlay_config(base: AppConfig, **patches: object) -> AppConfig:
    data = deepcopy(base.to_dict())
    for key, value in patches.items():
        section, _, name = key.partition("__")
        if not name:
            data[key] = value
        else:
            data.setdefault(section, {})
            data[section][name] = value
    return AppConfig.from_dict(data)
