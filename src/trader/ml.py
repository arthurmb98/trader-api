from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trader.ports import Predictor

FEATURE_COLS = ["body", "upper_wick", "lower_wick", "rng", "hour_frac", "dow"]
TARGET_COLS = ["y_open", "y_high", "y_low", "y_close"]


def add_candle_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    o, h, l, c = df["Abertura"], df["Máximo"], df["Mínimo"], df["Fechamento"]
    df["body"] = c - o
    df["upper_wick"] = h - np.maximum(o, c)
    df["lower_wick"] = np.minimum(o, c) - l
    df["rng"] = h - l
    ts = pd.to_datetime(df["timestamp"])
    df["hour_frac"] = ts.dt.hour + ts.dt.minute / 60.0
    df["dow"] = ts.dt.dayofweek
    return df


def add_true_range(frame: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = frame.copy()
    prev_close = df["Fechamento"].shift(1)
    tr = pd.concat(
        [
            df["Máximo"] - df["Mínimo"],
            (df["Máximo"] - prev_close).abs(),
            (df["Mínimo"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(period, min_periods=period).mean()
    return df


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    nxt = df.shift(-1)
    close = df["Fechamento"]
    out = df.copy()
    out["y_open"] = nxt["Abertura"] - close
    out["y_high"] = nxt["Máximo"] - close
    out["y_low"] = nxt["Mínimo"] - close
    out["y_close"] = nxt["Fechamento"] - close
    return out


class CandleRegressor(Predictor):
    """Linear regression on candle shape -> next candle deltas (not absolute price)."""

    def __init__(self) -> None:
        self._pipe: Pipeline | None = None
        self._fitted = False
        self._train_rows = 0

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, frame: pd.DataFrame) -> dict[str, float]:
        featured = add_candle_features(frame)
        labeled = _add_targets(featured).dropna(subset=FEATURE_COLS + TARGET_COLS)
        if len(labeled) < 50:
            raise ValueError("Treino insuficiente para o regressor.")
        x = labeled[FEATURE_COLS].to_numpy(dtype=float)
        y = labeled[TARGET_COLS].to_numpy(dtype=float)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MultiOutputRegressor(LinearRegression())),
            ]
        )
        pipe.fit(x, y)
        pred = pipe.predict(x)
        mae = np.mean(np.abs(pred - y), axis=0)
        rmse = np.sqrt(np.mean((pred - y) ** 2, axis=0))
        self._pipe = pipe
        self._fitted = True
        self._train_rows = len(labeled)
        return {
            "train_rows": float(len(labeled)),
            "mae_open": float(mae[0]),
            "mae_high": float(mae[1]),
            "mae_low": float(mae[2]),
            "mae_close": float(mae[3]),
            "rmse_close": float(rmse[3]),
        }

    def predict_next_ohlc(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self._pipe is None:
            raise RuntimeError("Modelo não treinado. fit() só pode ocorrer no CSV de treino.")
        featured = add_candle_features(frame)
        x = featured[FEATURE_COLS].to_numpy(dtype=float)
        deltas = self._pipe.predict(x)
        close = featured["Fechamento"].to_numpy(dtype=float)
        pred_open = close + deltas[:, 0]
        pred_high = close + deltas[:, 1]
        pred_low = close + deltas[:, 2]
        pred_close = close + deltas[:, 3]
        # Keep OHLC consistent: high >= max(o,c), low <= min(o,c)
        pred_high = np.maximum(pred_high, np.maximum(pred_open, pred_close))
        pred_low = np.minimum(pred_low, np.minimum(pred_open, pred_close))
        out = pd.DataFrame(
            {
                "pred_open": pred_open,
                "pred_high": pred_high,
                "pred_low": pred_low,
                "pred_close": pred_close,
            },
            index=featured.index,
        )
        return out

    def score_predictions(self, frame: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, float]:
        actual_next = frame.shift(-1)
        aligned = predicted.iloc[:-1]
        actual = actual_next.iloc[:-1]
        err_close = aligned["pred_close"] - actual["Fechamento"]
        direction_pred = np.sign(aligned["pred_close"].to_numpy() - frame["Fechamento"].iloc[:-1].to_numpy())
        direction_real = np.sign(actual["Fechamento"].to_numpy() - frame["Fechamento"].iloc[:-1].to_numpy())
        valid = direction_pred != 0
        hit = float((direction_pred[valid] == direction_real[valid]).mean()) if valid.any() else 0.0
        return {
            "test_mae_close": float(np.mean(np.abs(err_close))),
            "test_rmse_close": float(np.sqrt(np.mean(np.square(err_close)))),
            "test_direction_hit": hit,
        }
