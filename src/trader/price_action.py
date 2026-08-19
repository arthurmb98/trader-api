from __future__ import annotations

import numpy as np
import pandas as pd

from trader.domain import Side

LOOKBACK_BY_TF = {"m1": 10, "m5": 5}


def lookback_for_timeframe(timeframe: str | None, default: int = 5) -> int:
    key = str(timeframe or "").strip().lower()
    return int(LOOKBACK_BY_TF.get(key, default))


def _window_sum(flag: np.ndarray, width: int) -> np.ndarray:
    n = len(flag)
    out = np.zeros(n, dtype=np.int16)
    if n == 0 or width <= 0:
        return out
    acc = np.cumsum(flag.astype(np.int16))
    out[:width] = acc[:width]
    if n > width:
        out[width:] = acc[width:] - acc[:-width]
    return out


def _pattern_flags(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(closes)
    body = closes - opens
    abs_body = np.maximum(np.abs(body), 1e-9)
    rng = np.maximum(highs - lows, 1e-9)
    upper = highs - np.maximum(opens, closes)
    lower = np.minimum(opens, closes) - lows

    bull_engulf = np.zeros(n, dtype=bool)
    bear_engulf = np.zeros(n, dtype=bool)
    if n > 1:
        bull_engulf[1:] = (
            (body[1:] > 0)
            & (body[:-1] < 0)
            & (opens[1:] <= closes[:-1])
            & (closes[1:] >= opens[:-1])
        )
        bear_engulf[1:] = (
            (body[1:] < 0)
            & (body[:-1] > 0)
            & (opens[1:] >= closes[:-1])
            & (closes[1:] <= opens[:-1])
        )
    bull_pin = (lower >= 2.0 * abs_body) & (closes >= (opens + highs) / 2.0)
    bear_pin = (upper >= 2.0 * abs_body) & (closes <= (opens + lows) / 2.0)
    both_wicks = (upper >= abs_body) & (lower >= abs_body) & (abs_body <= 0.25 * rng)
    return bull_engulf, bear_engulf, bull_pin | both_wicks, bear_pin | both_wicks


def strange_from_ohlc(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray | None = None,
    lookback: int = 5,
) -> np.ndarray:
    """True = skip the bar: conflict, indecision, spike, or chop in the lookback window."""
    n = len(closes)
    strange = np.zeros(n, dtype=bool)
    if n < lookback + 1:
        return strange

    body = closes - opens
    abs_body = np.maximum(np.abs(body), 1e-9)
    rng = highs - lows
    upper = highs - np.maximum(opens, closes)
    lower = np.minimum(opens, closes) - lows
    bull_engulf, bear_engulf, bull_rej, bear_rej = _pattern_flags(opens, highs, lows, closes)

    conflict = (bull_engulf & bear_engulf) | (bull_rej & bear_rej)
    indecision = (upper >= abs_body) & (lower >= abs_body) & (abs_body <= 0.25 * np.maximum(rng, 1e-9))
    spike = np.zeros(n, dtype=bool)
    if atrs is not None and len(atrs) == n:
        atr = np.asarray(atrs, dtype=float)
        spike = rng > (3.0 * np.maximum(atr, 1e-9))
    else:
        mean_rng = np.zeros(n, dtype=float)
        c = np.cumsum(rng)
        mean_rng[lookback:] = (c[lookback:] - c[:-lookback]) / float(lookback)
        spike[lookback:] = rng[lookback:] > (3.0 * np.maximum(mean_rng[lookback:], 1e-9))

    chop = (_window_sum(bull_engulf, lookback) >= 2) & (_window_sum(bear_engulf, lookback) >= 2)
    strange = conflict | indecision | spike | chop
    strange[:lookback] = True
    return strange


def strange_from_frame(
    frame: pd.DataFrame,
    lookback: int = 5,
    atrs: np.ndarray | None = None,
) -> np.ndarray:
    atr_col = atrs
    if atr_col is None and "atr" in frame.columns:
        atr_col = frame["atr"].to_numpy(dtype=float)
    return strange_from_ohlc(
        frame["Abertura"].to_numpy(dtype=float),
        frame["Máximo"].to_numpy(dtype=float),
        frame["Mínimo"].to_numpy(dtype=float),
        frame["Fechamento"].to_numpy(dtype=float),
        atrs=atr_col,
        lookback=lookback,
    )


def last_bar_is_strange(frame: pd.DataFrame, lookback: int = 5) -> bool:
    mask = strange_from_frame(frame, lookback=lookback)
    if len(mask) == 0:
        return True
    return bool(mask[-1])


def pa_sides_from_ohlc(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int = 5,
) -> np.ndarray:
    """+1 buy, -1 sell, 0 flat. Uses candles through index i (last closed bar)."""
    n = len(closes)
    sides = np.zeros(n, dtype=np.int8)
    if n < lookback + 1:
        return sides

    bull_engulf, bear_engulf, bull_pin, bear_pin = _pattern_flags(opens, highs, lows, closes)
    up_seq = np.zeros(n, dtype=bool)
    down_seq = np.zeros(n, dtype=bool)
    if n > 2:
        up_seq[2:] = (closes[2:] > closes[1:-1]) & (closes[1:-1] > closes[:-2])
        down_seq[2:] = (closes[2:] < closes[1:-1]) & (closes[1:-1] < closes[:-2])

    slope = np.zeros(n, dtype=float)
    slope[lookback:] = closes[lookback:] - closes[:-lookback]
    uptrend = up_seq | (slope > 0)
    downtrend = down_seq | (slope < 0)

    buy = bull_engulf | bull_pin | (uptrend & ~bear_engulf & ~bear_pin)
    sell = bear_engulf | bear_pin | (downtrend & ~bull_engulf & ~bull_pin)
    buy[:lookback] = False
    sell[:lookback] = False
    sides = np.where(buy & ~sell, 1, np.where(sell & ~buy, -1, 0)).astype(np.int8)
    return sides


def pa_sides_from_frame(frame: pd.DataFrame, lookback: int = 5) -> np.ndarray:
    return pa_sides_from_ohlc(
        frame["Abertura"].to_numpy(dtype=float),
        frame["Máximo"].to_numpy(dtype=float),
        frame["Mínimo"].to_numpy(dtype=float),
        frame["Fechamento"].to_numpy(dtype=float),
        lookback=lookback,
    )


def side_from_last_candles(frame: pd.DataFrame, lookback: int = 5) -> Side:
    sides = pa_sides_from_frame(frame, lookback=lookback)
    if len(sides) == 0:
        return Side.FLAT
    last = int(sides[-1])
    if last > 0:
        return Side.BUY
    if last < 0:
        return Side.SELL
    return Side.FLAT
