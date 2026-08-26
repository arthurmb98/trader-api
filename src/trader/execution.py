from __future__ import annotations

from trader.domain import Side
from trader.risk import round_to_tick


def _is_buy(side: Side | str) -> bool:
    return side is Side.BUY or str(side) == "BUY"


def delay_band_points(tick: float, delay_points: float) -> float:
    """Worse-side band in points, snapped to tick (WIN mini 15 pts = R$3)."""
    pts = float(delay_points or 0.0)
    if pts <= 0 or tick <= 0:
        return 0.0
    return max(float(tick), round_to_tick(pts, tick))


def planned_limit_entry(
    pred_open: float | None,
    last_close: float,
    tick: float,
    *,
    side: Side | str | None = None,
    delay_points: float = 0.0,
) -> float:
    """LIMIT for the next bar: pred_open, plus a shallow worse-side band for send delay."""
    raw = float(last_close)
    if pred_open is not None and pred_open == pred_open:
        raw = float(pred_open)
    base = round_to_tick(raw, tick)
    band = delay_band_points(tick, delay_points)
    if band <= 0 or side is None:
        return base
    if _is_buy(side):
        return base + band
    return base - band


def limit_fill_price(
    side: Side | str,
    limit: float,
    open_: float,
    high: float,
    low: float,
) -> float | None:
    """Fill at the limit or better if the bar trades through it. None = not filled."""
    if _is_buy(side):
        if low > limit:
            return None
        return min(limit, open_)
    if high < limit:
        return None
    return max(limit, open_)


def limit_hits_mark(side: Side | str, limit: float, mark: float | None) -> bool:
    if mark is None:
        return False
    if _is_buy(side):
        return float(mark) <= limit
    return float(mark) >= limit


def limit_fill_from_mark(side: Side | str, limit: float, mark: float) -> float:
    if _is_buy(side):
        return min(limit, float(mark))
    return max(limit, float(mark))
