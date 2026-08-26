from __future__ import annotations

from trader.domain import Side
from trader.risk import round_to_tick


def _is_buy(side: Side | str) -> bool:
    return side is Side.BUY or str(side) == "BUY"


def planned_limit_entry(pred_open: float | None, last_close: float, tick: float) -> float:
    """LIMIT price for the next bar: model pred_open, else last close."""
    raw = float(last_close)
    if pred_open is not None and pred_open == pred_open:
        raw = float(pred_open)
    return round_to_tick(raw, tick)


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
