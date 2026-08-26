from __future__ import annotations

from trader.config import AppConfig
from trader.domain import Side


def round_to_tick(points: float, tick: float) -> float:
    if tick <= 0:
        return points
    return round(points / tick) * tick


def _tighter_stop(buy: bool, current: float, candidate: float) -> float:
    return max(current, candidate) if buy else min(current, candidate)


def _nearer_take(buy: bool, current: float, candidate: float) -> float:
    return min(current, candidate) if buy else max(current, candidate)


def protect_levels(
    *,
    buy: bool,
    entry: float,
    stop: float,
    take: float,
    orig_stop: float,
    mark: float | None,
    extreme: float,
    tick: float,
    be_trigger: float,
    be_lock: float,
    invalidate: bool,
    bar_high: float | None,
    bar_low: float | None,
    invalidate_tp: float,
    trail_enabled: bool,
    trail_trigger: float,
    trail_distance: float,
    orig_take: float,
) -> tuple[float, float, float]:
    """Tighten SL/TP without reversing. Prefer a small locked gain over a wide original target."""
    new_stop = stop
    new_take = take
    new_extreme = extreme
    if mark is not None:
        if buy:
            new_extreme = max(extreme, mark)
            fav = mark - entry
        else:
            new_extreme = min(extreme, mark)
            fav = entry - mark
        if be_trigger > 0 and fav >= be_trigger:
            lock = round_to_tick(entry + be_lock if buy else entry - be_lock, tick)
            new_stop = _tighter_stop(buy, new_stop, lock)

    if invalidate:
        lock = round_to_tick(entry + be_lock if buy else entry - be_lock, tick)
        new_stop = _tighter_stop(buy, new_stop, lock)
        if buy and bar_low is not None:
            new_stop = _tighter_stop(buy, new_stop, round_to_tick(bar_low, tick))
        if not buy and bar_high is not None:
            new_stop = _tighter_stop(buy, new_stop, round_to_tick(bar_high, tick))
        if invalidate_tp > 0:
            small = round_to_tick(entry + invalidate_tp if buy else entry - invalidate_tp, tick)
            new_take = _nearer_take(buy, new_take, small)

    if trail_enabled and mark is not None:
        if buy and new_extreme - entry >= trail_trigger:
            new_stop = _tighter_stop(buy, new_stop, round_to_tick(new_extreme - trail_distance, tick))
        elif (not buy) and entry - new_extreme >= trail_trigger:
            new_stop = _tighter_stop(buy, new_stop, round_to_tick(new_extreme + trail_distance, tick))

    if buy:
        new_stop = max(new_stop, orig_stop)
        new_take = min(new_take, orig_take)
    else:
        new_stop = min(new_stop, orig_stop)
        new_take = max(new_take, orig_take)

    if mark is not None:
        if buy:
            new_stop = min(new_stop, round_to_tick(mark - tick, tick))
            new_stop = max(new_stop, orig_stop)
        else:
            new_stop = max(new_stop, round_to_tick(mark + tick, tick))
            new_stop = min(new_stop, orig_stop)

    if buy and new_take <= new_stop:
        new_take = round_to_tick(new_stop + tick, tick)
    if not buy and new_take >= new_stop:
        new_take = round_to_tick(new_stop - tick, tick)
    return new_stop, new_take, new_extreme


class RiskCalculator:
    """Strategy object: converts a config into stop/gain distances in points."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.risk
        self.tick = float(config.instrument.tick_size)

    def distances(self, atr: float | None) -> tuple[float, float]:
        mode = self.cfg.mode
        if mode == "atr":
            atr_val = float(atr or 0.0)
            stop = max(self.tick, atr_val * self.cfg.atr_stop_mult)
            gain = max(self.tick, atr_val * self.cfg.atr_gain_mult)
        elif mode == "rr":
            stop = float(self.cfg.stop_points)
            gain = stop * float(self.cfg.rr_ratio)
        else:
            stop = float(self.cfg.stop_points)
            gain = float(self.cfg.gain_points)
        return round_to_tick(stop, self.tick), round_to_tick(gain, self.tick)

    def levels(self, side: Side, entry: float, atr: float | None) -> tuple[float, float]:
        stop_pts, gain_pts = self.distances(atr)
        if side is Side.BUY:
            return entry - stop_pts, entry + gain_pts
        return entry + stop_pts, entry - gain_pts
