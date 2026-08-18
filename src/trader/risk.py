from __future__ import annotations

from trader.config import AppConfig
from trader.domain import Side


def round_to_tick(points: float, tick: float) -> float:
    if tick <= 0:
        return points
    return round(points / tick) * tick


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
