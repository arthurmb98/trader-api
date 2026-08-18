from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd

from trader.config import AppConfig
from trader.domain import Side, StudyMetrics, Trade
from trader.risk import RiskCalculator, round_to_tick


def _parse_hhmm(value: str) -> time:
    parts = value.split(":")
    return time(int(parts[0]), int(parts[1]))


def _in_window(ts: datetime, start: time, end: time) -> bool:
    clock = ts.time()
    if start <= end:
        return start <= clock <= end
    return clock >= start or clock <= end


class SessionFilter:
    def __init__(self, config: AppConfig) -> None:
        f = config.filters
        self.start = _parse_hhmm(f.session_start)
        self.end = _parse_hhmm(f.session_end)
        self.skip_lunch = f.skip_lunch
        self.lunch_start = _parse_hhmm(f.lunch_start)
        self.lunch_end = _parse_hhmm(f.lunch_end)
        self.gold_only = f.gold_hours_only
        self.gold = [(_parse_hhmm("09:15"), _parse_hhmm("11:00")), (_parse_hhmm("14:30"), _parse_hhmm("17:00"))]

    def allows(self, ts: datetime) -> bool:
        clock = ts.time()
        if not (self.start <= clock <= self.end):
            return False
        if self.skip_lunch and self.lunch_start <= clock < self.lunch_end:
            return False
        if self.gold_only and not any(a <= clock <= b for a, b in self.gold):
            return False
        return True

    def flatten_day(self, ts: datetime) -> bool:
        return ts.time() >= self.end


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(values))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


class BacktestEngine:
    """Walks the TEST frame only. Predictions must already be computed with a frozen model."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.session = SessionFilter(config)
        self.risk = RiskCalculator(config)

    def run(self, test: pd.DataFrame, predicted: pd.DataFrame) -> StudyMetrics:
        acc = self.config.account
        flt = self.config.filters
        exe = self.config.execution
        tick = float(self.config.instrument.tick_size)
        bank = float(acc.initial_bank)
        peak = bank
        max_dd = 0.0
        trades: list[Trade] = []
        timestamps: list[datetime] = list(test["timestamp"])
        equity: list[dict] = [{"t": timestamps[0].isoformat(), "bank": bank}]
        position: dict | None = None
        day_key = None
        day_pnl = 0.0
        trades_today = 0
        fade = exe.direction == "fade"
        daily_loss_money = float(self.config.risk.daily_loss_points) * acc.point_value * acc.contracts
        opens = test["Abertura"].to_numpy(dtype=float)
        highs = test["Máximo"].to_numpy(dtype=float)
        lows = test["Mínimo"].to_numpy(dtype=float)
        closes = test["Fechamento"].to_numpy(dtype=float)
        atrs = test["atr"].to_numpy(dtype=float) if "atr" in test.columns else np.full(len(test), np.nan)
        pred_open_a = predicted["pred_open"].to_numpy(dtype=float)
        pred_high_a = predicted["pred_high"].to_numpy(dtype=float)
        pred_low_a = predicted["pred_low"].to_numpy(dtype=float)
        pred_close_a = predicted["pred_close"].to_numpy(dtype=float)
        allowed = np.array([self.session.allows(ts) for ts in timestamps])
        flatten = np.array([self.session.flatten_day(ts) for ts in timestamps])
        max_gap = flt.max_gap_points
        min_range = flt.min_predicted_range
        min_body = flt.min_predicted_body
        limit_inside = exe.entry_mode == "limit_inside"
        offset = float(exe.entry_offset_points)
        max_trades = self.config.risk.max_trades_per_day
        use_daily_loss = self.config.risk.daily_loss_points > 0

        n = len(test)
        for i in range(1, n):
            ts = timestamps[i]
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            atr = float(atrs[i]) if not np.isnan(atrs[i]) else None
            prev_ts = timestamps[i - 1]
            prev_close = closes[i - 1]

            key = ts.date()
            if key != day_key:
                day_key = key
                day_pnl = 0.0
                trades_today = 0
                if position is not None:
                    trades.append(self._close(position, prev_ts, float(prev_close), bank, acc, "fim_do_dia"))
                    bank += trades[-1].pnl
                    day_pnl = 0.0
                    position = None

            if position is not None:
                exit_price, reason = self._manage(position, o, h, l, c)
                if exit_price is None and flatten[i]:
                    exit_price, reason = c, "fim_da_sessao"
                if exit_price is not None:
                    trade = self._close(position, ts, exit_price, bank, acc, reason)
                    trades.append(trade)
                    bank += trade.pnl
                    day_pnl += trade.pnl
                    peak = max(peak, bank)
                    max_dd = max(max_dd, peak - bank)
                    equity.append({"t": ts.isoformat(), "bank": round(bank, 2)})
                    position = None

            if position is not None:
                continue
            if not allowed[i]:
                continue
            if bank < max(50.0, acc.contract_cost * acc.contracts + 20.0):
                continue
            if use_daily_loss and day_pnl <= -abs(daily_loss_money):
                continue
            if trades_today >= max_trades:
                continue
            pred_open = pred_open_a[i - 1]
            pred_high = pred_high_a[i - 1]
            pred_low = pred_low_a[i - 1]
            pred_close = pred_close_a[i - 1]
            if np.isnan(pred_close):
                continue

            gap = abs(pred_open - prev_close)
            pred_range = pred_high - pred_low
            pred_body = abs(pred_close - pred_open)
            if max_gap is not None and gap > max_gap:
                continue
            if pred_range < min_range:
                continue
            if pred_body < min_body:
                continue

            side = Side.BUY if pred_close >= prev_close else Side.SELL
            if fade:
                side = side.opposite()

            if limit_inside:
                if side is Side.BUY:
                    entry = round_to_tick(pred_high - offset, tick)
                    filled = l <= entry
                    if filled:
                        entry = min(entry, o)
                else:
                    entry = round_to_tick(pred_low + offset, tick)
                    filled = h >= entry
                    if filled:
                        entry = max(entry, o)
                if not filled:
                    continue
            else:
                entry = o

            stop, take = self.risk.levels(side, entry, atr)
            position = {
                "side": side,
                "entry": float(entry),
                "stop": float(stop),
                "take": float(take),
                "time": ts,
                "hour": ts.hour,
                "extreme": float(entry),
            }
            trades_today += 1
            exit_price, reason = self._manage(position, o, h, l, c)
            if exit_price is not None:
                trade = self._close(position, ts, exit_price, bank, acc, reason)
                trades.append(trade)
                bank += trade.pnl
                day_pnl += trade.pnl
                peak = max(peak, bank)
                max_dd = max(max_dd, peak - bank)
                equity.append({"t": ts.isoformat(), "bank": round(bank, 2)})
                position = None

        if position is not None:
            trade = self._close(position, timestamps[-1], float(closes[-1]), bank, acc, "fim_dos_dados")
            trades.append(trade)
            bank += trade.pnl
            equity.append({"t": timestamps[-1].isoformat(), "bank": round(bank, 2)})

        return self._metrics(test, trades, equity, acc.initial_bank, bank, max_dd)

    def _manage(
        self,
        position: dict,
        _o: float,
        h: float,
        l: float,
        _c: float,
    ) -> tuple[float | None, str]:
        side: Side = position["side"]
        stop = position["stop"]
        take = position["take"]
        risk = self.config.risk
        if risk.trailing_enabled:
            if side is Side.BUY:
                position["extreme"] = max(position["extreme"], h)
                if position["extreme"] - position["entry"] >= risk.trailing_trigger_points:
                    new_stop = position["extreme"] - risk.trailing_distance_points
                    position["stop"] = max(position["stop"], new_stop)
                    stop = position["stop"]
            else:
                position["extreme"] = min(position["extreme"], l)
                if position["entry"] - position["extreme"] >= risk.trailing_trigger_points:
                    new_stop = position["extreme"] + risk.trailing_distance_points
                    position["stop"] = min(position["stop"], new_stop)
                    stop = position["stop"]

        hit_stop = l <= stop if side is Side.BUY else h >= stop
        hit_take = h >= take if side is Side.BUY else l <= take
        if hit_stop and hit_take:
            return stop, "stop_conservador"
        if hit_stop:
            return stop, "stop"
        if hit_take:
            return take, "gain"
        return None, ""

    def _close(self, position: dict, ts: datetime, price: float, bank: float, acc, reason: str) -> Trade:
        side: Side = position["side"]
        entry = float(position["entry"])
        points = (price - entry) if side is Side.BUY else (entry - price)
        pnl = points * acc.point_value * acc.contracts - acc.contract_cost * acc.contracts
        result = "win" if pnl > 0 else "loss"
        return Trade(
            side=side,
            entry_time=position["time"],
            exit_time=ts,
            entry=entry,
            exit=float(price),
            points=float(points),
            pnl=float(pnl),
            result=result,
            reason=reason,
            hour=int(position["hour"]),
        )

    def _metrics(
        self,
        test: pd.DataFrame,
        trades: list[Trade],
        equity: list[dict],
        initial_bank: float,
        final_bank: float,
        max_dd: float,
    ) -> StudyMetrics:
        wins = [t for t in trades if t.result == "win"]
        losses = [t for t in trades if t.result != "win"]
        win_pnl = [t.pnl for t in wins]
        loss_pnl = [t.pnl for t in losses]
        gross_win = sum(win_pnl)
        gross_loss = abs(sum(loss_pnl))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        hourly: dict[str, dict] = {}
        for t in trades:
            bucket = hourly.setdefault(str(t.hour), {"trades": 0, "wins": 0, "pnl": 0.0})
            bucket["trades"] += 1
            bucket["wins"] += int(t.result == "win")
            bucket["pnl"] = round(bucket["pnl"] + t.pnl, 2)
        n_trades = len(trades)
        return StudyMetrics(
            n_candles=len(test),
            n_trades=n_trades,
            n_wins=len(wins),
            n_losses=len(losses),
            win_rate=100.0 * len(wins) / n_trades if n_trades else 0.0,
            net_pnl=float(final_bank - initial_bank),
            final_bank=float(final_bank),
            initial_bank=float(initial_bank),
            avg_win=_mean(win_pnl),
            avg_loss=_mean(loss_pnl),
            median_win=_median(win_pnl),
            median_loss=_median(loss_pnl),
            avg_points_win=_mean([t.points for t in wins]),
            avg_points_loss=_mean([t.points for t in losses]),
            profit_factor=float(profit_factor),
            max_drawdown=float(max_dd),
            max_drawdown_pct=100.0 * max_dd / initial_bank if initial_bank else 0.0,
            expectancy=_mean([t.pnl for t in trades]),
            trades_per_candle_pct=100.0 * n_trades / len(test) if len(test) else 0.0,
            hourly=hourly,
            equity=equity,
            trades=[t.to_dict() for t in trades],
        )
