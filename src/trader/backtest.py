from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd

from trader.config import AppConfig
from trader.domain import Side, StudyMetrics, Trade
from trader.risk import RiskCalculator, protect_levels, round_to_tick
from trader.execution import limit_fill_price, planned_limit_entry


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


LOT_FIXED = "fixed"
LOT_SCALED = "scaled"
LOT_STEP = 1000.0


def parse_lot(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {LOT_SCALED, "compound", "growing"}:
        return LOT_SCALED
    return LOT_FIXED


def contracts_for_bank(bank: float, initial_bank: float, cap: int = 16) -> int:
    """+1 contract every multiple of the initial bank (500→1c, 1000→2c, 1500→3c…)."""
    if initial_bank <= 0:
        return 1
    return min(cap, max(1, int(bank // initial_bank)))


def size_contracts(bank: float, lot: str, cap: int = 16) -> int:
    """Always 1 mini, or +1 mini for every R$1.000 of bank."""
    if parse_lot(lot) != LOT_SCALED:
        return 1
    return contracts_for_bank(bank, LOT_STEP, cap)


class BacktestEngine:
    """Walks the TEST frame only. Predictions must already be computed with a frozen model."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.session = SessionFilter(config)
        self.risk = RiskCalculator(config)

    def run(
        self,
        test: pd.DataFrame,
        predicted: pd.DataFrame,
        compound: bool = False,
        pa_sides: np.ndarray | None = None,
        strange_mask: np.ndarray | None = None,
        compound_step: float | None = None,
    ) -> StudyMetrics:
        acc = self.config.account
        flt = self.config.filters
        exe = self.config.execution
        tick = float(self.config.instrument.tick_size)
        initial_bank = float(acc.initial_bank)
        bank = initial_bank
        peak = bank
        max_dd = 0.0
        max_contracts = 1
        timestamps: list[datetime] = list(test["timestamp"])
        contracts_path: list[dict] = [{"t": timestamps[0].isoformat(), "contracts": 1, "bank": bank}]
        trades: list[Trade] = []
        equity: list[dict] = [{"t": timestamps[0].isoformat(), "bank": bank}]
        position: dict | None = None
        day_key = None
        day_pnl = 0.0
        trades_today = 0
        fade = exe.direction == "fade"
        decision = str(getattr(exe, "decision", "ml") or "ml")
        use_guard = decision == "ml_guard"
        use_pa = decision == "price_action_ml"
        pa_arr = pa_sides if pa_sides is not None else None
        strange_arr = strange_mask if strange_mask is not None else None
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
        last_logged_contracts = 1

        def size_now() -> int:
            if compound:
                step = float(compound_step) if compound_step else initial_bank
                return contracts_for_bank(bank, step)
            return max(1, int(acc.contracts))

        n = len(test)
        for i in range(1, n):
            ts = timestamps[i]
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            atr = float(atrs[i]) if not np.isnan(atrs[i]) else None
            prev_ts = timestamps[i - 1]
            prev_close = closes[i - 1]

            key = ts.date()
            if key != day_key:
                if position is not None:
                    trades.append(self._close(position, prev_ts, float(prev_close), "fim_do_dia"))
                    bank += trades[-1].pnl
                    peak = max(peak, bank)
                    max_dd = max(max_dd, peak - bank)
                    equity.append({"t": prev_ts.isoformat(), "bank": round(bank, 2)})
                    position = None
                day_key = key
                day_pnl = 0.0
                trades_today = 0

            if position is not None:
                invalidate = False
                open_i = int(position.get("open_i") or i)
                if i > open_i:
                    prev_px = closes[i - 1]
                    pred_px = pred_close_a[i - 1]
                    strange = bool(use_guard and strange_arr is not None and i - 1 < len(strange_arr) and strange_arr[i - 1])
                    if strange or np.isnan(pred_px):
                        invalidate = True
                    else:
                        nxt = Side.BUY if pred_px >= prev_px else Side.SELL
                        if fade:
                            nxt = nxt.opposite()
                        if use_pa and pa_arr is not None and i - 1 < len(pa_arr):
                            pa = int(pa_arr[i - 1])
                            if pa == 0:
                                invalidate = True
                            else:
                                nxt = Side.BUY if pa > 0 else Side.SELL
                        if not invalidate:
                            invalidate = nxt is not position["side"]
                self._protect_position(position, h, l, invalidate=invalidate)
                exit_price, reason = self._manage(position, o, h, l, c)
                if exit_price is None and flatten[i]:
                    exit_price, reason = c, "fim_da_sessao"
                if exit_price is not None:
                    trade = self._close(position, ts, exit_price, reason)
                    trades.append(trade)
                    bank += trade.pnl
                    day_pnl += trade.pnl
                    peak = max(peak, bank)
                    max_dd = max(max_dd, peak - bank)
                    equity.append({"t": ts.isoformat(), "bank": round(bank, 2)})
                    position = None

            if position is not None:
                continue
            if flatten[i]:
                continue
            if not allowed[i]:
                continue
            n_contracts = size_now()
            if bank < max(50.0, acc.contract_cost * n_contracts + 20.0):
                continue
            daily_loss_money = float(self.config.risk.daily_loss_points) * acc.point_value * n_contracts
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

            if use_guard:
                if strange_arr is None or i - 1 >= len(strange_arr) or bool(strange_arr[i - 1]):
                    continue
            if use_pa:
                if pa_arr is None or i - 1 >= len(pa_arr):
                    continue
                pa = int(pa_arr[i - 1])
                if pa == 0:
                    continue
                ml_dir = 1 if pred_close >= prev_close else -1
                if pa != ml_dir:
                    continue
                side = Side.BUY if pa > 0 else Side.SELL
            else:
                side = Side.BUY if pred_close >= prev_close else Side.SELL
                if fade:
                    side = side.opposite()

            if limit_inside:
                if side is Side.BUY:
                    entry = round_to_tick(pred_high - offset, tick)
                    filled_px = limit_fill_price(side, entry, o, h, l)
                else:
                    entry = round_to_tick(pred_low + offset, tick)
                    filled_px = limit_fill_price(side, entry, o, h, l)
                if filled_px is None:
                    continue
                entry = filled_px
            else:
                entry = planned_limit_entry(float(pred_open), float(prev_close), tick)
                filled_px = limit_fill_price(side, entry, o, h, l)
                if filled_px is None:
                    continue
                entry = filled_px

            stop, take = self.risk.levels(side, entry, atr)
            if n_contracts != last_logged_contracts:
                contracts_path.append({"t": ts.isoformat(), "contracts": n_contracts, "bank": round(bank, 2)})
                last_logged_contracts = n_contracts
            max_contracts = max(max_contracts, n_contracts)
            position = {
                "side": side,
                "entry": float(entry),
                "stop": float(stop),
                "take": float(take),
                "time": ts,
                "hour": ts.hour,
                "extreme": float(entry),
                "contracts": n_contracts,
                "open_i": i,
                "orig_stop": float(stop),
                "orig_take": float(take),
            }
            trades_today += 1
            self._protect_position(position, h, l, invalidate=False)
            exit_price, reason = self._manage(position, o, h, l, c)
            if exit_price is not None:
                trade = self._close(position, ts, exit_price, reason)
                trades.append(trade)
                bank += trade.pnl
                day_pnl += trade.pnl
                peak = max(peak, bank)
                max_dd = max(max_dd, peak - bank)
                equity.append({"t": ts.isoformat(), "bank": round(bank, 2)})
                position = None

        if position is not None:
            trade = self._close(position, timestamps[-1], float(closes[-1]), "fim_dos_dados")
            trades.append(trade)
            bank += trade.pnl
            equity.append({"t": timestamps[-1].isoformat(), "bank": round(bank, 2)})

        metrics = self._metrics(test, trades, equity, initial_bank, bank, max_dd)
        metrics.max_contracts = max_contracts
        metrics.contracts_path = contracts_path
        return metrics

    def _protect_position(self, position: dict, h: float, l: float, *, invalidate: bool) -> None:
        side: Side = position["side"]
        buy = side is Side.BUY
        risk = self.config.risk
        tick = float(self.config.instrument.tick_size)
        position.setdefault("orig_stop", float(position["stop"]))
        position.setdefault("orig_take", float(position["take"]))
        mark = h if buy else l
        new_stop, new_take, new_extreme = protect_levels(
            buy=buy,
            entry=float(position["entry"]),
            stop=float(position["stop"]),
            take=float(position["take"]),
            orig_stop=float(position["orig_stop"]),
            orig_take=float(position["orig_take"]),
            mark=float(mark),
            extreme=float(position.get("extreme") or position["entry"]),
            tick=tick,
            be_trigger=float(risk.be_trigger_points),
            be_lock=float(risk.be_lock_points),
            invalidate=invalidate,
            bar_high=float(h),
            bar_low=float(l),
            invalidate_tp=float(risk.invalidate_tp_points),
            trail_enabled=bool(risk.trailing_enabled),
            trail_trigger=float(risk.trailing_trigger_points),
            trail_distance=float(risk.trailing_distance_points),
        )
        position["stop"] = new_stop
        position["take"] = new_take
        position["extreme"] = new_extreme

    def _manage(
        self,
        position: dict,
        _o: float,
        h: float,
        l: float,
        _c: float,
    ) -> tuple[float | None, str]:
        side: Side = position["side"]
        stop = float(position["stop"])
        take = float(position["take"])
        hit_stop = l <= stop if side is Side.BUY else h >= stop
        hit_take = h >= take if side is Side.BUY else l <= take
        if hit_stop and hit_take:
            return stop, "stop_conservador"
        if hit_stop:
            return stop, "stop"
        if hit_take:
            return take, "gain"
        return None, ""

    def _close(self, position: dict, ts: datetime, price: float, reason: str) -> Trade:
        side: Side = position["side"]
        acc = self.config.account
        n_contracts = int(position.get("contracts") or acc.contracts or 1)
        entry = float(position["entry"])
        points = (price - entry) if side is Side.BUY else (entry - price)
        pnl = points * acc.point_value * n_contracts - acc.contract_cost * n_contracts
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
            contracts=n_contracts,
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
