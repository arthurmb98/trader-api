from __future__ import annotations

import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any

import joblib
import pandas as pd

from trader.backtest import BacktestEngine
from trader.config import AppConfig, load_config, save_config
from trader.data import load_candles, prepare_study_csvs, sanitize_test
from trader.ml import CandleRegressor, add_true_range
from trader.paths import CONFIGS_DIR, RESULTS_DIR, ROOT
from trader.price_action import lookback_for_timeframe, strange_from_frame
from trader.signals import overlay_config

MIN_TRADES = 40
BANKS = (500, 1000)
DD_CAPS = {500: 35.0, 1000: 40.0}
DAILY_OPTS = (80.0, 150.0, 250.0, 0.0)
CASES = ("last_candle", "last_candles")
CASE_SLUG = {"last_candle": "lc", "last_candles": "candles"}
CASE_LABEL = {
    "last_candle": "Último candle",
    "last_candles": "Últimos candles",
}
TF_KEYS = ("m1", "m5")
TF_LABEL = {"m1": "1 min", "m5": "5 min"}
SCORE_HARD_REJECT = -2_000_000_000.0
SCORE_SOFT_REJECT = -1_000_000_000.0
TIMEFRAMES = (
    ("m1", "datasets/WIN_1min_train.csv", "datasets/WIN_1min_test.csv"),
    ("m5", "datasets/WIN_5min_train.csv", "datasets/WIN_5min_test.csv"),
)
OLD_FROZEN = (
    "best_m1_a.yaml",
    "best_m1_b.yaml",
    "best_m5_a.yaml",
    "best_m5_b.yaml",
    "best_m1_1k.yaml",
    "best_m5_1k.yaml",
)

_BT_TEST = None
_BT_PRED = None
_BT_STRANGE = None
_BT_LEAK = None


def _daily_opts(_bank: float | None = None) -> tuple[float, ...]:
    return DAILY_OPTS


def _dd_cap(bank: float) -> float:
    return float(DD_CAPS.get(int(bank), 40.0))


def _compact_metrics(m) -> dict[str, Any]:
    data = m.to_dict()
    trades = data.pop("trades", [])
    equity = data.pop("equity", [])
    data["equity"] = equity
    data["trades"] = trades
    return data


def _setup_key(row: dict[str, Any]) -> tuple:
    risk = row["params"]["risk"]
    exe = row["params"]["execution"]
    tf = row["params"]["data"]["timeframe"]
    return (
        tf,
        risk["mode"],
        exe.get("direction"),
        round(float(risk.get("stop_points") or 0), 1),
        bool(risk.get("trailing_enabled")),
    )


def _is_clone(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        abs(a["metrics"]["net_pnl"] - b["metrics"]["net_pnl"]) < 0.05
        and a["metrics"]["n_trades"] == b["metrics"]["n_trades"]
    )


def _pick_diverse(ranked: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for row in ranked:
        duplicate = False
        key = _setup_key(row)
        for prev in picked:
            if _is_clone(row, prev) or _setup_key(prev) == key:
                duplicate = True
                break
        if duplicate:
            continue
        picked.append(row)
        if len(picked) >= k:
            break
    return picked


def _drawdown_abs(metrics: dict[str, Any], initial_bank: float) -> float:
    raw = metrics.get("max_drawdown")
    if raw is not None:
        return max(float(raw), 0.0)
    pct = max(float(metrics.get("max_drawdown_pct") or 0.0), 0.0)
    return pct / 100.0 * float(initial_bank)


def _is_viable(score: float) -> bool:
    return score > SCORE_SOFT_REJECT / 2


def _passes_hard_floor(score: float) -> bool:
    return score > SCORE_HARD_REJECT / 2


def _score(metrics: dict[str, Any], initial_bank: float) -> float:
    trades = metrics["n_trades"]
    pnl = float(metrics["net_pnl"])
    dd_pct = max(float(metrics["max_drawdown_pct"]), 0.0)
    dd_abs = _drawdown_abs(metrics, initial_bank)
    if trades < MIN_TRADES or dd_abs >= float(initial_bank):
        return SCORE_HARD_REJECT + pnl
    pf = min(float(metrics["profit_factor"]), 6.0)
    quality = pnl * (1.0 + pf) / (1.0 + dd_pct / 40.0)
    if dd_pct > _dd_cap(initial_bank):
        return SCORE_SOFT_REJECT + quality
    return quality


def build_grid(
    base: AppConfig,
    initial_bank: float | None = None,
    *,
    decision: str = "ml",
) -> list[AppConfig]:
    if initial_bank is not None:
        base = overlay_config(base, account__initial_bank=float(initial_bank))
    daily_opts = _daily_opts()
    configs: list[AppConfig] = []
    seen: set[str] = set()

    def add(cfg: AppConfig) -> None:
        key = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
        if key in seen:
            return
        seen.add(key)
        configs.append(cfg)

    fixed_pairs = [
        (80, 80),
        (80, 160),
        (100, 50),
        (100, 100),
        (100, 200),
        (120, 240),
        (150, 150),
        (200, 200),
    ]

    for (stop, gain), direction, gold, gap, daily in product(
        fixed_pairs,
        ("follow", "fade"),
        (True, False),
        (40.0, None),
        daily_opts,
    ):
        add(
            overlay_config(
                base,
                risk__mode="fixed",
                risk__stop_points=float(stop),
                risk__gain_points=float(gain),
                risk__trailing_enabled=False,
                risk__daily_loss_points=float(daily),
                filters__gold_hours_only=gold,
                filters__max_gap_points=gap,
                filters__min_predicted_body=0.0,
                execution__direction=direction,
                execution__entry_mode="market_open",
                execution__entry_offset_points=0.0,
                execution__decision=decision,
            )
        )

    for stop, gain, gold, trigger, daily in product(
        (100.0, 120.0),
        (200.0, 240.0),
        (True, False),
        (60.0, 100.0),
        daily_opts,
    ):
        if stop == 100.0 and gain != 200.0:
            continue
        if stop == 120.0 and gain != 240.0:
            continue
        add(
            overlay_config(
                base,
                risk__mode="fixed",
                risk__stop_points=stop,
                risk__gain_points=gain,
                risk__trailing_enabled=True,
                risk__trailing_trigger_points=trigger,
                risk__trailing_distance_points=50.0,
                risk__daily_loss_points=float(daily),
                filters__gold_hours_only=gold,
                filters__max_gap_points=40.0,
                filters__min_predicted_body=0.0,
                execution__direction="follow",
                execution__entry_mode="market_open",
                execution__entry_offset_points=0.0,
                execution__decision=decision,
            )
        )

    return configs


def _slim_equity(equity: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(equity) <= 400:
        return equity
    step = max(1, len(equity) // 400)
    slim = equity[::step]
    if slim[-1] is not equity[-1]:
        slim.append(equity[-1])
    return slim


def _year_breakdown(trades: list[dict[str, Any]], initial_bank: float) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        year = str(trade.get("exit_time", ""))[:4]
        if year.isdigit():
            buckets.setdefault(year, []).append(trade)
    out: dict[str, dict[str, Any]] = {}
    for year, rows in sorted(buckets.items()):
        rows = sorted(rows, key=lambda item: str(item.get("exit_time", "")))
        pnl = float(sum(float(item["pnl"]) for item in rows))
        wins = sum(1 for item in rows if item.get("result") == "win")
        bank = float(initial_bank)
        peak = bank
        max_dd = 0.0
        for item in rows:
            bank += float(item["pnl"])
            peak = max(peak, bank)
            max_dd = max(max_dd, peak - bank)
        out[year] = {
            "n_trades": len(rows),
            "n_wins": wins,
            "win_rate": round(100.0 * wins / len(rows), 2) if rows else 0.0,
            "net_pnl": round(pnl, 2),
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(100.0 * max_dd / initial_bank, 2) if initial_bank else 0.0,
        }
    return out


def _bucket_trades(trades: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        label = key_fn(trade)
        if label:
            buckets.setdefault(label, []).append(trade)
    rows: list[dict[str, Any]] = []
    for label, items in sorted(buckets.items()):
        pnl = float(sum(float(item["pnl"]) for item in items))
        wins = sum(1 for item in items if item.get("result") == "win")
        rows.append({"t": label, "pnl": round(pnl, 2), "n_trades": len(items), "n_wins": wins})
    return rows


def _period_breakdown(trades: list[dict[str, Any]]) -> dict[str, Any]:
    def _dt(trade: dict[str, Any]) -> datetime | None:
        raw = str(trade.get("exit_time") or "")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None

    def day_key(trade: dict[str, Any]) -> str | None:
        dt = _dt(trade)
        return dt.date().isoformat() if dt else None

    def week_key(trade: dict[str, Any]) -> str | None:
        dt = _dt(trade)
        if not dt:
            return None
        iso = dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    def month_key(trade: dict[str, Any]) -> str | None:
        dt = _dt(trade)
        return dt.strftime("%Y-%m") if dt else None

    daily = _bucket_trades(trades, day_key)
    weekly = _bucket_trades(trades, week_key)
    monthly = _bucket_trades(trades, month_key)

    def _extremes(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"best": None, "worst": None, "avg": 0.0, "positive_pct": 0.0}
        best = max(rows, key=lambda row: row["pnl"])
        worst = min(rows, key=lambda row: row["pnl"])
        positive = sum(1 for row in rows if row["pnl"] > 0)
        return {
            "best": best,
            "worst": worst,
            "avg": round(float(mean([row["pnl"] for row in rows])), 2),
            "positive_pct": round(100.0 * positive / len(rows), 1),
        }

    return {
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
        "summary": {
            "day": _extremes(daily),
            "week": _extremes(weekly),
            "month": _extremes(monthly),
            "n_days": len(daily),
        },
    }


def _metrics_public(metrics: dict[str, Any]) -> dict[str, Any]:
    trades = metrics.get("trades", [])
    return {
        **{k: v for k, v in metrics.items() if k not in {"trades", "equity"}},
        "equity": _slim_equity(metrics.get("equity", [])),
        "n_trade_events": len(trades),
        "max_contracts": metrics.get("max_contracts", 1),
        "contracts_path": metrics.get("contracts_path", []),
    }


def _pack_side(cfg: AppConfig, metrics: dict[str, Any]) -> dict[str, Any]:
    trades = metrics.get("trades", [])
    bank = float(cfg.account.initial_bank)
    return {
        "metrics": _metrics_public(metrics),
        "by_year": _year_breakdown(trades, bank),
        "by_period": _period_breakdown(trades),
        "max_contracts": int(metrics.get("max_contracts") or 1),
        "contracts_path": metrics.get("contracts_path") or [],
    }


def _setup_label(params: dict[str, Any]) -> str:
    risk = params["risk"]
    exe = params["execution"]
    filters = params["filters"]
    tf = params["data"]["timeframe"]
    direction = "follow" if exe.get("direction") == "follow" else "fade"
    daily = float(risk.get("daily_loss_points") or 0)
    daily_s = "sem daily" if daily <= 0 else f"daily {int(daily)}"
    trail = " trailing" if risk.get("trailing_enabled") else ""
    gold = " · ouro" if filters.get("gold_hours_only") else ""
    guard = " · guarda" if exe.get("decision") == "ml_guard" else ""
    body = f"{int(float(risk.get('stop_points') or 0))}/{int(float(risk.get('gain_points') or 0))}"
    return f"{tf} {direction} {body}{trail} · {daily_s}{gold}{guard}"


def _month_count(side: dict[str, Any]) -> int:
    months = (side.get("by_period") or {}).get("monthly") or []
    return max(len(months), 1)


def _avg_month(side: dict[str, Any]) -> float:
    return float(side["metrics"]["net_pnl"]) / _month_count(side)


def _select_winners(ranked: list[dict[str, Any]], n_winners: int = 2) -> list[dict[str, Any]]:
    viable = [row for row in ranked if _is_viable(row["score"])]
    survivors = [row for row in ranked if _passes_hard_floor(row["score"])]
    picked = _pick_diverse(viable, n_winners)
    if len(picked) < n_winners:
        picked = _pick_diverse(survivors, n_winners)
    if len(picked) < n_winners:
        extra: list[dict[str, Any]] = []
        for row in ranked:
            if any(row["name"] == prev["name"] or _is_clone(row, prev) for prev in picked + extra):
                continue
            extra.append(row)
            if len(picked) + len(extra) >= n_winners:
                break
        picked = picked + extra
    return picked[:n_winners]


def _summarize_run(
    cfg: AppConfig,
    metrics: dict[str, Any],
    leakage: dict[str, Any],
    keep_trades: bool = False,
) -> dict[str, Any]:
    trades = metrics.get("trades", [])
    packed = {
        "name": cfg.name,
        "params": cfg.to_dict(),
        "score": _score(metrics, float(cfg.account.initial_bank)),
        "leakage": leakage,
        "metrics": {
            **{k: v for k, v in metrics.items() if k not in {"trades", "equity"}},
            "equity": _slim_equity(metrics.get("equity", [])),
            "n_trade_events": len(trades),
        },
        "trades": trades if keep_trades else [],
    }
    if keep_trades:
        packed["by_year"] = _year_breakdown(trades, float(cfg.account.initial_bank))
    return packed


def _fit_predictor(timeframe: str, train: pd.DataFrame) -> tuple[CandleRegressor, dict[str, Any], Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = RESULTS_DIR / f"model_{timeframe}.joblib"
    if model_path.exists():
        model_path.unlink()
    predictor = CandleRegressor()
    train_scores = predictor.fit(train)
    joblib.dump(predictor, model_path)
    return predictor, train_scores, model_path


def _bt_job(item: tuple) -> dict[str, Any]:
    cfg_dict, compound, use_guard, name = item
    cfg = AppConfig.from_dict(cfg_dict)
    cfg.name = name
    metrics = BacktestEngine(cfg).run(
        _BT_TEST,
        _BT_PRED,
        compound=compound,
        strange_mask=_BT_STRANGE if use_guard else None,
    )
    compact = _compact_metrics(metrics)
    packed = _summarize_run(cfg, compact, _BT_LEAK, keep_trades=False)
    packed["model_test"] = None
    return packed


def _leaderboard_rows(ranked: list[dict[str, Any]], n: int = 40) -> list[dict[str, Any]]:
    rows = []
    for row in ranked[:n]:
        rows.append(
            {
                "name": row["name"],
                "score": row["score"],
                "net_pnl": row["metrics"]["net_pnl"],
                "win_rate": row["metrics"]["win_rate"],
                "n_trades": row["metrics"]["n_trades"],
                "profit_factor": row["metrics"]["profit_factor"],
                "max_drawdown": row["metrics"]["max_drawdown"],
                "max_drawdown_pct": row["metrics"]["max_drawdown_pct"],
                "timeframe": row["params"]["data"]["timeframe"],
                "params": {
                    "risk": row["params"]["risk"],
                    "filters": row["params"]["filters"],
                    "execution": row["params"]["execution"],
                    "account": row["params"]["account"],
                    "data": row["params"]["data"],
                },
            }
        )
    return rows


def _compact_runs(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "score": row["score"],
            "net_pnl": row["metrics"]["net_pnl"],
            "win_rate": row["metrics"]["win_rate"],
            "n_trades": row["metrics"]["n_trades"],
            "profit_factor": row["metrics"]["profit_factor"],
            "max_drawdown": row["metrics"]["max_drawdown"],
            "max_drawdown_pct": row["metrics"]["max_drawdown_pct"],
            "direction": row["params"]["execution"]["direction"],
            "mode": row["params"]["risk"]["mode"],
            "gold_hours_only": row["params"]["filters"]["gold_hours_only"],
            "entry_mode": row["params"]["execution"]["entry_mode"],
            "decision": row["params"]["execution"].get("decision", "ml"),
            "stop_points": row["params"]["risk"]["stop_points"],
            "gain_points": row["params"]["risk"]["gain_points"],
            "timeframe": row["params"]["data"]["timeframe"],
        }
        for row in ranked
    ]


def _eval_grid(
    configs: list[AppConfig],
    test: pd.DataFrame,
    predicted: pd.DataFrame,
    leakage: dict[str, Any],
    *,
    compound: bool,
    strange_mask,
    tag: str,
    model_test: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    global _BT_TEST, _BT_PRED, _BT_STRANGE, _BT_LEAK
    _BT_TEST, _BT_PRED, _BT_STRANGE, _BT_LEAK = test, predicted, strange_mask, leakage
    items = []
    for i, cfg in enumerate(configs):
        cfg.name = f"{tag}_{i:04d}"
        cfg.account.contracts = 1
        use_guard = str(cfg.execution.decision) == "ml_guard"
        items.append((cfg.to_dict(), compound, use_guard, cfg.name))

    runs: list[dict[str, Any]] = []
    workers = min(8, os.cpu_count() or 2)
    try:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(_bt_job, item): idx for idx, item in enumerate(items)}
            done = 0
            ordered: list[dict[str, Any] | None] = [None] * len(items)
            for fut in as_completed(futures):
                idx = futures[fut]
                ordered[idx] = fut.result()
                done += 1
                if done % 40 == 0 or done == len(items):
                    print(f"  {tag}: {done}/{len(items)}", flush=True)
            runs = [row for row in ordered if row is not None]
    except Exception as exc:
        print(f"  pool falhou ({exc}); rodando em série", flush=True)
        runs = []
        for i, item in enumerate(items):
            packed = _bt_job(item)
            runs.append(packed)
            if (i + 1) % 40 == 0 or i + 1 == len(items):
                print(f"  {tag}: {i + 1}/{len(items)}", flush=True)

    for packed in runs:
        packed["model_test"] = model_test
        packed["trades"] = []
    return runs


def _empty_case_block() -> dict[str, Any]:
    return {
        "winners": {tf: [] for tf in TF_KEYS},
        "n_configs": 0,
        "n_viable": 0,
        "leaderboard": [],
        "all_runs_compact": [],
    }


def _winners_by_tf(winners: Any) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {tf: [] for tf in TF_KEYS}
    if isinstance(winners, dict) and any(key in winners for key in TF_KEYS):
        for tf in TF_KEYS:
            out[tf] = list(winners.get(tf) or [])
        return out
    if isinstance(winners, list):
        for winner in winners:
            tf = str(winner.get("params", {}).get("data", {}).get("timeframe") or "m5")
            out.setdefault(tf, []).append(winner)
    return out


def _prediction_cache() -> dict[str, tuple]:
    cache: dict[str, tuple] = {}
    for tf, train_csv, test_csv in TIMEFRAMES:
        train = load_candles(ROOT / train_csv)
        test_raw = load_candles(ROOT / test_csv)
        test, _leak = sanitize_test(
            train, test_raw, Path(train_csv).name, Path(test_csv).name, match_ohlc=False
        )
        test = add_true_range(test, 14)
        predictor = joblib.load(RESULTS_DIR / f"model_{tf}.joblib")
        predicted = predictor.predict_next_ohlc(test)
        lookback = lookback_for_timeframe(tf)
        strange = strange_from_frame(test, lookback=lookback)
        cache[tf] = (test, predicted, strange)
    return cache


def enrich_winners(cases: dict[str, dict[str, dict[str, Any]]], cache: dict[str, tuple] | None = None) -> None:
    cache = cache if cache is not None else _prediction_cache()
    for case in CASES:
        use_guard = case == "last_candles"
        for bank in BANKS:
            by_tf = _winners_by_tf(cases[case][str(bank)]["winners"])
            cases[case][str(bank)]["winners"] = by_tf
            for tf in TF_KEYS:
                for winner in by_tf[tf]:
                    test, predicted, strange = cache[tf]
                    cfg = AppConfig.from_dict(winner["params"])
                    cfg.account.initial_bank = float(bank)
                    cfg.account.contracts = 1
                    mask = strange if use_guard else None
                    print(f"  replay {case} {tf} {winner.get('params', {}).get('name')} banca {bank}", flush=True)
                    fixed = BacktestEngine(cfg).run(test, predicted, compound=False, strange_mask=mask)
                    scaled = BacktestEngine(cfg).run(test, predicted, compound=True, strange_mask=mask)
                    packed_fixed = _pack_side(cfg, _compact_metrics(fixed))
                    packed_scaled = _pack_side(cfg, _compact_metrics(scaled))
                    winner["trades"] = []
                    winner["metrics"] = packed_fixed["metrics"]
                    winner["by_year"] = packed_fixed["by_year"]
                    winner["by_period"] = packed_fixed["by_period"]
                    winner["lot_fixed"] = packed_fixed
                    winner["lot_scaled"] = packed_scaled


def build_insights(cases: dict[str, dict[str, dict[str, Any]]]) -> dict[str, list[str]]:
    worked: list[str] = []
    failed: list[str] = []
    for case in CASES:
        for bank in BANKS:
            by_tf = _winners_by_tf(cases[case][str(bank)]["winners"])
            for tf in TF_KEYS:
                winners = by_tf[tf]
                label = f"{CASE_LABEL[case]} · {TF_LABEL[tf]} · R$ {bank}"
                if not winners:
                    failed.append(f"{label}: nenhum setup passou o piso de trades/drawdown.")
                    continue
                best = winners[0]
                fixed = best.get("lot_fixed") or best
                avg = _avg_month(fixed)
                worked.append(
                    f"{label}: média {avg:.0f} R$/mês no melhor setup ({best['params']['name']}), 1 mini."
                )
                years = (fixed.get("by_year") or best.get("by_year")) or {}
                y26 = years.get("2026")
                if y26 and y26["net_pnl"] < 0:
                    failed.append(f"{label}: 2026 negativo (R$ {y26['net_pnl']:.0f}).")
    improve = [
        "Paper 20–30 pregões antes de capital real. Não operar em Copom, payroll ou IPC.",
    ]
    return {"worked": worked, "failed": failed, "improve": improve}


def build_parecer(cases: dict[str, dict[str, dict[str, Any]]], tf_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    m1_hit = round(100.0 * float(tf_meta["m1"]["model_test"]["test_direction_hit"]), 1)
    m5_hit = round(100.0 * float(tf_meta["m5"]["model_test"]["test_direction_hit"]), 1)
    monthly = []
    by_case = []
    for case in CASES:
        for bank in BANKS:
            by_tf = _winners_by_tf(cases[case][str(bank)]["winners"])
            for tf in TF_KEYS:
                winners = by_tf[tf]
                if not winners:
                    by_case.append(
                        {
                            "case": case,
                            "bank": int(bank),
                            "timeframe": tf,
                            "name": None,
                            "label": None,
                            "avg_fixed": None,
                            "avg_scaled": None,
                            "n_months": 0,
                        }
                    )
                    continue
                best = winners[0]
                fixed = best.get("lot_fixed") or best
                scaled = best.get("lot_scaled") or best
                by_case.append(
                    {
                        "case": case,
                        "bank": int(bank),
                        "name": best["params"].get("name", best["name"]),
                        "timeframe": tf,
                        "label": _setup_label(best["params"]),
                        "avg_fixed": round(_avg_month(fixed), 2),
                        "avg_scaled": round(_avg_month(scaled), 2),
                        "n_months": _month_count(fixed),
                    }
                )
                for winner in winners:
                    side = winner.get("lot_fixed") or winner
                    scaled_w = winner.get("lot_scaled") or winner
                    monthly.append(
                        {
                            "case": case,
                            "bank": int(bank),
                            "name": winner["params"].get("name", winner["name"]),
                            "timeframe": tf,
                            "label": _setup_label(winner["params"]),
                            "avg_fixed": round(_avg_month(side), 2),
                            "avg_scaled": round(_avg_month(scaled_w), 2),
                            "dd_pct": round(float(side["metrics"]["max_drawdown_pct"]), 1),
                            "dd_abs": round(float(side["metrics"]["max_drawdown"]), 2),
                            "net_pnl": round(float(side["metrics"]["net_pnl"]), 2),
                            "n_months": _month_count(side),
                        }
                    )
    return {
        "headline": "Dois jeitos de decidir. O lote é outra pergunta.",
        "ml_hit": {"m1": m1_hit, "m5": m5_hit},
        "n_months_note": "Média mensal = P&L do teste ÷ meses com pelo menos um trade.",
        "by_case": by_case,
        "monthly": monthly,
        "strategy": [
            f"O ML acerta a direção em {m1_hit}% no 1 min e {m5_hit}% no 5 min — perto de cara ou coroa. O valor está no risco, não na cor.",
            "Dois casos: último candle e últimos candles. Banca, lote e tempo gráfico (1 min / 5 min) não são casos.",
            "Cada tempo gráfico tem os 2 melhores setups — o segundo aparece mesmo com prejuízo. Guarda: 10 no 1 min, 5 no 5 min. Ranking em 1 mini.",
            "Stop diário em pontos é o mesmo nas duas bancas. Fill na abertura do candle seguinte.",
        ],
        "dd_floor": "Tombo maior ou igual à banca tira a preferência no ranking; o segundo melhor continua visível mesmo no negativo.",
    }


def freeze_named(winner: dict[str, Any], name: str) -> str:
    cfg = AppConfig.from_dict(winner["params"])
    cfg.name = name
    winner["params"]["name"] = name
    path = CONFIGS_DIR / f"{name}.yaml"
    save_config(cfg, path)
    return str(path.relative_to(ROOT))


def freeze_winners(cases: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    written: list[str] = []
    labels = ("a", "b")
    for case in CASES:
        slug = CASE_SLUG[case]
        for bank in BANKS:
            by_tf = _winners_by_tf(cases[case][str(bank)]["winners"])
            for tf in TF_KEYS:
                for i, winner in enumerate(by_tf[tf][:2]):
                    written.append(freeze_named(winner, f"best_{slug}_{tf}_{bank}_{labels[i]}"))
    keep = {Path(path).name for path in written}
    for leftover in OLD_FROZEN:
        path = CONFIGS_DIR / leftover
        if path.exists() and path.name not in keep:
            path.unlink()
    for path in CONFIGS_DIR.glob("best_*.yaml"):
        if path.name not in keep:
            path.unlink()
    return written


def _ensure_split_csvs() -> None:
    needed = [ROOT / csv for _, train, test in TIMEFRAMES for csv in (train, test)]
    if all(path.exists() for path in needed):
        return
    print("CSVs de treino/teste ausentes — gerando a partir dos dumps WIN$D.")
    prepare_study_csvs()


def _dump_payload(payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "studies.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    web_public = ROOT / "web" / "public"
    web_public.mkdir(parents=True, exist_ok=True)
    (web_public / "studies.json").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    return out


HOW_IT_WORKS = [
    "Sinal no fechamento do candle; fill na abertura do próximo.",
    "Último candle: o ML escolhe compra ou venda.",
    "Últimos candles: o ML escolhe; padrões só cancelam mercado estranho (10 no 1 min, 5 no 5 min).",
    "Banca 500 ou 1.000. Tempo: 1 min ou 5 min, cada um com os 2 melhores. Lote: 1 mini no ranking, ou +1 a cada múltiplo da banca.",
]


def run_all() -> Path:
    _ensure_split_csvs()
    tf_meta: dict[str, dict[str, Any]] = {}
    cases: dict[str, dict[str, dict[str, Any]]] = {
        case: {str(bank): _empty_case_block() for bank in BANKS} for case in CASES
    }
    pred_cache: dict[str, tuple] = {}

    for timeframe, train_csv, test_csv in TIMEFRAMES:
        print(f"Fit {timeframe}: {train_csv} → {test_csv}")
        train = load_candles(ROOT / train_csv)
        test_raw = load_candles(ROOT / test_csv)
        test, leak_report = sanitize_test(
            train, test_raw, Path(train_csv).name, Path(test_csv).name, match_ohlc=False
        )
        leakage = leak_report.to_dict()
        test = add_true_range(test, 14)
        predictor, train_scores, model_path = _fit_predictor(timeframe, train)
        predicted = predictor.predict_next_ohlc(test)
        model_test_scores = predictor.score_predictions(test, predicted)
        lookback = lookback_for_timeframe(timeframe)
        strange = strange_from_frame(test, lookback=lookback)
        pred_cache[timeframe] = (test, predicted, strange)
        tf_meta[timeframe] = {
            "leakage": leakage,
            "model_test": model_test_scores,
            "model_train": train_scores,
            "model_path": str(model_path.relative_to(ROOT)),
            "lookback": lookback,
        }

        for bank in BANKS:
            base = load_config()
            base.data.timeframe = timeframe
            base.data.train_csv = train_csv
            base.data.test_csv = test_csv
            base.account.initial_bank = float(bank)
            base.account.contracts = 1

            ml_grid = build_grid(base, initial_bank=bank, decision="ml")
            guard_grid = build_grid(base, initial_bank=bank, decision="ml_guard")
            print(f"Grid último candle {timeframe} R$ {bank}: {len(ml_grid)} configs", flush=True)
            runs_lc = _eval_grid(
                ml_grid, test, predicted, leakage,
                compound=False, strange_mask=None, tag=f"lc_{timeframe}_{bank}",
                model_test=model_test_scores,
            )
            print(f"Grid últimos candles {timeframe} R$ {bank}: {len(guard_grid)} configs", flush=True)
            runs_guard = _eval_grid(
                guard_grid, test, predicted, leakage,
                compound=False, strange_mask=strange, tag=f"candles_{timeframe}_{bank}",
                model_test=model_test_scores,
            )

            for case, runs, n_cfg in (
                ("last_candle", runs_lc, len(ml_grid)),
                ("last_candles", runs_guard, len(guard_grid)),
            ):
                block = cases[case][str(bank)]
                block["all_runs_compact"].extend(_compact_runs(sorted(runs, key=lambda r: r["score"], reverse=True)))
                block["n_configs"] += n_cfg
                block["_pending"] = block.get("_pending", []) + runs

    for case in CASES:
        for bank in BANKS:
            block = cases[case][str(bank)]
            pending = block.pop("_pending", [])
            by_tf: dict[str, list] = {}
            for tf in TF_KEYS:
                tf_runs = [row for row in pending if row["params"]["data"]["timeframe"] == tf]
                ranked = sorted(tf_runs, key=lambda row: row["score"], reverse=True)
                picked = _select_winners(ranked, 2)
                by_tf[tf] = picked
                for winner in picked:
                    winner["leakage"] = tf_meta[tf]["leakage"]
                    winner["model_test"] = tf_meta[tf]["model_test"]
            block["winners"] = by_tf
            block["n_viable"] = sum(1 for row in pending if _is_viable(row["score"]))
            block["leaderboard"] = _leaderboard_rows(sorted(pending, key=lambda row: row["score"], reverse=True), 40)

    print("Replay dos vencedores: 1 mini e lote que sobe")
    enrich_winners(cases, cache=pred_cache)
    written = freeze_winners(cases)
    n_configs_total = sum(cases[case][str(bank)]["n_configs"] for case in CASES for bank in BANKS)
    m1_ref = tf_meta["m1"]["leakage"]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "Estudo educacional com o contínuo WIN$ (mini índice B3). Treino até 31/12/2024. "
            "Teste: 02/01/2025 em diante. O modelo não vê o teste. Não é recomendação de investimento."
        ),
        "instrument": {
            "name": "Mini índice B3 (WIN$ contínuo)",
            "point_value": 0.20,
            "tick": 5,
            "contracts": 1,
            "train_period_m1": m1_ref,
            "train_period_m5": tf_meta["m5"]["leakage"],
        },
        "banks": list(BANKS),
        "cases": list(CASES),
        "case_labels": CASE_LABEL,
        "timeframes_list": list(TF_KEYS),
        "timeframe_labels": TF_LABEL,
        "lookback": {"m1": lookback_for_timeframe("m1"), "m5": lookback_for_timeframe("m5")},
        "n_configs_total": n_configs_total,
        "timeframes": tf_meta,
        "studies": {
            case: {
                str(bank): {k: v for k, v in cases[case][str(bank)].items() if k != "winners"}
                for bank in BANKS
            }
            for case in CASES
        },
        "winners": {
            case: {str(bank): cases[case][str(bank)]["winners"] for bank in BANKS} for case in CASES
        },
        "insights": build_insights(cases),
        "parecer": build_parecer(cases, tf_meta),
        "frozen_configs": written,
        "how_it_works": HOW_IT_WORKS,
        "mt5": {
            "ready": True,
            "default_enabled": False,
            "steps": [
                "Abrir o MetaTrader 5 e autorizar algo trading.",
                "Na YAML vencedora, ligar mt5.enabled: true e conferir o símbolo.",
                "Validar POST /api/signal em paper antes de POST /api/orders.",
            ],
        },
    }
    out = _dump_payload(payload)
    print(f"Resultados em {out}")
    print("Configs congeladas:", written)
    return out


def enrich_saved_study() -> Path:
    out = RESULTS_DIR / "studies.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    cases = {
        case: {str(bank): {"winners": payload["winners"][case][str(bank)]} for bank in BANKS}
        for case in payload.get("cases", CASES)
    }
    enrich_winners(cases)
    payload["winners"] = {
        case: {str(bank): cases[case][str(bank)]["winners"] for bank in BANKS} for case in cases
    }
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    out = _dump_payload(payload)
    print(f"Estudo enriquecido em {out}")
    return out


def rerank_saved_study() -> Path:
    print("O ranking deste estudo depende do grid novo. Rode python -m trader study.")
    return RESULTS_DIR / "studies.json"

