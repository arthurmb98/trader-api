from __future__ import annotations

import json
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
from trader.signals import overlay_config

MIN_TRADES = 40
BANKS = (500, 1000, 5000)
DD_CAPS = {500: 35.0, 1000: 40.0, 5000: 50.0}
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


def _daily_opts(bank: float) -> tuple[float, ...]:
    if bank <= 500:
        return (80.0, 150.0, 0.0)
    if bank <= 1000:
        return (150.0, 250.0, 0.0)
    return (250.0, 400.0, 0.0)


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
    return (risk["mode"], exe["direction"], round(float(risk.get("stop_points") or 0), 1))


def _pick_diverse(ranked: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for row in ranked:
        duplicate = False
        key = _setup_key(row)
        for prev in picked:
            same_pnl = abs(row["metrics"]["net_pnl"] - prev["metrics"]["net_pnl"]) < 0.05
            same_n = row["metrics"]["n_trades"] == prev["metrics"]["n_trades"]
            if same_pnl and same_n:
                duplicate = True
                break
            if _setup_key(prev) == key:
                duplicate = True
                break
        if duplicate:
            continue
        picked.append(row)
        if len(picked) >= k:
            break
    return picked or ranked[:k]


def _score(metrics: dict[str, Any], initial_bank: float) -> float:
    trades = metrics["n_trades"]
    dd = max(float(metrics["max_drawdown_pct"]), 0.0)
    if trades < MIN_TRADES or dd > _dd_cap(initial_bank):
        return -1e9 + float(metrics["net_pnl"])
    pf = min(float(metrics["profit_factor"]), 6.0)
    return float(metrics["net_pnl"]) * (1.0 + pf) / (1.0 + dd / 40.0)


def build_grid(base: AppConfig, initial_bank: float | None = None) -> list[AppConfig]:
    if initial_bank is not None:
        base = overlay_config(base, account__initial_bank=float(initial_bank))
    bank = float(base.account.initial_bank)
    daily_opts = _daily_opts(bank)
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
        (150, 300),
        (200, 200),
        (200, 400),
        (250, 250),
    ]
    if bank <= 500:
        fixed_pairs = [(s, g) for s, g in fixed_pairs if s < 200]

    for (stop, gain), direction, gold, gap, entry_mode, daily in product(
        fixed_pairs,
        ("follow", "fade"),
        (True, False),
        (15.0, 40.0, None),
        ("market_open", "limit_inside"),
        daily_opts,
    ):
        offset = 120.0 if entry_mode == "limit_inside" else 0.0
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
                execution__entry_mode=entry_mode,
                execution__entry_offset_points=offset,
            )
        )

    rr_stops = (80.0, 120.0) if bank <= 500 else (80.0, 120.0, 180.0)
    for stop, rr, direction, gold, gap, daily in product(
        rr_stops,
        (1.5, 2.0, 3.0),
        ("follow", "fade"),
        (True, False),
        (40.0, None),
        daily_opts,
    ):
        add(
            overlay_config(
                base,
                risk__mode="rr",
                risk__stop_points=stop,
                risk__rr_ratio=rr,
                risk__gain_points=stop * rr,
                risk__trailing_enabled=False,
                risk__daily_loss_points=float(daily),
                filters__gold_hours_only=gold,
                filters__max_gap_points=gap,
                filters__min_predicted_body=0.0,
                execution__direction=direction,
                execution__entry_mode="market_open",
                execution__entry_offset_points=0.0,
            )
        )

    for stop_m, gain_m, direction, gold, daily in product(
        (1.0, 1.5, 2.0),
        (1.5, 2.0, 3.0),
        ("follow", "fade"),
        (True, False),
        daily_opts,
    ):
        add(
            overlay_config(
                base,
                risk__mode="atr",
                risk__atr_stop_mult=stop_m,
                risk__atr_gain_mult=gain_m,
                risk__trailing_enabled=False,
                risk__daily_loss_points=float(daily),
                filters__gold_hours_only=gold,
                filters__max_gap_points=40.0,
                filters__min_predicted_body=0.0,
                execution__direction=direction,
                execution__entry_mode="market_open",
                execution__entry_offset_points=0.0,
            )
        )

    trail_stop = 100.0 if bank <= 500 else 120.0
    trail_gain = 200.0 if bank <= 500 else 240.0
    for direction, gold, trigger, daily in product(("follow",), (True, False), (60.0, 100.0), daily_opts):
        add(
            overlay_config(
                base,
                risk__mode="fixed",
                risk__stop_points=trail_stop,
                risk__gain_points=trail_gain,
                risk__trailing_enabled=True,
                risk__trailing_trigger_points=trigger,
                risk__trailing_distance_points=50.0,
                risk__daily_loss_points=float(daily),
                filters__gold_hours_only=gold,
                filters__max_gap_points=40.0,
                filters__min_predicted_body=0.0,
                execution__direction=direction,
                execution__entry_mode="market_open",
                execution__entry_offset_points=0.0,
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


def enrich_winners(studies: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Replay each frozen winner: 1-contract periods + compound 2^n sizing."""
    cache: dict[str, tuple] = {}
    for bank in BANKS:
        for tf, train_csv, test_csv in TIMEFRAMES:
            block = studies[str(bank)].get(tf)
            if not block:
                continue
            if tf not in cache:
                train = load_candles(ROOT / train_csv)
                test_raw = load_candles(ROOT / test_csv)
                test, _leak = sanitize_test(
                    train, test_raw, Path(train_csv).name, Path(test_csv).name, match_ohlc=False
                )
                test = add_true_range(test, 14)
                predictor = joblib.load(RESULTS_DIR / f"model_{tf}.joblib")
                predicted = predictor.predict_next_ohlc(test)
                cache[tf] = (test, predicted)
            test, predicted = cache[tf]
            for winner in block["winners"]:
                cfg = AppConfig.from_dict(winner["params"])
                cfg.account.contracts = 1
                cfg.account.initial_bank = float(bank)
                print(f"  replay {winner.get('params', {}).get('name', winner['name'])} banca {bank} {tf}", flush=True)
                flat = BacktestEngine(cfg).run(test, predicted, compound=False)
                packed = _pack_side(cfg, _compact_metrics(flat))
                winner["metrics"] = packed["metrics"]
                winner["by_year"] = packed["by_year"]
                winner["by_period"] = packed["by_period"]
                winner["trades"] = []
                compounded = BacktestEngine(cfg).run(test, predicted, compound=True)
                winner["compound"] = _pack_side(cfg, _compact_metrics(compounded))


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


def run_timeframe(
    timeframe: str,
    train_csv: str,
    test_csv: str,
    initial_bank: float = 1000.0,
    n_winners: int = 2,
    predictor: CandleRegressor | None = None,
    train_scores: dict[str, Any] | None = None,
    model_path: Path | None = None,
    train: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    leakage: dict[str, Any] | None = None,
    predicted: pd.DataFrame | None = None,
    model_test_scores: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = load_config()
    base.data.timeframe = timeframe
    base.data.train_csv = train_csv
    base.data.test_csv = test_csv
    base.account.initial_bank = float(initial_bank)
    base.account.contracts = 1

    if train is None or test is None or leakage is None:
        train = load_candles(base.resolve_csv(train_csv))
        test_raw = load_candles(base.resolve_csv(test_csv))
        test, leak_report = sanitize_test(
            train, test_raw, Path(train_csv).name, Path(test_csv).name, match_ohlc=False
        )
        leakage = leak_report.to_dict()
        test = add_true_range(test, base.risk.atr_period)

    if predictor is None:
        predictor, train_scores, model_path = _fit_predictor(timeframe, train)
    assert predictor is not None
    assert train_scores is not None
    assert model_path is not None
    if predicted is None or model_test_scores is None:
        predicted = predictor.predict_next_ohlc(test)
        model_test_scores = predictor.score_predictions(test, predicted)

    grid = build_grid(base, initial_bank=initial_bank)
    runs: list[dict[str, Any]] = []
    tag = f"{timeframe}_{int(initial_bank)}"
    for i, cfg in enumerate(grid):
        cfg.name = f"{tag}_{i:04d}"
        cfg.data.timeframe = timeframe
        cfg.data.train_csv = train_csv
        cfg.data.test_csv = test_csv
        cfg.account.initial_bank = float(initial_bank)
        cfg.account.contracts = 1
        metrics = BacktestEngine(cfg).run(test, predicted)
        compact = _compact_metrics(metrics)
        packed = _summarize_run(cfg, compact, leakage, keep_trades=False)
        packed["model_train"] = train_scores
        packed["model_test"] = model_test_scores
        packed["_trades"] = compact["trades"]
        runs.append(packed)
        if (i + 1) % 50 == 0:
            print(f"  {tag}: {i + 1}/{len(grid)} configs", flush=True)

    ranked = sorted(runs, key=lambda row: row["score"], reverse=True)
    viable = [row for row in ranked if row["score"] > -1e8]
    winners = _pick_diverse(viable or ranked, n_winners)
    winner_names = {row["name"] for row in winners}
    for row in runs:
        trades = row.pop("_trades", [])
        if row["name"] in winner_names:
            row["by_year"] = _year_breakdown(trades, float(initial_bank))
        row["trades"] = []

    return {
        "timeframe": timeframe,
        "initial_bank": float(initial_bank),
        "train_file": Path(train_csv).name,
        "test_file": Path(test_csv).name,
        "resampled_test": False,
        "leakage": leakage,
        "model_train": train_scores,
        "model_test": model_test_scores,
        "model_path": str(model_path.relative_to(ROOT)),
        "n_configs": len(grid),
        "n_viable": len(viable),
        "winners": winners,
        "leaderboard": [
            {
                "name": row["name"],
                "score": row["score"],
                "net_pnl": row["metrics"]["net_pnl"],
                "win_rate": row["metrics"]["win_rate"],
                "n_trades": row["metrics"]["n_trades"],
                "profit_factor": row["metrics"]["profit_factor"],
                "max_drawdown": row["metrics"]["max_drawdown"],
                "max_drawdown_pct": row["metrics"]["max_drawdown_pct"],
                "params": {
                    "risk": row["params"]["risk"],
                    "filters": row["params"]["filters"],
                    "execution": row["params"]["execution"],
                    "account": row["params"]["account"],
                },
            }
            for row in ranked[:40]
        ],
        "all_runs_compact": [
            {
                "name": row["name"],
                "score": row["score"],
                "net_pnl": row["metrics"]["net_pnl"],
                "win_rate": row["metrics"]["win_rate"],
                "n_trades": row["metrics"]["n_trades"],
                "profit_factor": row["metrics"]["profit_factor"],
                "max_drawdown_pct": row["metrics"]["max_drawdown_pct"],
                "direction": row["params"]["execution"]["direction"],
                "mode": row["params"]["risk"]["mode"],
                "gold_hours_only": row["params"]["filters"]["gold_hours_only"],
                "entry_mode": row["params"]["execution"]["entry_mode"],
                "stop_points": row["params"]["risk"]["stop_points"],
                "gain_points": row["params"]["risk"]["gain_points"],
            }
            for row in ranked
        ],
    }


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows]
    return float(mean(vals)) if vals else 0.0


def build_insights(studies: dict[str, dict[str, dict[str, Any]]]) -> dict[str, list[str]]:
    worked: list[str] = []
    failed: list[str] = []
    improve: list[str] = []

    for bank, tfs in studies.items():
        for tf_label, key in (("1 minuto", "m1"), ("5 minutos", "m5")):
            block = tfs.get(key)
            if not block:
                continue
            label = f"R$ {int(bank):,} {tf_label}".replace(",", ".")
            rows = block["all_runs_compact"]
            if not rows:
                continue
            follow = [r for r in rows if r["direction"] == "follow"]
            fade = [r for r in rows if r["direction"] == "fade"]
            gold = [r for r in rows if r["gold_hours_only"]]
            all_day = [r for r in rows if not r["gold_hours_only"]]
            atr = [r for r in rows if r["mode"] == "atr"]
            fixed = [r for r in rows if r["mode"] == "fixed"]
            best = block["winners"][0]["metrics"] if block["winners"] else None

            if _avg(follow, "net_pnl") > _avg(fade, "net_pnl"):
                worked.append(f"{label}: seguir a previsão rendeu mais do que o fade.")
            else:
                worked.append(f"{label}: o fade (contra a previsão) se saiu melhor que seguir a cor do candle.")

            if _avg(gold, "net_pnl") > _avg(all_day, "net_pnl"):
                worked.append(f"{label}: horário-ouro (9h15–11h e 14h30–17h) melhorou o resultado.")
            else:
                failed.append(f"{label}: filtrar só o horário-ouro não superou o pregão inteiro.")

            if atr and _avg(atr, "net_pnl") > _avg(fixed, "net_pnl"):
                worked.append(f"{label}: ATR se adaptou melhor que stop/gain fixos.")
            elif atr:
                failed.append(f"{label}: ATR não superou stop e gain fixos neste período.")

            if best:
                worked.append(
                    f"{label}: melhor setup {best['n_trades']} operações, acerto {best['win_rate']:.1f}%, "
                    f"lucro R$ {best['net_pnl']:.2f}."
                )
                years = block["winners"][0].get("by_year") or {}
                y25 = years.get("2025")
                y26 = years.get("2026")
                if y25 and y26:
                    if y25["net_pnl"] > 0 and y26["net_pnl"] > 0:
                        worked.append(
                            f"{label}: o nº 1 foi positivo em 2025 (R$ {y25['net_pnl']:.0f}) e em 2026 (R$ {y26['net_pnl']:.0f})."
                        )
                    elif y26["net_pnl"] < 0:
                        failed.append(
                            f"{label}: o nº 1 lucrou no teste cheio, mas 2026 ficou negativo (R$ {y26['net_pnl']:.0f})."
                        )

    improve.extend(
        [
            "A série é o contínuo WIN$ (não um vencimento só): gaps de rolagem existem; o filtro de gap do grid tenta evitá-los.",
            "Validar no simulador do MT5 (paper) por 20–30 pregões antes de capital real.",
            "Não aumentar contratos nas bancas de R$ 500 e R$ 1.000. Stop diário proporcional à banca.",
            "Incluir filtro de volume/ATR do dia e não operar em notícia (Copom, payroll, IPC).",
        ]
    )
    return {"worked": worked, "failed": failed, "improve": improve}


def freeze_named(winner: dict[str, Any], name: str) -> str:
    cfg = AppConfig.from_dict(winner["params"])
    cfg.name = name
    winner["params"]["name"] = name
    path = CONFIGS_DIR / f"{name}.yaml"
    save_config(cfg, path)
    return str(path.relative_to(ROOT))


def freeze_winners(studies: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    written: list[str] = []
    labels = ("a", "b")
    for bank in BANKS:
        for tf in ("m1", "m5"):
            winners = studies[str(bank)][tf]["winners"]
            for i, winner in enumerate(winners[:2]):
                written.append(freeze_named(winner, f"best_{tf}_{bank}_{labels[i]}"))
    for leftover in OLD_FROZEN:
        path = CONFIGS_DIR / leftover
        if path.exists():
            path.unlink()
    return written


def _ensure_split_csvs() -> None:
    needed = [ROOT / csv for _, train, test in TIMEFRAMES for csv in (train, test)]
    if all(path.exists() for path in needed):
        return
    print("CSVs de treino/teste ausentes — gerando a partir dos dumps WIN$D.")
    prepare_study_csvs()


def _public_block(block: dict[str, Any]) -> dict[str, Any]:
    skip = {"winners"}
    return {k: v for k, v in block.items() if k not in skip}


def run_all() -> Path:
    _ensure_split_csvs()
    studies: dict[str, dict[str, dict[str, Any]]] = {str(bank): {} for bank in BANKS}
    tf_meta: dict[str, dict[str, Any]] = {}

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
        tf_meta[timeframe] = {
            "leakage": leakage,
            "model_test": model_test_scores,
            "model_train": train_scores,
        }

        for bank in BANKS:
            print(f"Estudo {timeframe} banca R$ {bank}")
            block = run_timeframe(
                timeframe,
                train_csv,
                test_csv,
                initial_bank=bank,
                n_winners=2,
                predictor=predictor,
                train_scores=train_scores,
                model_path=model_path,
                train=train,
                test=test,
                leakage=leakage,
                predicted=predicted,
                model_test_scores=model_test_scores,
            )
            studies[str(bank)][timeframe] = block

    insights = build_insights(studies)
    print("Replay dos vencedores: períodos + sizing composto")
    enrich_winners(studies)
    written = freeze_winners(studies)
    m1_ref = studies[str(BANKS[1])]["m1"]
    n_configs_total = sum(studies[str(bank)][tf]["n_configs"] for bank in BANKS for tf in ("m1", "m5"))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "Estudo educacional com o contínuo WIN$ (mini índice B3). Treino: primeiro candle da massa "
            "(18/08/2021) até 31/12/2024. Teste: 02/01/2025 até 18/08/2026. O modelo não vê o teste. "
            "Resultado passado não garante resultado futuro. Não é recomendação de investimento. "
            "Comece no simulador do MT5."
        ),
        "instrument": {
            "name": "Mini índice B3 (WIN$ contínuo)",
            "point_value": 0.20,
            "tick": 5,
            "contracts": 1,
            "train_period_m1": m1_ref["leakage"],
            "train_period_m5": studies[str(BANKS[1])]["m5"]["leakage"],
        },
        "banks": list(BANKS),
        "n_configs_total": n_configs_total,
        "timeframes": tf_meta,
        "studies": {bank: {tf: _public_block(block) for tf, block in tfs.items()} for bank, tfs in studies.items()},
        "winners": {
            bank: {"m1": tfs["m1"]["winners"], "m5": tfs["m5"]["winners"]} for bank, tfs in studies.items()
        },
        "insights": insights,
        "frozen_configs": written,
        "how_it_works": [
            "O modelo olha o candle que acabou de fechar (corpo, pavios, range e horário) e estima o próximo candle.",
            "Se a previsão é de alta, o robô compra; se é de baixa, vende. O fade inverte isso.",
            "Stop e alvo saem da configuração (fixo, risco/retorno ou ATR). Ranking oficial usa 1 contrato.",
            "Treino termina em 2024; teste começa em 2025. Anti-join remove do teste qualquer candle já visto no treino.",
            "O contínuo WIN$ junta vencimentos: gaps de rolagem existem. Combinações com gap grande são penalizadas no grid.",
            "Há uma simulação extra que dobra contratos quando a banca dobra (2×, 4×, 8×). Isso não escolhe o ranking.",
        ],
        "mt5": {
            "ready": True,
            "default_enabled": False,
            "steps": [
                "Abrir o MetaTrader 5 e autorizar algo trading.",
                "Em configs/live.yaml (ou na YAML vencedora da banca), ligar mt5.enabled: true e conferir o símbolo.",
                "Subir a API e chamar POST /api/orders só depois de validar POST /api/signal em paper.",
            ],
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "studies.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    web_public = ROOT / "web" / "public"
    web_public.mkdir(parents=True, exist_ok=True)
    (web_public / "studies.json").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Resultados em {out}")
    print("Configs congeladas:", written)
    return out


def enrich_saved_study() -> Path:
    out = RESULTS_DIR / "studies.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    studies = {
        str(bank): {
            "m1": {"winners": tfs["m1"]},
            "m5": {"winners": tfs["m5"]},
        }
        for bank, tfs in payload["winners"].items()
    }
    print("Replay dos vencedores: períodos + sizing composto")
    enrich_winners(studies)
    payload["winners"] = {
        bank: {"m1": tfs["m1"]["winners"], "m5": tfs["m5"]["winners"]} for bank, tfs in studies.items()
    }
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["how_it_works"] = [
        "O modelo olha o candle que acabou de fechar (corpo, pavios, range e horário) e estima o próximo candle.",
        "Se a previsão é de alta, o robô compra; se é de baixa, vende. O fade inverte isso.",
        "Stop e alvo saem da configuração (fixo, risco/retorno ou ATR). Ranking oficial usa 1 contrato.",
        "Treino termina em 2024; teste começa em 2025. Anti-join remove do teste qualquer candle já visto no treino.",
        "O contínuo WIN$ junta vencimentos: gaps de rolagem existem. Combinações com gap grande são penalizadas no grid.",
        "Há uma simulação extra que dobra contratos quando a banca dobra (2×, 4×, 8×). Isso não escolhe o ranking.",
    ]
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    web_public = ROOT / "web" / "public"
    (web_public / "studies.json").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Estudo enriquecido em {out}")
    return out
