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
# Hard reject (never publish): below this. Soft reject (over DD% cap): still ranked among survivors.
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
        cache[tf] = (test, predicted)
    return cache


def _replay_cfg(cfg: AppConfig, test, predicted) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg.account.contracts = 1
    flat = BacktestEngine(cfg).run(test, predicted, compound=False)
    compounded = BacktestEngine(cfg).run(test, predicted, compound=True)
    return _pack_side(cfg, _compact_metrics(flat)), _pack_side(cfg, _compact_metrics(compounded))


def enrich_winners(studies: dict[str, dict[str, dict[str, Any]]], cache: dict[str, tuple] | None = None) -> None:
    """Replay each frozen winner: 1-contract periods + compound 2^n sizing."""
    cache = cache if cache is not None else _prediction_cache()
    for bank in BANKS:
        for tf, _train_csv, _test_csv in TIMEFRAMES:
            block = studies[str(bank)].get(tf)
            if not block:
                continue
            test, predicted = cache[tf]
            for winner in block["winners"]:
                cfg = AppConfig.from_dict(winner["params"])
                cfg.account.initial_bank = float(bank)
                print(f"  replay {winner.get('params', {}).get('name', winner['name'])} banca {bank} {tf}", flush=True)
                packed, compounded = _replay_cfg(cfg, test, predicted)
                winner["metrics"] = packed["metrics"]
                winner["by_year"] = packed["by_year"]
                winner["by_period"] = packed["by_period"]
                winner["trades"] = []
                winner["compound"] = compounded


def _setup_label(params: dict[str, Any]) -> str:
    risk = params["risk"]
    exe = params["execution"]
    filters = params["filters"]
    direction = "follow" if exe.get("direction") == "follow" else "fade"
    daily = float(risk.get("daily_loss_points") or 0)
    daily_s = "sem daily" if daily <= 0 else f"daily {int(daily)}"
    trail = " trailing" if risk.get("trailing_enabled") else ""
    gold = " · ouro" if filters.get("gold_hours_only") else ""
    if risk.get("mode") == "atr":
        body = f"ATR {risk.get('atr_stop_mult')}×/{risk.get('atr_gain_mult')}×"
    else:
        body = f"{int(float(risk.get('stop_points') or 0))}/{int(float(risk.get('gain_points') or 0))}"
    return f"{direction} {body}{trail} · {daily_s}{gold}"


def _month_count(side: dict[str, Any]) -> int:
    months = (side.get("by_period") or {}).get("monthly") or []
    return max(len(months), 1)


def _avg_month(side: dict[str, Any]) -> float:
    return float(side["metrics"]["net_pnl"]) / _month_count(side)


def _metrics_from_row(row: dict[str, Any], bank: float) -> dict[str, Any]:
    dd_pct = float(row.get("max_drawdown_pct") or 0.0)
    if row.get("max_drawdown") is not None:
        dd_abs = float(row["max_drawdown"])
    else:
        dd_abs = dd_pct / 100.0 * float(bank)
    return {
        "n_trades": int(row.get("n_trades") or 0),
        "net_pnl": float(row.get("net_pnl") or 0.0),
        "profit_factor": float(row.get("profit_factor") or 0.0),
        "max_drawdown": dd_abs,
        "max_drawdown_pct": dd_pct,
    }


def _full_params_from_leaderboard(
    entry: dict[str, Any],
    timeframe: str,
    train_csv: str,
    test_csv: str,
    bank: float,
) -> dict[str, Any]:
    data = load_config().to_dict()
    data["name"] = entry["name"]
    data["account"].update(entry["params"].get("account") or {})
    data["account"]["initial_bank"] = float(bank)
    data["account"]["contracts"] = 1
    data["risk"].update(entry["params"]["risk"])
    data["filters"].update(entry["params"]["filters"])
    data["execution"].update(entry["params"]["execution"])
    data["data"]["timeframe"] = timeframe
    data["data"]["train_csv"] = train_csv
    data["data"]["test_csv"] = test_csv
    return data


def _run_from_leaderboard(
    entry: dict[str, Any],
    bank: float,
    timeframe: str,
    train_csv: str,
    test_csv: str,
) -> dict[str, Any]:
    metrics = _metrics_from_row(entry, bank)
    params = _full_params_from_leaderboard(entry, timeframe, train_csv, test_csv, bank)
    return {
        "name": entry["name"],
        "params": params,
        "score": _score(metrics, float(bank)),
        "metrics": metrics,
        "trades": [],
    }


def _select_winners(ranked: list[dict[str, Any]], n_winners: int = 2) -> list[dict[str, Any]]:
    viable = [row for row in ranked if _is_viable(row["score"])]
    survivors = [row for row in ranked if _passes_hard_floor(row["score"])]
    return _pick_diverse(viable, n_winners) or _pick_diverse(survivors, n_winners)


def replay_cross_bank(
    studies: dict[str, dict[str, dict[str, Any]]],
    cache: dict[str, tuple] | None = None,
) -> list[dict[str, Any]]:
    """Same risk/filters/execution on every bank so R$ 5.000 is not compared to another setup."""
    cache = cache if cache is not None else _prediction_cache()
    refs: list[tuple[str, dict[str, Any]]] = []
    for tf in ("m1", "m5"):
        winners = studies.get("1000", {}).get(tf, {}).get("winners") or []
        if winners:
            refs.append((tf, winners[0]))
    out: list[dict[str, Any]] = []
    for tf, winner in refs:
        test, predicted = cache[tf]
        by_bank: dict[str, Any] = {}
        for bank in BANKS:
            cfg = AppConfig.from_dict(winner["params"])
            cfg.account.initial_bank = float(bank)
            print(f"  mesma config {tf} {_setup_label(winner['params'])} banca {bank}", flush=True)
            packed, compounded = _replay_cfg(cfg, test, predicted)
            by_bank[str(bank)] = {**packed, "compound": compounded}
        out.append(
            {
                "id": f"{tf}_{winner['params']['execution']['direction']}_{winner['params']['risk'].get('mode')}",
                "timeframe": tf,
                "label": _setup_label(winner["params"]),
                "source_name": winner["params"].get("name", winner["name"]),
                "note": (
                    "Mesmos stop, alvo, trailing e stop diário. Só muda a banca inicial. "
                    "Com 1 contrato o P&L em reais é o mesmo; o tombo em % cai na banca maior. "
                    "O composto acelera mais cedo na banca pequena."
                ),
                "params": {
                    "risk": winner["params"]["risk"],
                    "filters": winner["params"]["filters"],
                    "execution": winner["params"]["execution"],
                },
                "by_bank": by_bank,
            }
        )
    return out


def build_parecer(
    studies: dict[str, dict[str, dict[str, Any]]],
    tf_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    m1_hit = round(100.0 * float(tf_meta["m1"]["model_test"]["test_direction_hit"]), 1)
    m5_hit = round(100.0 * float(tf_meta["m5"]["model_test"]["test_direction_hit"]), 1)
    monthly: list[dict[str, Any]] = []
    highlight_name = ""
    best_1c = -1e18
    for bank in BANKS:
        for tf in ("m1", "m5"):
            for winner in studies[str(bank)][tf].get("winners") or []:
                side = winner
                compound = winner.get("compound") or winner
                dd_pct = float(side["metrics"]["max_drawdown_pct"])
                note = ""
                if float(side["metrics"]["net_pnl"]) > best_1c:
                    best_1c = float(side["metrics"]["net_pnl"])
                    highlight_name = winner["params"].get("name", winner["name"])
                if tf == "m5" and dd_pct <= 35:
                    note = "melhor equilíbrio desta banca" if bank == 500 else ""
                monthly.append(
                    {
                        "bank": int(bank),
                        "name": winner["params"].get("name", winner["name"]),
                        "timeframe": tf,
                        "label": _setup_label(winner["params"]),
                        "avg_1c": round(_avg_month(side), 2),
                        "avg_compound": round(_avg_month(compound), 2),
                        "dd_pct_1c": round(dd_pct, 1),
                        "dd_abs_1c": round(float(side["metrics"]["max_drawdown"]), 2),
                        "net_pnl_1c": round(float(side["metrics"]["net_pnl"]), 2),
                        "net_pnl_compound": round(float(compound["metrics"]["net_pnl"]), 2),
                        "max_contracts": int(compound.get("max_contracts") or 1),
                        "n_months": _month_count(side),
                        "note": note,
                    }
                )
    for row in monthly:
        if row["name"] == highlight_name:
            row["note"] = (row["note"] + " · " if row["note"] else "") + "melhor 1 contrato do estudo"

    m5_1000 = (studies.get("1000", {}).get("m5", {}).get("winners") or [None])[0]
    m5_5000 = (studies.get("5000", {}).get("m5", {}).get("winners") or [None])[0]
    same_m5 = False
    if m5_1000 and m5_5000:
        same_m5 = _setup_key(m5_1000) == _setup_key(m5_5000) and abs(
            float(m5_1000["metrics"]["net_pnl"]) - float(m5_5000["metrics"]["net_pnl"])
        ) < 0.05

    why_5k = [
        {
            "title": "Score em % da banca",
            "body": (
                "Tombo de R$ 530 é 106% de R$ 500 (inoperável) e só 10,6% de R$ 5.000. "
                "Em R$ 5.000 o ranking premia setups calmos; em R$ 500 o teto de DD percentual é tão duro "
                "que quase ninguém passa — o nº 1 antigo de 1 min tinha DD 348% e só venceu no fallback por P&L bruto. "
                "Agora qualquer setup com tombo ≥ banca é rejeitado."
            ),
        },
        {
            "title": "Stop diário diferente no grid",
            "body": (
                "O grid de R$ 500 testa daily 80/150/0; o de R$ 5.000 testa 250/400/0. "
                "Sem teto diário o P&L e o tombo incham juntos. Por isso o ranking de cada banca escolhe outro setup."
            ),
        },
        {
            "title": "Composto começa tarde em R$ 5.000",
            "body": (
                "2 contratos só quando a banca chega a R$ 10.000; 4 contratos a R$ 20.000. "
                "Em R$ 500, 2 contratos já em R$ 1.000 — o composto explode no papel e o DD também."
            ),
        },
        {
            "title": "1 contrato não cresce com a banca",
            "body": (
                "Com 1 contrato o lucro em reais é o da máquina (ponto = R$ 0,20), não da banca. "
                + (
                    "O nº 1 de 5 min em R$ 1.000 e R$ 5.000 é o mesmo setup: P&L idêntico. A banca maior não piora o 1 contrato."
                    if same_m5
                    else "A tabela “mesma config” abaixo isola esse efeito: só muda a banca inicial."
                )
            ),
        },
    ]
    strategy = [
        f"O modelo não prevê cor (acerto de direção {m1_hit}% em 1 min e {m5_hit}% em 5 min). "
        "Lucro vem de assimetria stop/alvo, trailing e filtro de gap — não de acertar o candle.",
        "5 min follow 120/240 com trailing é o padrão mais estável e o candidato a setup de trabalho.",
        "1 min só faz sentido como fade ou scalps 1:1 com stop diário ligado. ATR fade sem daily e DD > banca é overfitting.",
        "Horário-ouro no grid médio não ganhou do pregão inteiro; vencedores que ainda o usam são artefato de seleção.",
        "Custo de R$ 1 por operação já está no P&L. ~2.000–3.000 trades em ~20 meses = teto de 8 ops/dia quase todo dia; "
        "custo e slippage reais comem esse edge.",
    ]
    improvements = [
        "Piso duro: max_drawdown < banca inicial (já aplicado neste ranking).",
        "Comparar a mesma config nas 3 bancas com 1 contrato (tabela abaixo) para não misturar o ranking.",
        "Em R$ 5.000, começar com 2–4 contratos — senão o capital fica ocioso; o composto atual só acelera depois de dobrar a banca.",
        "Obrigar stop diário no 5 min que hoje tem daily=0 (o melhor P&L). Ex.: 150–250 pts, proporcional à banca.",
        "Subir custo/slippage no backtest (R$ 2–4/op ou 1 tick) e ver quem sobrevive.",
        "Paper 20–30 pregões no 5 min follow 120/240 trailing antes de composto.",
        "Não ligar composto em R$ 500/1.000 até o 1 contrato sobreviver no paper: o gráfico composto assume margem para 16 mini WIN.",
    ]
    return {
        "headline": "Por que R$ 5.000 parece pior — e o que o estudo realmente diz",
        "ml_hit": {"m1": m1_hit, "m5": m5_hit},
        "n_months_note": "Média = P&L total / meses com trade no teste. Não é garantia de mês cheio na conta real.",
        "why_5k": why_5k,
        "monthly": monthly,
        "strategy": strategy,
        "improvements": improvements,
        "dd_floor": "Setups com tombo maior ou igual à banca inicial são rejeitados no ranking.",
    }


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
    winners = _select_winners(ranked, n_winners)
    viable = [row for row in ranked if _is_viable(row["score"])]
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
                "max_drawdown": row["metrics"]["max_drawdown"],
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
            rows = block.get("all_runs_compact") or []
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
            "Setups com tombo maior ou igual à banca inicial não entram no ranking (piso duro).",
            "Validar no simulador do MT5 (paper) por 20–30 pregões antes de capital real.",
            "Não aumentar contratos nas bancas de R$ 500 e R$ 1.000. Stop diário proporcional à banca.",
            "Incluir filtro de volume/ATR do dia e não operar em notícia (Copom, payroll, IPC).",
            "Paper 20–30 pregões no 5 min follow 120/240 com trailing antes de ligar composto.",
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
    keep = {Path(path).name for path in written}
    for leftover in OLD_FROZEN:
        path = CONFIGS_DIR / leftover
        if path.exists() and path.name not in keep:
            path.unlink()
    for path in CONFIGS_DIR.glob("best_m[15]_*.yaml"):
        if path.name not in keep:
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


HOW_IT_WORKS = [
    "O modelo olha o candle que acabou de fechar (corpo, pavios, range e horário) e estima o próximo candle.",
    "Se a previsão é de alta, o robô compra; se é de baixa, vende. O fade inverte isso.",
    "Stop e alvo saem da configuração (fixo, risco/retorno ou ATR). Ranking oficial usa 1 contrato.",
    "Treino termina em 2024; teste começa em 2025. Anti-join remove do teste qualquer candle já visto no treino.",
    "O contínuo WIN$ junta vencimentos: gaps de rolagem existem. Combinações com gap grande são penalizadas no grid.",
    "Há uma simulação extra que dobra contratos quando a banca dobra (2×, 4×, 8×). Isso não escolhe o ranking.",
    "Setups com tombo maior ou igual à banca inicial são rejeitados. O ranking não mistura a mesma config nas 3 bancas — a tabela de parecer faz essa comparação.",
]


def _dump_payload(payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "studies.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    web_public = ROOT / "web" / "public"
    web_public.mkdir(parents=True, exist_ok=True)
    (web_public / "studies.json").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def _attach_study_extras(payload: dict[str, Any], studies: dict[str, dict[str, dict[str, Any]]]) -> None:
    cache = _prediction_cache()
    print("Replay dos vencedores: períodos + sizing composto")
    enrich_winners(studies, cache=cache)
    payload["frozen_configs"] = freeze_winners(studies)
    print("Mesma config nas 3 bancas")
    payload["cross_bank"] = replay_cross_bank(studies, cache=cache)
    payload["parecer"] = build_parecer(studies, payload["timeframes"])
    payload["insights"] = build_insights(studies)
    payload["winners"] = {
        bank: {"m1": tfs["m1"]["winners"], "m5": tfs["m5"]["winners"]} for bank, tfs in studies.items()
    }
    payload["how_it_works"] = HOW_IT_WORKS
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()


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
    _attach_study_extras(payload, studies)
    out = _dump_payload(payload)
    print(f"Resultados em {out}")
    print("Configs congeladas:", payload["frozen_configs"])
    return out


def enrich_saved_study() -> Path:
    out = RESULTS_DIR / "studies.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    studies = {}
    for bank, tfs in payload["winners"].items():
        studies[str(bank)] = {}
        for tf in ("m1", "m5"):
            pub = (payload.get("studies") or {}).get(str(bank), {}).get(tf) or {}
            studies[str(bank)][tf] = {**pub, "winners": tfs[tf]}
    _attach_study_extras(payload, studies)
    out = _dump_payload(payload)
    print(f"Estudo enriquecido em {out}")
    return out


def rerank_saved_study() -> Path:
    """Re-pick winners from the saved leaderboard with the DD < bank floor; no full grid."""
    out = RESULTS_DIR / "studies.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    studies: dict[str, dict[str, dict[str, Any]]] = {}
    tf_files = {tf: (train, test) for tf, train, test in TIMEFRAMES}

    for bank in BANKS:
        studies[str(bank)] = {}
        for tf in ("m1", "m5"):
            train_csv, test_csv = tf_files[tf]
            pub = payload["studies"][str(bank)][tf]
            for row in pub.get("all_runs_compact") or []:
                metrics = _metrics_from_row(row, float(bank))
                row["score"] = _score(metrics, float(bank))
                row["max_drawdown"] = metrics["max_drawdown"]
            pub["n_viable"] = sum(1 for row in pub.get("all_runs_compact") or [] if _is_viable(row["score"]))
            for entry in pub.get("leaderboard") or []:
                metrics = _metrics_from_row(entry, float(bank))
                entry["score"] = _score(metrics, float(bank))
            pub["leaderboard"] = sorted(pub.get("leaderboard") or [], key=lambda item: item["score"], reverse=True)
            ranked = [
                _run_from_leaderboard(entry, float(bank), tf, train_csv, test_csv)
                for entry in pub["leaderboard"]
            ]
            winners = _select_winners(ranked, 2)
            leak = payload["timeframes"][tf]["leakage"]
            model_test = payload["timeframes"][tf]["model_test"]
            for winner in winners:
                winner["leakage"] = leak
                winner["model_test"] = model_test
            studies[str(bank)][tf] = {**pub, "winners": winners}

    payload["studies"] = {
        bank: {tf: _public_block(block) for tf, block in tfs.items()} for bank, tfs in studies.items()
    }
    payload["n_configs_total"] = sum(
        payload["studies"][str(bank)][tf]["n_configs"] for bank in BANKS for tf in ("m1", "m5")
    )
    _attach_study_extras(payload, studies)
    out = _dump_payload(payload)
    print(f"Ranking refeito em {out}")
    print("Configs congeladas:", payload["frozen_configs"])
    return out
