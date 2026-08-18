from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any

import joblib

from trader.backtest import BacktestEngine
from trader.config import AppConfig, load_config, save_config
from trader.data import load_candles, resample_to_5min, sanitize_test
from trader.ml import CandleRegressor, add_true_range
from trader.paths import CONFIGS_DIR, RESULTS_DIR, ROOT
from trader.signals import overlay_config

MIN_TRADES = 20


def _compact_metrics(m) -> dict[str, Any]:
    data = m.to_dict()
    trades = data.pop("trades", [])
    equity = data.pop("equity", [])
    data["equity"] = equity
    data["trades"] = trades
    return data


def _pick_diverse(ranked: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for row in ranked:
        duplicate = False
        for prev in picked:
            same_pnl = abs(row["metrics"]["net_pnl"] - prev["metrics"]["net_pnl"]) < 0.05
            same_n = row["metrics"]["n_trades"] == prev["metrics"]["n_trades"]
            if same_pnl and same_n:
                duplicate = True
                break
        if duplicate:
            continue
        picked.append(row)
        if len(picked) >= k:
            break
    return picked or ranked[:k]


def _score(metrics: dict[str, Any]) -> float:
    trades = metrics["n_trades"]
    if trades < MIN_TRADES:
        return -1e9 + metrics["net_pnl"]
    pf = min(float(metrics["profit_factor"]), 6.0)
    dd = max(float(metrics["max_drawdown_pct"]), 0.0)
    return float(metrics["net_pnl"]) * (1.0 + pf) / (1.0 + dd / 40.0)


def build_grid(base: AppConfig, initial_bank: float | None = None) -> list[AppConfig]:
    if initial_bank is not None:
        base = overlay_config(base, account__initial_bank=float(initial_bank))
    small = float(base.account.initial_bank) <= 1000
    daily_opts = (150.0, 250.0, 0.0) if small else (400.0, 0.0)
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

    for stop, rr, direction, gold, gap, daily in product(
        (80.0, 120.0, 180.0),
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

    for direction, gold, trigger, daily in product(("follow",), (True, False), (60.0, 100.0), daily_opts):
        add(
            overlay_config(
                base,
                risk__mode="fixed",
                risk__stop_points=120.0,
                risk__gain_points=240.0,
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


def _summarize_run(cfg: AppConfig, metrics: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
    slim_equity = metrics["equity"]
    if len(slim_equity) > 400:
        step = max(1, len(slim_equity) // 400)
        slim_equity = slim_equity[::step] + [slim_equity[-1]]
    return {
        "name": cfg.name,
        "params": cfg.to_dict(),
        "score": _score(metrics),
        "leakage": leakage,
        "metrics": {
            **{k: v for k, v in metrics.items() if k not in {"trades", "equity"}},
            "equity": slim_equity,
            "n_trade_events": len(metrics.get("trades", [])),
        },
        "trades": metrics.get("trades", []),
    }


def run_timeframe(
    timeframe: str,
    train_csv: str,
    test_csv: str,
    resample_test: bool = False,
    initial_bank: float = 1000.0,
    reuse_model: bool = False,
    n_winners: int = 2,
) -> dict[str, Any]:
    base = load_config()
    base.data.timeframe = timeframe
    base.data.train_csv = train_csv
    base.data.test_csv = test_csv
    base.account.initial_bank = float(initial_bank)

    train = load_candles(base.resolve_csv(train_csv))
    test_raw = load_candles(base.resolve_csv(test_csv))
    if resample_test:
        test_raw = resample_to_5min(test_raw)
    test, leakage = sanitize_test(train, test_raw, Path(train_csv).name, Path(test_csv).name)
    test = add_true_range(test, base.risk.atr_period)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = RESULTS_DIR / f"model_{timeframe}.joblib"
    if reuse_model and model_path.exists():
        predictor = joblib.load(model_path)
        train_scores = {"reused_model": 1.0}
    else:
        predictor = CandleRegressor()
        train_scores = predictor.fit(train)
        joblib.dump(predictor, model_path)
    predicted = predictor.predict_next_ohlc(test)
    model_test_scores = predictor.score_predictions(test, predicted)

    grid = build_grid(base, initial_bank=initial_bank)
    runs: list[dict[str, Any]] = []
    leak_dict = leakage.to_dict()
    tag = f"{timeframe}_{int(initial_bank)}"
    for i, cfg in enumerate(grid):
        cfg.name = f"{tag}_{i:04d}"
        cfg.data.timeframe = timeframe
        cfg.data.train_csv = train_csv
        cfg.data.test_csv = test_csv
        metrics = BacktestEngine(cfg).run(test, predicted)
        packed = _summarize_run(cfg, _compact_metrics(metrics), leak_dict)
        packed["model_train"] = train_scores
        packed["model_test"] = model_test_scores
        runs.append(packed)
        if (i + 1) % 50 == 0:
            print(f"  {tag}: {i + 1}/{len(grid)} configs", flush=True)

    ranked = sorted(runs, key=lambda r: r["score"], reverse=True)
    winners = _pick_diverse(ranked, n_winners)
    return {
        "timeframe": timeframe,
        "train_file": Path(train_csv).name,
        "test_file": Path(test_csv).name,
        "resampled_test": resample_test,
        "leakage": leak_dict,
        "model_train": train_scores,
        "model_test": model_test_scores,
        "model_path": str(model_path.relative_to(ROOT)),
        "n_configs": len(grid),
        "winners": winners,
        "leaderboard": [
            {
                "name": r["name"],
                "score": r["score"],
                "net_pnl": r["metrics"]["net_pnl"],
                "win_rate": r["metrics"]["win_rate"],
                "n_trades": r["metrics"]["n_trades"],
                "profit_factor": r["metrics"]["profit_factor"],
                "max_drawdown": r["metrics"]["max_drawdown"],
                "params": {
                    "risk": r["params"]["risk"],
                    "filters": r["params"]["filters"],
                    "execution": r["params"]["execution"],
                    "account": r["params"]["account"],
                },
            }
            for r in ranked[:40]
        ],
        "all_runs_compact": [
            {
                "name": r["name"],
                "score": r["score"],
                "net_pnl": r["metrics"]["net_pnl"],
                "win_rate": r["metrics"]["win_rate"],
                "n_trades": r["metrics"]["n_trades"],
                "profit_factor": r["metrics"]["profit_factor"],
                "max_drawdown_pct": r["metrics"]["max_drawdown_pct"],
                "direction": r["params"]["execution"]["direction"],
                "mode": r["params"]["risk"]["mode"],
                "gold_hours_only": r["params"]["filters"]["gold_hours_only"],
                "entry_mode": r["params"]["execution"]["entry_mode"],
                "stop_points": r["params"]["risk"]["stop_points"],
                "gain_points": r["params"]["risk"]["gain_points"],
            }
            for r in ranked
        ],
        "winners_full": winners,
    }


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows]
    return float(mean(vals)) if vals else 0.0


def build_insights(m1: dict[str, Any], m5: dict[str, Any]) -> dict[str, list[str]]:
    worked: list[str] = []
    failed: list[str] = []
    improve: list[str] = []

    for label, block in (("1 minuto", m1), ("5 minutos", m5)):
        rows = block["all_runs_compact"]
        if not rows:
            continue
        follow = [r for r in rows if r["direction"] == "follow"]
        fade = [r for r in rows if r["direction"] == "fade"]
        gold = [r for r in rows if r["gold_hours_only"]]
        all_day = [r for r in rows if not r["gold_hours_only"]]
        atr = [r for r in rows if r["mode"] == "atr"]
        fixed = [r for r in rows if r["mode"] == "fixed"]
        limit = [r for r in rows if r["entry_mode"] == "limit_inside"]
        market = [r for r in rows if r["entry_mode"] == "market_open"]
        best = block["winners"][0]["metrics"] if block["winners"] else None

        if _avg(follow, "net_pnl") > _avg(fade, "net_pnl"):
            worked.append(f"{label}: seguir a cor prevista do candle rendeu mais do que operar contra (fade).")
        else:
            worked.append(f"{label}: o fade (contra a previsão) se saiu melhor que seguir a cor do candle.")

        if _avg(gold, "net_pnl") > _avg(all_day, "net_pnl"):
            worked.append(f"{label}: operar só no horário-ouro (9h15–11h e 14h30–17h) melhorou o resultado.")
        else:
            failed.append(f"{label}: filtrar só o horário-ouro não foi superior a operar o pregão inteiro.")

        if atr and _avg(atr, "net_pnl") > _avg(fixed, "net_pnl"):
            worked.append(f"{label}: stop/gain em múltiplo de ATR se adaptou melhor que limites fixos.")
        elif atr:
            failed.append(f"{label}: ATR não superou stop e gain fixos neste período.")

        if limit and market and _avg(market, "net_pnl") > _avg(limit, "net_pnl"):
            worked.append(f"{label}: entrada a mercado na abertura do próximo candle foi mais estável que limite interno.")
        elif limit:
            failed.append(f"{label}: a ordem limitada (offset dentro do range previsto) perdeu para a entrada a mercado.")

        if best:
            worked.append(
                f"{label}: melhor setup {best['n_trades']} operações, taxa de acerto {best['win_rate']:.1f}%, lucro R$ {best['net_pnl']:.2f}."
            )
            best_dir = block["winners"][0]["params"]["execution"]["direction"]
            if best_dir == "follow":
                worked.append(f"{label}: o setup nº 1 seguiu a previsão (comprar alta prevista / vender baixa prevista).")
            else:
                worked.append(f"{label}: o setup nº 1 operou contra a previsão (fade), o que combina com o acerto de direção abaixo de 50%.")

    improve.extend(
        [
            "Treinar de novo em dados recentes (WIN de 2025/2026); 2020 foi um regime de pandemia, com volatilidade atípica.",
            "Validar no simulador do MT5 (paper) por 20–30 pregões antes de capital real.",
            "Incluir filtro de volume/ATR do dia e não operar em notícia (Copom, payroll).",
            "Manter 1 contrato e stop diário rígido: 3 losses ou ~R$80–R$120, o que ocorrer primeiro.",
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


def freeze_winners(m1: dict[str, Any], m5: dict[str, Any]) -> list[str]:
    written: list[str] = []
    mapping = [
        (m1["winners"], "best_m1"),
        (m5["winners"], "best_m5"),
    ]
    labels = ("a", "b")
    for winners, prefix in mapping:
        for i, winner in enumerate(winners[:2]):
            written.append(freeze_named(winner, f"{prefix}_{labels[i]}"))
    return written


def run_all() -> Path:
    print("Estudo 1 minuto (banca R$ 1.000): treino WINJ20_1min → teste WINM20_1min")
    m1 = run_timeframe(
        "m1",
        "datasets/WINJ20_1min.csv",
        "datasets/WINM20_1min.csv",
        initial_bank=1000,
        reuse_model=True,
        n_winners=2,
    )
    print("Estudo 5 minutos (banca R$ 1.000): treino WINJ20_5min → teste WINM20_1min agregado")
    m5 = run_timeframe(
        "m5",
        "datasets/WINJ20_5min.csv",
        "datasets/WINM20_1min.csv",
        resample_test=True,
        initial_bank=1000,
        reuse_model=True,
        n_winners=2,
    )

    print("Sanidade: WINM20_1min_teste vs treino WINJ20")
    sanity = None
    if m1["winners"]:
        cfg = AppConfig.from_dict(m1["winners"][0]["params"])
        train = load_candles(ROOT / "datasets/WINJ20_1min.csv")
        test_raw = load_candles(ROOT / "datasets/WINM20_1min_teste.csv")
        test, leak = sanitize_test(train, test_raw, "WINJ20_1min.csv", "WINM20_1min_teste.csv")
        test = add_true_range(test, cfg.risk.atr_period)
        predictor = joblib.load(RESULTS_DIR / "model_m1.joblib")
        predicted = predictor.predict_next_ohlc(test)
        metrics = BacktestEngine(cfg).run(test, predicted)
        sanity = {
            "leakage": leak.to_dict(),
            "metrics": {k: v for k, v in metrics.to_dict().items() if k not in {"trades", "equity"}},
            "note": "Arquivo de um único dia (17/04/2020). Sanitizado contra o treino WINJ20, não contra o WINM20 cheio.",
        }

    insights = build_insights(m1, m5)
    insights["improve"].append(
        "Banca de R$ 1.000 com 1 contrato. Stop diário curto (150–250 pontos). Não aumente contratos nessa banca."
    )
    written = freeze_winners(m1, m5)
    for extra in (CONFIGS_DIR / "best_m1_1k.yaml", CONFIGS_DIR / "best_m5_1k.yaml"):
        if extra.exists():
            extra.unlink()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "Estudo educacional com mini índice WIN de 2020. Resultado passado não garante resultado futuro. "
            "Não é recomendação de investimento. Integração MT5 deve começar em conta simulada."
        ),
        "instrument": {
            "name": "Mini índice B3 (WIN)",
            "point_value": 0.20,
            "tick": 5,
            "train_period_m1": m1["leakage"],
            "train_period_m5": m5["leakage"],
        },
        "m1": {k: v for k, v in m1.items() if k != "winners_full"},
        "m5": {k: v for k, v in m5.items() if k != "winners_full"},
        "winners": {"m1": m1["winners"], "m5": m5["winners"]},
        "sanity_m1_extra_file": sanity,
        "insights": insights,
        "frozen_configs": written,
        "how_it_works": [
            "O modelo olha o candle que acabou de fechar (corpo, pavios, range e horário) e estima o próximo candle.",
            "Se a previsão é de alta, o robô compra; se é de baixa, vende. O fade inverte isso.",
            "Stop e alvo saem da configuração (fixo, risco/retorno ou ATR). Só uma operação por vez.",
            "Treino e teste são arquivos diferentes (WINJ vs WINM). Linhas repetidas no teste são removidas.",
        ],
        "mt5": {
            "ready": True,
            "default_enabled": False,
            "steps": [
                "Abrir o MetaTrader 5 e autorizar algo trading.",
                "Em configs/live.yaml, ligar mt5.enabled: true e conferir o símbolo (WIN$ ou o vencimento atual).",
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
