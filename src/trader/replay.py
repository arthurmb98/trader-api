from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from trader.backtest import BacktestEngine
from trader.broker import Mt5Broker
from trader.config import AppConfig, load_config
from trader.data import frame_from_candles, load_candles
from trader.ml import CandleRegressor, add_candle_features, add_true_range
from trader.paths import CONFIGS_DIR, DATASETS_DIR, RESULTS_DIR, ROOT
from trader.price_action import lookback_for_timeframe, strange_from_frame

DEFAULT_CONFIG = "best_candles_m5_1000_a"
DEFAULT_FROM = "2026-08-17"
DEFAULT_TO = "2026-08-21"
DEFAULT_CSV = DATASETS_DIR / "mt5_m5_week.csv"
TEST_FALLBACK = DATASETS_DIR / "WIN_5min_test.csv"
MAC_HINT = """No MetaTrader 5 (Mac ou Windows), com a demo logada:
  1. Market Watch → duplo clique em WIN$ (ou o vencimento, ex. WINQ26)
  2. Timeframe M5
  3. Clique direito no gráfico → Exportar / Salvar como CSV
     de ~10/08/2026 até 21/08/2026 (precisa de candles antes da segunda)
  4. Salve em datasets/mt5_m5_week.csv

Ou compile mt5/ExportM5Week.mq5 no MetaEditor, rode no gráfico M5
e copie o arquivo de MQL5/Files para datasets/mt5_m5_week.csv.

No Windows, com o terminal aberto: PYTHONPATH=src python -m trader replay --source mt5
"""
EXPORT_HINT = "CSV do MT5 não encontrado.\n\n" + MAC_HINT


def load_named_config(name: str) -> AppConfig:
    path = Path(name)
    if not path.exists():
        path = CONFIGS_DIR / name
    if not path.exists():
        path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config {name} não encontrada em {CONFIGS_DIR}")
    return load_config(path)


def ensure_model(timeframe: str, train_csv: Path) -> CandleRegressor:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"model_{timeframe}.joblib"
    if path.exists():
        model = joblib.load(path)
        if getattr(model, "is_fitted", False):
            print(f"Modelo: {path.relative_to(ROOT)}", flush=True)
            return model
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Sem modelo em {path} e sem CSV de treino {train_csv}. "
            "Rode python -m trader prepare-data."
        )
    print(f"Treinando modelo {timeframe} só em {train_csv.relative_to(ROOT)} …", flush=True)
    train = load_candles(train_csv)
    model = CandleRegressor()
    scores = model.fit(train)
    joblib.dump(model, path)
    print(
        f"Gravado {path.relative_to(ROOT)} ({int(scores['train_rows'])} candles de treino).",
        flush=True,
    )
    return model


def _end_exclusive(day: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(day.date()) + pd.Timedelta(days=1)


def _load_csv_candles(csv_path: Path) -> tuple[pd.DataFrame, str]:
    if csv_path.exists():
        return load_candles(csv_path), str(csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path)
    if csv_path.resolve() != TEST_FALLBACK.resolve() and TEST_FALLBACK.exists():
        print(
            f"Aviso: {csv_path} ausente. Usando {TEST_FALLBACK.relative_to(ROOT)} "
            "(pode faltar quarta a sexta da semana).",
            flush=True,
        )
        print(EXPORT_HINT, flush=True)
        return load_candles(TEST_FALLBACK), str(TEST_FALLBACK.relative_to(ROOT))
    raise FileNotFoundError(EXPORT_HINT)


def _load_mt5_candles(cfg: AppConfig, start: pd.Timestamp, end: pd.Timestamp, warmup_days: int) -> pd.DataFrame:
    mt5 = cfg.mt5
    broker = Mt5Broker(mt5.symbol, mt5.magic, mt5.deviation, mt5.filling, mt5.comment)
    date_from = (start - pd.Timedelta(days=warmup_days)).to_pydatetime()
    date_to = (_end_exclusive(end) - pd.Timedelta(seconds=1)).to_pydatetime()
    broker.connect()
    try:
        candles = broker.copy_rates_range(mt5.symbol, cfg.data.timeframe, date_from, date_to)
    finally:
        broker.shutdown()
    return frame_from_candles(candles, source_file="mt5")


def load_replay_frame(
    cfg: AppConfig,
    source: str,
    csv_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    warmup_days: int,
) -> tuple[pd.DataFrame, str]:
    kind = source.strip().lower()
    if kind == "mt5":
        return _load_mt5_candles(cfg, start, end, warmup_days), f"mt5:{cfg.mt5.symbol}"
    if kind == "csv":
        return _load_csv_candles(csv_path)
    if kind == "auto":
        if csv_path.exists():
            return _load_csv_candles(csv_path)
        try:
            return _load_mt5_candles(cfg, start, end, warmup_days), f"mt5:{cfg.mt5.symbol}"
        except RuntimeError as exc:
            print(f"MT5 indisponível ({exc}). Tentando CSV…", flush=True)
            return _load_csv_candles(csv_path)
    raise ValueError(f"source deve ser csv, mt5 ou auto — recebi {source!r}")


def slice_week(
    frame: pd.DataFrame,
    predicted: pd.DataFrame,
    strange,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    ts = pd.to_datetime(frame["timestamp"])
    mask = (ts >= start) & (ts < _end_exclusive(end))
    if not bool(mask.any()):
        raise ValueError(
            f"Nenhum candle entre {start.date()} e {end.date()}. "
            f"Série vai de {ts.min()} a {ts.max()}."
        )
    week = frame.loc[mask].reset_index(drop=True)
    week_pred = predicted.loc[mask].reset_index(drop=True)
    week_strange = strange[mask.to_numpy()]
    return week, week_pred, week_strange


def format_report(
    cfg: AppConfig,
    source_label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    n_warmup: int,
    n_week: int,
    metrics,
) -> str:
    lines = [
        f"Config: {cfg.name}",
        f"Decisão: {cfg.execution.decision} · {cfg.execution.direction} · "
        f"stop {cfg.risk.stop_points:.0f} / alvo {cfg.risk.gain_points:.0f}"
        + (" · trailing" if cfg.risk.trailing_enabled else ""),
        f"Janela: {start.date()} → {end.date()}  (candles da semana: {n_week}, warmup: {n_warmup})",
        f"Fonte: {source_label}",
        f"Banca inicial: R$ {metrics.initial_bank:,.2f}",
        f"Banca final:   R$ {metrics.final_bank:,.2f}",
        f"P&L:           R$ {metrics.net_pnl:,.2f}",
        f"Drawdown:      R$ {metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.1f}%)",
        f"Trades: {metrics.n_trades}  wins {metrics.n_wins}  losses {metrics.n_losses}  "
        f"acerto {metrics.win_rate:.1f}%  PF {metrics.profit_factor:.2f}",
        "",
    ]
    trades = metrics.trades or []
    if not trades:
        lines.append("Nenhum trade na janela.")
        return "\n".join(lines)
    header = f"{'entrada':<20} {'saida':<20} {'lado':<5} {'entry':>9} {'exit':>9} {'pts':>8} {'pnl':>9}  motivo"
    lines.append(header)
    lines.append("-" * len(header))
    for t in trades:
        entry_t = str(t.get("entry_time", ""))[:19].replace("T", " ")
        exit_t = str(t.get("exit_time", ""))[:19].replace("T", " ")
        lines.append(
            f"{entry_t:<20} {exit_t:<20} {t.get('side', ''):<5} "
            f"{float(t.get('entry', 0)):9.1f} {float(t.get('exit', 0)):9.1f} "
            f"{float(t.get('points', 0)):8.1f} {float(t.get('pnl', 0)):9.2f}  "
            f"{t.get('reason', '')}"
        )
    return "\n".join(lines)


def run_replay(
    config_name: str = DEFAULT_CONFIG,
    start_s: str = DEFAULT_FROM,
    end_s: str = DEFAULT_TO,
    csv: str | Path | None = None,
    source: str = "csv",
    warmup_days: int = 5,
) -> str:
    cfg = load_named_config(config_name)
    start = pd.Timestamp(start_s)
    end = pd.Timestamp(end_s)
    csv_path = Path(csv) if csv else DEFAULT_CSV
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    train_csv = cfg.resolve_csv(cfg.data.train_csv)
    model = ensure_model(cfg.data.timeframe, train_csv)

    frame, source_label = load_replay_frame(cfg, source, csv_path, start, end, warmup_days)
    warmup_from = start - pd.Timedelta(days=warmup_days)
    ts = pd.to_datetime(frame["timestamp"])
    usable = frame.loc[(ts >= warmup_from) & (ts < _end_exclusive(end))].copy().reset_index(drop=True)
    if usable.empty:
        raise ValueError(f"Série vazia após recorte de warmup ({warmup_from.date()} → {end.date()}).")

    featured = add_true_range(add_candle_features(usable), cfg.risk.atr_period)
    predicted = model.predict_next_ohlc(featured)
    lookback = lookback_for_timeframe(cfg.data.timeframe)
    strange = strange_from_frame(featured, lookback=lookback)
    week, week_pred, week_strange = slice_week(featured, predicted, strange, start, end)
    n_warmup = max(0, len(featured) - len(week))

    use_guard = str(getattr(cfg.execution, "decision", "ml") or "ml") == "ml_guard"
    metrics = BacktestEngine(cfg).run(
        week,
        week_pred,
        compound=False,
        strange_mask=week_strange if use_guard else None,
    )
    report = format_report(cfg, source_label, start, end, n_warmup, len(week), metrics)
    print(report, flush=True)
    return report


def replay_from_args(args) -> None:
    try:
        run_replay(
            config_name=args.config,
            start_s=args.start,
            end_s=args.end,
            csv=args.csv,
            source=args.source,
            warmup_days=args.warmup_days,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\n\n{MAC_HINT}") from exc
