from __future__ import annotations

from pathlib import Path

import pandas as pd

from trader.domain import LeakageReport
from trader.paths import DATASETS_DIR, ROOT

OHLC = ["Abertura", "Máximo", "Mínimo", "Fechamento"]
EN_OHLC = {"open": "Abertura", "high": "Máximo", "low": "Mínimo", "close": "Fechamento"}
WRITE_COLS = ["Ativo", "Data", "Hora", *OHLC, "Volume"]
SPLIT_TS = pd.Timestamp("2025-01-01")
RAW_SOURCES = {
    "1min": "WIN$D(M1).csv",
    "5min": "WIN$D(M5).csv",
}
SPLIT_FILES = {
    "1min": ("WIN_1min_train.csv", "WIN_1min_test.csv"),
    "5min": ("WIN_5min_train.csv", "WIN_5min_test.csv"),
}


def _read_raw(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for sep in (",", ";"):
        for encoding in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=encoding)
                if len(df.columns) == 1:
                    continue
                return df
            except Exception as exc:  # noqa: BLE001
                last_error = exc
    raise ValueError(f"Não foi possível ler {path}: {last_error}")


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    required_pt = {"Ativo", "Data", "Hora", *OHLC}
    if required_pt <= set(df.columns):
        return df
    if {"time", "open", "high", "low", "close"} <= set(lower):
        out = pd.DataFrame()
        out["timestamp"] = pd.to_datetime(df[lower["time"]], errors="coerce")
        out["Ativo"] = "WIN$"
        out["Data"] = out["timestamp"].dt.strftime("%d/%m/%Y")
        out["Hora"] = out["timestamp"].dt.strftime("%H:%M:%S")
        for en, pt in EN_OHLC.items():
            out[pt] = pd.to_numeric(df[lower[en]], errors="coerce")
        if "real_volume" in lower:
            out["Volume"] = pd.to_numeric(df[lower["real_volume"]], errors="coerce")
        elif "tick_volume" in lower:
            out["Volume"] = pd.to_numeric(df[lower["tick_volume"]], errors="coerce")
        else:
            out["Volume"] = 0.0
        return out
    raise ValueError(f"CSV sem colunas reconhecidas: {list(df.columns)}")


def load_candles(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    df = _normalize_schema(_read_raw(csv_path))
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["Data"].astype(str).str.strip() + " " + df["Hora"].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
        )
    df = df.dropna(subset=["timestamp"]).copy()
    for col in OHLC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=OHLC)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df.sort_values("timestamp").drop_duplicates(subset=["Ativo", "timestamp"])
    df = df.reset_index(drop=True)
    df["source_file"] = csv_path.name
    return df


def write_candles(frame: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    export = frame.copy()
    if "Data" not in export.columns or "Hora" not in export.columns:
        ts = pd.to_datetime(export["timestamp"])
        export["Data"] = ts.dt.strftime("%d/%m/%Y")
        export["Hora"] = ts.dt.strftime("%H:%M:%S")
    if "Ativo" not in export.columns:
        export["Ativo"] = "WIN$"
    if "Volume" not in export.columns:
        export["Volume"] = 0.0
    export[WRITE_COLS].to_csv(out, index=False)
    return out


def split_by_wall(frame: pd.DataFrame, wall: pd.Timestamp = SPLIT_TS) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(frame["timestamp"])
    train = frame.loc[ts < wall].copy().reset_index(drop=True)
    test = frame.loc[ts >= wall].copy().reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(f"Recorte {wall.date()} deixou treino ou teste vazio.")
    return train, test


def prepare_study_csvs() -> dict[str, dict[str, str]]:
    """Convert MT5 dumps into Portuguese train/test CSVs (wall = 2025-01-01)."""
    written: dict[str, dict[str, str]] = {}
    for tf, raw_name in RAW_SOURCES.items():
        raw_path = DATASETS_DIR / raw_name
        if not raw_path.exists():
            raise FileNotFoundError(f"Massa bruta ausente: {raw_path}")
        frame = load_candles(raw_path)
        train, test = split_by_wall(frame)
        train_name, test_name = SPLIT_FILES[tf]
        train_path = write_candles(train, DATASETS_DIR / train_name)
        test_path = write_candles(test, DATASETS_DIR / test_name)
        written[tf] = {
            "source": str(raw_path.relative_to(ROOT)),
            "train": str(train_path.relative_to(ROOT)),
            "test": str(test_path.relative_to(ROOT)),
            "n_train": str(len(train)),
            "n_test": str(len(test)),
            "train_start": pd.Timestamp(train["timestamp"].min()).isoformat(),
            "train_end": pd.Timestamp(train["timestamp"].max()).isoformat(),
            "test_start": pd.Timestamp(test["timestamp"].min()).isoformat(),
            "test_end": pd.Timestamp(test["timestamp"].max()).isoformat(),
        }
        print(
            f"{tf}: {len(train)} treino ({written[tf]['train_start']} → {written[tf]['train_end']}) · "
            f"{len(test)} teste ({written[tf]['test_start']} → {written[tf]['test_end']})",
            flush=True,
        )
    return written


def resample_to_5min(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = []
    for symbol, grp in frame.groupby("Ativo", sort=False):
        g = grp.set_index("timestamp").sort_index()
        agg = g.resample("5min", label="left", closed="left").agg(
            {
                "Ativo": "first",
                "Abertura": "first",
                "Máximo": "max",
                "Mínimo": "min",
                "Fechamento": "last",
                "Volume": "sum" if "Volume" in g.columns else "count",
            }
        )
        agg = agg.dropna(subset=["Abertura", "Fechamento"]).reset_index()
        agg["Ativo"] = symbol
        agg["Data"] = agg["timestamp"].dt.strftime("%d/%m/%Y")
        agg["Hora"] = agg["timestamp"].dt.strftime("%H:%M:%S")
        out.append(agg)
    return pd.concat(out, ignore_index=True) if out else frame.iloc[0:0].copy()


def _ohlc_tuple(frame: pd.DataFrame):
    return zip(
        frame["Abertura"].round(2),
        frame["Máximo"].round(2),
        frame["Mínimo"].round(2),
        frame["Fechamento"].round(2),
    )


def sanitize_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_file: str,
    test_file: str,
    match_ohlc: bool = True,
) -> tuple[pd.DataFrame, LeakageReport]:
    """Remove from test any candle already present in train (key, optionally identical OHLC)."""
    n_original = len(test)
    train_keys = set(zip(train["Ativo"], train["timestamp"]))
    test_keys = list(zip(test["Ativo"], test["timestamp"]))
    by_key = pd.Series([k in train_keys for k in test_keys], index=test.index)
    if match_ohlc:
        train_ohlc = set(_ohlc_tuple(train))
        test_fp = list(_ohlc_tuple(test))
        by_ohlc = pd.Series([fp in train_ohlc for fp in test_fp], index=test.index)
    else:
        by_ohlc = pd.Series(False, index=test.index)
    drop = by_key | by_ohlc
    cleaned = test.loc[~drop].copy().reset_index(drop=True)
    if cleaned.empty:
        raise ValueError(
            f"Teste ficou vazio após anti-join contra o treino ({train_file} vs {test_file})."
        )

    def _fmt(series: pd.Series) -> str | None:
        if series.empty:
            return None
        return pd.Timestamp(series.min()).isoformat()

    report = LeakageReport(
        n_train=len(train),
        n_test_original=n_original,
        n_removed=int(drop.sum()),
        n_test_clean=len(cleaned),
        removed_by_key=int(by_key.sum()),
        removed_by_ohlc=int((by_ohlc & ~by_key).sum()),
        train_file=train_file,
        test_file=test_file,
        train_start=_fmt(train["timestamp"]),
        train_end=pd.Timestamp(train["timestamp"].max()).isoformat() if len(train) else None,
        test_start=_fmt(cleaned["timestamp"]),
        test_end=pd.Timestamp(cleaned["timestamp"].max()).isoformat() if len(cleaned) else None,
    )
    return cleaned, report
