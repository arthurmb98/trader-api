from __future__ import annotations

from pathlib import Path

import pandas as pd

from trader.domain import LeakageReport
from trader.paths import ROOT

OHLC = ["Abertura", "Máximo", "Mínimo", "Fechamento"]
EN_OHLC = {"Abertura": "open", "Máximo": "high", "Mínimo": "low", "Fechamento": "close"}


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


def load_candles(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    df = _read_raw(csv_path)
    df.columns = [c.strip() for c in df.columns]
    required = {"Ativo", "Data", "Hora", *OHLC}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} sem colunas {missing}")

    df["timestamp"] = pd.to_datetime(
        df["Data"].astype(str).str.strip() + " " + df["Hora"].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"]).copy()
    for col in OHLC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=OHLC)
    df = df.sort_values("timestamp").drop_duplicates(subset=["Ativo", "timestamp"])
    df = df.reset_index(drop=True)
    df["source_file"] = csv_path.name
    return df


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


def sanitize_test(train: pd.DataFrame, test: pd.DataFrame, train_file: str, test_file: str) -> tuple[pd.DataFrame, LeakageReport]:
    """Remove from test any candle already present in train (key or identical OHLC)."""
    n_original = len(test)
    train_keys = set(zip(train["Ativo"], train["timestamp"]))
    train_ohlc = set(_ohlc_tuple(train))

    test_keys = list(zip(test["Ativo"], test["timestamp"]))
    by_key = pd.Series([k in train_keys for k in test_keys], index=test.index)
    test_fp = list(_ohlc_tuple(test))
    by_ohlc = pd.Series([fp in train_ohlc for fp in test_fp], index=test.index)
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
