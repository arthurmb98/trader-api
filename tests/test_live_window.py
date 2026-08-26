from datetime import date

import pandas as pd

from trader.data import merge_candle_frames
from trader.live import coverage_fetch_range, live_meta, validate_live_params
from trader.replay import load_named_config


def test_winner_loads_delay_band() -> None:
    cfg = load_named_config("best_candles_m5_1000_a")
    assert float(cfg.execution.entry_delay_points) == 10.0


def test_coverage_fetches_days_after_csv_end() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2025-01-02 10:00"),
                pd.Timestamp("2026-08-18 17:00"),
            ]
        }
    )
    gap = coverage_fetch_range(frame, date(2026, 8, 10), date(2026, 8, 26), warmup_days=5)
    assert gap == (date(2026, 8, 18), date(2026, 8, 26))


def test_coverage_none_when_window_inside_csv() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-08-01 10:00"),
                pd.Timestamp("2026-08-18 17:00"),
            ]
        }
    )
    assert coverage_fetch_range(frame, date(2026, 8, 10), date(2026, 8, 18), warmup_days=5) is None


def test_merge_candle_frames_dedupes() -> None:
    a = pd.DataFrame(
        {
            "Ativo": ["WIN$"],
            "timestamp": [pd.Timestamp("2026-08-18 17:00")],
            "Abertura": [1.0],
            "Máximo": [2.0],
            "Mínimo": [0.5],
            "Fechamento": [1.5],
            "Volume": [1.0],
        }
    )
    b = pd.DataFrame(
        {
            "Ativo": ["WIN$", "WIN$"],
            "timestamp": [pd.Timestamp("2026-08-18 17:00"), pd.Timestamp("2026-08-19 09:20")],
            "Abertura": [9.0, 3.0],
            "Máximo": [9.0, 4.0],
            "Mínimo": [9.0, 2.0],
            "Fechamento": [9.0, 3.5],
            "Volume": [2.0, 3.0],
        }
    )
    merged = merge_candle_frames(a, b)
    assert len(merged) == 2
    assert float(merged.iloc[0]["Fechamento"]) == 9.0


def test_paper_window_allows_today_not_csv_last_day() -> None:
    today = date.today()
    start = date(today.year, today.month, 1) if today.day > 1 else date(today.year, today.month, 1)
    case, tf, bank, window_start, window_end = validate_live_params(
        "last_candles",
        "m5",
        1000,
        "paper",
        start,
        today,
    )
    assert case == "last_candles"
    assert tf == "m5"
    assert bank == 1000
    assert window_end == today
    meta = live_meta("m5")
    assert meta["max_date"] >= today.isoformat()
    assert meta["max_date"] >= meta["csv_max_date"]
