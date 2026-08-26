from datetime import datetime, timedelta

import pandas as pd

from trader.backtest import BacktestEngine
from trader.domain import Side
from trader.replay import load_named_config


def _bars() -> tuple[pd.DataFrame, pd.DataFrame]:
    start = datetime(2026, 8, 26, 9, 15)
    rows = []
    preds = []
    # 09:15 close 178550; 09:20 dumps 590 pts after a SELL fill at open.
    ohlc = [
        (178600.0, 178620.0, 178540.0, 178550.0),
        (178545.0, 178720.0, 177950.0, 178330.0),
        (178330.0, 178350.0, 178300.0, 178320.0),
    ]
    for i, (o, h, l, c) in enumerate(ohlc):
        ts = start + timedelta(minutes=5 * i)
        rows.append(
            {
                "timestamp": ts,
                "Abertura": o,
                "Máximo": h,
                "Mínimo": l,
                "Fechamento": c,
                "atr": 80.0,
            }
        )
        preds.append(
            {
                "pred_open": c,
                "pred_high": c + 80.0,
                "pred_low": c - 80.0,
                "pred_close": c - 80.0,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(preds)


def test_entry_bar_does_not_bank_the_full_m5_range() -> None:
    cfg = load_named_config("best_candles_m5_1000_a")
    test, pred = _bars()
    strange = [False] * len(test)
    metrics = BacktestEngine(cfg).run(test, pred, strange_mask=strange)
    same = [t for t in metrics.trades if t["entry_time"][:19] == t["exit_time"][:19]]
    assert not any(float(t["points"]) >= 400 for t in same)
    assert metrics.n_trades >= 1
    first = sorted(metrics.trades, key=lambda row: row["entry_time"])[0]
    assert first["side"] == Side.SELL.value
    assert float(first["points"]) < 400


def test_marks_can_exit_inside_the_same_m5() -> None:
    cfg = load_named_config("best_candles_m5_1000_a")
    test, pred = _bars()
    ts = datetime(2026, 8, 26, 9, 20)
    marks = [
        (ts, 178545.0),
        (ts + timedelta(seconds=12), 178340.0),
    ]
    metrics = BacktestEngine(cfg).run(test, pred, strange_mask=[False] * len(test), marks=marks)
    assert metrics.n_trades >= 1
    trade = sorted(metrics.trades, key=lambda row: row["entry_time"])[0]
    assert trade["side"] == Side.SELL.value
    assert "09:20" in trade["exit_time"]
    assert float(trade["points"]) <= 205
