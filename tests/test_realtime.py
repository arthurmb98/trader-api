from __future__ import annotations

from datetime import date, datetime

from trader.backtest import SessionFilter
from trader.broker import filling_from_mode
from trader.domain import Side, Signal
from trader.mt5_session import (
    SymbolCandidate,
    TRADE_MODE_FULL,
    bar_is_fresh,
    front_win_contract,
    next_gold_window,
    pick_win_symbol,
    planned_order,
    preferred_win_symbols,
    session_wait_reason,
)
from trader.feeds import parse_stream_payload
from trader.realtime import RealtimeEngine
from trader.replay import load_named_config
from trader.risk import RiskCalculator


def test_front_contract_rolls_after_august_expiry() -> None:
    assert front_win_contract(date(2026, 8, 24)) == "WINV26"
    assert front_win_contract(date(2026, 8, 10)) == "WINQ26"
    assert preferred_win_symbols(date(2026, 8, 24))[0] == "WINV26"


def test_pick_prefers_tradeable_front_month() -> None:
    picked = pick_win_symbol(
        [
            SymbolCandidate("WIN$", TRADE_MODE_FULL, 10),
            SymbolCandidate("WINV26", TRADE_MODE_FULL, 80),
            SymbolCandidate("WINQ26", 3, 5),
        ],
        date(2026, 8, 24),
    )
    assert picked == "WINV26"


def test_winner_levels_are_100_200() -> None:
    cfg = load_named_config("best_candles_m5_1000_a")
    risk = RiskCalculator(cfg)
    stop, take = risk.levels(Side.BUY, 140_000.0, None)
    assert stop == 139_900.0
    assert take == 140_200.0
    sell_stop, sell_take = risk.levels(Side.SELL, 140_000.0, None)
    assert sell_stop == 140_100.0
    assert sell_take == 139_800.0
    sig = Signal(Side.BUY, 140_000.0, stop, take, "seguir_previsao_proxima_abertura")
    plan = planned_order(sig, 140_000.0, stop, take)
    assert plan is not None
    assert plan["volume"] == 1.0
    assert plan["stop"] == 139_900.0
    assert planned_order(Signal(Side.FLAT, 0, 0, 0, "gap"), 140_000.0, 0, 0) is None


def test_gold_hours_and_fresh_bar() -> None:
    cfg = load_named_config("best_candles_m5_1000_a")
    session = SessionFilter(cfg)
    gold = datetime(2026, 8, 24, 9, 20)
    lunch = datetime(2026, 8, 24, 12, 0)
    night = datetime(2026, 8, 24, 0, 13)
    assert session.allows(gold)
    assert not session.allows(lunch)
    assert session.flatten_day(datetime(2026, 8, 24, 17, 0))
    assert bar_is_fresh(datetime(2026, 8, 24, 9, 15), datetime(2026, 8, 24, 9, 20))
    assert not bar_is_fresh(datetime(2026, 8, 21, 17, 50), night)
    assert session_wait_reason(
        connected=True,
        account=True,
        demo=True,
        symbol="WINV26",
        trade_allowed=True,
        now=night,
        last_bar=datetime(2026, 8, 21, 17, 50),
        in_position=False,
        session=session,
    ) == "mercado_fechado"
    assert session_wait_reason(
        connected=True,
        account=True,
        demo=True,
        symbol="WINV26",
        trade_allowed=True,
        now=lunch,
        last_bar=datetime(2026, 8, 24, 10, 55),
        in_position=False,
        session=session,
    ) == "fora_do_ouro"
    assert session_wait_reason(
        connected=True,
        account=True,
        demo=False,
        symbol="WINV26",
        trade_allowed=True,
        now=gold,
        last_bar=gold,
        in_position=False,
        session=session,
    ) == "conta_real"
    assert next_gold_window(night) == "2026-08-24T09:15"


def test_filling_prefers_return() -> None:
    assert filling_from_mode(4, True, True, False) == "RETURN"
    assert filling_from_mode(2, False, True, False) == "IOC"


def test_parse_stream_payload() -> None:
    name, candles = parse_stream_payload(
        {
            "symbol": "WINV26",
            "candles": [
                {"t": "2026-08-24T09:15:00", "open": 140000, "high": 140080, "low": 139940, "close": 140020}
            ],
        }
    )
    assert name == "WINV26"
    assert candles[0].close == 140020


def test_stream_is_paper_and_never_sends() -> None:
    import pandas as pd

    class Boom:
        def send(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("order_send nao pode rodar no stream")

    engine = RealtimeEngine()
    engine.set_source("stream")
    engine._broker = Boom()  # type: ignore[assignment]
    rows = []
    start = datetime(2026, 8, 24, 8, 0)
    price = 140_000.0
    for i in range(40):
        ts = start + pd.Timedelta(minutes=5 * i)
        rows.append(
            {
                "t": ts.isoformat(),
                "open": price,
                "high": price + 40,
                "low": price - 20,
                "close": price + 15,
            }
        )
        price += 10
    engine.ingest_candles({"symbol": "WINV26", "candles": rows})
    engine._prepare_policy()
    engine.tick(now=datetime(2026, 8, 24, 9, 20))
    feeds = engine.list_feeds()
    assert [item["key"] for item in feeds["feeds"]] == ["mt5", "stream"]
    assert engine.source == "stream"
    assert engine.snapshot()["mode"] == "paper"
    assert engine.feed_info.get("ingested") == 40

    sig = Signal(Side.BUY, 140_000.0, 139_900.0, 140_200.0, "teste")
    engine._open_paper(sig, 140_000.0, datetime(2026, 8, 24, 9, 15))
    assert engine.position is not None
    assert engine.position["stop"] == 139_900.0
    assert engine.position["take"] == 140_200.0
    assert engine.position.get("ticket") is None
    hit = pd.Series(
        {"Abertura": 140000.0, "Máximo": 140250.0, "Mínimo": 139990.0, "Fechamento": 140210.0}
    )
    engine._manage_open(None, hit, datetime(2026, 8, 24, 9, 20), datetime(2026, 8, 24, 9, 20))
    assert engine.position is None
    assert engine.trades[-1]["result"] == "win"
    assert engine.trades[-1]["points"] == 200.0
