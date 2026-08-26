from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timedelta, timezone

os.environ["WIN_DISABLE_YAHOO"] = "1"

from trader.backtest import SessionFilter
from trader.broker import filling_from_mode
from trader.domain import Candle, Side, Signal
from trader.mt5_session import (
    SymbolCandidate,
    TRADE_MODE_FULL,
    bar_is_fresh,
    bar_is_live,
    bar_is_today,
    front_win_contract,
    next_gold_window,
    pick_win_symbol,
    planned_order,
    preferred_win_symbols,
    session_wait_reason,
)
from trader.feeds import StreamFeed, clock_demo_candles, parse_stream_payload, parse_yahoo_chart
from trader.realtime import (
    ALIGN_SLACK_MS,
    RealtimeEngine,
    cloud_snapshot,
    is_m5_close_slot,
    seconds_until_aligned,
)
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
    assert not bar_is_fresh(datetime(2026, 8, 24, 11, 0), gold)
    assert bar_is_today(gold, gold)
    assert not bar_is_today(datetime(2026, 8, 14, 9, 35), gold)
    assert bar_is_live(datetime(2026, 8, 24, 9, 15), gold)
    assert not bar_is_live(datetime(2026, 8, 14, 9, 35), gold)
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
    engine.set_source("mt5")
    engine.set_source("stream")
    assert engine.source == "stream"
    assert engine.wait_reason in {"aguardando_candle", "pronto", "fora_do_ouro", "mercado_fechado"}
    assert engine.feed_info.get("ingested") == 40

    sig = Signal(Side.BUY, 140_000.0, 139_900.0, 140_200.0, "teste")
    engine._open_paper(sig, 140_000.0, datetime(2026, 8, 24, 9, 15))
    assert engine.position is not None
    assert engine.lot == "scaled"
    assert engine.position["contracts"] == 1
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
    engine.bank = 2000
    engine._open_paper(sig, 140_000.0, datetime(2026, 8, 24, 9, 25))
    assert engine.position is not None
    assert engine.position["contracts"] == 2


def test_stream_falls_back_to_demo_clock() -> None:
    feed = StreamFeed()
    candles = feed.last_closed_candles("WIN$", "m5", 80, now=datetime(2026, 8, 24, 1, 50))
    assert len(candles) == 80
    assert feed.origin == "demo"

    engine = RealtimeEngine()
    engine.set_source("stream")
    engine._prepare_policy()
    engine.tick(now=datetime(2026, 8, 24, 1, 50))
    engine.reset()
    engine.tick(now=datetime(2026, 8, 24, 1, 50))
    assert engine.error is None
    assert engine.trades == []
    assert engine.position is None
    assert engine.bank == engine.initial_bank == 1000
    assert engine.snapshot()["net_pnl"] == 0
    assert engine.snapshot()["mode"] == "paper"
    assert engine.wait_reason == "mercado_fechado"
    engine.trades = [
        {
            "side": "BUY",
            "entry_time": "2026-08-14T09:35:00",
            "exit_time": "2026-08-14T09:35:00",
            "pnl": 60.0,
        }
    ]
    engine.bank = 1060
    engine._keep_today_only(date(2026, 8, 24))
    assert engine.trades == []
    assert engine.bank == 1000


def test_paper_waits_at_night_and_never_sends() -> None:
    import pandas as pd

    class Boom:
        def send(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("order_send nao pode rodar no paper")

    engine = RealtimeEngine()
    engine.set_order_mode("paper")
    engine._broker = Boom()  # type: ignore[assignment]
    engine._prepare_policy()
    engine.tick(now=datetime(2026, 8, 24, 2, 10))
    assert engine.order_mode == "paper"
    assert engine.trades == []
    assert engine.bank == 1000
    assert engine.snapshot()["mode"] == "paper"
    assert engine.wait_reason == "mercado_fechado"

    rows = []
    price = 140_000.0
    start = datetime(2026, 8, 24, 6, 0)
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
    engine.tick(now=datetime(2026, 8, 24, 9, 20))
    assert engine.snapshot()["mode"] == "paper"
    assert engine.position is None or engine.position.get("ticket") is None

    engine.set_order_mode("mt5")
    engine.mt5_info["demo"] = False
    try:
        engine._send_signal(Boom(), Signal(Side.BUY, 140_000.0, 139_900.0, 140_200.0, "x"), 140_000.0)  # type: ignore[arg-type]
        raise AssertionError("mt5 sem demo nao pode enviar")
    except RuntimeError as exc:
        assert "real" in str(exc).lower() or "simular" in str(exc).lower() or "Recusado" in str(exc)


def test_parse_yahoo_chart_keeps_closed_5m_only() -> None:
    now = datetime(2026, 8, 24, 10, 20)
    closed_utc = datetime(2026, 8, 24, 13, 15, tzinfo=timezone.utc)
    open_utc = datetime(2026, 8, 24, 13, 20, tzinfo=timezone.utc)
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "^BVSP", "gmtoffset": -10800},
                    "timestamp": [int(closed_utc.timestamp()), int(open_utc.timestamp())],
                    "indicators": {
                        "quote": [
                            {
                                "open": [140000.0, 140050.0],
                                "high": [140080.0, 140090.0],
                                "low": [139940.0, 140000.0],
                                "close": [140020.0, 140060.0],
                                "volume": [10, 11],
                            }
                        ]
                    },
                }
            ]
        }
    }
    candles = parse_yahoo_chart(payload, now)
    assert len(candles) == 1
    assert candles[0].symbol == "^BVSP"
    assert candles[0].timestamp == datetime(2026, 8, 24, 10, 15)
    assert candles[0].close == 140020.0


def test_demo_releases_0920_after_that_clock() -> None:
    rows = []
    price = 140_000.0
    start = datetime(2026, 8, 21, 9, 0)
    for i in range(24):
        ts = start + timedelta(minutes=5 * i)
        rows.append(Candle("WIN$", ts, price, price + 40, price - 20, price + 15, 1))
        price += 10
    early = clock_demo_candles(rows, datetime(2026, 8, 24, 9, 19), 80)
    today_early = [c.timestamp for c in early if c.timestamp.date() == date(2026, 8, 24)]
    assert datetime(2026, 8, 24, 9, 20) not in today_early
    late = clock_demo_candles(rows, datetime(2026, 8, 24, 9, 25), 80)
    today_late = {c.timestamp for c in late if c.timestamp.date() == date(2026, 8, 24)}
    assert datetime(2026, 8, 24, 9, 20) in today_late


def test_yahoo_feed_used_when_last_bar_is_live() -> None:
    now = datetime(2026, 8, 24, 10, 20)
    closed_utc = datetime(2026, 8, 24, 13, 15, tzinfo=timezone.utc)

    def fake_yahoo(_url: str) -> dict:
        return {
            "chart": {
                "result": [
                    {
                        "meta": {"symbol": "^BVSP", "gmtoffset": -10800},
                        "timestamp": [int(closed_utc.timestamp())],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [140000.0],
                                    "high": [140080.0],
                                    "low": [139940.0],
                                    "close": [140020.0],
                                    "volume": [10],
                                }
                            ]
                        },
                    }
                ]
            }
        }

    os.environ.pop("WIN_DISABLE_YAHOO", None)
    try:
        feed = StreamFeed(yahoo_fetch=fake_yahoo)
        candles = feed.last_closed_candles("WIN$", "m5", 80, now=now)
        assert feed.origin == "yahoo"
        assert candles[-1].timestamp == datetime(2026, 8, 24, 10, 15)
    finally:
        os.environ["WIN_DISABLE_YAHOO"] = "1"


def test_seconds_until_aligned_hits_next_ten_plus_slack() -> None:
    slack = ALIGN_SLACK_MS / 1000.0
    wait = seconds_until_aligned(47.2, interval=10, slack_ms=ALIGN_SLACK_MS)
    assert abs(wait - (50.0 + slack - 47.2)) < 1e-9
    wait_on_mark = seconds_until_aligned(50.0, interval=10, slack_ms=ALIGN_SLACK_MS)
    assert abs(wait_on_mark - slack) < 1e-9
    wait_after_slack = seconds_until_aligned(50.0 + slack, interval=10, slack_ms=ALIGN_SLACK_MS)
    assert abs(wait_after_slack - 10.0) < 1e-9
    wait_end = seconds_until_aligned(59.9, interval=10, slack_ms=ALIGN_SLACK_MS)
    assert abs(wait_end - (60.0 + slack - 59.9)) < 1e-9


def test_m5_close_catchup_only_on_zero_slot() -> None:
    assert is_m5_close_slot(datetime(2026, 8, 24, 14, 35, 0))
    assert is_m5_close_slot(datetime(2026, 8, 24, 14, 35, 0, 250000))
    assert not is_m5_close_slot(datetime(2026, 8, 24, 14, 35, 10))
    assert not is_m5_close_slot(datetime(2026, 8, 24, 14, 36, 0))
    assert is_m5_close_slot(datetime(2026, 8, 24, 14, 30, 9))


def test_replay_today_walks_closed_demo_bars() -> None:
    early = RealtimeEngine()
    early.replay_today(datetime(2026, 8, 24, 9, 19))
    late = RealtimeEngine()
    late.replay_today(datetime(2026, 8, 24, 9, 26))
    assert late.last_bar_time == "2026-08-24T09:20:00"
    assert late.last_bar_time != early.last_bar_time
    assert late._frame is not None
    assert len(late._frame) >= 2


def test_cloud_snapshot_arms_without_background_loop() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        path = handle.name
    os.environ["WIN_CLOUD_STATE"] = path
    try:
        idle = cloud_snapshot()
        assert idle["running"] is False
        armed = cloud_snapshot(arm=True)
        assert armed["running"] is True
        assert armed["order_mode"] == "paper"
        paused = cloud_snapshot(pause=True)
        assert paused["running"] is False
        cleared = cloud_snapshot(reset=True)
        assert cleared["running"] is False
        assert cleared["n_trades"] == 0
    finally:
        os.environ.pop("WIN_CLOUD_STATE", None)
        try:
            os.unlink(path)
        except OSError:
            pass

