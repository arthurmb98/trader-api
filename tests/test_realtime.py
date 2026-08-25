from __future__ import annotations

from datetime import date, datetime, timedelta

from trader.backtest import SessionFilter
from trader.broker import filling_from_mode
from trader.domain import Side, Signal
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
from trader.feeds import StreamFeed, parse_stream_payload
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


def test_mt5_unix_keeps_server_clock() -> None:
    from datetime import timezone

    from trader.broker import _mt5_time

    ts = datetime(2026, 8, 25, 10, 30)
    unix = int(ts.replace(tzinfo=timezone.utc).timestamp())
    assert _mt5_time(unix) == ts


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


def test_stream_falls_back_to_local_csv() -> None:
    feed = StreamFeed()
    candles = feed.last_closed_candles("WIN$", "m5", 80)
    assert len(candles) == 80
    assert feed.origin == "file"
    assert feed.ready()
    assert feed.status()["file"] == "WIN_5min_test.csv"

    engine = RealtimeEngine()
    engine.set_source("stream")
    engine._prepare_policy()
    engine.tick(now=datetime(2026, 8, 24, 1, 50))
    engine.reset()
    engine.tick(now=datetime(2026, 8, 24, 9, 20))
    engine.tick(now=datetime(2026, 8, 24, 9, 25))
    assert engine.error is None
    assert engine.trades == []
    assert engine.position is None
    assert engine.bank == engine.initial_bank == 1000
    assert engine.snapshot()["net_pnl"] == 0
    assert engine.snapshot()["mode"] == "paper"
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
    engine.set_source("stream")
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


class _FakeWinBroker:
    symbol = "WINV26"
    filling = "IOC"

    def __init__(self) -> None:
        self.sent = 0
        self.demo = True
        self.allow_send = False
        self._candles: list = []

    def connect(self, **kwargs):  # noqa: ANN003
        del kwargs
        return None

    def account_payload(self) -> dict:
        return {
            "account": True,
            "demo": self.demo,
            "login": 123,
            "server": "GENIAL-DEMO" if self.demo else "GenialInvestimentos-PRD",
            "balance": 5000.0,
            "equity": 5000.0,
        }

    def terminal_payload(self) -> dict:
        return {"trade_allowed": True, "connected": True}

    def list_win_symbols(self) -> list[SymbolCandidate]:
        return [SymbolCandidate("WINV26", TRADE_MODE_FULL, 80), SymbolCandidate("WIN$", TRADE_MODE_FULL, 10)]

    def has_symbol(self, name: str) -> bool:
        return name.upper().startswith("WIN")

    def use_symbol(self, symbol: str) -> None:
        self.symbol = symbol

    def last_closed_candles(self, symbol: str, timeframe: str, count: int) -> list:
        del symbol, timeframe
        return list(self._candles[-max(count, 1) :])

    def quote(self) -> dict:
        if not self._candles:
            return {"bid": 140000.0, "ask": 140005.0, "last": 140000.0, "time": None}
        last = self._candles[-1]
        return {
            "bid": float(last.close) - 5,
            "ask": float(last.close) + 5,
            "last": float(last.close),
            "time": last.timestamp.isoformat(),
        }

    def open_positions(self) -> list:
        return []

    def send(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.sent += 1
        if not self.allow_send:
            raise AssertionError("order_send nao pode rodar no paper")
        return {"retcode": 10009, "order": 77, "deal": 77, "comment": "ok"}


def _win_candles(start: datetime, n: int = 80):
    from trader.domain import Candle

    rows = []
    price = 140_000.0
    for i in range(n):
        ts = start + timedelta(minutes=5 * i)
        rows.append(
            Candle(
                symbol="WINV26",
                timestamp=ts,
                open=price,
                high=price + 40,
                low=price - 20,
                close=price + 15,
                volume=10,
            )
        )
        price += 10
    return rows


def test_paper_mt5_waits_at_night_and_never_sends() -> None:
    engine = RealtimeEngine()
    engine.set_order_mode("paper")
    fake = _FakeWinBroker()
    fake._candles = _win_candles(datetime(2026, 8, 24, 2, 40))
    engine._broker = fake  # type: ignore[assignment]
    engine._prepare_policy()
    engine.tick(now=datetime(2026, 8, 24, 2, 10))
    assert engine.order_mode == "paper"
    assert engine.source == "mt5"
    assert engine.trades == []
    assert engine.bank == 1000
    assert engine.snapshot()["mode"] == "paper"
    assert engine.wait_reason == "mercado_fechado"
    assert fake.sent == 0

    engine.tick(now=datetime(2026, 8, 24, 9, 20))
    assert engine.snapshot()["mode"] == "paper"
    assert engine.position is None or engine.position.get("ticket") is None
    assert fake.sent == 0

    engine.set_order_mode("mt5")
    engine.mt5_info["demo"] = False
    try:
        engine._send_signal(fake, Signal(Side.BUY, 140_000.0, 139_900.0, 140_200.0, "x"), 140_000.0)  # type: ignore[arg-type]
        raise AssertionError("mt5 sem demo nao pode enviar")
    except RuntimeError as exc:
        assert "real" in str(exc).lower() or "simular" in str(exc).lower() or "Recusado" in str(exc)


def test_open_position_blocks_signal_and_lists_open_order() -> None:
    engine = RealtimeEngine()
    engine.position = {
        "side": Side.BUY,
        "entry": 140_000.0,
        "stop": 139_900.0,
        "take": 140_200.0,
        "time": datetime(2026, 8, 24, 9, 25),
        "hour": 9,
        "extreme": 140_000.0,
        "contracts": 1,
        "reason": "mt5",
        "ticket": 42,
    }
    engine.last_signal = Signal(Side.SELL, 140_050.0, 140_150.0, 139_850.0, "seguir_previsao_proxima_abertura")
    engine.quote = {"bid": 140_040.0, "ask": 140_050.0, "last": 140_045.0}
    snap = engine.snapshot()
    assert snap["skip_reason"] and "BUY" in snap["skip_reason"]
    assert snap["trades"][0]["result"] == "open"
    assert snap["trades"][0]["side"] == "BUY"
    engine.set_order_mode("paper")
    assert engine.position is None


def test_armed_at_night_waits_for_tomorrows_gold() -> None:
    from trader.mt5_session import next_gold_window

    night = datetime(2026, 8, 24, 23, 0)
    assert next_gold_window(night) == "2026-08-25T09:15"
    engine = RealtimeEngine()
    engine.set_order_mode("paper")
    fake = _FakeWinBroker()
    fake._candles = _win_candles(datetime(2026, 8, 24, 14, 30))
    engine._broker = fake  # type: ignore[assignment]
    engine._prepare_policy()
    engine.tick(now=night)
    assert engine.wait_reason == "mercado_fechado"
    assert engine._can_enter(night, live_mt5=False) is False
    assert fake.sent == 0

    fake._candles = _win_candles(datetime(2026, 8, 25, 8, 0), n=16)
    morning = datetime(2026, 8, 25, 9, 20)
    engine.tick(now=morning)
    assert engine._session is not None
    assert engine._session.allows(morning)
    assert engine.wait_reason in {"pronto", "em_posicao", "aguardando_candle"}
    assert fake.sent == 0



def test_real_account_falls_back_to_paper_and_never_sends() -> None:
    engine = RealtimeEngine()
    engine.set_order_mode("mt5")
    fake = _FakeWinBroker()
    fake.demo = False
    fake._candles = _win_candles(datetime(2026, 8, 24, 2, 40))
    engine._broker = fake  # type: ignore[assignment]
    engine._prepare_policy()
    engine.tick(now=datetime(2026, 8, 24, 9, 20))
    assert engine.order_mode == "mt5"
    assert engine.source == "mt5"
    assert fake.sent == 0
    assert engine.position is None or engine.position.get("ticket") is None
    try:
        engine._send_signal(fake, Signal(Side.BUY, 140_000.0, 139_900.0, 140_200.0, "x"), 140_000.0)  # type: ignore[arg-type]
        raise AssertionError("conta real nao pode enviar")
    except RuntimeError as exc:
        assert "real" in str(exc).lower() or "simular" in str(exc).lower() or "Recusado" in str(exc)


class _AlwaysBuy:
    def from_candles(self, frame):  # noqa: ANN001
        del frame
        return Signal(Side.BUY, 140_000.0, 139_900.0, 140_200.0, "teste_ouro")


def test_armed_through_lunch_opens_paper_at_gold() -> None:
    engine = RealtimeEngine()
    engine.set_order_mode("paper")
    fake = _FakeWinBroker()
    fake.demo = False
    fake._candles = _win_candles(datetime(2026, 8, 25, 8, 0), n=78)
    engine._broker = fake  # type: ignore[assignment]
    engine._prepare_policy()
    engine._policy = _AlwaysBuy()  # type: ignore[assignment]
    engine._enter_now = True
    engine.tick(now=datetime(2026, 8, 25, 14, 29))
    assert engine.position is None
    assert engine._enter_now is True
    assert engine.wait_reason == "fora_do_ouro"
    assert fake.sent == 0
    engine.tick(now=datetime(2026, 8, 25, 14, 30))
    assert engine.position is not None
    assert engine.position.get("ticket") is None
    assert engine.wait_reason in {"em_posicao", "pronto"}
    assert fake.sent == 0


def test_paper_does_not_instant_close_on_wide_entry_bar() -> None:
    from trader.domain import Candle

    engine = RealtimeEngine()
    engine.set_order_mode("paper")
    fake = _FakeWinBroker()
    fake.demo = False
    price = 140_000.0
    start = datetime(2026, 8, 25, 14, 0)
    rows = []
    for i in range(8):
        ts = start + timedelta(minutes=5 * i)
        rows.append(
            Candle(
                symbol="WINV26",
                timestamp=ts,
                open=price,
                high=price + 400,
                low=price - 400,
                close=price + 10,
                volume=10,
            )
        )
        price += 10
    fake._candles = rows
    engine._broker = fake  # type: ignore[assignment]
    engine._prepare_policy()
    engine._policy = _AlwaysBuy()  # type: ignore[assignment]
    engine._enter_now = True
    engine.tick(now=datetime(2026, 8, 25, 14, 36))
    assert engine.position is not None
    assert engine.trades == []
    engine.tick(now=datetime(2026, 8, 25, 14, 36, 30))
    assert engine.position is not None
    assert engine.trades == []
    assert fake.sent == 0


def test_prd_sends_on_real_account_in_gold() -> None:
    engine = RealtimeEngine()
    engine.set_order_mode("prd")
    fake = _FakeWinBroker()
    fake.demo = False
    fake.allow_send = True
    fake._candles = _win_candles(datetime(2026, 8, 25, 8, 0), n=78)
    engine._broker = fake  # type: ignore[assignment]
    engine._prepare_policy()
    engine._policy = _AlwaysBuy()  # type: ignore[assignment]
    engine._enter_now = True
    engine.tick(now=datetime(2026, 8, 25, 14, 30))
    assert engine.order_mode == "prd"
    assert engine.snapshot()["can_send"] is True
    assert fake.sent == 1
    assert engine.position is not None
    assert engine.position.get("ticket") == 77


def test_enviar_on_real_still_refuses_send() -> None:
    engine = RealtimeEngine()
    engine.set_order_mode("mt5")
    fake = _FakeWinBroker()
    fake.demo = False
    fake.allow_send = True
    engine._prepare_policy()
    engine.mt5_info["demo"] = False
    try:
        engine._send_signal(fake, Signal(Side.BUY, 140_000.0, 139_900.0, 140_200.0, "x"), 140_000.0)  # type: ignore[arg-type]
        raise AssertionError("Enviar na conta real nao pode order_send")
    except RuntimeError as exc:
        assert "real" in str(exc).lower() or "Produção" in str(exc) or "prd" in str(exc).lower()
    assert fake.sent == 0

