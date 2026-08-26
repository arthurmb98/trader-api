from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from trader.paths import CONFIGS_DIR, ROOT


def _merge(dc_type: type, data: dict[str, Any] | None) -> Any:
    if data is None:
        return dc_type()
    hints = get_type_hints(dc_type)
    allowed = set(hints)
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"Chaves desconhecidas em {dc_type.__name__}: {extra}")
    kwargs: dict[str, Any] = {}
    for name, typ in hints.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(typ) and isinstance(value, dict):
            kwargs[name] = _merge(typ, value)
        else:
            kwargs[name] = value
    return dc_type(**kwargs)


@dataclass
class AccountConfig:
    initial_bank: float = 1000.0
    contracts: int = 1
    point_value: float = 0.20
    contract_cost: float = 1.0


@dataclass
class InstrumentConfig:
    symbol: str = "WIN$"
    tick_size: int = 5


@dataclass
class DataConfig:
    train_csv: str = "datasets/WIN_1min_train.csv"
    test_csv: str = "datasets/WIN_1min_test.csv"
    timeframe: str = "m1"


@dataclass
class RiskConfig:
    mode: str = "fixed"
    stop_points: float = 100.0
    gain_points: float = 200.0
    rr_ratio: float = 2.0
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    atr_gain_mult: float = 2.0
    trailing_enabled: bool = False
    trailing_trigger_points: float = 80.0
    trailing_distance_points: float = 50.0
    be_trigger_points: float = 25.0
    be_lock_points: float = 10.0
    invalidate_tp_points: float = 30.0
    daily_loss_points: float = 400.0
    max_trades_per_day: int = 8


@dataclass
class FilterConfig:
    session_start: str = "09:15"
    session_end: str = "17:00"
    skip_lunch: bool = True
    lunch_start: str = "11:00"
    lunch_end: str = "14:30"
    gold_hours_only: bool = True
    min_predicted_range: float = 40.0
    max_gap_points: float | None = 40.0
    min_predicted_body: float = 0.0


@dataclass
class ExecutionConfig:
    direction: str = "follow"
    entry_mode: str = "market_open"
    entry_offset_points: float = 0.0
    decision: str = "ml"


@dataclass
class Mt5Config:
    enabled: bool = False
    symbol: str = "WIN$"
    magic: int = 20260818
    deviation: int = 20
    filling: str = "IOC"
    comment: str = "trader-api"


@dataclass
class AppConfig:
    name: str = "default"
    account: AccountConfig = field(default_factory=AccountConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    mt5: Mt5Config = field(default_factory=Mt5Config)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        return _merge(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_csv(self, relative: str) -> Path:
        path = Path(relative)
        if not path.is_absolute():
            path = ROOT / path
        return path


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg_path = Path(path) if path else CONFIGS_DIR / "default.yaml"
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _merge(AppConfig, raw)


def save_config(config: AppConfig, path: str | Path) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.to_dict(), fh, sort_keys=False, allow_unicode=True)
    return out
