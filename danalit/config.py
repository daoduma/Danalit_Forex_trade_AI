"""Typed configuration loaded from config/settings.yaml and config/instruments.yaml.

Secrets are NEVER stored in files: MT5 credentials come from the environment
variables DANALIT_MT5_LOGIN / DANALIT_MT5_SERVER / DANALIT_MT5_PASSWORD, the
FRED key from FRED_API_KEY, Telegram from DANALIT_TG_TOKEN / DANALIT_TG_CHAT_ID.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from danalit.constants import TIMEFRAMES

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


class PathsConfig(BaseModel):
    data_store: Path = Path("data_store")
    models_store: Path = Path("models_store")
    reports: Path = Path("reports")
    logs: Path = Path("logs")

    def absolute(self, name: str) -> Path:
        p: Path = getattr(self, name)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def db_path(self) -> Path:
        return self.absolute("data_store") / "danalit.db"


class BrokerConfig(BaseModel):
    magic_number: int = 20260701
    leverage: int = Field(500, gt=0)
    # 'cents' on a cent account: balances/equity are cents while capital-tier
    # boundaries stay in USD; risk code converts for tier lookups only.
    account_units: str = "usd"

    @field_validator("account_units")
    @classmethod
    def _units(cls, v: str) -> str:
        if v not in ("usd", "cents"):
            raise ValueError("account_units must be 'usd' or 'cents'")
        return v

    @property
    def units_per_usd(self) -> float:
        return 100.0 if self.account_units == "cents" else 1.0

    # Credentials live only in the environment.
    @property
    def login(self) -> Optional[int]:
        v = os.environ.get("DANALIT_MT5_LOGIN")
        return int(v) if v else None

    @property
    def server(self) -> Optional[str]:
        return os.environ.get("DANALIT_MT5_SERVER")

    @property
    def password(self) -> Optional[str]:
        return os.environ.get("DANALIT_MT5_PASSWORD")


class RiskConfig(BaseModel):
    risk_per_trade: float = Field(0.0075, gt=0, lt=0.05)
    max_positions: int = Field(2, ge=1)
    max_positions_per_instrument: int = Field(1, ge=1)
    max_total_risk: float = Field(0.015, gt=0, lt=0.10)
    daily_loss_limit: float = Field(0.03, gt=0, lt=0.5)
    weekly_loss_limit: float = Field(0.06, gt=0, lt=0.5)
    max_drawdown: float = Field(0.15, gt=0, lt=1.0)
    consecutive_loss_brake: int = Field(4, ge=2)
    brake_risk_factor: float = Field(0.5, gt=0, le=1.0)
    brake_hours: int = Field(24, ge=1)
    min_lot_risk_cap_mult: float = Field(1.5, ge=1.0)

    @model_validator(mode="after")
    def _sanity(self) -> "RiskConfig":
        if self.weekly_loss_limit < self.daily_loss_limit:
            raise ValueError("weekly_loss_limit must be >= daily_loss_limit")
        if self.max_total_risk < self.risk_per_trade:
            raise ValueError("max_total_risk must be >= risk_per_trade")
        return self


class TradingConfig(BaseModel):
    primary_timeframe: str = "M15"
    loop_interval_sec: int = Field(30, ge=5)
    dry_run: bool = True
    weekend_flatten_utc: str = "20:30"

    @field_validator("primary_timeframe")
    @classmethod
    def _tf(cls, v: str) -> str:
        if v not in TIMEFRAMES:
            raise ValueError(f"unknown timeframe {v!r}; expected one of {list(TIMEFRAMES)}")
        return v

    @field_validator("weekend_flatten_utc")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        if not _TIME_RE.match(v):
            raise ValueError("weekend_flatten_utc must be HH:MM (24h UTC)")
        return v


class NewsConfig(BaseModel):
    poll_minutes: int = Field(5, ge=1)
    calendar_poll_minutes: int = Field(30, ge=5)
    blackout_minutes: int = Field(15, ge=0)


class LabelingConfig(BaseModel):
    k_tp: float = Field(2.0, gt=0)
    k_sl: float = Field(1.0, gt=0)
    horizon_bars: int = Field(96, ge=1)
    dead_zone_atr: float = Field(0.25, ge=0)


class DatasetConfig(BaseModel):
    embargo_days: int = Field(5, ge=0)


class CapitalTier(BaseModel):
    max_equity: Optional[float] = None  # None = open-ended top tier
    instruments: list[str]
    risk_per_trade: float = Field(gt=0, lt=0.05)
    set_aside_pct: float = Field(ge=0, le=1.0)


class CapitalConfig(BaseModel):
    tiers: list[CapitalTier]
    tier_hysteresis_days: int = Field(5, ge=1)
    min_withdrawal: float = Field(25.0, gt=0)

    @model_validator(mode="after")
    def _tiers_ordered(self) -> "CapitalConfig":
        if not self.tiers:
            raise ValueError("capital.tiers must not be empty")
        bounds = [t.max_equity for t in self.tiers]
        if bounds[-1] is not None:
            raise ValueError("last capital tier must have max_equity: null (open-ended)")
        finite = [b for b in bounds[:-1]]
        if any(b is None for b in finite) or finite != sorted(finite):
            raise ValueError("capital tiers must be ordered by ascending max_equity, only last open-ended")
        return self


class Settings(BaseModel):
    paths: PathsConfig = PathsConfig()
    broker: BrokerConfig = BrokerConfig()
    risk: RiskConfig = RiskConfig()
    trading: TradingConfig = TradingConfig()
    news: NewsConfig = NewsConfig()
    labeling: LabelingConfig = LabelingConfig()
    dataset: DatasetConfig = DatasetConfig()
    capital: CapitalConfig


class InstrumentConfig(BaseModel):
    broker_symbol: str
    pip_size: float = Field(gt=0)
    digits: int = Field(ge=0)
    contract_size: float = Field(gt=0)
    min_lot: float = Field(gt=0)
    lot_step: float = Field(gt=0)
    max_lot: float = Field(gt=0)
    spread_estimate_pips: float = Field(ge=0)
    commission_per_lot: float = 0.0
    swap_long_pips: float = 0.0
    swap_short_pips: float = 0.0
    sessions: list[str] = []
    enabled: bool = True
    news_currencies: list[str]
    round_number_grid: float = Field(gt=0)

    @field_validator("news_currencies")
    @classmethod
    def _nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("news_currencies must not be empty")
        return v


class AppConfig(BaseModel):
    settings: Settings
    instruments: dict[str, InstrumentConfig]

    @model_validator(mode="after")
    def _cross_validate(self) -> "AppConfig":
        if not self.instruments:
            raise ValueError("instruments.yaml defined no instruments")
        known = set(self.instruments)
        for tier in self.settings.capital.tiers:
            unknown = set(tier.instruments) - known
            if unknown:
                raise ValueError(f"capital tier references unknown instruments: {sorted(unknown)}")
        return self

    def enabled_instruments(self) -> list[str]:
        return [k for k, v in self.instruments.items() if v.enabled]


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def load_config(
    settings_path: Optional[Path] = None,
    instruments_path: Optional[Path] = None,
) -> AppConfig:
    settings_path = settings_path or CONFIG_DIR / "settings.yaml"
    instruments_path = instruments_path or CONFIG_DIR / "instruments.yaml"
    settings = Settings(**_read_yaml(settings_path))
    instruments = {
        name: InstrumentConfig(**block)
        for name, block in _read_yaml(instruments_path).items()
    }
    return AppConfig(settings=settings, instruments=instruments)
