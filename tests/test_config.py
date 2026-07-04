"""Prompt 1: config loads, validates, and reads secrets from the environment."""

import pytest
from pydantic import ValidationError

from danalit.config import (
    CapitalConfig,
    CapitalTier,
    RiskConfig,
    TradingConfig,
    load_config,
)


def test_config_loads_and_matches_roadmap_values():
    cfg = load_config()
    s = cfg.settings
    assert s.risk.risk_per_trade == 0.0075
    assert s.risk.max_positions == 2
    assert s.risk.max_total_risk == 0.015
    assert s.risk.daily_loss_limit == 0.03
    assert s.risk.weekly_loss_limit == 0.06
    assert s.risk.max_drawdown == 0.15
    assert s.risk.consecutive_loss_brake == 4
    assert s.trading.primary_timeframe == "M15"
    assert s.trading.loop_interval_sec == 30
    assert s.trading.dry_run is True
    assert s.news.poll_minutes == 5
    assert set(cfg.instruments) == {"EURUSD", "XAUUSD", "US100"}
    assert cfg.instruments["EURUSD"].pip_size == 0.0001
    assert cfg.instruments["XAUUSD"].news_currencies == ["USD"]
    assert cfg.instruments["US100"].news_currencies == ["USD"]
    assert cfg.enabled_instruments() == ["EURUSD", "XAUUSD", "US100"]


def test_broker_credentials_come_from_env(monkeypatch):
    cfg = load_config()
    monkeypatch.delenv("DANALIT_MT5_LOGIN", raising=False)
    assert cfg.settings.broker.login is None
    monkeypatch.setenv("DANALIT_MT5_LOGIN", "12345678")
    monkeypatch.setenv("DANALIT_MT5_SERVER", "Broker-Demo")
    monkeypatch.setenv("DANALIT_MT5_PASSWORD", "s3cret")
    assert cfg.settings.broker.login == 12345678
    assert cfg.settings.broker.server == "Broker-Demo"
    assert cfg.settings.broker.password == "s3cret"


def test_invalid_timeframe_rejected():
    with pytest.raises(ValidationError):
        TradingConfig(primary_timeframe="M7")


def test_weekly_limit_must_cover_daily():
    with pytest.raises(ValidationError):
        RiskConfig(daily_loss_limit=0.05, weekly_loss_limit=0.03)


def test_capital_tiers_must_be_ordered_and_open_ended():
    good = CapitalConfig(
        tiers=[
            CapitalTier(max_equity=50, instruments=["EURUSD"], risk_per_trade=0.0075, set_aside_pct=0),
            CapitalTier(max_equity=None, instruments=["EURUSD"], risk_per_trade=0.01, set_aside_pct=0.4),
        ]
    )
    assert good.tiers[-1].max_equity is None
    with pytest.raises(ValidationError):
        CapitalConfig(
            tiers=[
                CapitalTier(max_equity=None, instruments=["EURUSD"], risk_per_trade=0.01, set_aside_pct=0.4),
                CapitalTier(max_equity=50, instruments=["EURUSD"], risk_per_trade=0.0075, set_aside_pct=0),
            ]
        )


def test_capital_tier_unknown_instrument_rejected(tmp_path):
    import yaml
    from danalit.config import CONFIG_DIR

    settings = yaml.safe_load((CONFIG_DIR / "settings.yaml").read_text(encoding="utf-8"))
    settings["capital"]["tiers"][0]["instruments"] = ["DOGEUSD"]
    bad = tmp_path / "settings.yaml"
    bad.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(settings_path=bad)
