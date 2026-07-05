"""Live trading entry point (dry-run by default — see settings.yaml trading.dry_run).

Wires the real components: MT5 gateway, champion models from the registry,
feature pipeline over the price store, risk manager, trade manager, journal.

    python scripts/run_trading.py             # runs the loop until Ctrl+C
    python scripts/run_trading.py --once      # single tick (smoke)
    python scripts/run_trading.py --resume    # explicit resume after HALT
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danalit.config import load_config  # noqa: E402
from danalit.data import price_store  # noqa: E402
from danalit.data.collector_daemon import heartbeat_age_seconds  # noqa: E402
from danalit.features.technical import build_features  # noqa: E402
from danalit.journal.journal import Journal  # noqa: E402
from danalit.models import registry  # noqa: E402
from danalit.risk.risk_manager import RiskManager  # noqa: E402
from danalit.timeutil import utc_now  # noqa: E402
from danalit.trading.mt5_gateway import MT5Gateway  # noqa: E402
from danalit.trading.orchestrator import Orchestrator  # noqa: E402
from danalit.trading.signal_engine import SignalEngine  # noqa: E402
from danalit.trading.trade_manager import ManageParams, TradeManager  # noqa: E402


def make_feature_provider(cfg, gateway):
    """Refresh recent bars from the terminal, then compute the feature tail."""
    tf = cfg.settings.trading.primary_timeframe

    def provider(instrument: str) -> dict:
        # refresh the last ~3 days of M1 from the terminal into the store
        try:
            from danalit.data import mt5_history_ingest

            start = (utc_now() - pd.Timedelta(days=3)).to_pydatetime()
            mt5_history_ingest.fetch_m1(instrument,
                                        cfg.instruments[instrument].broker_symbol, start)
            price_store.build_all_timeframes(instrument)
        except Exception:
            pass  # stale-data veto will fire if this keeps failing
        f = build_features(instrument, include_news=True,
                           start=str(utc_now() - pd.Timedelta(days=30)))
        bars = price_store.read_bars(instrument, tf,
                                     start=utc_now() - pd.Timedelta(days=5))
        row = f.iloc[-1]
        bar_time = f.index[-1]
        close = float(bars[bars["time_utc"] == bar_time]["close"].iloc[-1])
        interval_min = {"M15": 15}.get(tf, 15)
        age = (pd.Timestamp(utc_now()) - bar_time).total_seconds() / 60 / interval_min - 1
        # ATR in price units for the engine barriers
        row = row.copy()
        row["atr"] = row.get("atr_norm", 0.0) * close
        return {
            "bar_time": bar_time, "features_row": row, "close": close,
            "spread": cfg.instruments[instrument].spread_estimate_pips
            * cfg.instruments[instrument].pip_size,
            "bar_age_intervals": max(age, 0.0),
            "collector_age_sec": heartbeat_age_seconds() or 1e9,
        }

    return provider


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    gateway = MT5Gateway(cfg)

    # Preflight gate: the orchestrator refuses the TRADING state on failure.
    from danalit.preflight import print_table, run_preflight

    passed, rows = run_preflight()
    print("preflight:")
    print_table(rows)
    if not passed:
        print("PREFLIGHT FAILED — refusing to start. Fix the FAIL rows above.")
        return 1

    lab = cfg.settings.labeling
    orch = Orchestrator(
        cfg=cfg,
        gateway=gateway,
        risk_manager=RiskManager(cfg),
        signal_engine=SignalEngine(k_tp=lab.k_tp, k_sl=lab.k_sl),
        trade_manager=TradeManager(ManageParams(hold_bars=lab.horizon_bars)),
        journal=Journal(),
        feature_provider=make_feature_provider(cfg, gateway),
        model_loader=lambda name: registry.load_latest(name),
    )
    state = orch.startup()
    print(f"startup -> {state} (mode={orch.mode})")
    if args.resume:
        print(f"resume -> {orch.resume()}")
    if args.once:
        orch.tick()
        print(f"tick done, state={orch.state}")
        return 0
    orch.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
