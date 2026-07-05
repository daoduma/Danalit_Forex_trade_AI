"""Manual integration test against a DEMO account: connect, validate symbols,
open 0.01 EURUSD with SL/TP, modify SL, partial close if possible, close, print deals.

    python scripts/gateway_smoke_test.py            # refuses on a live account
    python scripts/gateway_smoke_test.py --i-know-this-is-live   # you were warned
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from danalit.timeutil import utc_now  # noqa: E402
from danalit.trading.mt5_gateway import MT5Gateway  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--i-know-this-is-live", action="store_true")
    args = ap.parse_args()

    gw = MT5Gateway()
    gw.connect()
    account = gw.get_account()
    print(f"account {account.login} | balance {account.balance} {account.currency} "
          f"| {'DEMO' if account.is_demo else '*** LIVE ***'}")
    if not account.is_demo and not args.i_know_this_is_live:
        print("\n" + "!" * 70)
        print("!!  THIS IS NOT A DEMO ACCOUNT. Aborting the smoke test.        !!")
        print("!!  Re-run with --i-know-this-is-live only if you truly mean it. !!")
        print("!" * 70)
        return 2

    specs = gw.validate_symbols()
    for name, s in specs.items():
        print(f"  {name}: {s.broker_symbol} lot {s.min_lot}/{s.lot_step}, "
              f"stops_level {s.stops_level}")

    tick = gw.tick("EURUSD")
    sl, tp = tick.bid - 0.0050, tick.bid + 0.0050
    print(f"\nopening 0.01 EURUSD long @~{tick.ask} (sl {sl:.5f} tp {tp:.5f})")
    r = gw.market_order("EURUSD", 1, 0.01, sl=sl, tp=tp, comment="danalit:smoke")
    print("  ->", r)
    if not r.ok:
        return 1
    time.sleep(2)

    pos = next((p for p in gw.get_open_positions() if p.id == r.ticket), None)
    if pos:
        print(f"modifying SL to {tick.bid - 0.0040:.5f}")
        print("  ->", gw.modify_position_sltp(pos.id, sl=tick.bid - 0.0040, tp=tp))
        time.sleep(2)
        if pos.lots >= 0.02:
            print("partial close 50%:", gw.close_position(pos.id, 0.5))
            time.sleep(2)
        print("closing remainder:", gw.close_position(pos.id, 1.0))

    print("\ndeals in the last hour:")
    for d in gw.get_deals_history(pd.Timestamp(utc_now()) - pd.Timedelta(hours=1)):
        print("  ", d)
    gw.shutdown()
    print("\nsmoke test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
