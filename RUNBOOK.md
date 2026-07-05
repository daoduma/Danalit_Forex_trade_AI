# Danalit RUNBOOK

Operational procedures. If you can execute everything in this document, you can
operate the system. Keep it printed or on a second device — you will need it
exactly when the main machine is misbehaving.

---

## 1. Cold start (fresh boot or new machine)

1. `python -m danalit.preflight` — must end `PASS`. Fix any FAIL before continuing.
2. Start the MT5 terminal; confirm it is logged in to the **intended** account
   (demo vs cent-live — check the title bar).
3. `python scripts/run_collector.py` (or confirm the `Danalit Collector` task is running:
   `schtasks /Query /TN "Danalit Collector"`).
4. `python scripts/run_trading.py` — watch the log: `STARTING -> RECONCILING -> TRADING`.
   Any halt reason is printed and journaled.
5. Confirm from your phone: `/status` shows state TRADING, fresh heartbeats.

## 2. Clean shutdown

1. `/halt` from Telegram (or create a `HALT` file in the repo root) — stops new entries.
2. If positions should not ride overnight: `/halt_flat` instead (flattens).
3. Ctrl+C the orchestrator (or `schtasks /End /TN "Danalit Orchestrator"`).
4. The collector can keep running — its archive is valuable regardless.

## 3. Resume after a halt

1. Identify why it halted: `/status`, then `logs/orchestrator.log` and
   `SELECT * FROM system_events ORDER BY id DESC LIMIT 20`.
2. Fix the cause. Do not resume a halt you do not understand.
3. `/resume` from Telegram (deletes kill files, requests resume), or delete the
   `HALT`/`HALT_FLAT` files and run `python scripts/run_trading.py --resume`.
4. Resume re-runs reconciliation and freshness checks; it refuses while a risk
   halt (daily/weekly/breaker) is still active.

## 4. Drawdown breaker manual reset

The 15% breaker means the system flattened everything and disabled itself.
**Mandatory review before reset — write down the answers:**

- What sequence of trades caused it? (journal: last 20 trades with explanations)
- Was execution quality normal? (forward-test report: cost panel)
- Did the model drift? (PSI report; decision confidence distributions)
- Is the market in a regime the backtest never saw?
- Decision: resume as-is / resume at reduced risk / stop and retrain?

Then: `python -m danalit.risk.risk_manager --reset-breaker` and restart per §1.

## 5. Broker maintenance windows / weekends

- Fridays 20:30 UTC: positions are flattened by rule (config `weekend_flatten_utc`).
- Broker server maintenance (usually weekend): expect `no connection` retcodes;
  the gateway retries and the orchestrator halts on repeated failure. Do not
  intervene — verify normal operation after Sunday reopen (~21:00-22:00 UTC).

## 6. CRITICAL alerts — meaning and first three steps

| Alert | Meaning | First three steps |
|---|---|---|
| `DRAWDOWN BREAKER` | equity fell 15% from HWM; system flattened + disabled | 1) confirm flatten in MT5 terminal 2) run §4 review 3) reset only after review |
| `Orchestrator exception` | unhandled error; state HALTED | 1) read the traceback in logs 2) check disk/DB 3) fix, then §3 |
| `restart storm` | watchdog restarted a task 3x in 1h and stopped | 1) read that task's log 2) run its script by hand in a console 3) fix root cause; watchdog resumes automatically next window |
| `AUTO-ROLLBACK <inst>` | promoted model degraded in probation; champion restored | 1) verify champion pointer (`registry`) 2) read retrain report 3) investigate before re-promoting |
| `order failed` (repeated) | broker rejecting orders | 1) check retcodes in journal 2) check margin + symbol trade mode 3) halt if persistent |

## 7. Demo → live (cent account) switchover

Only after the go-live checklist passes (`python scripts/journal_report.py`).

1. Open the MT5 cent account; verify symbols, min lots, and that `instruments.yaml`
   matches the **new** broker's specs (the gateway refuses on mismatch — good).
2. Point env vars at the live account: `DANALIT_MT5_LOGIN/SERVER/PASSWORD`.
3. `config/settings.yaml`: `broker.account_units: cents` (already default).
4. Deliberately start at tier-1 risk: equity < $50 ⇒ EURUSD only at 0.75% — this
   happens automatically; do not override it.
5. `python scripts/gateway_smoke_test.py` — it will warn this is a live account;
   run it once with `--i-know-this-is-live` and 0.01 lots to verify round-trip.
6. Set `trading.dry_run: false`. Start per §1. Watch the first day closely.

## 8. BREAK GLASS — flatten everything manually

If Danalit is dead/unreachable and positions are open:

1. Open the MT5 terminal (desktop or phone app — install the phone app NOW,
   not during the emergency).
2. Trade tab → right-click each position → **Close Position**. Do every one.
3. Verify the Trade tab is empty and check the journal later for ghosts —
   startup reconciliation will mark them closed.
4. Create a `HALT` file in the repo root so a watchdog restart cannot re-enter.

## 9. Routine cadence

- **Daily (5 min):** read the digest; `/status`; data freshness; broker statement
  matches the journal.
- **Weekly (30 min):** review every trade with its logged explanation; review the
  skipped-signal/veto log; check the PSI drift report.
- **Monthly:** retrain gate report (`reports/retrain_*.md`); capital policy report
  (`python -m danalit.risk.capital report --equity ... --hwm ...`); withdraw
  set-aside if recommended.
