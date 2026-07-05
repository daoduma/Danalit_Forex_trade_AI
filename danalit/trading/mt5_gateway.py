"""MT5 broker gateway — the ONLY module allowed to import MetaTrader5.

Everything else consumes its normalized types (BrokerPosition, OrderResult,
Tick, AccountInfo, SymbolSpec). The MetaTrader5 module is injected in tests and
lazy-imported live, so the test suite runs without a terminal.

Retcode policy (MT5 trade server return codes):
  10004 requote / 10020 price changed / 10021 off quotes
      -> retry up to 3x with a fresh price and small backoff
  10024 too many requests / 10028 locked (context busy) / 10031 no connection
      -> short backoff retry
  10019 no money / 10016 invalid stops / 10018 market closed / anything else
      -> fail fast with a typed error in OrderResult
"""

from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from danalit.config import AppConfig, load_config
from danalit.logging_setup import setup_logging
from danalit.timeutil import utc_now

log = setup_logging("gateway")

# ---- retcodes (numeric so tests need no MT5 install) -----------------------
RC_DONE, RC_DONE_PARTIAL = 10009, 10010
RC_REQUOTE, RC_PRICE_CHANGED, RC_OFF_QUOTES = 10004, 10020, 10021
RC_TOO_FREQUENT, RC_LOCKED, RC_NO_CONNECTION = 10024, 10028, 10031
RC_INVALID_STOPS, RC_MARKET_CLOSED, RC_NO_MONEY = 10016, 10018, 10019

RETRY_FRESH_PRICE = {RC_REQUOTE, RC_PRICE_CHANGED, RC_OFF_QUOTES}
RETRY_BACKOFF = {RC_TOO_FREQUENT, RC_LOCKED, RC_NO_CONNECTION}
SUCCESS = {RC_DONE, RC_DONE_PARTIAL}

MAX_RETRIES = 3
ACCOUNT_TRADE_MODE_DEMO = 0


class GatewayError(RuntimeError):
    pass


class GatewayConfigError(GatewayError):
    """Config disagrees with live symbol specs — refuse to start."""


@dataclass
class Tick:
    bid: float
    ask: float
    time_utc: pd.Timestamp


@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin: float
    margin_free: float
    currency: str
    trade_mode: int
    login: int = 0

    @property
    def is_demo(self) -> bool:
        return self.trade_mode == ACCOUNT_TRADE_MODE_DEMO


@dataclass
class SymbolSpec:
    broker_symbol: str
    contract_size: float
    point: float
    digits: int
    min_lot: float
    lot_step: float
    max_lot: float
    stops_level: int      # min SL/TP distance in points
    freeze_level: int
    trade_mode: int
    filling_mode: int


@dataclass
class BrokerPosition:
    id: int               # broker ticket
    instrument: str
    side: int             # +1 / -1
    lots: float
    entry_price: float
    entry_time: pd.Timestamp
    sl: Optional[float]
    tp: Optional[float]
    comment: str = ""
    profit: float = 0.0
    # fields for TradeManager compatibility
    contract_size: float = 0.0
    initial_lots: float = 0.0
    best_price: float = 0.0
    worst_price: float = 0.0

    @property
    def signal_id(self) -> Optional[str]:
        return self.comment.split("danalit:", 1)[1] if "danalit:" in self.comment else None


@dataclass
class OrderResult:
    ok: bool
    retcode: Optional[int] = None
    ticket: Optional[int] = None
    price: Optional[float] = None
    error: str = ""
    attempts: int = 1


@dataclass
class ReconcileReport:
    orphans: list = field(default_factory=list)  # at broker, not in journal
    ghosts: list = field(default_factory=list)   # in journal, not at broker

    @property
    def clean(self) -> bool:
        return not self.orphans and not self.ghosts


def _load_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
        return mt5
    except ImportError as e:  # pragma: no cover
        raise GatewayError(
            "MetaTrader5 package not installed (pip install MetaTrader5; Windows only, "
            "requires a running MT5 terminal)."
        ) from e


class MT5Gateway:
    def __init__(self, cfg: Optional[AppConfig] = None, mt5=None,
                 sleep: Callable[[float], None] = _time.sleep):
        self.cfg = cfg or load_config()
        self._mt5 = mt5  # injected fake in tests; lazy-loaded live
        self._sleep = sleep
        self.specs: dict[str, SymbolSpec] = {}
        self._symbol_of = {k: v.broker_symbol for k, v in self.cfg.instruments.items()}
        self._instrument_of = {v: k for k, v in self._symbol_of.items()}
        self.magic = self.cfg.settings.broker.magic_number

    @property
    def mt5(self):
        if self._mt5 is None:
            self._mt5 = _load_mt5()
        return self._mt5

    # ------------------------------------------------------------ connection
    def connect(self) -> None:
        b = self.cfg.settings.broker
        kwargs = {}
        if b.login:
            kwargs = {"login": b.login, "server": b.server, "password": b.password}
        if not self.mt5.initialize(**kwargs):
            raise GatewayError(f"MT5 initialize failed: {self.mt5.last_error()} — "
                               "is the terminal installed, running and logged in?")
        log.info("gateway connected (account %s)", getattr(self.get_account(), "login", "?"))

    def ensure_connected(self) -> None:
        if self.mt5.terminal_info() is None:
            log.warning("terminal connection lost — reinitializing")
            self.connect()

    def shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    # --------------------------------------------------------------- symbols
    def validate_symbols(self) -> dict[str, SymbolSpec]:
        """Resolve + select every enabled symbol; refuse to start on config mismatch."""
        for name in self.cfg.enabled_instruments():
            inst = self.cfg.instruments[name]
            sym = inst.broker_symbol
            info = self.mt5.symbol_info(sym)
            if info is None:
                raise GatewayConfigError(f"{name}: broker symbol {sym!r} not found")
            if not getattr(info, "visible", True):
                self.mt5.symbol_select(sym, True)
                info = self.mt5.symbol_info(sym)
            spec = SymbolSpec(
                broker_symbol=sym,
                contract_size=float(info.trade_contract_size),
                point=float(info.point),
                digits=int(info.digits),
                min_lot=float(info.volume_min),
                lot_step=float(info.volume_step),
                max_lot=float(info.volume_max),
                stops_level=int(info.trade_stops_level),
                freeze_level=int(getattr(info, "trade_freeze_level", 0)),
                trade_mode=int(getattr(info, "trade_mode", 4)),
                filling_mode=int(getattr(info, "filling_mode", 1)),
            )
            problems = []
            if abs(spec.min_lot - inst.min_lot) > 1e-9:
                problems.append(f"min_lot config {inst.min_lot} != broker {spec.min_lot}")
            if abs(spec.lot_step - inst.lot_step) > 1e-9:
                problems.append(f"lot_step config {inst.lot_step} != broker {spec.lot_step}")
            if abs(spec.contract_size - inst.contract_size) > 1e-6:
                problems.append(
                    f"contract_size config {inst.contract_size} != broker {spec.contract_size}")
            if problems:
                raise GatewayConfigError(f"{name}: " + "; ".join(problems) +
                                         " — fix instruments.yaml before trading")
            self.specs[name] = spec
        return self.specs

    def tick(self, instrument: str) -> Tick:
        t = self.mt5.symbol_info_tick(self._symbol_of[instrument])
        if t is None:
            raise GatewayError(f"no tick for {instrument}")
        return Tick(bid=float(t.bid), ask=float(t.ask),
                    time_utc=pd.Timestamp(utc_now()))

    # ---------------------------------------------------------------- orders
    def _adjust_stops(self, instrument: str, side: int, price: float,
                      sl: Optional[float], tp: Optional[float]):
        """Respect stops_level minimum distances by pushing out; log adjustments."""
        spec = self.specs.get(instrument)
        if spec is None or spec.stops_level <= 0:
            return sl, tp
        min_dist = spec.stops_level * spec.point
        if sl is not None and abs(price - sl) < min_dist:
            new_sl = price - side * min_dist
            log.warning("%s: SL %.5f within stops_level — adjusted to %.5f",
                        instrument, sl, new_sl)
            sl = new_sl
        if tp is not None and abs(tp - price) < min_dist:
            new_tp = price + side * min_dist
            log.warning("%s: TP %.5f within stops_level — adjusted to %.5f",
                        instrument, tp, new_tp)
            tp = new_tp
        return sl, tp

    def market_order(self, instrument: str, side: int, lots: float,
                     sl: Optional[float] = None, tp: Optional[float] = None,
                     comment: str = "") -> OrderResult:
        mt5 = self.mt5
        sym = self._symbol_of[instrument]
        attempts = 0
        while attempts < MAX_RETRIES:
            attempts += 1
            self.ensure_connected()
            tick = self.tick(instrument)
            price = tick.ask if side > 0 else tick.bid
            adj_sl, adj_tp = self._adjust_stops(instrument, side, price, sl, tp)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": sym,
                "volume": float(lots),
                "type": mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": float(adj_sl) if adj_sl else 0.0,
                "tp": float(adj_tp) if adj_tp else 0.0,
                "deviation": 20,
                "magic": self.magic,
                "comment": comment[:31],  # MT5 comment limit
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling(instrument),
            }
            result = mt5.order_send(request)
            if result is None:
                return OrderResult(False, error=f"order_send returned None: "
                                   f"{mt5.last_error()}", attempts=attempts)
            rc = result.retcode
            if rc in SUCCESS:
                return OrderResult(True, rc, getattr(result, "order", None),
                                   getattr(result, "price", price), attempts=attempts)
            if rc in RETRY_FRESH_PRICE:
                log.warning("%s: retcode %s (requote/price) — retry %d/%d",
                            instrument, rc, attempts, MAX_RETRIES)
                self._sleep(0.2 * attempts)
                continue
            if rc in RETRY_BACKOFF:
                log.warning("%s: retcode %s (busy) — backoff retry %d/%d",
                            instrument, rc, attempts, MAX_RETRIES)
                self._sleep(0.5 * attempts)
                continue
            return OrderResult(False, rc, error=self._describe(rc), attempts=attempts)
        return OrderResult(False, rc, error=f"gave up after {attempts} attempts "
                           f"(last retcode {rc})", attempts=attempts)

    @staticmethod
    def _describe(rc: int) -> str:
        return {
            RC_NO_MONEY: "not enough money — order refused (fail fast)",
            RC_INVALID_STOPS: "invalid stops — SL/TP violates broker constraints",
            RC_MARKET_CLOSED: "market closed",
        }.get(rc, f"trade server retcode {rc}")

    def _filling(self, instrument: str) -> int:
        spec = self.specs.get(instrument)
        mt5 = self.mt5
        mode = spec.filling_mode if spec else 1
        # bitmask: 1 = FOK, 2 = IOC per SYMBOL_FILLING_*; prefer IOC when allowed
        if mode & 2:
            return mt5.ORDER_FILLING_IOC
        if mode & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def modify_position_sltp(self, ticket: int, sl: Optional[float],
                             tp: Optional[float]) -> OrderResult:
        mt5 = self.mt5
        pos = self._position_by_ticket(ticket)
        if pos is None:
            return OrderResult(False, error=f"position {ticket} not found")
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": self._symbol_of[pos.instrument],
            "sl": float(sl) if sl is not None else (pos.sl or 0.0),
            "tp": float(tp) if tp is not None else (pos.tp or 0.0),
        }
        result = mt5.order_send(request)
        if result is None:
            return OrderResult(False, error=str(mt5.last_error()))
        ok = result.retcode in SUCCESS
        return OrderResult(ok, result.retcode,
                           error="" if ok else self._describe(result.retcode))

    def close_position(self, ticket: int, fraction: float = 1.0,
                       comment: str = "") -> OrderResult:
        mt5 = self.mt5
        pos = self._position_by_ticket(ticket)
        if pos is None:
            return OrderResult(False, error=f"position {ticket} not found")
        spec = self.specs.get(pos.instrument)
        step = spec.lot_step if spec else 0.01
        lots = pos.lots if fraction >= 1.0 else max(
            round(int(pos.lots * fraction / step) * step, 8), step)
        tick = self.tick(pos.instrument)
        price = tick.bid if pos.side > 0 else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self._symbol_of[pos.instrument],
            "volume": float(lots),
            "type": mt5.ORDER_TYPE_SELL if pos.side > 0 else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(pos.instrument),
        }
        result = mt5.order_send(request)
        if result is None:
            return OrderResult(False, error=str(mt5.last_error()))
        ok = result.retcode in SUCCESS
        return OrderResult(ok, result.retcode, getattr(result, "order", None),
                           getattr(result, "price", price),
                           error="" if ok else self._describe(result.retcode))

    # ----------------------------------------------------------------- state
    def get_open_positions(self, all_magic: bool = False) -> list[BrokerPosition]:
        raw = self.mt5.positions_get() or []
        out = []
        for p in raw:
            if not all_magic and getattr(p, "magic", 0) != self.magic:
                continue
            instrument = self._instrument_of.get(p.symbol, p.symbol)
            spec = self.specs.get(instrument)
            out.append(BrokerPosition(
                id=p.ticket, instrument=instrument,
                side=1 if p.type == 0 else -1,
                lots=float(p.volume), entry_price=float(p.price_open),
                entry_time=pd.Timestamp(p.time, unit="s", tz="UTC"),
                sl=float(p.sl) or None, tp=float(p.tp) or None,
                comment=getattr(p, "comment", "") or "",
                profit=float(getattr(p, "profit", 0.0)),
                contract_size=spec.contract_size if spec else 0.0,
                initial_lots=float(p.volume),
                best_price=float(p.price_current), worst_price=float(p.price_current),
            ))
        return out

    def _position_by_ticket(self, ticket: int) -> Optional[BrokerPosition]:
        return next((p for p in self.get_open_positions(all_magic=True)
                     if p.id == ticket), None)

    def get_account(self) -> AccountInfo:
        a = self.mt5.account_info()
        if a is None:
            raise GatewayError("account_info() returned None — not logged in?")
        return AccountInfo(balance=float(a.balance), equity=float(a.equity),
                           margin=float(a.margin), margin_free=float(a.margin_free),
                           currency=a.currency, trade_mode=int(a.trade_mode),
                           login=int(getattr(a, "login", 0)))

    def get_deals_history(self, since: pd.Timestamp):
        return self.mt5.history_deals_get(since.to_pydatetime(),
                                          utc_now()) or []

    # ---------------------------------------------------------- reconcile
    def reconcile(self, journal_open: list[dict]) -> ReconcileReport:
        """journal_open: [{'ticket': int|None, 'signal_id': str, 'instrument': str}]"""
        broker = self.get_open_positions()
        by_ticket = {p.id: p for p in broker}
        by_signal = {p.signal_id: p for p in broker if p.signal_id}
        report = ReconcileReport()
        matched: set[int] = set()
        for j in journal_open:
            p = by_ticket.get(j.get("ticket")) or by_signal.get(j.get("signal_id"))
            if p is None:
                report.ghosts.append(j)
            else:
                matched.add(p.id)
        report.orphans = [p for p in broker if p.id not in matched]
        return report
