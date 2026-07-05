"""A programmable in-memory MetaTrader5 stand-in for gateway tests."""

from types import SimpleNamespace


class FakeMT5:
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 2
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2

    def __init__(self):
        self.initialized = False
        self.symbols = {}
        self.ticks = {}
        self.positions = []
        self.deals = []
        self.account = SimpleNamespace(
            balance=2000.0, equity=2000.0, margin=0.0, margin_free=2000.0,
            currency="USD", trade_mode=0, login=12345678)
        self.retcode_queue = []          # pop-left per order_send
        self.order_send_calls = []
        self._next_ticket = 1000

    # -- module surface -----------------------------------------------------
    def initialize(self, **kwargs):
        self.initialized = True
        return True

    def shutdown(self):
        self.initialized = False

    def last_error(self):
        return (0, "ok")

    def terminal_info(self):
        return SimpleNamespace(connected=True) if self.initialized else None

    def account_info(self):
        return self.account

    def symbol_info(self, sym):
        return self.symbols.get(sym)

    def symbol_select(self, sym, enable=True):
        if sym in self.symbols:
            self.symbols[sym].visible = True
            return True
        return False

    def symbol_info_tick(self, sym):
        return self.ticks.get(sym)

    def positions_get(self, **kwargs):
        return list(self.positions)

    def history_deals_get(self, a, b):
        return list(self.deals)

    def order_send(self, request):
        self.order_send_calls.append(dict(request))
        rc = self.retcode_queue.pop(0) if self.retcode_queue else 10009
        ticket = None
        if rc in (10009, 10010) and request.get("action") == self.TRADE_ACTION_DEAL:
            ticket = self._next_ticket
            self._next_ticket += 1
        return SimpleNamespace(retcode=rc, order=ticket,
                               price=request.get("price", 0.0), comment="")

    # -- helpers for tests ----------------------------------------------------
    def add_symbol(self, sym, contract_size=100_000, point=0.00001, digits=5,
                   volume_min=0.01, volume_step=0.01, volume_max=100.0,
                   trade_stops_level=0, filling_mode=2):
        self.symbols[sym] = SimpleNamespace(
            trade_contract_size=contract_size, point=point, digits=digits,
            volume_min=volume_min, volume_step=volume_step, volume_max=volume_max,
            trade_stops_level=trade_stops_level, trade_freeze_level=0,
            trade_mode=4, filling_mode=filling_mode, visible=True)

    def set_tick(self, sym, bid, ask, t=1_750_000_000):
        self.ticks[sym] = SimpleNamespace(bid=bid, ask=ask, time=t)

    def add_position(self, ticket, sym, side, volume, price_open, sl=0.0, tp=0.0,
                     magic=20260701, comment=""):
        self.positions.append(SimpleNamespace(
            ticket=ticket, symbol=sym, type=0 if side > 0 else 1, volume=volume,
            price_open=price_open, sl=sl, tp=tp, magic=magic, comment=comment,
            time=1_750_000_000, profit=0.0, price_current=price_open))
