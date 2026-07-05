"""Notifier: push alerts used by orchestrator/risk/gateway.

Levels: INFO (trade opens/closes — rate-limited and batched), WARNING (halts,
staleness, reconciliation findings), CRITICAL (crash, drawdown breaker —
always immediate). Telegram via plain Bot HTTP API (no heavy dependency);
graceful no-op when DANALIT_TG_TOKEN / DANALIT_TG_CHAT_ID are unset.

BotFather setup (2 minutes): message @BotFather -> /newbot -> copy the token
into DANALIT_TG_TOKEN. Message your bot once, then GET
https://api.telegram.org/bot<token>/getUpdates and copy chat.id into
DANALIT_TG_CHAT_ID.
"""

from __future__ import annotations

import os
import time as _time
from typing import Callable, Optional

from danalit.logging_setup import setup_logging

log = setup_logging("notifier")

LEVELS = ("INFO", "WARNING", "CRITICAL")
INFO_BATCH_SECONDS = 60.0


class Notifier:
    """Base/no-op notifier; also the interface."""

    def notify(self, level: str, title: str, body: str = "") -> None:
        log.info("[%s] %s | %s", level, title, body)


class TelegramNotifier(Notifier):
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        transport: Optional[Callable[[str], None]] = None,
        clock: Callable[[], float] = _time.monotonic,
        batch_seconds: float = INFO_BATCH_SECONDS,
    ):
        self.token = token or os.environ.get("DANALIT_TG_TOKEN")
        self.chat_id = chat_id or os.environ.get("DANALIT_TG_CHAT_ID")
        self.transport = transport or self._http_send
        self.clock = clock
        self.batch_seconds = batch_seconds
        self._info_buffer: list[str] = []
        self._last_info_flush = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def _http_send(self, text: str) -> None:  # pragma: no cover — network
        import requests

        requests.post(
            f"https://api.telegram.org/bot{self.token}/sendMessage",
            json={"chat_id": self.chat_id, "text": text[:4000]},
            timeout=15,
        ).raise_for_status()

    def _send(self, text: str) -> None:
        if not self.configured:
            log.debug("telegram unconfigured — dropping: %s", text[:80])
            return
        try:
            self.transport(text)
        except Exception as e:  # alerts must never crash the caller
            log.error("telegram send failed: %s", e)

    def notify(self, level: str, title: str, body: str = "") -> None:
        assert level in LEVELS, f"unknown level {level}"
        line = f"[{level}] {title}" + (f"\n{body}" if body else "")
        if level == "INFO":
            self._info_buffer.append(line)
            now = self.clock()
            if now - self._last_info_flush >= self.batch_seconds:
                self.flush()
            return
        # WARNING/CRITICAL flush pending INFO first, then send immediately
        self.flush()
        self._send(("🚨 " if level == "CRITICAL" else "⚠️ ") + line)

    def flush(self) -> None:
        if self._info_buffer:
            self._send("\n———\n".join(self._info_buffer))
            self._info_buffer.clear()
        self._last_info_flush = self.clock()


def make_notifier() -> Notifier:
    n = TelegramNotifier()
    if n.configured:
        return n
    log.info("Telegram env vars not set — notifications go to logs only")
    return Notifier()
