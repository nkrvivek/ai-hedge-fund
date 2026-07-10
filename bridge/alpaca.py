"""Minimal Alpaca PAPER trading client for the signal bridge.

Hard-quarantined: refuses any endpoint that is not paper-api.alpaca.markets,
and refuses accounts whose equity looks like a real book. This bridge is a
1-month experiment (user directive 2026-07-10) — no live-trading path exists
here on purpose.
"""

from __future__ import annotations

import os
from typing import Any

import requests

PAPER_HOST = "paper-api.alpaca.markets"
# The experiment account is the $100K Alpaca paper sim. Anything outside this
# band means we are pointed at the wrong account — refuse to trade.
EQUITY_SANITY_MIN = 20_000
EQUITY_SANITY_MAX = 500_000


class AlpacaPaperError(RuntimeError):
    pass


class AlpacaPaper:
    def __init__(self) -> None:
        base = os.environ.get("ALPACA_TRADING_ENDPOINT", "")
        if PAPER_HOST not in base:
            raise AlpacaPaperError(
                f"refusing endpoint {base!r} — bridge is paper-only ({PAPER_HOST})"
            )
        self.base = base.rstrip("/")
        key = os.environ.get("ALPACA_API_KEY_ID")
        secret = os.environ.get("ALPACA_API_SECRET")
        if not key or not secret:
            raise AlpacaPaperError("ALPACA_API_KEY_ID / ALPACA_API_SECRET missing")
        self.session = requests.Session()
        self.session.headers.update(
            {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        )

    def _req(self, method: str, path: str, **kw: Any) -> Any:
        resp = self.session.request(method, f"{self.base}{path}", timeout=30, **kw)
        if not resp.ok:
            raise AlpacaPaperError(f"{method} {path} {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.text else None

    def account(self) -> dict:
        acct = self._req("GET", "/account")
        equity = float(acct["equity"])
        if not (EQUITY_SANITY_MIN <= equity <= EQUITY_SANITY_MAX):
            raise AlpacaPaperError(
                f"account equity ${equity:,.0f} outside sanity band "
                f"[{EQUITY_SANITY_MIN}, {EQUITY_SANITY_MAX}] — wrong account?"
            )
        return acct

    def positions(self) -> dict[str, float]:
        """symbol -> signed market value (short positions negative)."""
        out: dict[str, float] = {}
        for p in self._req("GET", "/positions"):
            out[p["symbol"]] = float(p["market_value"])
        return out

    def submit_market_order(self, symbol: str, notional_or_qty: dict) -> dict:
        body = {"symbol": symbol, "type": "market", "time_in_force": "day", **notional_or_qty}
        return self._req("POST", "/orders", json=body)

    def latest_price(self, symbol: str) -> float | None:
        """Best-effort last trade price via the data API (paper key works)."""
        try:
            resp = self.session.get(
                f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest",
                timeout=15,
            )
            if resp.ok:
                return float(resp.json()["trade"]["p"])
        except Exception:
            pass
        return None
