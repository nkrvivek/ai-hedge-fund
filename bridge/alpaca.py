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

    def positions_full(self) -> list[dict]:
        """Raw position dicts — the index-hedge sleeve needs qty/cost_basis."""
        return self._req("GET", "/positions") or []

    def submit_market_order(self, symbol: str, notional_or_qty: dict) -> dict:
        body = {"symbol": symbol, "type": "market", "time_in_force": "day", **notional_or_qty}
        return self._req("POST", "/orders", json=body)

    def submit_limit_order(self, symbol: str, *, qty: int, side: str,
                           limit_price: float) -> dict:
        body = {"symbol": symbol, "qty": str(qty), "side": side, "type": "limit",
                "limit_price": str(round(limit_price, 2)), "time_in_force": "day"}
        return self._req("POST", "/orders", json=body)

    def option_contracts(self, underlying: str, *, type_: str, today,
                         dte_min: int, dte_max: int,
                         strike_min: float, strike_max: float) -> list[dict]:
        """Tradable option contracts from the trading API (paper host)."""
        from datetime import timedelta
        params = {
            "underlying_symbols": underlying,
            "type": type_,
            "status": "active",
            "expiration_date_gte": (today + timedelta(days=dte_min)).isoformat(),
            "expiration_date_lte": (today + timedelta(days=dte_max)).isoformat(),
            "strike_price_gte": str(round(strike_min, 2)),
            "strike_price_lte": str(round(strike_max, 2)),
            "limit": 200,
        }
        resp = self._req("GET", "/options/contracts", params=params) or {}
        return resp.get("option_contracts") or []

    def option_quote_latest(self, symbol: str) -> tuple[float, float] | None:
        """(bid, ask) from the options data API; None when unquoted."""
        try:
            resp = self.session.get(
                "https://data.alpaca.markets/v1beta1/options/quotes/latest",
                params={"symbols": symbol}, timeout=15,
            )
            if not resp.ok:
                return None
            q = (resp.json().get("quotes") or {}).get(symbol)
            if not q:
                return None
            bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
            return (bid, ask) if ask > 0 else None
        except Exception:
            return None

    def get_order(self, order_id: str) -> dict | None:
        """Fetch one order — used to read fill qty/avg price for the trade
        ledger. Best-effort: returns None on any failure rather than raising
        into the ledger path."""
        if not order_id:
            return None
        try:
            return self._req("GET", f"/orders/{order_id}")
        except Exception:  # noqa: BLE001 — ledger read must never break a run
            return None

    def close_position(self, symbol: str) -> dict | None:
        """Liquidate the full position via Alpaca's close-position endpoint —
        broker-sized, so a stale local market value can never oversell (the
        2026-07-23 META insufficient-qty failure class)."""
        return self._req("DELETE", f"/positions/{symbol}")

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
