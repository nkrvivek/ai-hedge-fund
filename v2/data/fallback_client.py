"""FundamentalsFallbackClient — primary DataClient with a fundamentals backstop.

Why this exists: HomeDataClient serves fundamentals from FMP, whose plan
gates some symbols behind a per-symbol paywall (HTTP 402 "not available
under your current subscription"). LLY hit this — 7 of its 8 committee
agents run on FMP fundamentals, so all 7 errored and the ticker was
excluded from the book (2026-07-22 audit; see bridge/run_daily.py
ticker_failure_ratios).

This wrapper delegates everything to a primary client, but for
get_financial_metrics it falls back to a secondary client (FDClient /
financialdatasets.ai) when the primary is blind — either it raised an
infrastructure error (the FMP 402 case) or it returned no rows. Only the
failing/empty tickers touch the secondary, so a shared quota there is spent
sparingly.

Fail-loud contract (v2/data/protocol.py) is preserved: if the primary
raised and the secondary can't cover, the primary's original error is
re-raised — a real outage never degrades into a silent empty.
"""

from __future__ import annotations

from typing import Any

from v2.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsRecord,
    FinancialMetrics,
    InsiderTrade,
    Price,
)
from v2.data.protocol import DataClient


class FundamentalsFallbackClient:
    """Wrap a primary DataClient; back its fundamentals with a secondary.

    Only get_financial_metrics is intercepted. Every other method delegates
    straight to the primary — prices, earnings, market cap, news and insider
    trades all keep their primary source.
    """

    def __init__(self, primary: DataClient, secondary: DataClient) -> None:
        self._primary = primary
        self._secondary = secondary
        self.fallback_tickers: list[str] = []  # tickers the secondary served

    # -- the one intercepted method ---------------------------------------
    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10,
    ) -> list[FinancialMetrics]:
        primary_err: Exception | None = None
        try:
            rows = self._primary.get_financial_metrics(ticker, end_date, period, limit)
            if rows:
                return rows
        except Exception as e:  # noqa: BLE001 - re-raised below if secondary can't cover
            primary_err = e

        # Primary is blind (empty result or raised). Try the secondary.
        try:
            alt = self._secondary.get_financial_metrics(ticker, end_date, period, limit)
        except Exception as sec_err:  # noqa: BLE001
            # Preserve the primary's real error over the secondary's, if any.
            raise primary_err if primary_err is not None else sec_err

        if alt:
            self.fallback_tickers.append(ticker)
            return alt

        # Both blind. Empty is only a valid answer if the primary didn't raise;
        # otherwise the data is genuinely unknown and we must fail loud.
        if primary_err is not None:
            raise primary_err
        return []

    # -- straight delegation ----------------------------------------------
    def get_prices(self, ticker: str, start_date: str, end_date: str, **kw: Any) -> list[Price]:
        return self._primary.get_prices(ticker, start_date, end_date, **kw)

    def get_news(self, ticker: str, end_date: str, start_date: str | None = None,
                 limit: int = 1000) -> list[CompanyNews]:
        return self._primary.get_news(ticker, end_date, start_date, limit)

    def get_insider_trades(self, ticker: str, end_date: str, start_date: str | None = None,
                           limit: int = 1000) -> list[InsiderTrade]:
        return self._primary.get_insider_trades(ticker, end_date, start_date, limit)

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        return self._primary.get_company_facts(ticker)

    def get_earnings(self, ticker: str) -> Earnings | None:
        return self._primary.get_earnings(ticker)

    def get_earnings_history(self, ticker: str, limit: int = 12) -> list[EarningsRecord]:
        return self._primary.get_earnings_history(ticker, limit)

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        return self._primary.get_market_cap(ticker, end_date)

    # -- context-manager parity: manage both underlying sessions ----------
    def __enter__(self) -> "FundamentalsFallbackClient":
        self._primary.__enter__()
        self._secondary.__enter__()
        return self

    def __exit__(self, *a: Any) -> None:
        try:
            self._secondary.__exit__(*a)
        finally:
            self._primary.__exit__(*a)
