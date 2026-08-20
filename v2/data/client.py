"""Financial Datasets API client."""

from __future__ import annotations

import logging
import os
import time

import requests

from v2.data.coverage import check_price_coverage
from v2.data.errors import FDClientError, FDCoverageError
from v2.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsRecord,
    FinancialMetrics,
    InsiderTrade,
    Price,
)

logger = logging.getLogger(__name__)

__all__ = ["FDClient", "FDClientError", "FDCoverageError"]


class FDClient:
    """Financial Datasets API client.

    Usage::

        with FDClient() as fd:
            prices = fd.get_prices("AAPL", "2024-01-01", "2024-12-31")
    """

    BASE_URL = "https://api.financialdatasets.ai"
    _RETRY_DELAYS = (5, 15, 30)
    # /prices/ serves 100 rows a page and ignores limit. Ten years of
    # daily bars is about 2,520 rows, so 40 pages covers any window a
    # caller has reason to ask for while still ending a server that
    # hands back a cursor forever.
    _MAX_PAGES = 40

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["X-API-Key"] = self._api_key

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> FDClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "day",
        interval_multiplier: int = 1,
        *,
        max_pages: int | None = None,
        require_full_range: bool = False,
    ) -> list[Price]:
        """Fetch OHLC price bars, following the cursor to the last page.

        The endpoint serves 100 rows a page and ignores ``limit``. It also
        clamps history to about a year on this plan, silently, which paging
        cannot undo. Pass ``require_full_range=True`` when a short window
        would make the answer wrong rather than merely smaller; otherwise the
        gap is logged and the rows are returned.
        """
        rows = self._get_paged("/prices/", {
            "ticker": ticker,
            "interval": interval,
            "interval_multiplier": interval_multiplier,
            "start_date": start_date,
            "end_date": end_date,
        }, response_key="prices", max_pages=max_pages or self._MAX_PAGES)
        prices = [Price(**row) for row in rows]
        check_price_coverage(
            prices, ticker, start_date, end_date,
            require_full_range=require_full_range, path="/prices/",
        )
        return prices

    # ------------------------------------------------------------------
    # Financial Metrics
    # ------------------------------------------------------------------

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        """Fetch financial metrics that were PUBLIC as of *end_date*.

        Point-in-time: filters on ``filing_date`` (when the SEC filing was
        accepted, ET) — not ``report_period`` (the fiscal period end, which
        precedes public availability by 3-6 weeks and would leak the future
        into a backtest). Rows without a known filing_date are excluded
        server-side, so everything returned was provably knowable on
        *end_date*.
        """
        data = self._get("/financial-metrics/", {
            "ticker": ticker,
            "filing_date_lte": end_date,
            "period": period,
            "limit": limit,
        }, response_key="financial_metrics")
        return [FinancialMetrics(**row) for row in data] if data else []

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def get_news(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[CompanyNews]:
        """Fetch company news."""
        params: dict = {"ticker": ticker, "end_date": end_date, "limit": limit}
        if start_date is not None:
            params["start_date"] = start_date
        data = self._get("/news/", params, response_key="news")
        return [CompanyNews(**row) for row in data] if data else []

    # ------------------------------------------------------------------
    # Insider Trades
    # ------------------------------------------------------------------

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        """Fetch insider trades."""
        params: dict = {"ticker": ticker, "filing_date_lte": end_date, "limit": limit}
        if start_date is not None:
            params["filing_date_gte"] = start_date
        data = self._get("/insider-trades/", params, response_key="insider_trades")
        return [InsiderTrade(**row) for row in data] if data else []

    # ------------------------------------------------------------------
    # Company Facts
    # ------------------------------------------------------------------

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        """Fetch company metadata (single record)."""
        resp = self._request("GET", "/company/facts/", params={"ticker": ticker})
        if resp is None:
            return None
        facts_data = resp.json().get("company_facts")
        return CompanyFacts(**facts_data) if facts_data else None

    # ------------------------------------------------------------------
    # Earnings
    # ------------------------------------------------------------------

    def get_earnings(self, ticker: str) -> Earnings | None:
        """Fetch latest earnings for a single ticker."""
        data = self._get("/earnings/", {"ticker": ticker}, response_key="earnings")
        if not data:
            return None
        row = data[0] if isinstance(data, list) else data
        return Earnings(**row)

    def get_earnings_history(
        self,
        ticker: str,
        limit: int = 12,
    ) -> list[EarningsRecord]:
        """Fetch historical earnings filings as a flat list.

        Returns one record per SEC filing (8-K, 10-Q, 10-K, 20-F).
        The same ``report_period`` may appear multiple times with
        different ``source_type`` values.
        """
        data = self._get("/earnings/", {
            "ticker": ticker,
            "limit": limit,
        }, response_key="earnings")
        return [EarningsRecord(**row) for row in data] if data else []

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        """Return market cap from company facts or financial metrics."""
        facts = self.get_company_facts(ticker)
        if facts is not None and facts.market_cap is not None:
            return facts.market_cap
        metrics = self.get_financial_metrics(ticker, end_date, limit=1)
        if metrics and metrics[0].market_cap is not None:
            return metrics[0].market_cap
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get(
        self,
        path: str,
        params: dict,
        response_key: str,
    ) -> list[dict] | None:
        """GET and extract *response_key* from the JSON payload."""
        resp = self._request("GET", path, params=params)
        if resp is None:
            return None
        return resp.json().get(response_key)

    def _get_paged(
        self,
        path: str,
        params: dict,
        response_key: str,
        max_pages: int,
    ) -> list[dict]:
        """GET and follow next_page_url until it runs out.

        The cursor URL already carries the query, so it is requested verbatim.
        Re-sending params alongside it would reset the walk to page one and
        loop forever. Two other things end the walk: a repeated cursor, which
        means the server has stopped advancing, and the page cap.
        """
        rows: list[dict] = []
        seen: set[str] = set()
        next_url: str | None = None

        for _ in range(max_pages):
            if next_url is None:
                resp = self._request("GET", path, params=params)
            else:
                resp = self._request("GET", path, url=next_url)
            if resp is None:
                break

            payload = resp.json()
            rows.extend(payload.get(response_key) or [])

            next_url = payload.get("next_page_url")
            if not next_url or next_url in seen:
                break
            seen.add(next_url)
        else:
            logger.warning(
                "%s: stopped at the %d-page cap with a cursor still open; "
                "rows beyond that point were not read",
                path, max_pages,
            )
        return rows

    def _request(
        self,
        method: str,
        path: str,
        url: str | None = None,
        **kwargs,
    ) -> requests.Response | None:
        """HTTP request with retry on 429.

        Fail-loud contract: raises FDClientError on network errors, HTTP
        errors, and exhausted rate-limit retries. Returns None ONLY for
        404 — "this data doesn't exist" is a data fact, not a failure.
        Silently returning empty on real failures poisons backtests
        (missing data reads as "no signal").
        """
        target = url or (self.BASE_URL + path)
        for attempt, delay in enumerate((*self._RETRY_DELAYS, None)):
            try:
                resp = self._session.request(
                    method, target, timeout=self._timeout, **kwargs,
                )
            except requests.RequestException as exc:
                raise FDClientError(
                    f"{method} {path} failed: {exc}", path=path,
                ) from exc

            if resp.status_code == 429 and delay is not None:
                logger.info(
                    "Rate limited (429), retrying in %ds (attempt %d/%d)",
                    delay, attempt + 1, len(self._RETRY_DELAYS),
                )
                time.sleep(delay)
                continue

            if resp.status_code == 404:
                return None

            if resp.status_code >= 400:
                raise FDClientError(
                    f"{method} {path} returned {resp.status_code}: {resp.text[:200]}",
                    status_code=resp.status_code, path=path,
                )

            return resp

        raise FDClientError(
            f"{method} {path} rate limited (429) after {len(self._RETRY_DELAYS)} retries",
            status_code=429, path=path,
        )
