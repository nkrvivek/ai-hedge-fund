"""UWFundamentalsClient — derive point-in-time FinancialMetrics from UW statements.

Why this exists: HomeDataClient serves the derived ratio metrics from FMP, whose
plan gates some symbols behind a per-symbol paywall (HTTP 402). LLY hit this — its
committee ran fully dead because FMP 402'd it and the only backstop
(financialdatasets.ai) keeps a shallow ~3-quarter window, below MIN_PERIODS=4.

UnusualWhales retains deep quarterly history (LLY: 5+ filed quarters) and we
already pay flat-rate for it. This client fetches UW's income, balance-sheet and
cash-flow statements and derives the fields the fundamentals snapshot consumes.
It is only used as the *secondary* in a FundamentalsFallbackClient chain — placed
ahead of FDClient so its deep history wins for paywalled names — so every other
DataClient method delegates to the primary and is never called here.

Point-in-time (protocol contract): a period is visible only once UW ingested it
(`inserted_at` <= end_date). Infrastructure failures RAISE; a genuinely empty
history returns []. Derivation never invents a number — any missing or
zero-denominator input yields None for that field, not a fabricated value.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from v2.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsRecord,
    FinancialMetrics,
    InsiderTrade,
    Price,
)

UW = "https://api.unusualwhales.com/api"


class UWFundamentalsError(RuntimeError):
    pass


def _num(v: Any) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # NaN guard
    except (TypeError, ValueError):
        return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Safe division: None if either side is missing or the denominator is 0."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _by_period(rows: list[dict]) -> dict[str, dict]:
    """Index quarterly statement rows by fiscal_date_ending (newest wins ties)."""
    out: dict[str, dict] = {}
    for r in rows or []:
        rtype = str(r.get("report_type") or "quarterly").lower()
        if rtype and rtype != "quarterly":
            continue
        period = r.get("fiscal_date_ending")
        if period:
            out[str(period)] = r
    return out


def derive_metrics(
    ticker: str,
    income: list[dict],
    balance: list[dict],
    cashflow: list[dict],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Pure derivation: three UW statement lists -> FinancialMetrics, newest first.

    Kept side-effect-free so the math is unit-testable without a network call.
    """
    inc = _by_period(income)
    bal = _by_period(balance)
    cf = _by_period(cashflow)

    # A period is usable only with an income row (revenue, margins) that UW had
    # ingested by end_date. Balance/cash-flow rows fill in where present.
    periods = sorted(
        (
            p for p, row in inc.items()
            if str(row.get("inserted_at") or "")[:10] <= end_date
        ),
        reverse=True,
    )

    out: list[FinancialMetrics] = []
    for p in periods:
        i = inc[p]
        b = bal.get(p, {})
        c = cf.get(p, {})

        revenue = _num(i.get("total_revenue"))
        gross = _num(i.get("gross_profit"))
        op_income = _num(i.get("operating_income"))
        net = _num(i.get("net_income") or i.get("net_income_from_continuing_operations"))

        equity = _num(b.get("total_shareholder_equity"))
        cur_assets = _num(b.get("total_current_assets"))
        cur_liab = _num(b.get("total_current_liabilities"))
        shares = _num(b.get("common_stock_shares_outstanding"))
        debt = _num(b.get("short_long_term_debt_total"))
        if debt is None:
            lt = _num(b.get("long_term_debt")) or 0.0
            st = _num(b.get("short_term_debt")) or _num(b.get("current_debt")) or 0.0
            debt = lt + st if (b.get("long_term_debt") or b.get("short_term_debt")
                               or b.get("current_debt")) else None

        opcf = _num(c.get("operating_cashflow"))
        capex = _num(c.get("capital_expenditures"))
        fcf = None
        if opcf is not None and capex is not None:
            fcf = opcf - abs(capex)  # capex sign varies by feed; treat as magnitude

        # Year-over-year revenue growth: same quarter four periods back, if filed.
        rev_growth = None
        if p in periods:
            idx = periods.index(p)
            if idx + 4 < len(periods):
                prior = _num(inc[periods[idx + 4]].get("total_revenue"))
                rev_growth = _ratio(revenue, prior)
                if rev_growth is not None:
                    rev_growth = round(rev_growth - 1, 6)

        out.append(FinancialMetrics(
            ticker=ticker,
            report_period=p,
            period=period,
            currency=i.get("reported_currency") or "USD",
            filing_date=str(i.get("inserted_at") or "")[:10] or None,
            gross_margin=_ratio(gross, revenue),
            operating_margin=_ratio(op_income, revenue),
            net_margin=_ratio(net, revenue),
            return_on_equity=_ratio(net, equity),
            debt_to_equity=_ratio(debt, equity),
            current_ratio=_ratio(cur_assets, cur_liab),
            earnings_per_share=_ratio(net, shares),
            book_value_per_share=_ratio(equity, shares),
            free_cash_flow_per_share=_ratio(fcf, shares),
            revenue_growth=rev_growth,
            # market_cap and P/E need a historical price UW statements don't carry;
            # left None (the snapshot renders them as "-" and never averages P/E).
        ))
        if len(out) >= limit:
            break
    return out


class UWFundamentalsClient:
    """Fetch UW statements and derive FinancialMetrics. Secondary-only use.

    Only get_financial_metrics does real work; the remaining DataClient methods
    exist for protocol parity and are never reached in the fallback chain.
    """

    def __init__(self) -> None:
        self.uw = os.environ.get("UW_TOKEN")
        if not self.uw:
            raise UWFundamentalsError("UW_TOKEN missing")
        self.session = requests.Session()

    def _get(self, path: str, params: dict | None = None) -> Any:
        resp = self.session.get(
            f"{UW}{path}",
            headers={"Authorization": f"Bearer {self.uw}"},
            params=params or {},
            timeout=30,
        )
        if resp.status_code == 404:
            return None
        if not resp.ok:
            raise UWFundamentalsError(f"GET {path} {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return (data or {}).get("data", data)

    def _fetch(self, ticker: str) -> tuple[list, list, list]:
        """Return (income, balance, cash_flow) statement lists for ticker."""
        inc = self._get(f"/stock/{ticker}/income-statements", {"limit": 20}) or []
        bal = self._get(f"/stock/{ticker}/balance-sheets", {"limit": 20}) or []
        cf = self._get(f"/stock/{ticker}/cash-flows", {"limit": 20}) or []
        return inc, bal, cf

    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10,
    ) -> list[FinancialMetrics]:
        income, balance, cashflow = self._fetch(ticker)
        return derive_metrics(ticker, income, balance, cashflow, end_date, period, limit)

    # -- protocol parity (unused as secondary; safe defaults) -----------------
    def get_prices(self, ticker: str, start_date: str, end_date: str, **kw: Any) -> list[Price]:
        raise UWFundamentalsError("UWFundamentalsClient serves fundamentals only")

    def get_news(self, ticker: str, end_date: str, start_date: str | None = None,
                 limit: int = 1000) -> list[CompanyNews]:
        return []

    def get_insider_trades(self, ticker: str, end_date: str, start_date: str | None = None,
                           limit: int = 1000) -> list[InsiderTrade]:
        return []

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        return None

    def get_earnings(self, ticker: str) -> Earnings | None:
        return None

    def get_earnings_history(self, ticker: str, limit: int = 12) -> list[EarningsRecord]:
        return []

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        return None

    def __enter__(self) -> "UWFundamentalsClient":
        return self

    def __exit__(self, *a: Any) -> None:
        self.session.close()
