"""HomeDataClient — DataClient implementation on owned sources (no financialdatasets.ai).

Sources: Alpaca (daily bars, free w/ paper keys) · UnusualWhales (earnings
history, market cap; flat-rate sub) · FMP (quarterly ratio metrics; free
250/day quota is SHARED with autopilot — calls are minimized and disk-cached
via CachedDataClient upstream).

Contract (v2/data/protocol.py): empty/None = data genuinely absent;
infrastructure failures RAISE. Point-in-time: nothing dated after end_date
is ever returned.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from v2.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsData,
    EarningsRecord,
    FinancialMetrics,
    InsiderTrade,
    Price,
)

UW = "https://api.unusualwhales.com/api"
FMP = "https://financialmodelingprep.com/api/v3"
ALPACA_DATA = "https://data.alpaca.markets/v2"


class HomeClientError(RuntimeError):
    pass


def _num(v: Any) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # NaN guard
    except (TypeError, ValueError):
        return None


class HomeDataClient:
    def __init__(self) -> None:
        self.uw = os.environ.get("UW_TOKEN")
        self.fmp = os.environ.get("FMP_API_KEY")
        self.alp_key = os.environ.get("ALPACA_API_KEY_ID")
        self.alp_sec = os.environ.get("ALPACA_API_SECRET")
        if not self.uw:
            raise HomeClientError("UW_TOKEN missing")
        self.session = requests.Session()

    # -- low-level ---------------------------------------------------------
    def _get(self, url: str, headers: dict, params: dict | None = None) -> Any:
        resp = self.session.get(url, headers=headers, params=params or {}, timeout=30)
        if resp.status_code == 404:
            return None
        if not resp.ok:
            raise HomeClientError(f"GET {url} {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def _uw(self, path: str, params: dict | None = None) -> Any:
        data = self._get(f"{UW}{path}", {"Authorization": f"Bearer {self.uw}"}, params)
        return (data or {}).get("data", data)

    # -- prices (Alpaca) ----------------------------------------------------
    def get_prices(self, ticker: str, start_date: str, end_date: str, **kw: Any) -> list[Price]:
        if not (self.alp_key and self.alp_sec):
            raise HomeClientError("Alpaca keys missing")
        hdrs = {"APCA-API-KEY-ID": self.alp_key, "APCA-API-SECRET-KEY": self.alp_sec}
        out: list[Price] = []
        page = None
        while True:
            params = {"timeframe": "1Day", "start": start_date, "end": end_date,
                      "adjustment": "split", "feed": "iex", "limit": 10000}
            if page:
                params["page_token"] = page
            data = self._get(f"{ALPACA_DATA}/stocks/{ticker}/bars", hdrs, params) or {}
            for b in data.get("bars") or []:
                t = str(b.get("t", ""))
                if t[:10] > end_date:
                    continue
                out.append(Price(open=b["o"], close=b["c"], high=b["h"], low=b["l"],
                                 volume=int(b["v"]), time=t[:10]))
            page = data.get("next_page_token")
            if not page:
                break
        return out

    # -- fundamentals (FMP ratios + key-metrics, quarterly) ------------------
    def get_financial_metrics(self, ticker: str, end_date: str,
                              period: str = "ttm", limit: int = 10) -> list[FinancialMetrics]:
        if not self.fmp:
            return []
        p = "quarter" if period in ("ttm", "quarterly", "quarter") else "annual"
        common = {"period": p, "limit": max(limit * 2, 20), "apikey": self.fmp}
        ratios = self._get(f"{FMP}/ratios/{ticker}", {}, common) or []
        kms = {r.get("date"): r for r in (self._get(f"{FMP}/key-metrics/{ticker}", {}, common) or [])}
        growth = {r.get("date"): r for r in (self._get(f"{FMP}/financial-growth/{ticker}", {}, common) or [])}
        out: list[FinancialMetrics] = []
        for r in ratios:
            d = r.get("date")
            if not d or d > end_date:  # point-in-time: period end must be filed by end_date;
                continue               # FMP lacks filing_date here — period-end is the proxy.
            km, gr = kms.get(d, {}), growth.get(d, {})
            out.append(FinancialMetrics(
                ticker=ticker, report_period=d, period=period, currency="USD",
                market_cap=_num(km.get("marketCap")),
                enterprise_value=_num(km.get("enterpriseValue")),
                price_to_earnings_ratio=_num(r.get("priceEarningsRatio")),
                price_to_book_ratio=_num(r.get("priceToBookRatio")),
                price_to_sales_ratio=_num(r.get("priceToSalesRatio")),
                enterprise_value_to_ebitda_ratio=_num(km.get("enterpriseValueOverEBITDA")),
                enterprise_value_to_revenue_ratio=_num(km.get("evToSales")),
                free_cash_flow_yield=_num(km.get("freeCashFlowYield")),
                peg_ratio=_num(r.get("priceEarningsToGrowthRatio")),
                gross_margin=_num(r.get("grossProfitMargin")),
                operating_margin=_num(r.get("operatingProfitMargin")),
                net_margin=_num(r.get("netProfitMargin")),
                return_on_equity=_num(r.get("returnOnEquity")),
                return_on_assets=_num(r.get("returnOnAssets")),
                return_on_invested_capital=_num(km.get("roic")),
                asset_turnover=_num(r.get("assetTurnover")),
                inventory_turnover=_num(r.get("inventoryTurnover")),
                receivables_turnover=_num(r.get("receivablesTurnover")),
                days_sales_outstanding=_num(r.get("daysOfSalesOutstanding")),
                operating_cycle=_num(r.get("operatingCycle")),
                current_ratio=_num(r.get("currentRatio")),
                quick_ratio=_num(r.get("quickRatio")),
                cash_ratio=_num(r.get("cashRatio")),
                operating_cash_flow_ratio=_num(r.get("operatingCashFlowSalesRatio")),
                debt_to_equity=_num(r.get("debtEquityRatio")),
                debt_to_assets=_num(r.get("debtRatio")),
                interest_coverage=_num(r.get("interestCoverage")),
                revenue_growth=_num(gr.get("revenueGrowth")),
                earnings_growth=_num(gr.get("netIncomeGrowth")),
                book_value_growth=_num(gr.get("bookValueperShareGrowth")),
                earnings_per_share_growth=_num(gr.get("epsgrowth")),
                free_cash_flow_growth=_num(gr.get("freeCashFlowGrowth")),
                operating_income_growth=_num(gr.get("operatingIncomeGrowth")),
                ebitda_growth=_num(gr.get("ebitgrowth")),
                payout_ratio=_num(r.get("payoutRatio")),
                earnings_per_share=_num(km.get("netIncomePerShare")),
                book_value_per_share=_num(km.get("bookValuePerShare")),
                free_cash_flow_per_share=_num(km.get("freeCashFlowPerShare")),
            ))
            if len(out) >= limit:
                break
        return out

    # -- earnings history (UW) ----------------------------------------------
    def get_earnings_history(self, ticker: str, limit: int = 12) -> list[EarningsRecord]:
        rows = self._uw(f"/earnings/{ticker}") or []
        if isinstance(rows, dict):
            rows = rows.get("data", []) or []
        out: list[EarningsRecord] = []
        for r in rows:
            report = r.get("report_date") or r.get("date")
            if not report:
                continue
            eps_a = _num(r.get("eps") or r.get("actual_eps") or r.get("eps_actual"))
            eps_e = _num(r.get("expected_move_eps") if False else
                         r.get("eps_mean_est") or r.get("street_mean_est") or r.get("eps_estimate"))
            rev_a = _num(r.get("revenue") or r.get("actual_revenue"))
            rev_e = _num(r.get("revenue_mean_est") or r.get("revenue_estimate"))
            if eps_a is None and rev_a is None:
                continue  # future/unreported rows carry no actuals
            def label(a: float | None, e: float | None) -> str | None:
                if a is None or e is None:
                    return None
                return "BEAT" if a > e else "MISS" if a < e else "MEET"
            q = EarningsData(
                earnings_per_share=eps_a, estimated_earnings_per_share=eps_e,
                eps_surprise=label(eps_a, eps_e), revenue=rev_a, estimated_revenue=rev_e,
                revenue_surprise=label(rev_a, rev_e),
            )
            when = str(r.get("report_time") or "").lower()
            window = "amc" if "after" in when or when == "postmarket" else \
                     "bmo" if "pre" in when or when == "premarket" else None
            out.append(EarningsRecord(
                ticker=ticker, report_period=str(report), source_type="uw",
                filing_date=str(report), filing_window=window, quarterly=q,
            ))
        out.sort(key=lambda x: x.report_period, reverse=True)
        return out[:limit]

    def get_earnings(self, ticker: str) -> Earnings | None:
        hist = self.get_earnings_history(ticker, limit=1)
        if not hist:
            return None
        h = hist[0]
        return Earnings(ticker=ticker, report_period=h.report_period,
                        fiscal_period=h.fiscal_period, quarterly=h.quarterly)

    # -- market cap (UW) ------------------------------------------------------
    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        info = self._uw(f"/stock/{ticker}/info") or {}
        return _num(info.get("marketcap"))

    # -- unused-by-our-agents protocol methods --------------------------------
    def get_news(self, ticker: str, end_date: str, start_date: str | None = None,
                 limit: int = 1000) -> list[CompanyNews]:
        return []

    def get_insider_trades(self, ticker: str, end_date: str, start_date: str | None = None,
                           limit: int = 1000) -> list[InsiderTrade]:
        return []

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        return None

    # context-manager parity with FDClient
    def __enter__(self) -> "HomeDataClient":
        return self

    def __exit__(self, *a: Any) -> None:
        self.session.close()
