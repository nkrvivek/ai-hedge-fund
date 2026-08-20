"""HomeDataClient — DataClient implementation on owned sources (no financialdatasets.ai).

Sources: Alpaca (daily bars, free w/ paper keys) · UnusualWhales (earnings
history, market cap; flat-rate sub) · FMP (ratio metrics; the key is SHARED
with autopilot and trade-refresh, so calls are minimized and disk-cached via
CachedDataClient upstream).

The FMP plan was upgraded on 2026-08-20 ($230/yr). The 250/day free quota that
this comment used to name is gone, and so is the per-symbol 402 that killed
whole committees: LLY and CBRS both answer 200 now. The new daily allowance is
NOT measured and FMP still sends no quota header, so nothing here should be
read as room to spend freely.

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

    # -- fundamentals (FMP; v3-quarterly for legacy keys, stable-annual for new) --
    def _fmp_rows(self, ticker: str, limit: int) -> tuple[list, dict, dict]:
        """Returns (ratio_rows, key_metrics_by_date, growth_by_date)."""
        common = {"period": "quarter", "limit": max(limit * 2, 20), "apikey": self.fmp}
        try:
            ratios = self._get(f"{FMP}/ratios/{ticker}", {}, common) or []
            kms = self._get(f"{FMP}/key-metrics/{ticker}", {}, common) or []
            gr = self._get(f"{FMP}/financial-growth/{ticker}", {}, common) or []
        except HomeClientError as e:
            if "Legacy Endpoint" not in str(e):
                raise
            # post-2025 FMP key: /stable API; quarter is premium on ratios/
            # key-metrics -> annual there, quarterly growth still allowed.
            stable = "https://financialmodelingprep.com/stable"
            # Re-measured 2026-08-20 on the paid plan. Two halves moved apart:
            # period=quarter on ratios and key-metrics still answers 402
            # "Premium Query Parameter", so annual stays. The limit=5 ceiling is
            # gone; annual limit=20 now returns 20 rows. Asking for 5 when the
            # caller asked for 10 was silently handing the committee half its
            # history, so the requested limit is passed through.
            n = max(limit, 5)
            ann = {"symbol": ticker, "period": "annual", "limit": n, "apikey": self.fmp}
            qtr = {"symbol": ticker, "period": "quarter", "limit": n, "apikey": self.fmp}
            ratios = self._get(f"{stable}/ratios", {}, ann) or []
            kms = self._get(f"{stable}/key-metrics", {}, ann) or []
            gr = self._get(f"{stable}/financial-growth", {}, qtr) or []
        return (ratios,
                {r.get("date"): r for r in kms},
                {r.get("date"): r for r in gr})

    def get_financial_metrics(self, ticker: str, end_date: str,
                              period: str = "ttm", limit: int = 10) -> list[FinancialMetrics]:
        if not self.fmp:
            return []
        ratios, kms, growth = self._fmp_rows(ticker, limit)

        def pk(*rows_keys: tuple) -> float | None:
            for row, keys in rows_keys:
                for k in keys:
                    v = _num(row.get(k))
                    if v is not None:
                        return v
            return None

        out: list[FinancialMetrics] = []
        gdates = sorted(growth)
        for r in ratios:
            d = r.get("date")
            if not d or d > end_date:  # point-in-time proxy: period end by end_date
                continue
            km = kms.get(d, {})
            gr = growth.get(d) or (growth.get(max((x for x in gdates if x <= d), default="")) or {})
            out.append(FinancialMetrics(
                ticker=ticker, report_period=d, period=period, currency="USD",
                market_cap=pk((km, ("marketCap",))),
                enterprise_value=pk((km, ("enterpriseValue",))),
                price_to_earnings_ratio=pk((r, ("priceEarningsRatio", "priceToEarningsRatio"))),
                price_to_book_ratio=pk((r, ("priceToBookRatio",))),
                price_to_sales_ratio=pk((r, ("priceToSalesRatio",))),
                enterprise_value_to_ebitda_ratio=pk((km, ("enterpriseValueOverEBITDA", "evToEBITDA"))),
                enterprise_value_to_revenue_ratio=pk((km, ("evToSales",))),
                free_cash_flow_yield=pk((km, ("freeCashFlowYield",))),
                peg_ratio=pk((r, ("priceEarningsToGrowthRatio", "priceToEarningsGrowthRatio"))),
                gross_margin=pk((r, ("grossProfitMargin",))),
                operating_margin=pk((r, ("operatingProfitMargin",))),
                net_margin=pk((r, ("netProfitMargin",))),
                return_on_equity=pk((r, ("returnOnEquity",)), (km, ("returnOnEquity",))),
                return_on_assets=pk((r, ("returnOnAssets",)), (km, ("returnOnAssets",))),
                return_on_invested_capital=pk((km, ("roic", "returnOnInvestedCapital"))),
                asset_turnover=pk((r, ("assetTurnover",))),
                inventory_turnover=pk((r, ("inventoryTurnover",))),
                receivables_turnover=pk((r, ("receivablesTurnover",))),
                days_sales_outstanding=pk((r, ("daysOfSalesOutstanding",)), (km, ("daysOfSalesOutstanding",))),
                operating_cycle=pk((r, ("operatingCycle",)), (km, ("operatingCycle",))),
                current_ratio=pk((r, ("currentRatio",))),
                quick_ratio=pk((r, ("quickRatio",))),
                cash_ratio=pk((r, ("cashRatio",))),
                operating_cash_flow_ratio=pk((r, ("operatingCashFlowSalesRatio",))),
                debt_to_equity=pk((r, ("debtEquityRatio", "debtToEquityRatio"))),
                debt_to_assets=pk((r, ("debtRatio", "debtToAssetsRatio"))),
                interest_coverage=pk((r, ("interestCoverage", "interestCoverageRatio"))),
                revenue_growth=pk((gr, ("revenueGrowth",))),
                earnings_growth=pk((gr, ("netIncomeGrowth",))),
                book_value_growth=pk((gr, ("bookValueperShareGrowth",))),
                earnings_per_share_growth=pk((gr, ("epsgrowth", "epsGrowth"))),
                free_cash_flow_growth=pk((gr, ("freeCashFlowGrowth",))),
                operating_income_growth=pk((gr, ("operatingIncomeGrowth",))),
                ebitda_growth=pk((gr, ("ebitgrowth", "ebitdaGrowth"))),
                payout_ratio=pk((r, ("payoutRatio", "dividendPayoutRatio"))),
                earnings_per_share=pk((km, ("netIncomePerShare",)), (r, ("netIncomePerShare",))),
                book_value_per_share=pk((km, ("bookValuePerShare",)), (r, ("bookValuePerShare",))),
                free_cash_flow_per_share=pk((km, ("freeCashFlowPerShare",)), (r, ("freeCashFlowPerShare",))),
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
