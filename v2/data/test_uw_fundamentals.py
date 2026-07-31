"""Tests for UWFundamentalsClient — deriving FinancialMetrics from UW statements.

The client exists to unstick FMP-paywalled tickers (LLY: FMP 402'd, FD had 3
periods < MIN_PERIODS=4, UW retains 5+ filed quarters). These tests pin the
derivation math and the point-in-time filter with canned statements — no
network.
"""

from __future__ import annotations

import pytest

from v2.data.uw_fundamentals import UWFundamentalsClient, derive_metrics
from v2.features.snapshot import MIN_PERIODS, build_snapshot


def _income(fiscal: str, inserted: str, revenue, gross, op, net) -> dict:
    return {
        "fiscal_date_ending": fiscal,
        "inserted_at": inserted,
        "report_type": "quarterly",
        "total_revenue": str(revenue),
        "gross_profit": str(gross),
        "operating_income": str(op),
        "net_income": str(net),
    }


def _balance(fiscal: str, inserted: str, equity, cur_assets, cur_liab,
             lt_debt, st_debt, shares) -> dict:
    return {
        "fiscal_date_ending": fiscal,
        "inserted_at": inserted,
        "report_type": "quarterly",
        "total_shareholder_equity": str(equity),
        "total_current_assets": str(cur_assets),
        "total_current_liabilities": str(cur_liab),
        "long_term_debt": str(lt_debt),
        "short_term_debt": str(st_debt),
        "common_stock_shares_outstanding": str(shares),
    }


def _cashflow(fiscal: str, inserted: str, opcf, capex) -> dict:
    return {
        "fiscal_date_ending": fiscal,
        "inserted_at": inserted,
        "report_type": "quarterly",
        "operating_cashflow": str(opcf),
        "capital_expenditures": str(capex),
    }


# Five filed quarters, newest first. Q1-2026 numbers chosen for clean ratios;
# 2025-03-31 revenue=800 makes the YoY growth for 2026-03-31 exactly 0.25.
_QUARTERS = [
    ("2026-03-31", "2026-05-02", 1000, 600, 300, 200, 800, 500, 250, 300, 100, 100, 250, 50),
    ("2025-12-31", "2026-02-06", 950, 560, 280, 180, 780, 480, 240, 300, 100, 100, 230, 45),
    ("2025-09-30", "2025-11-01", 900, 520, 260, 160, 760, 460, 230, 300, 100, 100, 210, 40),
    ("2025-06-30", "2025-08-02", 850, 500, 250, 150, 740, 440, 220, 300, 100, 100, 200, 40),
    ("2025-03-31", "2025-05-03", 800, 470, 230, 140, 720, 420, 210, 300, 100, 100, 190, 35),
]


def _statements():
    inc, bal, cf = [], [], []
    for (f, ins, rev, gp, op, net, eq, ca, cl, ltd, std, sh, ocf, cx) in _QUARTERS:
        inc.append(_income(f, ins, rev, gp, op, net))
        bal.append(_balance(f, ins, eq, ca, cl, ltd, std, sh))
        cf.append(_cashflow(f, ins, ocf, cx))
    return inc, bal, cf


def test_derive_matches_hand_computed_ratios():
    inc, bal, cf = _statements()
    rows = derive_metrics("LLY", inc, bal, cf, end_date="2026-07-31",
                          period="ttm", limit=20)
    assert [r.report_period for r in rows] == [
        "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31",
    ]
    top = rows[0]
    assert top.gross_margin == pytest.approx(0.6)
    assert top.operating_margin == pytest.approx(0.3)
    assert top.net_margin == pytest.approx(0.2)
    assert top.return_on_equity == pytest.approx(0.25)      # 200/800
    assert top.debt_to_equity == pytest.approx(0.5)         # (300+100)/800
    assert top.current_ratio == pytest.approx(2.0)          # 500/250
    assert top.earnings_per_share == pytest.approx(2.0)     # 200/100
    assert top.book_value_per_share == pytest.approx(8.0)   # 800/100
    assert top.free_cash_flow_per_share == pytest.approx(2.0)  # (250-50)/100
    assert top.revenue_growth == pytest.approx(0.25)        # 1000/800 - 1 YoY
    assert top.period == "ttm"
    assert top.ticker == "LLY"


def test_point_in_time_excludes_unfiled_periods():
    inc, bal, cf = _statements()
    # As-of before Q1-2026 was inserted (2026-05-02): newest visible is Q4-2025.
    rows = derive_metrics("LLY", inc, bal, cf, end_date="2026-03-15",
                          period="ttm", limit=20)
    assert rows[0].report_period == "2025-12-31"
    assert "2026-03-31" not in [r.report_period for r in rows]


def test_oldest_quarter_has_no_yoy_growth():
    inc, bal, cf = _statements()
    rows = derive_metrics("LLY", inc, bal, cf, end_date="2026-07-31",
                          period="ttm", limit=20)
    assert rows[-1].report_period == "2025-03-31"
    assert rows[-1].revenue_growth is None  # no quarter four back


def test_build_snapshot_clears_min_periods_on_uw_data(monkeypatch):
    """The whole point: UW's deep history builds a snapshot where FD's 3 rows
    would have raised InsufficientData."""
    inc, bal, cf = _statements()
    client = UWFundamentalsClient()
    monkeypatch.setattr(client, "_fetch", lambda t: (inc, bal, cf))
    snap = build_snapshot("LLY", "2026-07-31", client, periods=20)
    assert len(snap.periods) >= MIN_PERIODS
    assert snap.roe_avg is not None
    assert snap.debt_to_equity_latest == pytest.approx(0.5)


def test_missing_denominator_yields_none_not_crash():
    inc = [_income("2026-03-31", "2026-05-02", 0, 0, 0, 0)]          # revenue 0
    bal = [_balance("2026-03-31", "2026-05-02", 0, 500, 0, 0, 0, 0)]  # equity/shares 0
    cf = [_cashflow("2026-03-31", "2026-05-02", 100, 20)]
    rows = derive_metrics("LLY", inc, bal, cf, end_date="2026-07-31",
                          period="ttm", limit=20)
    r = rows[0]
    assert r.gross_margin is None      # div by zero revenue -> None
    assert r.return_on_equity is None  # div by zero equity -> None
    assert r.current_ratio is None     # div by zero current liabilities -> None
