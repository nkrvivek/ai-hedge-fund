"""Tests for FundamentalsFallbackClient.

No network. Stub primary/secondary clients exercise every branch of
get_financial_metrics plus the straight-delegation and context-manager
paths.
"""

from __future__ import annotations

import pytest

from v2.data.fallback_client import FundamentalsFallbackClient
from v2.data.models import FinancialMetrics


def _metric(ticker: str) -> FinancialMetrics:
    return FinancialMetrics(ticker=ticker, report_period="2026-06-30", period="ttm", currency="USD")


class _Stub:
    """Configurable stand-in for a DataClient's fundamentals + lifecycle."""

    def __init__(self, *, rows=None, raises=None):
        self._rows = rows if rows is not None else []
        self._raises = raises
        self.calls = 0
        self.entered = False
        self.exited = False

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return list(self._rows)

    # lifecycle
    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *a):
        self.exited = True

    # one delegated method, to prove pass-through
    def get_market_cap(self, ticker, end_date):
        return 123.0


def test_primary_hit_never_touches_secondary():
    primary = _Stub(rows=[_metric("AAPL")])
    secondary = _Stub(rows=[_metric("AAPL")])
    fb = FundamentalsFallbackClient(primary, secondary)

    rows = fb.get_financial_metrics("AAPL", "2026-07-29")

    assert len(rows) == 1
    assert secondary.calls == 0
    assert fb.fallback_tickers == []


def test_primary_raises_secondary_covers():
    """The LLY / FMP-402 case: primary blows up, secondary serves it."""
    primary = _Stub(raises=RuntimeError("GET /stable/ratios 402: not available"))
    secondary = _Stub(rows=[_metric("LLY")])
    fb = FundamentalsFallbackClient(primary, secondary)

    rows = fb.get_financial_metrics("LLY", "2026-07-29")

    assert len(rows) == 1
    assert secondary.calls == 1
    assert fb.fallback_tickers == ["LLY"]


def test_primary_empty_secondary_covers():
    primary = _Stub(rows=[])
    secondary = _Stub(rows=[_metric("LLY")])
    fb = FundamentalsFallbackClient(primary, secondary)

    rows = fb.get_financial_metrics("LLY", "2026-07-29")

    assert len(rows) == 1
    assert fb.fallback_tickers == ["LLY"]


def test_primary_raises_secondary_also_raises_reraises_primary():
    """Real outage must stay loud — primary's error wins, never a silent []."""
    primary_err = RuntimeError("402 paywall")
    primary = _Stub(raises=primary_err)
    secondary = _Stub(raises=ConnectionError("fd down"))
    fb = FundamentalsFallbackClient(primary, secondary)

    with pytest.raises(RuntimeError, match="402 paywall"):
        fb.get_financial_metrics("LLY", "2026-07-29")


def test_both_empty_returns_empty_no_raise():
    """Primary didn't raise and returned []; that is a valid 'no data' answer."""
    primary = _Stub(rows=[])
    secondary = _Stub(rows=[])
    fb = FundamentalsFallbackClient(primary, secondary)

    rows = fb.get_financial_metrics("XYZ", "2026-07-29")

    assert rows == []
    assert fb.fallback_tickers == []


def test_nested_chain_prefers_uw_over_fd_for_paywalled_ticker():
    """run_daily wiring: Home -> UW -> FD. When FMP (home) 402s a ticker, UW's
    deep history must serve it and FD must never be consulted — UW's 5 rows clear
    MIN_PERIODS where FD's shallow 3 would not."""
    home = _Stub(raises=RuntimeError("GET /stable/ratios 402: not available"))
    uw = _Stub(rows=[_metric("LLY") for _ in range(5)])
    fd = _Stub(rows=[_metric("LLY") for _ in range(3)])

    chain = FundamentalsFallbackClient(FundamentalsFallbackClient(home, uw), fd)
    rows = chain.get_financial_metrics("LLY", "2026-07-31")

    assert len(rows) == 5      # UW's deep history, not FD's shallow 3
    assert uw.calls == 1
    assert fd.calls == 0       # FD never reached


def test_nested_chain_falls_to_fd_only_when_uw_also_blind():
    """FD remains a real last resort: reached only if both FMP and UW are blind."""
    home = _Stub(raises=RuntimeError("402 paywall"))
    uw = _Stub(rows=[])        # UW has no coverage for this name
    fd = _Stub(rows=[_metric("ZZZ") for _ in range(4)])

    chain = FundamentalsFallbackClient(FundamentalsFallbackClient(home, uw), fd)
    rows = chain.get_financial_metrics("ZZZ", "2026-07-31")

    assert len(rows) == 4
    assert uw.calls == 1 and fd.calls == 1


def test_delegation_passthrough():
    primary = _Stub(rows=[])
    secondary = _Stub(rows=[])
    fb = FundamentalsFallbackClient(primary, secondary)

    assert fb.get_market_cap("AAPL", "2026-07-29") == 123.0


def test_context_manager_enters_and_exits_both():
    primary = _Stub(rows=[])
    secondary = _Stub(rows=[])
    fb = FundamentalsFallbackClient(primary, secondary)

    with fb as ctx:
        assert ctx is fb
        assert primary.entered and secondary.entered

    assert primary.exited and secondary.exited
