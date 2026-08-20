"""What the engine does when the feed holds less history than the window wants.

The estimation window is 250 trading days deep. The FinancialDatasets plan
serves about one year, roughly 250 trading days in total, so the deepest
window that fits leaves no room for an event to sit in front of it. Every
event fails the est_start >= 0 check, every ticker lands in skipped_tickers,
and the result reads as "no earnings effect at these companies" when what
happened is "the feed is a year long".

These tests run on a synthetic feed. No API key, no network.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from v2.data.models import EarningsRecord, Price
from v2.event_study import compute_car


def _trading_days(n: int, end: date) -> list[date]:
    """n weekdays ending on or before *end*, oldest first."""
    days: list[date] = []
    d = end
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


class FakeClient:
    """Serves a fixed number of trading days and earnings on chosen days.

    Prices drift deterministically so the market model has something to fit.
    """

    def __init__(self, n_days: int, event_offsets: list[int], *, tickers=("AAPL",)):
        self.days = _trading_days(n_days, date(2026, 8, 19))
        self.event_offsets = event_offsets
        self._tickers = tickers
        self.price_calls: list[tuple] = []

    def get_prices(self, ticker, start_date, end_date, *a, **kw):
        self.price_calls.append((ticker, start_date, end_date))
        seed = sum(ord(c) for c in ticker)
        out = []
        for i, d in enumerate(self.days):
            if not (start_date <= d.isoformat() <= end_date):
                continue
            close = 100.0 + (i * 0.05) + ((i * seed) % 7) * 0.1
            out.append(Price(
                open=close, close=close, high=close, low=close,
                volume=1_000, time=f"{d.isoformat()}T00:00:00Z",
            ))
        return out

    def get_earnings_history(self, ticker, limit=12):
        if ticker not in self._tickers:
            return []
        return [
            EarningsRecord(
                ticker=ticker,
                report_period=self.days[off].isoformat(),
                source_type="10-Q",
                filing_date=self.days[off].isoformat(),
            )
            for off in self.event_offsets
        ]


# ---------------------------------------------------------------------------
# The failure being fixed
# ---------------------------------------------------------------------------

def test_a_one_year_feed_still_produces_events(FakeFeed=FakeClient):
    """250 trading days, events in the back half. Before narrowing this
    returned zero events and called the ticker skipped."""
    client = FakeFeed(250, [140, 200, 240])

    result = compute_car(["AAPL"], client, rng_seed=42, n_bootstrap=100)

    assert len(result.events) > 0
    assert "AAPL" not in result.skipped_tickers


def test_the_window_actually_used_is_recorded_on_the_result():
    """Narrowing changes the statistics, so it has to be visible."""
    client = FakeClient(250, [140, 200, 240])

    result = compute_car(["AAPL"], client, rng_seed=42, n_bootstrap=100)

    window = result.estimation_window
    assert window is not None
    assert window.narrowed is True
    assert window.start > -250          # shallower than configured
    assert window.n_days == window.end - window.start + 1
    assert "250" in window.reason or "narrow" in window.reason.lower()


def test_plentiful_history_leaves_the_configured_window_alone():
    client = FakeClient(600, [300, 420, 540])

    result = compute_car(["AAPL"], client, rng_seed=42, n_bootstrap=100)

    assert result.estimation_window.start == -250
    assert result.estimation_window.narrowed is False


def test_narrowing_can_be_refused():
    """A caller that would rather have nothing than a shallower model."""
    client = FakeClient(250, [140, 200, 240])

    result = compute_car(
        ["AAPL"], client, rng_seed=42, n_bootstrap=100, allow_narrow_window=False,
    )

    assert result.events == []
    assert result.skip_reasons["AAPL"] == "short-history"


def test_a_feed_too_short_for_any_usable_window_refuses_rather_than_fitting_noise():
    """Below the stated floor a market model is not worth fitting."""
    client = FakeClient(100, [80, 90])

    result = compute_car(["AAPL"], client, rng_seed=42, n_bootstrap=100)

    assert result.events == []
    assert result.skip_reasons["AAPL"] == "short-history"


# ---------------------------------------------------------------------------
# Skip reasons: the two meanings that printed the same
# ---------------------------------------------------------------------------

def test_no_earnings_history_and_no_prices_are_different_findings():
    class NoEarnings(FakeClient):
        def get_earnings_history(self, ticker, limit=12):
            return []

    class NoPrices(FakeClient):
        def get_prices(self, ticker, start_date, end_date, *a, **kw):
            if ticker == "SPY":
                return super().get_prices(ticker, start_date, end_date)
            return []

    no_earnings = compute_car(["AAPL"], NoEarnings(600, [300]), n_bootstrap=100)
    no_prices = compute_car(["AAPL"], NoPrices(600, [300]), n_bootstrap=100)

    assert no_earnings.skip_reasons["AAPL"] == "no-earnings-history"
    assert no_prices.skip_reasons["AAPL"] == "no-prices"
    assert no_earnings.skipped_tickers == ["AAPL"]
    assert no_prices.skipped_tickers == ["AAPL"]


def test_prices_that_arrive_but_fit_no_event_are_not_called_missing_data():
    """Rows came back, hundreds of them, and still no event could be fitted:
    one filing sits too close to the front of the series, the other is dated
    past the last bar. Neither is missing data, and calling it that would
    file a calendar problem as a company with nothing to report."""

    class OddFilings(FakeClient):
        def get_earnings_history(self, ticker, limit=12):
            records = super().get_earnings_history(ticker, limit)
            return records + [
                EarningsRecord(
                    ticker=ticker,
                    report_period="2026-12-01",
                    source_type="10-Q",
                    filing_date="2026-12-01",   # after the last served bar
                )
            ]

    client = OddFilings(600, [100])

    result = compute_car(["AAPL"], client, rng_seed=42, n_bootstrap=100)

    assert result.events == []
    assert result.skip_reasons["AAPL"] == "no-usable-events"


def test_skipped_tickers_still_lists_every_skip():
    """The old field keeps its meaning so existing readers do not break."""
    client = FakeClient(600, [300], tickers=("AAPL",))

    result = compute_car(["AAPL", "MSFT"], client, rng_seed=42, n_bootstrap=100)

    assert result.skipped_tickers == ["MSFT"]
    assert set(result.skip_reasons) == {"MSFT"}


def test_a_dead_market_proxy_names_every_ticker_with_a_reason():
    class NoSpy(FakeClient):
        def get_prices(self, ticker, start_date, end_date, *a, **kw):
            if ticker == "SPY":
                return []
            return super().get_prices(ticker, start_date, end_date)

    result = compute_car(["AAPL", "MSFT"], NoSpy(600, [300]), n_bootstrap=100)

    assert result.skipped_tickers == ["AAPL", "MSFT"]
    assert result.skip_reasons == {"AAPL": "no-market-prices", "MSFT": "no-market-prices"}
