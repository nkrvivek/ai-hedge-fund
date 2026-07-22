"""CachedDataClient tests — counting fake, no network."""

import v2.data.cached as cached_module
from v2.data.cached import CachedDataClient
from v2.data.models import CompanyFacts, Earnings, EarningsRecord, Price


class CountingClient:
    """Counts calls; returns canned data."""

    def __init__(self, facts=None):
        self.calls = 0
        self._facts = facts

    def get_prices(self, ticker, start_date, end_date, interval="day", interval_multiplier=1):
        self.calls += 1
        return [Price(open=1.0, close=2.0, high=2.0, low=1.0, volume=100,
                      time=f"{start_date}T00:00:00Z")]

    def get_company_facts(self, ticker):
        self.calls += 1
        return self._facts

    def get_market_cap(self, ticker, end_date):
        self.calls += 1
        return 3.0e12

    def get_earnings_history(self, ticker, limit=12):
        self.calls += 1
        return [EarningsRecord(ticker=ticker, report_period="2025-06-30", source_type="8-K",
                               filing_date="2025-08-01")]

    def get_earnings(self, ticker):
        self.calls += 1
        return Earnings(ticker=ticker, report_period="2025-06-30")


class _FrozenDate:
    """Stand-in for datetime.date — freezes date.today().isoformat() in cached.py."""

    def __init__(self, iso: str):
        self._iso = iso

    def today(self):
        return self

    def isoformat(self):
        return self._iso


def test_cache_hit_skips_wrapped_client(tmp_path):
    inner = CountingClient()
    fd = CachedDataClient(inner, cache_dir=tmp_path)

    first = fd.get_prices("AAPL", "2024-01-01", "2024-12-31")
    second = fd.get_prices("AAPL", "2024-01-01", "2024-12-31")

    assert inner.calls == 1
    assert isinstance(second[0], Price)  # re-hydrated to the pydantic model
    assert second[0].close == first[0].close


def test_different_params_different_entries(tmp_path):
    inner = CountingClient()
    fd = CachedDataClient(inner, cache_dir=tmp_path)

    fd.get_prices("AAPL", "2024-01-01", "2024-12-31")
    fd.get_prices("AAPL", "2024-01-01", "2025-12-31")  # different end date

    assert inner.calls == 2


def test_refresh_busts_cache(tmp_path):
    inner = CountingClient()
    CachedDataClient(inner, cache_dir=tmp_path).get_prices("AAPL", "2024-01-01", "2024-12-31")
    CachedDataClient(inner, cache_dir=tmp_path, refresh=True).get_prices(
        "AAPL", "2024-01-01", "2024-12-31")

    assert inner.calls == 2


def test_none_item_is_cached(tmp_path):
    """A cached None (ticker without facts) must not re-hit the API."""
    inner = CountingClient(facts=None)
    fd = CachedDataClient(inner, cache_dir=tmp_path)

    assert fd.get_company_facts("ZZZZ") is None
    assert fd.get_company_facts("ZZZZ") is None
    assert inner.calls == 1


def test_item_rehydrates(tmp_path):
    inner = CountingClient(facts=CompanyFacts(ticker="AAPL", sector="Tech"))
    fd = CachedDataClient(inner, cache_dir=tmp_path)

    fd.get_company_facts("AAPL")
    facts = fd.get_company_facts("AAPL")

    assert inner.calls == 1
    assert isinstance(facts, CompanyFacts)
    assert facts.sector == "Tech"


def test_scalar_cached(tmp_path):
    inner = CountingClient()
    fd = CachedDataClient(inner, cache_dir=tmp_path)

    assert fd.get_market_cap("AAPL", "2024-12-31") == 3.0e12
    assert fd.get_market_cap("AAPL", "2024-12-31") == 3.0e12
    assert inner.calls == 1


def test_earnings_history_cache_hit_same_day(tmp_path, monkeypatch):
    """Same-day cache hit still works — the fix must not defeat caching."""
    monkeypatch.setattr(cached_module, "date", _FrozenDate("2026-07-22"))
    inner = CountingClient()
    fd = CachedDataClient(inner, cache_dir=tmp_path)

    fd.get_earnings_history("LLY")
    fd.get_earnings_history("LLY")

    assert inner.calls == 1


def test_earnings_history_refetches_on_new_day(tmp_path, monkeypatch):
    """2026-07-22 audit: the key had no date component, so a ticker's
    first-ever fetch was cached forever (GH Actions restore-key chain
    inherits every prior day's cache file) — a new earnings filing would
    never be seen again. A new calendar day must force a refetch."""
    inner = CountingClient()

    monkeypatch.setattr(cached_module, "date", _FrozenDate("2026-07-22"))
    CachedDataClient(inner, cache_dir=tmp_path).get_earnings_history("LLY")

    monkeypatch.setattr(cached_module, "date", _FrozenDate("2026-07-23"))
    CachedDataClient(inner, cache_dir=tmp_path).get_earnings_history("LLY")

    assert inner.calls == 2


def test_get_earnings_refetches_on_new_day(tmp_path, monkeypatch):
    """Same frozen-cache gap and fix for the sibling get_earnings method."""
    inner = CountingClient()

    monkeypatch.setattr(cached_module, "date", _FrozenDate("2026-07-22"))
    CachedDataClient(inner, cache_dir=tmp_path).get_earnings("LLY")

    monkeypatch.setattr(cached_module, "date", _FrozenDate("2026-07-23"))
    CachedDataClient(inner, cache_dir=tmp_path).get_earnings("LLY")

    assert inner.calls == 2
