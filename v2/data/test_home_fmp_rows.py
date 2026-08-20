"""HomeDataClient._fmp_rows against the plan we actually pay for.

Written 2026-08-20, the day the FMP plan was upgraded. Two constraints had been
folded into one comment ("free stable tier: limit capped at 5") and only one of
them survived the upgrade, so each is pinned separately here.
"""
from __future__ import annotations

import pytest

from v2.data.home_client import HomeDataClient, HomeClientError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("UW_TOKEN", "t")
    monkeypatch.setenv("FMP_API_KEY", "k")
    return HomeDataClient()


def _legacy_then_capture(calls):
    """First call raises the v3 Legacy Endpoint error, the rest record params."""
    def _get(self, url, headers, params=None):
        if "/api/v3/" in url:
            raise HomeClientError("GET %s 403: Legacy Endpoint" % url)
        calls.append((url, dict(params or {})))
        return []
    return _get


def test_ratios_and_key_metrics_stay_annual_because_quarter_is_still_premium(client, monkeypatch):
    # Measured 2026-08-20 on the paid plan: period=quarter on /stable/ratios and
    # /stable/key-metrics answers 402 "Premium Query Parameter". Growth does not.
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(HomeDataClient, "_get", _legacy_then_capture(calls))
    client._fmp_rows("AAPL", limit=10)

    by_path = {url.rsplit("/", 1)[-1]: p for url, p in calls if "/stable/" in url}
    assert by_path["ratios"]["period"] == "annual"
    assert by_path["key-metrics"]["period"] == "annual"
    assert by_path["financial-growth"]["period"] == "quarter"


def test_the_requested_limit_is_passed_through_now_that_the_cap_is_gone(client, monkeypatch):
    # The free tier capped every stable call at 5 rows, so a caller asking for 10
    # got 5 and was never told. Annual limit=20 returns 20 rows on the paid plan.
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(HomeDataClient, "_get", _legacy_then_capture(calls))
    client._fmp_rows("AAPL", limit=10)

    assert [p["limit"] for url, p in calls if "/stable/" in url] == [10, 10, 10]


def test_a_small_request_still_asks_for_the_five_row_floor(client, monkeypatch):
    # Fewer rows than five is never worth a round trip; the floor is cheap.
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(HomeDataClient, "_get", _legacy_then_capture(calls))
    client._fmp_rows("AAPL", limit=2)

    assert {p["limit"] for url, p in calls if "/stable/" in url} == {5}


def test_a_non_legacy_error_is_raised_and_never_retried_on_stable(client, monkeypatch):
    # The stable fallback exists for one specific v3 error. Any other failure is
    # an infrastructure fault and must propagate, per v2/data/protocol.py.
    def _boom(self, url, headers, params=None):
        raise HomeClientError("GET %s 500: upstream is down" % url)

    monkeypatch.setattr(HomeDataClient, "_get", _boom)
    with pytest.raises(HomeClientError):
        client._fmp_rows("AAPL", limit=10)
