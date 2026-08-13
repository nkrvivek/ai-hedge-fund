"""Tests for the shared watch list reader.

What is under test is the failure shape, not the happy path. This book is on
probation (DJ-20260810-05) and every name that enters the universe costs eight
LLM calls, so the two questions that matter are "can a bad read shrink the pool"
and "can a good read blow the cost up". Neither may happen quietly.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from bridge.shared_watchlist import STALE_AFTER_HOURS, read_shared_watchlist

NOW = datetime(2026, 8, 13, 21, 0, tzinfo=timezone.utc)
FLOOR = ["AMD", "MU", "PLTR"]


def _payload(tickers: list[str], age_hours: float = 2.0, source: str = "r2") -> str:
    as_of = (NOW - timedelta(hours=age_hours)).isoformat()
    return json.dumps({
        "tickers": tickers, "count": len(tickers),
        "as_of": as_of, "source": source, "stale": False, "detail": "ok",
    })


def _fetcher(body: str | None, raises: bool = False):
    def fetch(url: str) -> str:
        if raises:
            raise OSError("connection reset by peer")
        assert body is not None
        return body
    return fetch


# ── never shrink the pool ────────────────────────────────────────────────────


def test_an_unreachable_worker_falls_back_to_the_floor():
    # A pool handed zero names ranks zero movers and rotates nothing in. That
    # looks exactly like a quiet tape, which is why this cannot fail closed.
    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher(None, raises=True), now=NOW)

    # Assert
    assert read.tickers == tuple(FLOOR)
    assert read.source == "fallback"
    assert "connection reset" in read.detail


def test_a_body_that_is_not_json_falls_back_to_the_floor():
    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher("<html>401</html>"), now=NOW)

    # Assert
    assert read.tickers == tuple(FLOOR)
    assert read.source == "fallback"


def test_the_worker_reporting_its_own_fallback_is_treated_as_no_list():
    # The route answers 200 with source:"fallback" and no tickers when the
    # published object is missing. A 200 is not a read.
    # Arrange
    body = json.dumps({"tickers": [], "source": "fallback", "as_of": None,
                       "detail": "current.json missing"})

    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher(body), now=NOW)

    # Assert
    assert read.tickers == tuple(FLOOR)
    assert read.source == "fallback"
    assert "current.json missing" in read.detail


def test_the_floor_is_unioned_never_replaced():
    # The curated pool is the reason a fresh pick gets scored instead of
    # abstained-and-dropped. The shared list adds to it; it does not stand in
    # for it.
    # Arrange
    body = _payload(["NVDA", "AAOI"])

    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher(body), now=NOW)

    # Assert
    for t in FLOOR:
        assert t in read.tickers
    assert "NVDA" in read.tickers and "AAOI" in read.tickers
    assert read.source == "r2"


def test_the_floor_keeps_its_order_at_the_front():
    # rank_movers sorts by move magnitude so order does not decide who is
    # picked, but a stable pool makes two runs comparable in the log.
    # Arrange
    body = _payload(["NVDA"])

    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher(body), now=NOW)

    # Assert
    assert read.tickers[: len(FLOOR)] == tuple(FLOOR)


def test_names_are_upper_cased_and_deduplicated():
    # Arrange
    body = _payload(["amd", "NVDA", "nvda", "", "  "])

    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher(body), now=NOW)

    # Assert
    assert read.tickers.count("AMD") == 1
    assert read.tickers.count("NVDA") == 1
    assert all(t.strip() for t in read.tickers)


# ── never blow the cost up ───────────────────────────────────────────────────


def test_the_added_names_are_capped_and_the_drop_is_reported():
    # Only build_universe's k_fresh names ever reach the committee, so a long
    # pool is not itself a cost. But an unbounded pool is an unbounded snapshot
    # URL, and a silent truncation reads as full coverage. Cap it, and say so.
    # Arrange
    many = 400
    body = _payload([f"T{i:03d}" for i in range(many)])

    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher(body), now=NOW, max_added=250)

    # Assert
    assert len(read.tickers) == len(FLOOR) + 250
    assert "250 of 400" in read.detail


def test_a_pool_inside_the_cap_reports_no_truncation():
    # Arrange
    body = _payload(["NVDA", "AAOI"])

    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher(body), now=NOW, max_added=250)

    # Assert
    assert " of " not in read.detail


# ── freshness ────────────────────────────────────────────────────────────────


def test_a_stale_list_is_used_and_flagged():
    # Dropping it shrinks the pool, which is the failure this whole design
    # guards against. The names are still real.
    # Arrange
    body = _payload(["NVDA"], age_hours=STALE_AFTER_HOURS + 6)

    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher(body), now=NOW)

    # Assert
    assert read.stale is True
    assert "NVDA" in read.tickers
    assert read.source == "r2"


def test_a_list_replenished_this_morning_is_not_stale():
    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher(_payload(["NVDA"], 5)), now=NOW)

    # Assert
    assert read.stale is False


def test_an_unparseable_timestamp_counts_as_stale():
    # Defaulting an unreadable stamp to fresh is how a dead cron reads healthy
    # forever.
    # Arrange
    body = json.dumps({"tickers": ["NVDA"], "as_of": "whenever", "source": "r2"})

    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher(body), now=NOW)

    # Assert
    assert read.stale is True
    assert "NVDA" in read.tickers


def test_the_worker_flagging_stale_is_believed_even_on_a_fresh_stamp():
    # The publisher and this reader carry the same bound, but if they ever
    # disagree the stale side wins.
    # Arrange
    body = json.dumps({"tickers": ["NVDA"], "as_of": NOW.isoformat(),
                       "source": "r2", "stale": True})

    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher(body), now=NOW)

    # Assert
    assert read.stale is True


# ── configuration ────────────────────────────────────────────────────────────


def test_no_configured_endpoint_is_a_clean_fallback_not_a_crash():
    # The bridge must run in a checkout with no worker secret at all.
    # Act
    read = read_shared_watchlist(FLOOR, base_url="", token="", now=NOW)

    # Assert
    assert read.tickers == tuple(FLOOR)
    assert read.source == "fallback"
    assert "not configured" in read.detail


def test_the_token_travels_in_the_query_string_the_worker_expects():
    # Arrange
    seen: list[str] = []

    def fetch(url: str) -> str:
        seen.append(url)
        return _payload(["NVDA"])

    # Act
    read_shared_watchlist(FLOOR, fetch=fetch, base_url="https://w.example",
                          token="sekrit", now=NOW)

    # Assert
    assert seen == ["https://w.example/watchlist?token=sekrit"]


def test_the_default_fetcher_sends_a_named_user_agent():
    # Found in live verification 2026-08-13: urllib's default UA
    # ("Python-urllib/3.x") is answered 403 Forbidden at the Cloudflare edge
    # before the worker sees the request, while the same URL under curl
    # returns 200. The failure reads as a bad token and is not one.
    # Arrange
    import urllib.request

    from bridge import shared_watchlist as sw

    captured: list[urllib.request.Request] = []

    class _Resp:
        def read(self) -> bytes:
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *a) -> None:
            return None

    def fake_urlopen(req, timeout=None):
        captured.append(req)
        return _Resp()

    original = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        # Act
        sw._http_get("https://w.example/watchlist?token=x")
    finally:
        urllib.request.urlopen = original

    # Assert
    assert captured[0].get_header("User-agent") == sw.USER_AGENT
    assert "urllib" not in captured[0].get_header("User-agent").lower()


def test_the_result_is_immutable():
    # Act
    read = read_shared_watchlist(FLOOR, fetch=_fetcher(_payload(["NVDA"])), now=NOW)

    # Assert
    assert isinstance(read.tickers, tuple)
