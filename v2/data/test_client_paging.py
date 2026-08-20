"""FDClient paging and coverage contract. Mocked HTTP, no API key required.

Measured against the live API on 2026-08-20: /prices/ returns at most 100 rows
per page and hands back a next_page_url. The client discarded that URL, so a
request for 2023-01-01..2026-08-20 answered 100 rows ending 2026-01-12 and
nothing in the payload said the rest existed.

The plan also clamps history to about one year. Asking for 2023-01-01 served
2025-08-20 no matter how many pages were followed. That is a different fault
from the page cap and needs its own signal, because paging cannot fix it.
"""

import pytest

from v2.data import FDClient, FDCoverageError


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def client():
    c = FDClient(api_key="test-key")
    yield c
    c.close()


def _stub(client, responses):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    client._session.request = fake_request
    return calls


def _bar(day: str) -> dict:
    return {
        "time": f"{day}T00:00:00Z", "open": 1.0, "high": 1.0,
        "low": 1.0, "close": 1.0, "volume": 1,
    }


def _page(days, next_url=None):
    return _FakeResponse(200, {
        "ticker": "SPY", "prices": [_bar(d) for d in days],
        "next_page_url": next_url,
    })


# ---------------------------------------------------------------------------
# Paging
# ---------------------------------------------------------------------------

def test_every_page_is_followed_until_the_cursor_runs_out(client):
    calls = _stub(client, [
        _page(["2026-01-02", "2026-01-05"], "https://api.financialdatasets.ai/prices/?cursor=A"),
        _page(["2026-01-06", "2026-01-07"], "https://api.financialdatasets.ai/prices/?cursor=B"),
        _page(["2026-01-08"], None),
    ])

    prices = client.get_prices("SPY", "2026-01-01", "2026-01-08")

    assert [p.time[:10] for p in prices] == [
        "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
    ]
    assert len(calls) == 3
    assert calls[1]["url"].endswith("cursor=A")
    assert calls[2]["url"].endswith("cursor=B")


def test_a_cursor_is_followed_verbatim_and_not_re_signed_with_params(client):
    """The cursor already encodes the query. Re-sending params would reset it
    to page one and loop forever."""
    calls = _stub(client, [
        _page(["2026-01-02"], "https://api.financialdatasets.ai/prices/?cursor=A"),
        _page(["2026-01-05"], None),
    ])

    client.get_prices("SPY", "2026-01-01", "2026-01-08")

    assert calls[0].get("params") is not None
    assert calls[1].get("params") is None


def test_paging_stops_at_the_page_cap_rather_than_spinning(client):
    """A server that always hands back a cursor must not hang the caller."""
    responses = [
        _page([f"2026-01-{d:02d}"], f"https://api.financialdatasets.ai/prices/?cursor=X{d}")
        for d in range(1, 40)
    ]
    _stub(client, responses)

    prices = client.get_prices("SPY", "2026-01-01", "2026-12-31", max_pages=3)

    assert len(prices) == 3


def test_a_page_that_repeats_its_cursor_ends_the_walk(client):
    """Same URL twice means the server is not advancing. Stop, do not loop."""
    same = "https://api.financialdatasets.ai/prices/?cursor=SAME"
    _stub(client, [_page(["2026-01-02"], same), _page(["2026-01-05"], same)])

    prices = client.get_prices("SPY", "2026-01-01", "2026-01-08")

    assert [p.time[:10] for p in prices] == ["2026-01-02", "2026-01-05"]


def test_no_cursor_means_one_request(client):
    calls = _stub(client, [_page(["2026-01-02"], None)])
    client.get_prices("SPY", "2026-01-01", "2026-01-08")
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_a_short_read_still_returns_its_rows_by_default(client):
    """Callers that ask for a window wider than the feed holds are common and
    are not doing anything wrong. They get the rows and a logged warning."""
    _stub(client, [_page(["2026-06-01", "2026-06-02"], None)])

    prices = client.get_prices("SPY", "2023-01-01", "2026-08-20")

    assert len(prices) == 2


def test_require_full_range_raises_when_the_start_was_clamped(client):
    """The live fault: history capped at a year, start silently moved forward."""
    _stub(client, [_page(["2025-08-20", "2026-08-19"], None)])

    with pytest.raises(FDCoverageError) as exc:
        client.get_prices("SPY", "2023-01-01", "2026-08-20", require_full_range=True)

    msg = str(exc.value)
    assert "2023-01-01" in msg and "2025-08-20" in msg


def test_require_full_range_raises_when_the_end_falls_short(client):
    _stub(client, [_page(["2026-01-02", "2026-03-01"], None)])

    with pytest.raises(FDCoverageError) as exc:
        client.get_prices("SPY", "2026-01-01", "2026-08-20", require_full_range=True)

    assert "2026-03-01" in str(exc.value)


def test_require_full_range_accepts_a_range_that_covers_the_request(client):
    _stub(client, [_page(["2026-01-02", "2026-08-19"], None)])

    prices = client.get_prices(
        "SPY", "2026-01-02", "2026-08-19", require_full_range=True,
    )

    assert len(prices) == 2


def test_an_empty_answer_under_require_full_range_raises_rather_than_reading_as_no_data(client):
    """Zero rows for a requested window is the loudest possible short read."""
    _stub(client, [_page([], None)])

    with pytest.raises(FDCoverageError):
        client.get_prices("SPY", "2026-01-01", "2026-08-20", require_full_range=True)
