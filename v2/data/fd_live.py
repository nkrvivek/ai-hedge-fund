"""Whether the live financialdatasets.ai suites can reach the API at all.

financialdatasets.ai is prepaid. When the balance runs out it answers 402 on
every route, and the `skipif` guards on the live suites still pass because the
key is set. On 2026-08-20 the balance reached 0.00 and 38 tests across three
files went red at once, every one of them reporting the same billing state and
none of them reporting a code fault.

A 402 is a skip that names its reason. Every other status returns None and
reaches the tests, because a suite that skips itself whenever the API is
unhappy cannot tell a dead client from an unpaid one.
"""

from __future__ import annotations

from v2.data.errors import FDClientError

PROBE_TICKER = "AAPL"
PROBE_DAY = "2024-01-02"


def why_fd_cannot_answer(client) -> str | None:
    """The reason the live FD suite must skip, or None to let it run."""
    try:
        client.get_prices(PROBE_TICKER, PROBE_DAY, PROBE_DAY)
    except FDClientError as exc:
        if exc.status_code == 402:
            return (
                "financialdatasets.ai answered 402: the prepaid balance is "
                "spent, so these live tests cannot reach the API"
            )
    except Exception:  # noqa: BLE001 - let the tests themselves report it
        return None
    return None
