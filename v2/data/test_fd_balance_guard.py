"""A key with no balance is not a configured key.

financialdatasets.ai is prepaid. When the balance runs out it answers 402 on
every route, and on 2026-08-20 that turned all 35 live smoke tests in
test_client.py red at once. Every one of them was reporting the same billing
state, and none of them was reporting a code fault.

So a 402 is a skip that names its reason. Every other status stays a failure,
because a suite that skips itself whenever the API is unhappy is a suite that
cannot tell a dead client from an unpaid one.
"""

import pytest

from v2.data.errors import FDClientError
from v2.data.fd_live import why_fd_cannot_answer


class _Probe:
    def __init__(self, raises: Exception | None = None) -> None:
        self._raises = raises
        self.calls = 0

    def get_prices(self, *_a, **_k):
        self.calls += 1
        if self._raises:
            raise self._raises
        return ["a bar"]


def test_an_exhausted_balance_is_a_skip_that_names_itself():
    probe = _Probe(FDClientError("GET /prices returned 402: no balance",
                                 status_code=402, path="/prices"))

    reason = why_fd_cannot_answer(probe)

    assert reason is not None
    assert "402" in reason
    assert "balance" in reason.lower()


def test_any_other_failure_stays_loud():
    # A 500 or a broken client must reach the tests and fail them. Skipping
    # here would hide a real outage behind an unpaid invoice.
    probe = _Probe(FDClientError("GET /prices returned 500", status_code=500))

    assert why_fd_cannot_answer(probe) is None


def test_a_working_key_does_not_skip():
    probe = _Probe()

    assert why_fd_cannot_answer(probe) is None
    assert probe.calls == 1
