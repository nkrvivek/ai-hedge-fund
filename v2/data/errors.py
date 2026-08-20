"""Data-layer exceptions.

They live here rather than in client.py so that coverage.py can raise one
without importing the client, which imports coverage.
"""

from __future__ import annotations


class FDClientError(Exception):
    """An API request failed for infrastructure reasons (auth, rate limit,
    server error, network). Distinct from "no data exists" — that returns
    empty. A backtest must crash on this, not treat it as no-data.
    """

    def __init__(self, message: str, *, status_code: int | None = None, path: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.path = path


class FDCoverageError(FDClientError):
    """The response was well formed but covered less than was asked for.

    Measured on 2026-08-20: a request for 2023-01-01 through 2026-08-20 was
    answered with 2025-08-20 onward and no field anywhere in the payload
    marked the missing two and a half years. Nothing distinguishes that from
    a company that listed in August 2025, so a caller reading the rows learns
    something false about the company instead of something true about the
    plan.

    Raised only when a caller says it needs the whole window. Callers that
    can work with what exists get the rows and a logged warning.
    """

    def __init__(self, message: str, *, path: str | None = None, coverage=None) -> None:
        super().__init__(message, path=path)
        self.coverage = coverage
