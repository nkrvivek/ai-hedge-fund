"""Did the rows that came back cover the window that was asked for?

Two separate truncations sit behind this, and only one of them is signalled.
The page cap of 100 rows hands back a next_page_url, so a client that follows
the cursor recovers everything. The history cap of roughly one year moves
start_date forward and says nothing at all, so no amount of paging reaches it.

The two ends of a window are not the same kind of miss. A start that arrives
late means history the caller asked for is not on the plan. An end that stops
short is usually today's bar, which has not closed. Both are reported, and the
caller decides which one matters.
"""

from __future__ import annotations

import logging
from datetime import date

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Weekends and market holidays mean the served edges rarely land on the exact
# dates asked for. Christmas plus a weekend is the widest ordinary gap at four
# calendar days; seven leaves room without hiding a real truncation, which in
# the measured case ran to 962 days.
TOLERANCE_DAYS = 7


def _parse(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


class PriceCoverage(BaseModel):
    """What was asked for against what arrived."""

    requested_start: str
    requested_end: str
    served_start: str | None = None
    served_end: str | None = None
    rows: int = 0

    @property
    def is_empty(self) -> bool:
        return self.rows == 0

    @property
    def start_short_by_days(self) -> int:
        """Calendar days of history missing from the front of the window."""
        want, got = _parse(self.requested_start), _parse(self.served_start)
        if want is None or got is None:
            return 0
        return max(0, (got - want).days)

    @property
    def end_short_by_days(self) -> int:
        """Calendar days missing from the back. Small values are ordinary:
        the most recent bar has not closed yet."""
        want, got = _parse(self.requested_end), _parse(self.served_end)
        if want is None or got is None:
            return 0
        return max(0, (want - got).days)

    @property
    def is_short(self) -> bool:
        if self.is_empty:
            return True
        return (
            self.start_short_by_days > TOLERANCE_DAYS
            or self.end_short_by_days > TOLERANCE_DAYS
        )

    def describe(self) -> str:
        if self.is_empty:
            return (
                f"requested {self.requested_start}..{self.requested_end}, "
                f"served nothing"
            )
        parts = []
        if self.start_short_by_days > TOLERANCE_DAYS:
            parts.append(
                f"history starts at {self.served_start}, "
                f"{self.start_short_by_days} days after the requested "
                f"{self.requested_start}"
            )
        if self.end_short_by_days > TOLERANCE_DAYS:
            parts.append(
                f"history ends at {self.served_end}, "
                f"{self.end_short_by_days} days before the requested "
                f"{self.requested_end}"
            )
        if not parts:
            return (
                f"requested {self.requested_start}..{self.requested_end}, "
                f"served {self.served_start}..{self.served_end} ({self.rows} rows)"
            )
        return "; ".join(parts) + f" ({self.rows} rows)"


def price_coverage(prices, requested_start: str, requested_end: str) -> PriceCoverage:
    """Measure a price list against the window that was requested.

    *prices* is any sequence of objects carrying a ``time`` string. Rows are
    not assumed to be sorted.
    """
    days = sorted(p.time[:10] for p in prices if getattr(p, "time", None))
    return PriceCoverage(
        requested_start=requested_start,
        requested_end=requested_end,
        served_start=days[0] if days else None,
        served_end=days[-1] if days else None,
        rows=len(prices),
    )


def check_price_coverage(
    prices,
    label: str,
    requested_start: str,
    requested_end: str,
    *,
    require_full_range: bool = False,
    path: str | None = None,
) -> PriceCoverage:
    """Measure, then either raise or log. Returns the measurement either way.

    Callers that can work with a shorter window get the rows and a warning.
    Only a caller that says the whole window matters gets an exception.
    """
    from v2.data.errors import FDCoverageError

    coverage = price_coverage(prices, requested_start, requested_end)
    if coverage.is_short:
        message = f"{label} prices: {coverage.describe()}"
        if require_full_range:
            raise FDCoverageError(message, path=path, coverage=coverage)
        logger.warning("%s", message)
    return coverage
