"""PEAD alpha model — Post-Earnings Announcement Drift.

Forms a view based on earnings surprises: bullish after a BEAT, bearish
after a MISS, on the theory that the market underreacts and the stock
keeps drifting in the surprise direction for days/weeks.

This is the quant counterpart to an LLM investor agent — same AlphaModel
interface, pure Python math. It only forms a *view* (conviction); the
backtest harness / portfolio construction decides timing and sizing.
"""

from __future__ import annotations

from datetime import datetime

from v2.data.protocol import DataClient
from v2.data.models import EarningsRecord
from v2.models import Signal
from v2.signals.base import QuantModel

_RETROSPECTIVE_CUTOFF_DAYS = 45   # drop filings whose data is stale vs report period
_SOURCE_PRIORITY = {"8-K": 0, "10-Q": 1, "10-K": 2, "20-F": 3}  # 8-K = earliest announcement


class PEADModel(QuantModel):
    """Long after an EPS BEAT, short after a MISS.

    `predict(ticker, date)` returns ±1.0 conviction if a qualifying earnings
    surprise was filed within `signal_window_days` of `date`, else 0.0 (no view).
    Conviction magnitude is fixed ±1 for v0 — scaling by surprise size is a
    future enhancement.
    """

    def __init__(
        self,
        *,
        earnings_limit: int = 8,
        signal_window_days: int = 4,
    ) -> None:
        self._earnings_limit = earnings_limit
        self._signal_window_days = signal_window_days
        # Cache earnings history per ticker — predict is called once per
        # trading day during a backtest, so we fetch each ticker only once.
        self._cache: dict[str, list[EarningsRecord]] = {}

    @property
    def name(self) -> str:
        return "pead"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        as_of = _parse_date(date)
        records = self._records(ticker, data_client)
        events = self._qualifying_events(records)

        # Four ways to hold no view, and they are not the same fact. The first
        # says nothing was read; the rest are readings. A committee handed a
        # bare 0.0 cannot tell an empty feed from a company that reported in
        # line, and would weigh the two the same.
        if not records:
            return self._neutral(
                ticker, date, "no-earnings-data",
                "no earnings history was returned for this ticker",
            )
        if not events:
            return self._neutral(
                ticker, date, "no-surprise-recorded",
                f"{len(records)} earnings record(s) read, none carrying a BEAT or MISS",
            )

        # Point-in-time: only consider filings on or before `date` (no lookahead)
        past = [e for e in events if _parse_date(e["filing_date"]) <= as_of]
        if not past:
            return self._neutral(
                ticker, date, "not-yet-reported",
                "every earnings filing on record was filed after this date",
            )

        # Most recent qualifying event as of `date`
        event = max(past, key=lambda e: e["filing_date"])
        filed = _parse_date(event["filing_date"])
        age = (as_of - filed).days

        # Only fire if the event is fresh (we just learned about it)
        if age > self._signal_window_days:
            return self._neutral(
                ticker, date, "event-stale",
                (
                    f"last {event['surprise']} filed {event['filing_date']}, "
                    f"{age} days ago, outside the {self._signal_window_days}-day window"
                ),
                # The event travels with the neutral. A reader that wants to
                # know what the last print was should not have to re-fetch.
                days_since_filing=age,
                eps_surprise=event["surprise"],
                source_type=event["source_type"],
                report_period=event["report_period"],
                filing_date=event["filing_date"],
            )

        surprise = event["surprise"]
        value = 1.0 if surprise == "BEAT" else -1.0
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=value,
            reasoning=(
                f"{surprise} on {event['report_period']} earnings "
                f"(filed {event['filing_date']}, {event['source_type']})"
            ),
            metadata={
                "eps_surprise": surprise,
                "source_type": event["source_type"],
                "report_period": event["report_period"],
                "filing_date": event["filing_date"],
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _neutral(self, ticker: str, date: str, reason: str, why: str, **extra) -> Signal:
        """A 0.0 that says which of the four silences this is."""
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=0.0,
            reasoning=why,
            metadata={"neutral_reason": reason, **extra},
        )

    def _records(self, ticker: str, data_client: DataClient) -> list[EarningsRecord]:
        """The raw history, fetched once per ticker.

        Kept apart from the cleaning below so an empty feed and a feed whose
        rows all fell out of the filters do not read alike.
        """
        if ticker not in self._cache:
            self._cache[ticker] = data_client.get_earnings_history(ticker, limit=self._earnings_limit)
        return self._cache[ticker]

    def _qualifying_events(self, records: list[EarningsRecord]) -> list[dict]:
        """Return BEAT/MISS events, deduped + retrospective-filtered.

        Mirrors the Week 3 PEAD cleaning: one event per (report_period),
        preferring the 8-K (the actual announcement) over later 10-Q/K
        filings, and dropping retrospective rows whose filing date is far
        after the report period (the extractor sometimes parses prior-quarter
        comparison data from a current 8-K).
        """
        best: dict[str, tuple[int, EarningsRecord]] = {}
        for r in records:
            if not r.filing_date or not r.quarterly:
                continue
            surprise = r.quarterly.eps_surprise
            if surprise not in ("BEAT", "MISS"):
                continue

            # 45-day retrospective filter
            lag = (_parse_date(r.filing_date) - _parse_date(r.report_period)).days
            if lag >= _RETROSPECTIVE_CUTOFF_DAYS:
                continue

            # Keep the highest-priority filing per report period (8-K wins)
            priority = _SOURCE_PRIORITY.get(r.source_type, 99)
            if r.report_period not in best or priority < best[r.report_period][0]:
                best[r.report_period] = (priority, r)

        return [
            {
                "filing_date": r.filing_date,
                "report_period": r.report_period,
                "source_type": r.source_type,
                "surprise": r.quarterly.eps_surprise,
            }
            for _, r in best.values()
        ]


def _parse_date(s: str):
    return datetime.strptime(s[:10], "%Y-%m-%d").date()
