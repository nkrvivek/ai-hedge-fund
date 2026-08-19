"""Tests for alpha models (AlphaModel/QuantModel + PEADModel)."""

from __future__ import annotations

from v2.data.models import EarningsData, EarningsRecord
from v2.signals import PEADModel, QuantModel
from v2.signals.base import AlphaModel
from v2.models import Signal


class MockFDClient:
    """Returns canned earnings history for testing without API calls."""

    def __init__(self, earnings=None):
        self._earnings = earnings or []

    def get_earnings_history(self, ticker, limit=12):
        return self._earnings


def _rec(report_period, filing_date, surprise, source_type="8-K"):
    return EarningsRecord(
        ticker="TEST", report_period=report_period, source_type=source_type,
        filing_date=filing_date,
        quarterly=EarningsData(eps_surprise=surprise) if surprise else None,
    )


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class TestInterface:
    def test_quant_model_is_alpha_model(self):
        assert issubclass(QuantModel, AlphaModel)
        assert issubclass(PEADModel, QuantModel)

    def test_name(self):
        assert PEADModel().name == "pead"

    def test_helpers(self):
        assert QuantModel._safe_float(None) == 0.0
        assert QuantModel._safe_float("3.5") == 3.5
        assert QuantModel._normalize_to_signal(2.0) == 1.0
        assert QuantModel._normalize_to_signal(-2.0) == -1.0


# ---------------------------------------------------------------------------
# PEADModel.predict
# ---------------------------------------------------------------------------

class TestPEADPredict:
    def test_beat_fires_long(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 1.0
        assert sig.model_name == "pead"
        assert "BEAT" in sig.reasoning

    def test_miss_fires_short(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "MISS")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == -1.0

    def test_meet_is_neutral(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "MEET")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 0.0

    def test_no_earnings_is_neutral(self):
        fd = MockFDClient([])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 0.0

    def test_stale_event_is_neutral(self):
        # Event filed 30 days before the query date — outside the freshness window
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-31", fd)
        assert sig.value == 0.0

    def test_point_in_time_ignores_future_filings(self):
        # A filing dated after the query date must not be visible (no lookahead)
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-07-15", fd)
        assert sig.value == 0.0

    def test_freshness_window_bridges_weekend(self):
        # Filed Saturday 2025-08-02; queried Monday 2025-08-04 (2 days) → still fresh
        fd = MockFDClient([_rec("2025-06-30", "2025-08-02", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-04", fd)
        assert sig.value == 1.0

    def test_45_day_retrospective_filter(self):
        # Filing is 100+ days after the report period → retrospective, excluded
        fd = MockFDClient([_rec("2025-12-31", "2026-04-13", "BEAT")])
        sig = PEADModel().predict("TEST", "2026-04-13", fd)
        assert sig.value == 0.0

    def test_dedup_prefers_8k(self):
        # Same report period via 8-K and 10-Q; 8-K should be the chosen source
        fd = MockFDClient([
            _rec("2025-06-30", "2025-08-01", "BEAT", source_type="8-K"),
            _rec("2025-06-30", "2025-08-02", "BEAT", source_type="10-Q"),
        ])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 1.0
        assert sig.metadata["source_type"] == "8-K"

    def test_returns_signal_type(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert isinstance(sig, Signal)


# ---------------------------------------------------------------------------
# PEADModel neutrals — a 0.0 that says why
#
# Every branch above returns the same bare 0.0, so "the vendor gave us nothing"
# and "the company reported three months ago and nothing has happened since"
# reach the committee looking identical. One is a data gap and the other is a
# reading. Only the second is evidence about the stock.
# ---------------------------------------------------------------------------

class TestPEADNeutralReasons:
    def test_no_earnings_data_says_nothing_was_read(self):
        sig = PEADModel().predict("TEST", "2025-08-01", MockFDClient([]))
        assert sig.value == 0.0
        assert sig.metadata["neutral_reason"] == "no-earnings-data"
        assert "no earnings" in sig.reasoning.lower()

    def test_records_without_a_surprise_are_not_a_missing_feed(self):
        # The company reported and the print was in line. That is a reading, and
        # it is not the same as never having heard from the vendor.
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "MEET")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 0.0
        assert sig.metadata["neutral_reason"] == "no-surprise-recorded"

    def test_a_retrospective_only_history_reads_as_no_surprise_not_no_data(self):
        # The rows arrived; the 45-day filter dropped them. Something was read.
        fd = MockFDClient([_rec("2025-12-31", "2026-04-13", "BEAT")])
        sig = PEADModel().predict("TEST", "2026-04-13", fd)
        assert sig.metadata["neutral_reason"] == "no-surprise-recorded"

    def test_a_filing_still_in_the_future_says_not_yet_reported(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-07-15", fd)
        assert sig.metadata["neutral_reason"] == "not-yet-reported"

    def test_a_stale_event_says_how_stale_it_is(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-31", fd)
        assert sig.metadata["neutral_reason"] == "event-stale"
        assert sig.metadata["days_since_filing"] == 30
        # The event itself travels with the neutral. A reader that wants to know
        # what the last print was should not have to re-fetch the history.
        assert sig.metadata["filing_date"] == "2025-08-01"
        assert sig.metadata["eps_surprise"] == "BEAT"

    def test_a_firing_signal_carries_no_neutral_reason(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert "neutral_reason" not in sig.metadata
