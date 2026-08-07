"""Tests for the ai-hedge-fund learning loop.

The book ran for four weeks, wrote a conviction for every name every morning,
and never once checked whether those convictions were right. This scores them.

Prices arrive as a plain {symbol: {date: close}} mapping rather than a fetch
callback. That is deliberate: the autopilot book's scorer took a per-ticker
quote function under a 250-call cap, walked its history in file order, spent
the whole budget on rows that could never score, and reported n_scored 0 for
eight weeks (fixed 2026-08-07). One batched read has no budget to misspend.
"""
from __future__ import annotations

import json

from bridge.learning import (
    CONVICTION_FLOOR,
    close_on_or_after,
    append_log,
    daily_rows,
    log_row,
    score_convictions,
    symbols_in,
    window_for,
)

TODAY = "2026-08-07"


def _row(asof, convictions, **kw):
    row = {"asof": asof, "equity": 100_000.0, "dry_run": False,
           "convictions": convictions, "targets": {}, "orders": []}
    row.update(kw)
    return row


# --- daily_rows: one view per day, newest last ------------------------------


def test_the_last_row_of_a_day_is_that_days_view():
    """The bridge wrote five rows on 2026-07-17. A day has one final view."""
    rows = [_row("2026-07-17", {"MU": 0.1}), _row("2026-07-17", {"MU": 0.9})]

    assert [r["convictions"] for r in daily_rows(rows)] == [{"MU": 0.9}]


def test_rows_come_back_in_date_order_whatever_order_they_arrived():
    rows = [_row("2026-08-05", {}), _row("2026-07-10", {})]

    assert [r["asof"] for r in daily_rows(rows)] == ["2026-07-10", "2026-08-05"]


def test_a_dry_run_row_is_not_a_decision():
    rows = [_row("2026-07-10", {"MU": 0.9}, dry_run=True)]

    assert daily_rows(rows) == []


def test_a_row_with_no_date_is_dropped_not_dated_now():
    assert daily_rows([{"convictions": {"MU": 0.9}}]) == []


# --- close_on_or_after: sessions, not calendar days -------------------------


SERIES = {"2026-08-03": 10.0, "2026-08-04": 11.0, "2026-08-07": 12.0}


def test_an_exact_session_is_used_as_is():
    assert close_on_or_after(SERIES, "2026-08-04") == 11.0


def test_a_weekend_date_rolls_forward_to_the_next_session():
    """2026-08-05 and 06 are missing from the series; 08-07 is the next print."""
    assert close_on_or_after(SERIES, "2026-08-05") == 12.0


def test_a_gap_longer_than_the_lookahead_is_unpriced_not_guessed():
    assert close_on_or_after(SERIES, "2026-07-20") is None


def test_a_date_past_the_end_of_the_series_is_unpriced():
    assert close_on_or_after(SERIES, "2026-08-10") is None


# --- score_convictions ------------------------------------------------------


def _closes(**by_symbol):
    return {sym: dict(series) for sym, series in by_symbol.items()}


def test_a_long_call_that_rose_is_a_hit():
    rows = [_row("2026-07-27", {"MU": 0.8})]
    closes = _closes(MU={"2026-07-27": 100.0, "2026-08-01": 110.0})

    a = score_convictions(rows, closes, TODAY, mature_days=5)

    assert a["n_scored"] == 1
    assert a["hit_rate"] == 1.0


def test_a_short_call_that_fell_is_a_hit():
    rows = [_row("2026-07-27", {"TSLA": -0.8})]
    closes = _closes(TSLA={"2026-07-27": 100.0, "2026-08-01": 90.0})

    assert score_convictions(rows, closes, TODAY, mature_days=5)["hit_rate"] == 1.0


def test_a_long_call_that_fell_is_a_miss():
    rows = [_row("2026-07-27", {"MU": 0.8})]
    closes = _closes(MU={"2026-07-27": 100.0, "2026-08-01": 90.0})

    a = score_convictions(rows, closes, TODAY, mature_days=5)

    assert a["n_scored"] == 1
    assert a["hit_rate"] == 0.0


def test_a_flat_close_is_a_miss_for_either_direction():
    """The call said the name would move. It did not. That is not a hit."""
    rows = [_row("2026-07-27", {"MU": 0.8, "TSLA": -0.8})]
    closes = _closes(MU={"2026-07-27": 100.0, "2026-08-01": 100.0},
                     TSLA={"2026-07-27": 50.0, "2026-08-01": 50.0})

    a = score_convictions(rows, closes, TODAY, mature_days=5)

    assert a["n_scored"] == 2
    assert a["hit_rate"] == 0.0


def test_a_conviction_under_the_floor_is_not_a_call():
    """QCOM and NOW sat at 0.0 for weeks. Scoring them would inflate the count
    with picks nobody made."""
    rows = [_row("2026-07-27", {"QCOM": 0.0, "NVDA": 0.01})]
    closes = _closes(QCOM={"2026-07-27": 1.0, "2026-08-01": 2.0},
                     NVDA={"2026-07-27": 1.0, "2026-08-01": 2.0})

    a = score_convictions(rows, closes, TODAY, mature_days=5,
                          floor=CONVICTION_FLOOR)

    assert a["n_scored"] == 0
    assert a["hit_rate"] is None


def test_a_row_too_recent_to_have_resolved_is_not_scored():
    rows = [_row("2026-08-06", {"MU": 0.8})]
    closes = _closes(MU={"2026-08-06": 100.0})

    a = score_convictions(rows, closes, TODAY, mature_days=5)

    assert a["n_matured"] == 0
    assert a["n_scored"] == 0


def test_a_pick_with_no_price_is_counted_as_unpriced_not_dropped():
    """A silent drop makes a data outage look like a book with nothing to say."""
    rows = [_row("2026-07-27", {"MU": 0.8, "ARM": 0.8})]
    closes = _closes(MU={"2026-07-27": 100.0, "2026-08-01": 110.0})

    a = score_convictions(rows, closes, TODAY, mature_days=5)

    assert a["n_scored"] == 1
    assert a["n_unpriced"] == 1


def test_hit_rate_is_none_when_nothing_scored_never_zero():
    """0.0 means every call was wrong. None means no call was measured. A rail
    that cannot tell those apart reports the wrong failure."""
    a = score_convictions([], {}, TODAY)

    assert a["n_scored"] == 0 and a["hit_rate"] is None


def test_per_symbol_records_calls_and_hits():
    rows = [_row("2026-07-20", {"MU": 0.8}), _row("2026-07-27", {"MU": 0.8})]
    closes = _closes(MU={"2026-07-20": 100.0, "2026-07-25": 110.0,
                         "2026-07-27": 110.0, "2026-08-01": 100.0})

    by = score_convictions(rows, closes, TODAY, mature_days=5)["by_symbol"]

    assert by["MU"] == {"n": 2, "hits": 1}


def test_the_forward_return_is_carried_on_each_pick():
    rows = [_row("2026-07-27", {"MU": 0.8})]
    closes = _closes(MU={"2026-07-27": 100.0, "2026-08-01": 110.0})

    pick = score_convictions(rows, closes, TODAY, mature_days=5)["picks"][0]

    assert pick["symbol"] == "MU"
    assert round(pick["ret"], 4) == 0.1
    assert pick["hit"] is True
    assert pick["conviction"] == 0.8


# --- symbols_in / window_for: what to fetch ---------------------------------


def test_symbols_are_collected_across_every_row():
    rows = [_row("2026-07-27", {"MU": 0.8}), _row("2026-07-28", {"ARM": -0.9})]

    assert symbols_in(rows, floor=CONVICTION_FLOOR) == ["ARM", "MU"]


def test_symbols_under_the_floor_are_never_fetched():
    rows = [_row("2026-07-27", {"MU": 0.8, "QCOM": 0.0})]

    assert symbols_in(rows, floor=CONVICTION_FLOOR) == ["MU"]


def test_the_fetch_window_spans_the_first_row_to_the_last_exit():
    rows = [_row("2026-07-27", {"MU": 0.8}), _row("2026-07-28", {"MU": 0.8})]

    start, end = window_for(rows, mature_days=5)

    assert start == "2026-07-27"
    assert end >= "2026-08-02"  # last row + maturity + the weekend lookahead


def test_the_window_is_none_when_there_is_nothing_to_score():
    assert window_for([], mature_days=5) is None


# --- log_row / append_log ---------------------------------------------------


def test_the_log_row_carries_the_scored_count_the_rail_reads():
    a = score_convictions([_row("2026-07-27", {"MU": 0.8})],
                          _closes(MU={"2026-07-27": 100.0, "2026-08-01": 110.0}),
                          TODAY, mature_days=5)

    row = log_row(a, TODAY)

    assert row["date"] == TODAY
    assert row["n_scored"] == 1
    assert row["hit_rate"] == 1.0


def test_the_log_row_names_the_best_and_worst_names():
    rows = [_row("2026-07-27", {"MU": 0.8, "TSLA": -0.8})]
    closes = _closes(MU={"2026-07-27": 100.0, "2026-08-01": 130.0},
                     TSLA={"2026-07-27": 100.0, "2026-08-01": 130.0})

    row = log_row(score_convictions(rows, closes, TODAY, mature_days=5), TODAY)

    assert row["best"][0]["symbol"] == "MU"
    assert row["worst"][0]["symbol"] == "TSLA"


def test_a_run_that_scored_nothing_still_writes_a_row(tmp_path):
    """A missing row and a row saying zero are different claims. The first
    reads as 'nobody ran it', which is the failure this loop exists to end."""
    path = tmp_path / "learning_log.jsonl"

    assert append_log(log_row(score_convictions([], {}, TODAY), TODAY), path) is True

    written = json.loads(path.read_text().strip())
    assert written["n_scored"] == 0 and written["hit_rate"] is None


def test_the_log_is_append_only(tmp_path):
    path = tmp_path / "learning_log.jsonl"
    append_log(log_row(score_convictions([], {}, TODAY), TODAY), path)
    append_log(log_row(score_convictions([], {}, "2026-08-08"), "2026-08-08"), path)

    assert len(path.read_text().strip().splitlines()) == 2


def test_a_log_write_failure_never_raises_into_the_run(tmp_path):
    """Same rule as the trade ledger: bookkeeping may not break the book."""
    unwritable = tmp_path / "nope" / "x" / "learning_log.jsonl"
    (tmp_path / "nope").write_text("i am a file, not a directory")

    assert append_log({"date": TODAY}, unwritable) is False


# --- the live ledger --------------------------------------------------------


def test_the_live_ledger_still_yields_scorable_days():
    """The measurement this loop was built on: 26 rows across 19 dates — the
    bridge reran itself five times on 2026-07-17 and twice on 07-10 and 07-31 —
    every one of them carrying convictions nobody ever scored."""
    from pathlib import Path

    ledger = Path(__file__).parent / "ledger.jsonl"
    rows = [json.loads(line) for line in
            ledger.read_text().splitlines() if line.strip()]

    days = daily_rows(rows)

    assert len(days) >= 19, "the bridge ledger lost days"
    assert symbols_in(days, floor=CONVICTION_FLOOR), "no day carries a call"


# --- run_learning: the whole pass, guarded ----------------------------------


class _Client:
    """Stands in for AlpacaPaper. Records what the loop asked for."""

    def __init__(self, closes=None, boom=False):
        self.closes, self.boom, self.asked = closes or {}, boom, None

    def daily_closes(self, symbols, start, end):
        if self.boom:
            raise RuntimeError("data API down")
        self.asked = (symbols, start, end)
        return self.closes


def _ledger(tmp_path, rows):
    path = tmp_path / "ledger.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_the_pass_scores_the_ledger_and_writes_one_row(tmp_path):
    from bridge.learning import run_learning

    ledger = _ledger(tmp_path, [_row("2026-07-27", {"MU": 0.8})])
    log = tmp_path / "learning_log.jsonl"
    client = _Client({"MU": {"2026-07-27": 100.0, "2026-08-01": 110.0}})

    row = run_learning(client, TODAY, ledger, log)

    assert row["n_scored"] == 1 and row["hit_rate"] == 1.0
    assert json.loads(log.read_text().strip())["n_scored"] == 1


def test_only_the_called_names_and_the_needed_window_are_fetched(tmp_path):
    from bridge.learning import run_learning

    ledger = _ledger(tmp_path, [_row("2026-07-27", {"MU": 0.8, "QCOM": 0.0})])
    client = _Client()

    run_learning(client, TODAY, ledger, tmp_path / "log.jsonl")

    symbols, start, end = client.asked
    assert symbols == ["MU"]
    assert start == "2026-07-27" and end >= "2026-08-01"


def test_a_price_outage_still_writes_a_row_saying_nothing_scored(tmp_path):
    """The failure this loop exists to end is silence. An outage that skips the
    row is indistinguishable from a run nobody made."""
    from bridge.learning import run_learning

    ledger = _ledger(tmp_path, [_row("2026-07-27", {"MU": 0.8})])
    log = tmp_path / "learning_log.jsonl"

    row = run_learning(_Client(boom=True), TODAY, ledger, log)

    assert row["n_scored"] == 0 and row["n_unpriced"] == 1
    assert log.read_text().strip(), "no row written"


def test_an_unreadable_ledger_never_raises_into_the_daily_run(tmp_path):
    from bridge.learning import run_learning

    row = run_learning(_Client(), TODAY, tmp_path / "missing.jsonl",
                       tmp_path / "log.jsonl")

    assert row["n_days"] == 0


def test_the_window_never_asks_for_bars_that_do_not_exist_yet():
    """Alpaca refuses a window ending in the future, and the refusal costs the
    whole history rather than the few days that have not happened."""
    rows = [_row("2026-08-06", {"MU": 0.8})]

    _, end = window_for(rows, mature_days=5, today=TODAY)

    assert end == TODAY
