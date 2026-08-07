"""Did the committee's conviction predict the move? — ai-hedge-fund book only.

The book wrote a conviction for all 25 names every morning for four weeks and
never checked one of them. This scores each call against what the name actually
did, and appends one row per run to `learning_log.jsonl`. Nothing is applied
automatically: the loop reports to a human, and the registry row in
trade-refresh's `learning-loops.json` says so (`applied_ref: null`).

The measurement is narrow on purpose. A conviction carries a sign and a size;
only the sign is falsifiable without a position-sizing model, so a call is
scored as right when the name moved the way the conviction pointed over the
next `MATURE_DAYS` sessions. A flat close is a miss — the call said the name
would move.

Prices come in as a plain {symbol: {date: close}} mapping. The autopilot book's
scorer instead took a per-ticker quote callback under a 250-call cap, walked
its history in file order, spent the whole budget on rows that could never be
scored, and reported `n_scored` 0 for eight weeks (found and fixed 2026-08-07).
One batched read has no budget to misspend.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

# One trading week forward. Short enough that a daily committee's call is still
# the thing being measured, long enough that a single session's noise is not.
MATURE_DAYS = 5

# Below this the committee has not made a call. QCOM and NOW sat at exactly 0.0
# for weeks; scoring those would pad the count with picks nobody made.
CONVICTION_FLOOR = 0.10

# How far past a requested date to look for the next print. Covers a weekend
# plus a holiday; beyond that the price is missing, not late.
MAX_LOOKAHEAD_DAYS = 5

LEARNING_LOG = Path(__file__).parent / "learning_log.jsonl"


def _parse(day: str) -> date:
    return date.fromisoformat(day)


def daily_rows(rows: list[dict]) -> list[dict]:
    """One decision per day, in date order.

    The bridge wrote five rows on 2026-07-17 alone (retries, reruns). A day has
    one final view, so the last row for a date wins. Dry runs decided nothing
    and undated rows cannot be scored against anything."""
    by_date: dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, dict) or row.get("dry_run"):
            continue
        asof = row.get("asof")
        if not asof:
            continue
        try:
            _parse(asof)
        except (TypeError, ValueError):
            continue
        by_date[asof] = row
    return [by_date[k] for k in sorted(by_date)]


def close_on_or_after(series: dict, day: str,
                      max_lookahead: int = MAX_LOOKAHEAD_DAYS) -> float | None:
    """The close on `day`, or the next print within `max_lookahead` days.

    Requested dates land on weekends and holidays. Rolling forward a bounded
    number of days finds the next session; rolling forward without a bound
    would silently compare a July decision against an August price."""
    if not series:
        return None
    target = _parse(day)
    for step in range(max_lookahead + 1):
        value = series.get((target + timedelta(days=step)).isoformat())
        if value is not None:
            return float(value)
    return None


def symbols_in(rows: list[dict], floor: float = CONVICTION_FLOOR) -> list[str]:
    """Every name that was actually called, sorted. This is the fetch list."""
    names = {sym for row in rows or []
             for sym, conviction in (row.get("convictions") or {}).items()
             if abs(float(conviction or 0.0)) >= floor}
    return sorted(names)


def window_for(rows: list[dict], mature_days: int = MATURE_DAYS,
               today: str | None = None) -> tuple[str, str] | None:
    """The date range the price fetch must cover, or None when there is
    nothing to score. The end reaches past the last exit by the lookahead, so
    a decision made before a long weekend still finds its exit print, and is
    then clamped to today: the newest rows have not matured anyway, and asking
    Alpaca for bars ending in the future is refused outright, which costs the
    whole history rather than the few days that do not exist yet."""
    days = [row.get("asof") for row in rows or [] if row.get("asof")]
    if not days:
        return None
    start, last = min(days), max(days)
    end = _parse(last) + timedelta(days=mature_days + MAX_LOOKAHEAD_DAYS)
    if today:
        end = min(end, _parse(today))
    return start, end.isoformat()


def score_convictions(rows: list[dict], closes: dict, today: str,
                      mature_days: int = MATURE_DAYS,
                      floor: float = CONVICTION_FLOOR) -> dict:
    """Score every matured call against what the name did.

    `closes` is {symbol: {date: close}}. A pick whose entry or exit price is
    missing counts as unpriced rather than dropped — a silent drop makes a data
    outage look like a book with nothing to say."""
    t_today = _parse(today)
    picks: list[dict] = []
    by_symbol: dict[str, dict] = {}
    n_matured = n_unpriced = 0

    for row in daily_rows(rows):
        asof = row["asof"]
        if (t_today - _parse(asof)).days < mature_days:
            continue
        n_matured += 1
        exit_day = (_parse(asof) + timedelta(days=mature_days)).isoformat()
        for symbol, raw in sorted((row.get("convictions") or {}).items()):
            conviction = float(raw or 0.0)
            if abs(conviction) < floor:
                continue
            series = closes.get(symbol) or {}
            entry = close_on_or_after(series, asof)
            exit_px = close_on_or_after(series, exit_day)
            if not entry or exit_px is None or entry <= 0:
                n_unpriced += 1
                continue
            ret = exit_px / entry - 1.0
            hit = ret > 0 if conviction > 0 else ret < 0
            picks.append({"date": asof, "symbol": symbol,
                          "conviction": conviction, "entry": entry,
                          "exit": exit_px, "ret": ret, "hit": hit})
            tally = by_symbol.setdefault(symbol, {"n": 0, "hits": 0})
            tally["n"] += 1
            tally["hits"] += 1 if hit else 0

    hits = sum(1 for p in picks if p["hit"])
    return {
        "n_days": len(daily_rows(rows)),
        "n_matured": n_matured,
        "n_scored": len(picks),
        "n_unpriced": n_unpriced,
        # None, never 0.0: nothing measured is a different claim from every
        # call being wrong, and a reader that cannot tell them apart chases
        # the wrong failure.
        "hit_rate": round(hits / len(picks), 3) if picks else None,
        "by_symbol": by_symbol,
        "picks": picks,
    }


def log_row(analysis: dict, today: str, top: int = 3) -> dict:
    """The one row this run appends. `n_scored` is what the trade-refresh
    learning rail reads (`scored_field` in `learning-loops.json`)."""
    picks = sorted(analysis.get("picks") or [],
                   key=lambda p: p["ret"] * (1 if p["conviction"] > 0 else -1),
                   reverse=True)
    trim = [{"symbol": p["symbol"], "date": p["date"],
             "conviction": round(p["conviction"], 3), "ret": round(p["ret"], 4)}
            for p in picks]
    return {
        "date": today,
        "n_days": analysis.get("n_days", 0),
        "n_matured": analysis.get("n_matured", 0),
        "n_scored": analysis.get("n_scored", 0),
        "n_unpriced": analysis.get("n_unpriced", 0),
        "hit_rate": analysis.get("hit_rate"),
        "mature_days": MATURE_DAYS,
        "floor": CONVICTION_FLOOR,
        "best": trim[:top],
        "worst": trim[-top:][::-1],
        "by_symbol": analysis.get("by_symbol", {}),
    }


def append_log(row: dict, path: Path | str = LEARNING_LOG) -> bool:
    """Append one row, guarded. Same rule as the trade ledger: bookkeeping may
    never break the book."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        return True
    except Exception as e:  # noqa: BLE001 — a log write must never break a run
        print(f"learning log write FAILED (non-fatal): {e}")
        return False


def run_learning(client, today: str, ledger_path: Path | str,
                 log_path: Path | str = LEARNING_LOG) -> dict:
    """Score the book's own history and append a row. Returns the row.

    Guarded end to end — a learning pass that raises would take the daily
    bridge down with it, which is a worse outcome than an unscored week."""
    try:
        rows = [json.loads(line) for line in
                Path(ledger_path).read_text().splitlines() if line.strip()]
    except Exception as e:  # noqa: BLE001
        print(f"learning: ledger unreadable (non-fatal): {e}")
        rows = []

    days = daily_rows(rows)
    window = window_for(days, today=today)
    closes: dict = {}
    if window and days:
        try:
            closes = client.daily_closes(symbols_in(days), *window)
        except Exception as e:  # noqa: BLE001
            print(f"learning: price fetch failed (non-fatal): {e}")

    row = log_row(score_convictions(days, closes, today), today)
    append_log(row, log_path)
    return row
