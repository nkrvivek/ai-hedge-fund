"""On-disk byte counter for the shared FMP key.

Five processes hold one key and none of them can see the others. A per-process
counter that only lives in memory answers `unmeasured` every time the rail asks,
which is the failure `uw-consumers.json` records having shipped with.

So each process writes its own running total to a file nobody else writes to:

    ~/.fmp-spend/<YYYY-MM-DD>/<consumer>.<pid>.json

One writer per file means no locking and no lost updates, and the rail sums the
day's directory without asking anyone to pipe anything. The format is the shared
thing here, not this code: `trade-refresh`, `autopilot-experiment` and `traderkit`
each write the same shape from their own repo, because a counter that has to be imported across
a project boundary is a counter that will not be installed. `trade-refresh`
owns the rail that reads this directory (`src.fmp_quota`).

Bytes, not calls. FMP's Starter plan caps a trailing 30 days of bandwidth at
20GB and does not cap calls per day at all.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

SCHEMA = 1
DEFAULT_ROOT = Path.home() / ".fmp-spend"

# One entry per (consumer, day) in this process. Cumulative, so the file it
# writes is a whole total and never a delta that could be applied twice.
_TOTALS: dict[tuple[str, str], dict] = {}
_WARNED = False


def spend_root(root: Path | None = None) -> Path:
    if root is not None:
        return root
    override = os.environ.get("FMP_SPEND_DIR")
    return Path(override) if override else DEFAULT_ROOT


def _warn_once(message: str) -> None:
    """A counter must never break the call it is counting, and must never go
    quiet about failing either."""
    global _WARNED
    if not _WARNED:
        print(f"[fmp-spend] {message}", file=sys.stderr)
        _WARNED = True


def record(
    consumer: str,
    n_bytes: int | None,
    *,
    limit_hit: bool = False,
    day: date | None = None,
    root: Path | None = None,
) -> None:
    """Add one response to this process's running total.

    `n_bytes` is None when the response carried no `content-length`. That is
    counted as a call with unknown size rather than as zero bytes, and the rail
    reports the unknown count so the total reads as the floor it is.
    """
    day = day or datetime.now(timezone.utc).date()
    key = (consumer, day.isoformat())
    total = _TOTALS.get(key) or {
        "schema": SCHEMA,
        "consumer": consumer,
        "day": day.isoformat(),
        "pid": os.getpid(),
        "calls": 0,
        "bytes": 0,
        "unsized_calls": 0,
        "limit_hit": False,
    }
    total = {
        **total,
        "calls": total["calls"] + 1,
        "bytes": total["bytes"] + (n_bytes if isinstance(n_bytes, int) and n_bytes > 0 else 0),
        "unsized_calls": total["unsized_calls"] + (0 if isinstance(n_bytes, int) and n_bytes >= 0 else 1),
        "limit_hit": total["limit_hit"] or bool(limit_hit),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    _TOTALS[key] = total

    directory = spend_root(root) / day.isoformat()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{consumer}.{os.getpid()}.json"
        with open(path, "w") as handle:
            json.dump(total, handle)
    except OSError as exc:
        _warn_once(f"cannot write the counter ({type(exc).__name__}); spend is unmeasured")


def read_day(day: date, root: Path | None = None) -> dict[str, dict]:
    """Sum every process's file for one day, keyed by consumer.

    An unreadable file is named rather than skipped: a file that will not parse
    is spend nobody can see, and dropping it would understate the day.
    """
    directory = spend_root(root) / day.isoformat()
    out: dict[str, dict] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        consumer = path.name.split(".")[0]
        row = out.setdefault(
            consumer,
            {"calls": 0, "bytes": 0, "unsized_calls": 0, "limit_hit": False,
             "processes": 0, "unreadable": 0},
        )
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            out[consumer] = {**row, "unreadable": row["unreadable"] + 1}
            continue
        if not isinstance(data, dict):
            out[consumer] = {**row, "unreadable": row["unreadable"] + 1}
            continue
        out[consumer] = {
            **row,
            "calls": row["calls"] + _int(data.get("calls")),
            "bytes": row["bytes"] + _int(data.get("bytes")),
            "unsized_calls": row["unsized_calls"] + _int(data.get("unsized_calls")),
            "limit_hit": row["limit_hit"] or bool(data.get("limit_hit")),
            "processes": row["processes"] + 1,
        }
    return out


def _int(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def touch(consumer: str, *, day: date | None = None, root: Path | None = None) -> None:
    """Write a zero row so a quiet process is measured rather than missing.

    Without this there is no way to tell a consumer that ran and made no calls
    from a consumer whose counter was never installed, and the rail would have
    to report both as unmeasured. A rail that reports a finding on every quiet
    day teaches the reader to skip it.
    """
    day = day or datetime.now(timezone.utc).date()
    key = (consumer, day.isoformat())
    if key in _TOTALS:
        return
    total = {
        "schema": SCHEMA,
        "consumer": consumer,
        "day": day.isoformat(),
        "pid": os.getpid(),
        "calls": 0,
        "bytes": 0,
        "unsized_calls": 0,
        "limit_hit": False,
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    _TOTALS[key] = total
    directory = spend_root(root) / day.isoformat()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / f"{consumer}.{os.getpid()}.json", "w") as handle:
            json.dump(total, handle)
    except OSError as exc:
        _warn_once(f"cannot write the counter ({type(exc).__name__}); spend is unmeasured")
