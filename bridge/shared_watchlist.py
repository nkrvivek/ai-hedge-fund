"""The shared watch list — this book's half of "one list, five books".

User, 2026-08-13, in trade-refresh: *"all the books, 5k, 200k, 100k ai hedge
fund, bildof, personal should follow a common watch list which is kept up to
date and replenished daily"*, then, on the delivery leg: *"we can't have these
lists updated only if a human runs it, it should be on cloud and replenished
and fed into all these books automatically and daily"*.

trade-refresh composes that list on a daily GitHub Actions cron and publishes it
to R2. The other four books read the object directly. This one cannot: it runs
in its own repo with no R2 credentials, and it is not getting any. The R2 S3
token that reaches `autopilot-state` can WRITE it, and handing a probation
experiment write access to the autopilot book's state so it can read one object
trades a real blast radius for a convenience. It reads over the worker's
token-gated `GET /watchlist` instead: one secret, no dependency, no write path.

WHERE IT LANDS, AND WHY NOT ANYWHERE ELSE.

Into `THEME_POOL`, the curated fresh-candidate pool. Not `UNIVERSE`: the fixed
core anchors are also the gate set for the dead-committee halt
(`llm_failure_ratio(..., gate_tickers=UNIVERSE)`), and a small-cap with no
filed statements added there would abstain every day and tip a false FATAL on a
healthy system. The pool is the right door because `build_universe` already
bounds it — top `k_fresh` movers above a price floor — so a pool of 160 names
costs one extra Alpaca snapshot call and not one extra LLM call. The committee
still scores the same number of names it scored yesterday.

That bound matters more here than anywhere else. This book is on probation
(DJ-20260810-05) until 2026-09-10 and dies if the hit rate misses 0.50. Widening
what it can *see* is the point; widening what it *trades per day* is a behaviour
change nobody asked for, and `k_fresh` is what keeps the second from following
the first.

NO CYCLE HERE. The composer's sources are the banger board, conviction, thesis,
pullback and the two moonshot screens. Nothing this book emits feeds it, so
unlike the autopilot reader there is no self-seeding loop to guard against. If
this book ever starts publishing into the list, that stops being true and this
comment is the place it was written down.

FAILING BACK, NOT CLOSED. Every path below returns the caller's own pool. A
screen handed zero names ranks zero movers and rotates nothing in, which looks
exactly like a quiet tape. A stale list is used and flagged rather than dropped,
for the same reason: dropping it shrinks coverage, which is the failure the
whole design guards against.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

WATCHLIST_PATH = "/watchlist"

# Must match STALE_AFTER_HOURS in trade-refresh's src/watchlist.py and
# WATCHLIST_STALE_AFTER_HOURS in its cloud/worker/shared_watchlist.ts. 40 hours
# rather than 24: the cron runs daily, and a bound tighter than the gap between
# two runs reports every ordinary morning as stale.
STALE_AFTER_HOURS = 40

# Ceiling on names added to the pool. The list runs ~115 today and the LLM cost
# is capped downstream by build_universe's k_fresh, so this is not a cost gate.
# It bounds the snapshot query string, and it is the line past which a
# quietly-doubled list stops being a watch list and becomes a market scan.
MAX_ADDED = 250

HTTP_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class SharedWatchlist:
    """Never shorter than the floor the caller passed in."""

    tickers: tuple[str, ...]
    as_of: str | None
    source: str  # "r2" | "fallback"
    stale: bool
    detail: str


def _clean(names: Iterable[object]) -> list[str]:
    """Upper-cased, de-duplicated, blanks dropped, order preserved."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        if not isinstance(raw, str):
            continue
        ticker = raw.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def _fallback(floor: Iterable[str], detail: str) -> SharedWatchlist:
    return SharedWatchlist(
        tickers=tuple(_clean(floor)), as_of=None,
        source="fallback", stale=False, detail=detail,
    )


def _http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as resp:
        return resp.read().decode("utf-8")


def _parse_as_of(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def read_shared_watchlist(
    floor: Iterable[str],
    fetch: Callable[[str], str] | None = None,
    base_url: str | None = None,
    token: str | None = None,
    now: datetime | None = None,
    max_added: int = MAX_ADDED,
) -> SharedWatchlist:
    """Read the published list and union it onto `floor`.

    `floor` is the caller's own pool and the guarantee: whatever goes wrong
    below, the returned pool is never shorter than what the caller already had.
    """
    now = now or datetime.now(timezone.utc)
    floor_names = _clean(floor)

    base = (base_url if base_url is not None else os.environ.get("TR_WORKER_URL", "")).rstrip("/")
    tok = token if token is not None else os.environ.get("TR_WORKER_TOKEN", "")
    if fetch is None:
        if not base or not tok:
            return _fallback(
                floor_names,
                "shared watch list not configured (TR_WORKER_URL / TR_WORKER_TOKEN) — pool only",
            )
        fetch = _http_get

    url = f"{base}{WATCHLIST_PATH}?token={tok}"
    try:
        body = fetch(url)
    except Exception as e:  # noqa: BLE001 — the screen is best-effort, the pool is safe
        return _fallback(floor_names, f"shared watch list unreachable ({e}) — pool only")

    try:
        payload = json.loads(body)
    except Exception:  # noqa: BLE001 — a 401 page is not JSON
        return _fallback(floor_names, "shared watch list did not return JSON — pool only")
    if not isinstance(payload, dict):
        return _fallback(floor_names, "shared watch list returned an unexpected shape — pool only")

    published = _clean(payload.get("tickers") or [])
    # A 200 is not a read. The route answers with source:"fallback" and no
    # tickers when the published object is missing, and that must land here as
    # "no list today", not as an empty universe.
    if not published or payload.get("source") == "fallback":
        why = payload.get("detail") or "no tickers published"
        return _fallback(floor_names, f"shared watch list unavailable ({why}) — pool only")

    added = [t for t in published if t not in set(floor_names)]
    truncated = len(added) > max_added
    if truncated:
        added = added[:max_added]

    as_of_raw = payload.get("as_of")
    as_of = as_of_raw if isinstance(as_of_raw, str) else None
    parsed = _parse_as_of(as_of_raw)
    # An unreadable stamp counts as stale, and the publisher's own flag can
    # only ever make it worse. Neither side gets to talk the other into fresh.
    stale = parsed is None or (now - parsed) > timedelta(hours=STALE_AFTER_HOURS)
    stale = stale or bool(payload.get("stale"))

    detail = f"shared watch list +{len(added)} names, as_of {as_of or 'unparseable'}"
    if truncated:
        detail += f"; capped at {max_added} of {len(published)} published"
    if stale:
        detail += (
            f"; STALE — not replenished inside {STALE_AFTER_HOURS}h, used anyway "
            "(dropping it would shrink the pool)"
        )

    return SharedWatchlist(
        tickers=tuple([*floor_names, *added]),
        as_of=as_of, source="r2", stale=stale, detail=detail,
    )
