"""Canonical per-fill trade ledger — ai-hedge-fund book only.

One JSON row per fill, append-only, in the shape TRADE_LEDGER_SCHEMA.md defines
(the same shape the autopilot books use). This module is the ONLY place a row
is built; the daily bridge funnels each fill through one adapter:

  - `record_for_rebalance` → each placed stock-rebalance order (EQUITY)
  - `record_for_hedge_open` → each XSP index-hedge leg opened (OPTION)
  - `records_for_hedge_close` → each hedge leg closed (OPTION)

`write_records` appends them, fully guarded — a ledger write must never break a
run. `backfill_records_from_daily` rebuilds rows, best-effort, from the old
1-row-per-day `ledger.jsonl`.

Books stay separate. This module writes only `ai_hedge_fund` and never reads
another book's ledger. Consistency across books comes from the shared shape,
not a shared store — see the schema doc.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SCHEMA_VERSION = 1
_PT = ZoneInfo("America/Los_Angeles")

BOOK = "ai_hedge_fund"
_BROKER = "alpaca_paper"

_BUY_SIDES = {"buy_to_open", "buy_to_close", "buy"}
_OCC_RE = re.compile(r"^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")

LEDGER_PATH = Path(__file__).parent / "trades.jsonl"


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_occ(osi: str | None) -> dict:
    """Split a compact OCC option symbol into instrument fields. {} on failure —
    never a fabricated field."""
    m = _OCC_RE.match((osi or "").strip().upper())
    if not m:
        return {}
    root, yy, mm, dd, right, strike = m.groups()
    return {
        "underlying": root,
        "expiry": f"20{yy}-{mm}-{dd}",
        "right": right,
        "strike": int(strike) / 1000.0,
    }


def build_trade_record(*, kind, side, symbol, asset_class="EQUITY",
                       underlying=None, right=None, strike=None, expiry=None,
                       qty, price, mult=1, gross_usd=None, fees_usd=0.0,
                       broker_order_id=None, realized_pnl_usd=None,
                       account_id=None, sleeve="committee",
                       source="run_daily.py", ts_utc=None, seq=0,
                       thesis_ref=None, proposal_id=None, dj_ref=None) -> dict:
    """Build one canonical trade row. Pure — no I/O, no clock unless ts_utc omitted.

    `gross_usd` derives from price × qty × mult, signed by `side` (buy = debit =
    negative), only when a caller passes none. A null price leaves gross null —
    numbers are never fabricated."""
    now = ts_utc or datetime.now(timezone.utc)
    q = abs(float(qty)) if qty is not None else None
    q = int(q) if (q is not None and q == int(q)) else q
    px = round(float(price), 4) if price is not None else None
    if gross_usd is None and px is not None and q is not None:
        sign = -1.0 if side in _BUY_SIDES else 1.0
        gross_usd = round(sign * px * q * (mult or 1), 2)
    compact = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rid = f"{BOOK}-{compact}-{broker_order_id or 'na'}-{seq}"
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": rid,
        "book": BOOK,
        "account": {"broker": _BROKER, "account_id": account_id},
        "ts_utc": _iso_z(now),
        "ts_pt": now.astimezone(_PT).isoformat(),
        "kind": kind,
        "instrument": {
            "asset_class": asset_class,
            "symbol": symbol,
            "underlying": underlying,
            "right": right,
            "strike": float(strike) if strike is not None else None,
            "expiry": expiry,
        },
        "side": side,
        "qty": q,
        "price": px,
        "mult": mult,
        "gross_usd": gross_usd,
        "fees_usd": fees_usd,
        "sleeve": sleeve,
        "thesis_ref": thesis_ref,
        "proposal_id": proposal_id,
        "broker_order_id": broker_order_id,
        "realized_pnl_usd": (round(float(realized_pnl_usd), 2)
                             if realized_pnl_usd is not None else None),
        "dj_ref": dj_ref,
        "source": source,
    }


def record_for_rebalance(placed: dict, fill: dict | None = None,
                         *, ts_utc=None, seq=0) -> dict | None:
    """One EQUITY row for a placed stock-rebalance order.

    `placed` is a run_daily order dict: {symbol, delta_usd, side, order_id,
    status, [liquidated], [skipped], [error]}. `fill` is the resolved Alpaca
    order, when fetched: {filled_qty, filled_avg_price}.

    Nothing filled → no row: a `skipped`/`error` order, or a zero-fill status,
    never books a trade. When qty/price aren't available we still record the
    intended notional (signed `delta_usd`) as an estimate tagged `:booked_est`,
    with qty/price null — the cash intent is real, the per-share split is not."""
    if not isinstance(placed, dict):
        return None
    if placed.get("skipped") or placed.get("error"):
        return None
    symbol = placed.get("symbol")
    side = placed.get("side")
    if not symbol or side not in ("buy", "sell"):
        return None

    liquidated = bool(placed.get("liquidated"))
    kind = "close" if (liquidated or side == "sell") else "fill"

    qty = price = gross = None
    source = "run_daily.py:rebalance"
    if fill:
        fq = fill.get("filled_qty")
        fp = fill.get("filled_avg_price")
        qty = float(fq) if fq not in (None, "") else None
        price = float(fp) if fp not in (None, "") else None
    if qty is None or price is None:
        # No confirmed fill split — book the signed notional as an estimate.
        delta = placed.get("delta_usd")
        if delta is None:
            return None
        gross = round(-abs(float(delta)) if side == "buy" else abs(float(delta)), 2)
        source += ":booked_est"

    return build_trade_record(
        kind=kind, side=side, symbol=symbol, asset_class="EQUITY",
        underlying=symbol, qty=qty, price=price, mult=1, gross_usd=gross,
        sleeve="committee", broker_order_id=placed.get("order_id"),
        source=source, ts_utc=ts_utc, seq=seq)


def record_for_hedge_open(hedge: dict, fill: dict | None = None,
                          *, ts_utc=None, seq=0) -> dict | None:
    """One OPTION row for an opened XSP index-hedge leg.

    `hedge` is the run_index_hedge return with action=="open": carries
    `contract` ({symbol, mid, ...}), `contracts` (qty), `order_id`, `status`.
    Price is the limit `mid` unless a real `filled_avg_price` is supplied."""
    if not isinstance(hedge, dict) or hedge.get("action") != "open":
        return None
    if hedge.get("dry_run") or hedge.get("status") == "dry_run":
        return None
    contract = hedge.get("contract") or {}
    osi = contract.get("symbol")
    n = hedge.get("contracts")
    if not osi or not n:
        return None
    price = None
    if fill and fill.get("filled_avg_price") not in (None, ""):
        price = float(fill["filled_avg_price"])
    if price is None:
        price = contract.get("mid")
    inst = _parse_occ(osi)
    return build_trade_record(
        kind="open", side="buy_to_open", symbol=osi, asset_class="OPTION",
        underlying=inst.get("underlying"), right=inst.get("right"),
        strike=inst.get("strike"), expiry=inst.get("expiry"),
        qty=n, price=price, mult=100, sleeve="index_hedge",
        broker_order_id=hedge.get("order_id"),
        source="index_hedge.py:open", ts_utc=ts_utc, seq=seq)


def records_for_hedge_close(hedge: dict, *, ts_utc=None) -> list[dict]:
    """OPTION close rows for each leg in a run_index_hedge close result. Price is
    unknown at this layer (broker-sized close), so it is null, not invented."""
    if not isinstance(hedge, dict) or hedge.get("action") != "close":
        return []
    out = []
    for i, leg in enumerate(hedge.get("closed") or []):
        sym = leg.get("symbol")
        if not sym or leg.get("error"):
            continue
        inst = _parse_occ(sym)
        out.append(build_trade_record(
            kind="close", side="sell_to_close", symbol=sym, asset_class="OPTION",
            underlying=inst.get("underlying"), right=inst.get("right"),
            strike=inst.get("strike"), expiry=inst.get("expiry"),
            qty=None, price=None, mult=100, sleeve="index_hedge",
            source="index_hedge.py:close", ts_utc=ts_utc, seq=i))
    return out


def backfill_records_from_daily(daily: dict) -> list[dict]:
    """Best-effort rows from one old-format `ledger.jsonl` daily row.

    The daily row records submitted orders (`delta_usd`, `side`, `order_id`,
    `status`), never fill qty/price — so every backfilled row has null qty/price,
    a signed-notional `gross_usd` estimate, and a `:backfill` source. Dry-run
    days and non-placed orders (skipped/error/nothing-filled) book nothing.
    Gaps stay gaps."""
    if not isinstance(daily, dict) or daily.get("dry_run"):
        return []
    ts = _daily_ts(daily)
    out = []
    for i, o in enumerate(daily.get("orders") or []):
        rec = record_for_rebalance(o, ts_utc=ts, seq=i)
        if rec:
            rec["source"] = "run_daily.py:backfill"
            out.append(rec)
    hedge = daily.get("index_hedge") or {}
    h = record_for_hedge_open(hedge, ts_utc=ts, seq=0)
    if h:
        h["source"] = "index_hedge.py:backfill"
        out.append(h)
    for r in records_for_hedge_close(hedge, ts_utc=ts):
        r["source"] = "index_hedge.py:backfill"
        out.append(r)
    return out


def _daily_ts(daily: dict) -> datetime:
    """Timestamp for a backfilled row: the daily row's own `ts`, else its
    `asof` date at UTC midnight. Falls back to now only if both are missing."""
    raw = daily.get("ts")
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    asof = daily.get("asof")
    if asof:
        try:
            d = date.fromisoformat(asof)
            return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def write_records(records: list[dict], path: Path | str = LEDGER_PATH) -> int:
    """Append rows to the book's own ledger, one JSON object per line. Guarded —
    returns the count written; a write failure prints and returns 0, never
    raising into a fill path."""
    rows = [r for r in (records or []) if r]
    if not rows:
        return 0
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return len(rows)
    except Exception as e:  # noqa: BLE001 — a ledger write must never break a run
        print(f"trade_ledger write FAILED (non-fatal): {e}")
        return 0
