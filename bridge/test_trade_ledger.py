"""Tests for the ai-hedge-fund canonical per-fill trade ledger."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from bridge.trade_ledger import (
    BOOK,
    backfill_records_from_daily,
    build_trade_record,
    record_for_hedge_open,
    record_for_rebalance,
    records_for_hedge_close,
    write_records,
)

TS = datetime(2026, 7, 28, 16, 25, 42, tzinfo=timezone.utc)


def test_build_record_signs_buy_as_debit():
    r = build_trade_record(kind="fill", side="buy", symbol="AAPL",
                           qty=10, price=200.0, mult=1, ts_utc=TS)
    assert r["book"] == BOOK
    assert r["account"]["broker"] == "alpaca_paper"
    assert r["gross_usd"] == -2000.0  # buy = cash out = negative
    assert r["ts_utc"].endswith("Z")
    assert r["ts_pt"].startswith("2026-07-28T09:25:42")  # PT, same instant


def test_build_record_signs_sell_as_credit():
    r = build_trade_record(kind="close", side="sell", symbol="AAPL",
                           qty=10, price=200.0, mult=1, ts_utc=TS)
    assert r["gross_usd"] == 2000.0


def test_build_record_null_price_leaves_gross_null():
    r = build_trade_record(kind="fill", side="buy", symbol="AAPL",
                           qty=None, price=None, ts_utc=TS)
    assert r["price"] is None
    assert r["gross_usd"] is None  # never fabricated


def test_rebalance_with_fill_uses_real_qty_price():
    placed = {"symbol": "NVDA", "delta_usd": 3000.0, "side": "buy",
              "order_id": "abc", "status": "filled"}
    fill = {"filled_qty": "20", "filled_avg_price": "150.25"}
    r = record_for_rebalance(placed, fill, ts_utc=TS)
    assert r["qty"] == 20
    assert r["price"] == 150.25
    assert r["gross_usd"] == -3005.0  # 20 * 150.25, buy negative
    assert r["source"] == "run_daily.py:rebalance"  # no :booked_est
    assert r["broker_order_id"] == "abc"


def test_rebalance_without_fill_books_notional_estimate():
    placed = {"symbol": "META", "delta_usd": -3070.51, "side": "sell",
              "order_id": "d1", "status": "accepted"}
    r = record_for_rebalance(placed, fill=None, ts_utc=TS)
    assert r["qty"] is None
    assert r["price"] is None
    assert r["gross_usd"] == 3070.51  # sell = credit, from |delta_usd|
    assert r["source"].endswith(":booked_est")
    assert r["kind"] == "close"  # sell reduces a long


def test_rebalance_liquidation_is_close():
    placed = {"symbol": "TSLA", "delta_usd": -500.0, "side": "sell",
              "order_id": "x", "status": "closed", "liquidated": True}
    r = record_for_rebalance(placed, ts_utc=TS)
    assert r["kind"] == "close"


def test_rebalance_skipped_or_error_books_nothing():
    assert record_for_rebalance(
        {"symbol": "AAPL", "side": "sell", "skipped": "nothing_held_long_only"}) is None
    assert record_for_rebalance(
        {"symbol": "AAPL", "side": "buy", "delta_usd": 100, "error": "boom"}) is None


def test_hedge_open_parses_occ_and_uses_mid():
    hedge = {"action": "open", "contracts": 3,
             "contract": {"symbol": "XSP260918P00600000", "mid": 4.20},
             "order_id": "h1", "status": "accepted"}
    r = record_for_hedge_open(hedge, ts_utc=TS)
    assert r["instrument"]["asset_class"] == "OPTION"
    assert r["instrument"]["underlying"] == "XSP"
    assert r["instrument"]["right"] == "P"
    assert r["instrument"]["strike"] == 600.0
    assert r["instrument"]["expiry"] == "2026-09-18"
    assert r["side"] == "buy_to_open"
    assert r["qty"] == 3
    assert r["price"] == 4.20
    assert r["gross_usd"] == -1260.0  # 3 * 4.20 * 100, debit
    assert r["sleeve"] == "index_hedge"


def test_hedge_open_dry_run_books_nothing():
    hedge = {"action": "open", "contracts": 3,
             "contract": {"symbol": "XSP260918P00600000", "mid": 4.20},
             "status": "dry_run"}
    assert record_for_hedge_open(hedge) is None


def test_hedge_close_qty_price_null_not_invented():
    hedge = {"action": "close", "closed": [{"symbol": "XSP260918P00600000"}]}
    rows = records_for_hedge_close(hedge, ts_utc=TS)
    assert len(rows) == 1
    assert rows[0]["side"] == "sell_to_close"
    assert rows[0]["qty"] is None
    assert rows[0]["price"] is None
    assert rows[0]["gross_usd"] is None


def test_backfill_from_daily_row():
    daily = {
        "ts": "2026-07-28T16:25:42.845932+00:00", "asof": "2026-07-28",
        "orders": [
            {"symbol": "GOOGL", "delta_usd": -1089.17, "side": "sell",
             "order_id": "o1", "status": "pending_new"},
            {"symbol": "META", "delta_usd": -3070.51, "side": "sell",
             "order_id": "o2", "status": "pending_new"},
            {"symbol": "AAPL", "delta_usd": 500.0, "side": "buy",
             "order_id": "o3", "skipped": "x"},  # skipped → no row
        ],
        "index_hedge": {"action": "none"},
        "dry_run": False,
    }
    rows = backfill_records_from_daily(daily)
    assert len(rows) == 2  # the skipped order books nothing
    assert all(r["source"] == "run_daily.py:backfill" for r in rows)
    assert all(r["qty"] is None and r["price"] is None for r in rows)
    assert rows[0]["ts_utc"] == "2026-07-28T16:25:42Z"


def test_backfill_dry_run_books_nothing():
    assert backfill_records_from_daily({"orders": [{"symbol": "A", "side": "buy",
                                                    "delta_usd": 1}],
                                        "dry_run": True}) == []


def test_write_records_appends_jsonl(tmp_path):
    p = tmp_path / "trades.jsonl"
    n = write_records([{"a": 1}, {"b": 2}], p)
    assert n == 2
    lines = p.read_text().strip().splitlines()
    assert [json.loads(x) for x in lines] == [{"a": 1}, {"b": 2}]
    # append, not overwrite
    write_records([{"c": 3}], p)
    assert len(p.read_text().strip().splitlines()) == 3


def test_write_records_empty_is_noop(tmp_path):
    p = tmp_path / "trades.jsonl"
    assert write_records([], p) == 0
    assert write_records([None], p) == 0
    assert not p.exists()
