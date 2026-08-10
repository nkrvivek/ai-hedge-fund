"""Probation gates for the 2026-08-10 -> 2026-09-10 extension (DJ-20260810-05).

Month-1 eval: -2.64% vs SPY +2.43%, hit rate 45.9% on 98 scored signals,
and a 72%-of-book turnover morning on overnight conviction whiplash. The
user extended one month on probation with two code conditions:

1. Committee-health gate: ANY committee failure on the core anchors blocks
   all trading for the day (signals still run, ledger still written, email
   still sent). Would have blocked 7/16-17 and the 8/10 2%-failed day.
2. Churn cap: at most CHURN_CAP of equity turned per day; over-cap order
   sets are scaled proportionally, never reordered or cherry-picked.

Condition 3 (hit rate must clear 0.50 by 2026-09-10) is an eval criterion,
not code — it lives in the decision journal.
"""
from __future__ import annotations

import re
from pathlib import Path

from bridge.run_daily import (
    CHURN_CAP,
    MIN_TRADE_USD,
    cap_churn,
    probation_no_trade_reason,
)

RUN_DAILY_SRC = (Path(__file__).parent / "run_daily.py").read_text()


def test_healthy_committee_trades():
    assert probation_no_trade_reason(0.0) is None


def test_any_failure_blocks_the_day():
    reason = probation_no_trade_reason(0.02)
    assert reason is not None
    assert "no trades" in reason
    assert "2%" in reason


def test_churn_under_cap_passes_through():
    orders = [{"symbol": "AAPL", "delta_usd": 1000.0, "side": "buy"}]
    capped, scale = cap_churn(orders, equity=100_000.0)
    assert scale == 1.0
    assert capped == orders


def test_churn_over_cap_scales_proportionally():
    # 50% of equity staged -> scaled to CHURN_CAP, sides preserved
    orders = [
        {"symbol": "AAPL", "delta_usd": 30_000.0, "side": "buy"},
        {"symbol": "TSLA", "delta_usd": -20_000.0, "side": "sell"},
    ]
    capped, scale = cap_churn(orders, equity=100_000.0)
    assert scale == CHURN_CAP * 100_000.0 / 50_000.0
    assert [o["symbol"] for o in capped] == ["AAPL", "TSLA"]
    assert capped[0]["delta_usd"] == 15_000.0
    assert capped[1]["delta_usd"] == -10_000.0
    assert capped[1]["side"] == "sell"
    # input list untouched (immutability)
    assert orders[0]["delta_usd"] == 30_000.0


def test_churn_scaling_drops_dust():
    orders = [
        {"symbol": "AAPL", "delta_usd": 50_000.0, "side": "buy"},
        {"symbol": "TSLA", "delta_usd": 300.0, "side": "buy"},
    ]
    capped, scale = cap_churn(orders, equity=100_000.0)
    # TSLA scales to ~$149 < MIN_TRADE_USD and is dropped
    assert scale < 1.0
    assert [o["symbol"] for o in capped] == ["AAPL"]
    assert abs(capped[0]["delta_usd"]) >= MIN_TRADE_USD


def test_churn_no_orders_is_a_noop():
    assert cap_churn([], equity=100_000.0) == ([], 1.0)


def test_placement_is_gated_on_probation_in_source():
    # Source contract (same pattern as autopilot's cron-dedupe tests): a
    # refactor of main() must not silently drop the probation gate from the
    # order-placement branch or the hedge dispatch.
    assert re.search(r"if not args\.dry_run and not probation\b", RUN_DAILY_SRC), (
        "order placement must be gated on the probation reason"
    )
    assert re.search(r"dry_run=args\.dry_run or bool\(probation\)", RUN_DAILY_SRC), (
        "the index-hedge sleeve must also stand down on a probation day"
    )


def test_ledger_records_probation_and_churn_in_source():
    assert '"probation_no_trade"' in RUN_DAILY_SRC
    assert '"churn_scale"' in RUN_DAILY_SRC
