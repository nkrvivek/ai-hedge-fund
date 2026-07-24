"""Index-hedge sleeve (XSP puts) — pure-core unit tests.

Rule doc: bridge/INDEX_HEDGE_RULE.md (approved 2026-07-24; start decision:
live now w/ arm threshold −5.0 until the 8/10 eval ends, then −1.5).
"""
from __future__ import annotations

from datetime import date

import pytest

from bridge import index_hedge as ih


# ── net_conviction ───────────────────────────────────────────────────────────
def test_net_conviction_sums_all_signs() -> None:
    assert ih.net_conviction({"A": -1.0, "B": -0.5, "C": 0.75}) == pytest.approx(-0.75)


def test_net_conviction_empty() -> None:
    assert ih.net_conviction({}) == 0.0


# ── OCC parsing ──────────────────────────────────────────────────────────────
def test_parse_occ() -> None:
    root, expiry, right, strike = ih.parse_occ("XSP260821P00610000")
    assert root == "XSP"
    assert expiry == date(2026, 8, 21)
    assert right == "P"
    assert strike == pytest.approx(610.0)


def test_parse_occ_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        ih.parse_occ("NOTANOCC")


# ── arm threshold schedule (user decision 2026-07-24) ────────────────────────
def test_arm_threshold_pre_eval_is_minus_5() -> None:
    assert ih.arm_threshold(date(2026, 8, 1)) == pytest.approx(-5.0)
    assert ih.arm_threshold(ih.EVAL_END) == pytest.approx(-5.0)


def test_arm_threshold_post_eval_is_minus_1_5() -> None:
    assert ih.arm_threshold(date(2026, 8, 11)) == pytest.approx(-1.5)


# ── hedge_action: arm / hold / stand-down ────────────────────────────────────
TODAY_PRE = date(2026, 7, 24)    # pre-eval → arm needs < −5.0
TODAY_POST = date(2026, 8, 12)   # post-eval → arm needs ≤ −1.5


def test_no_arm_pre_eval_at_minus_3_75() -> None:
    # The 7/23 reference read must NOT arm before 8/10 under the user's rule.
    act = ih.hedge_action(-3.75, TODAY_PRE, open_puts=[], equity=100_000)
    assert act["action"] == "none"


def test_arms_pre_eval_below_minus_5() -> None:
    act = ih.hedge_action(-5.5, TODAY_PRE, open_puts=[], equity=100_000)
    assert act["action"] == "open"
    assert act["contracts"] == 2  # int(5.5 / 2.0) = 2, cap 2


def test_arms_post_eval_at_minus_1_5() -> None:
    act = ih.hedge_action(-1.5, TODAY_POST, open_puts=[], equity=100_000)
    assert act["action"] == "open"
    assert act["contracts"] == 1  # size floor is 1 once armed


def test_size_caps_at_two_contracts() -> None:
    act = ih.hedge_action(-9.0, TODAY_POST, open_puts=[], equity=100_000)
    assert act["contracts"] == 2


def test_stand_down_closes_open_hedge() -> None:
    puts = [{"symbol": "XSP260918P00600000", "qty": 1, "cost_basis": -800.0,
             "market_value": -500.0}]
    act = ih.hedge_action(-0.4, TODAY_POST, open_puts=puts, equity=100_000)
    assert act["action"] == "close"
    assert act["symbols"] == ["XSP260918P00600000"]


def test_hysteresis_band_holds_existing() -> None:
    # Between stand-down (−0.5) and arm: existing hedge stays, no new entry.
    puts = [{"symbol": "XSP260918P00600000", "qty": 1, "cost_basis": -800.0,
             "market_value": -500.0}]
    act = ih.hedge_action(-1.0, TODAY_POST, open_puts=puts, equity=100_000)
    assert act["action"] == "hold"


def test_no_double_entry_when_hedge_open() -> None:
    puts = [{"symbol": "XSP260918P00600000", "qty": 1, "cost_basis": -800.0,
             "market_value": -500.0}]
    act = ih.hedge_action(-6.0, TODAY_PRE, open_puts=puts, equity=100_000)
    assert act["action"] == "hold"


# ── 21-DTE time exit beats everything ────────────────────────────────────────
def test_time_exit_at_21_dte() -> None:
    # expiry 2026-08-21, today 2026-08-01 → 20 DTE → close even though armed.
    puts = [{"symbol": "XSP260821P00600000", "qty": 1, "cost_basis": -800.0,
             "market_value": -700.0}]
    act = ih.hedge_action(-6.0, date(2026, 8, 1), open_puts=puts, equity=100_000)
    assert act["action"] == "close"
    assert "dte" in act["reason"]


# ── premium caps ─────────────────────────────────────────────────────────────
def test_tranche_premium_cap_blocks_entry() -> None:
    # 2 contracts * $9.00 mid * 100 = $1,800 > 1.5% of $100K ($1,500) → trim to 1.
    n = ih.contracts_within_tranche_cap(2, mid=9.00, equity=100_000)
    assert n == 1


def test_tranche_cap_can_zero_out() -> None:
    n = ih.contracts_within_tranche_cap(1, mid=16.00, equity=100_000)
    assert n == 0


def test_total_premium_cap_blocks_entry() -> None:
    # Open hedge premium $2,600 + new $700 > 3% of $100K → blocked.
    puts = [{"symbol": "XSP260918P00600000", "qty": 2, "cost_basis": -2600.0,
             "market_value": -2000.0}]
    assert not ih.within_total_premium_cap(puts, new_premium=700.0, equity=100_000)
    assert ih.within_total_premium_cap(puts, new_premium=300.0, equity=100_000)


# ── contract selection ───────────────────────────────────────────────────────
CONTRACTS = [
    {"symbol": "XSP260814P00610000", "strike_price": "610", "expiration_date": "2026-08-14"},
    {"symbol": "XSP260828P00610000", "strike_price": "610", "expiration_date": "2026-08-28"},
    {"symbol": "XSP260828P00595000", "strike_price": "595", "expiration_date": "2026-08-28"},
    {"symbol": "XSP260828P00560000", "strike_price": "560", "expiration_date": "2026-08-28"},
]


def test_select_contract_wants_30_45_dte_and_3_5_pct_otm() -> None:
    # today 7/24, spot 630: 8/14 = 21 DTE (out), 8/28 = 35 DTE (in).
    # 3-5% OTM band = 598.5-611.1 → 610 in band, 595 barely out, 560 way out.
    pick = ih.select_contract(
        CONTRACTS, spot=630.0, today=date(2026, 7, 24),
        quote_fn=lambda s: (8.00, 8.60),
    )
    assert pick["symbol"] == "XSP260828P00610000"
    assert pick["mid"] == pytest.approx(8.30)


def test_select_contract_spread_guard() -> None:
    # spread $2.00 on mid $5.00 = 40% > 10% → rejected, nothing eligible.
    pick = ih.select_contract(
        CONTRACTS, spot=630.0, today=date(2026, 7, 24),
        quote_fn=lambda s: (4.00, 6.00),
    )
    assert pick is None


def test_select_contract_no_quotes() -> None:
    pick = ih.select_contract(
        CONTRACTS, spot=630.0, today=date(2026, 7, 24),
        quote_fn=lambda s: None,
    )
    assert pick is None
