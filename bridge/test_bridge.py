from bridge.run_daily import (
    buy_order_body,
    composite,
    llm_failure_ratio,
    rebalance_orders,
    target_weights,
    ticker_failure_ratios,
)


def test_buy_uses_whole_share_qty_not_notional():
    # 2026-07-31 MSFT: notional buy 403'd (40310000). Whole-share qty avoids it.
    assert buy_order_body(9849.0, 464.2) == {"qty": "21", "side": "buy"}
    assert buy_order_body(150.0, 464.2) == {"skip": "sub_share_dust"}
    assert buy_order_body(9849.0, None) == {"skip": "no_price"}
    assert buy_order_body(9849.0, 0) == {"skip": "no_price"}


def test_held_short_detects_short_dust():
    # 2026-07-31 root cause: the book carried tiny fractional SHORT dust
    # (MSFT -0.0022 sh, MV -$1.04) left by prior notional-sell rounding. A
    # whole-share BUY 403s (40310000) while any fractional lot is open — proven
    # live: close_position then buy clears. The account is long-only, so a
    # negative market value is always unwanted dust to flatten before buying.
    from bridge.run_daily import held_short
    assert held_short(-1.04) is True
    assert held_short(-0.0001) is True
    assert held_short(0.0) is False
    assert held_short(9852.0) is False


def test_flatten_and_wait_closes_then_polls_until_flat():
    # Mirrors the verified fix: close the short, poll until Alpaca reports the
    # symbol flat, THEN the whole-share buy can clear. Back-to-back close+buy
    # would race the still-open lot.
    from bridge.run_daily import flatten_and_wait

    class _FakeBroker:
        def __init__(self):
            self.closed = []
            self._poll = 0

        def close_position(self, symbol):
            self.closed.append(symbol)
            return {"status": "pending_new"}

        def positions(self):
            self._poll += 1
            return {"MSFT": -1.04} if self._poll < 2 else {}

    b = _FakeBroker()
    assert flatten_and_wait(b, "MSFT", tries=5, pause=0) is True
    assert b.closed == ["MSFT"]


def test_composite_ignores_abstains():
    assert composite([0.6, 0.0]) == 0.6
    assert composite([0.0, 0.0]) == 0.0
    assert composite([0.5, -0.5]) == 0.0


def test_weights_capped_per_name_and_gross():
    w = target_weights({"A": 1.0, "B": -1.0, "C": 1.0})
    assert all(abs(x) <= 0.10 for x in w.values())
    assert sum(abs(x) for x in w.values()) <= 1.0 + 1e-9


def test_weights_long_only_drops_bearish_names():
    # 2026-07-23: Alpaca paper refuses shorts (403 40310000) — bearish
    # conviction maps to no position, never a negative weight.
    w = target_weights({"UP": 0.5, "DOWN": -0.8, "FLAT": 0.0})
    assert "DOWN" not in w and "FLAT" not in w
    assert w["UP"] > 0
    assert all(x > 0 for x in w.values())


def test_rebalance_bearish_name_sells_held_never_more():
    # Bearish name absent from targets: order sells exactly what is held.
    orders = rebalance_orders({"UP": 0.10}, {"DOWN": 3000.0}, 100_000)
    by = {o["symbol"]: o for o in orders}
    assert by["DOWN"]["side"] == "sell"
    assert abs(by["DOWN"]["delta_usd"]) == 3000.0
    # And a name with nothing held generates no sell at all.
    assert "GHOST" not in {o["symbol"] for o in rebalance_orders({}, {}, 100_000)}


def test_rebalance_diffs_and_dust_filter():
    orders = rebalance_orders({"A": 0.10}, {"A": 9000.0, "B": 5000.0}, 100_000)
    by = {o["symbol"]: o for o in orders}
    assert by["A"]["side"] == "buy" and by["A"]["delta_usd"] == 1000.0
    assert by["B"]["side"] == "sell" and by["B"]["delta_usd"] == -5000.0
    assert not rebalance_orders({"A": 0.05}, {"A": 4900.0}, 100_000)  # dust


def test_paper_endpoint_guard(monkeypatch):
    import pytest
    from bridge.alpaca import AlpacaPaper, AlpacaPaperError
    monkeypatch.setenv("ALPACA_TRADING_ENDPOINT", "https://api.alpaca.markets/v2")
    with pytest.raises(AlpacaPaperError):
        AlpacaPaper()


def test_rebalance_holds_excluded_ticker_as_is():
    """A per-ticker-excluded symbol gets no order at all — held as-is, not
    force-sold — even though it carries no target weight and the diff
    against current market value would otherwise read as a full close."""
    orders = rebalance_orders(
        {"A": 0.10}, {"A": 9000.0, "B": 5000.0}, 100_000, excluded=frozenset({"B"}),
    )
    symbols = {o["symbol"] for o in orders}
    assert "B" not in symbols
    assert "A" in symbols


# ---------------------------------------------------------------------------
# llm_failure_ratio / ticker_failure_ratios — real _abstain() path
#
# 2026-07-22 audit: LLMAgent._abstain() (v2/signals/llm_agent.py) writes
# reasoning prefixed "abstained: ...", not "LLM call failed" — the old
# string-match in llm_failure_ratio missed every abstain and undercounted a
# dead committee (verified: first post-gate run read true failure 87.5% as
# 8.75%). These build signals through the REAL predict()/_abstain() path
# (not hand-built dicts) so the fix is checked against the actual prefix.
# ---------------------------------------------------------------------------

class _RaisingLLM:
    """Fake LLMClient that always raises — drives the real _abstain() path."""

    model = "fake-model"

    def complete(self, system, user):
        raise TimeoutError("llm timed out")


class _EightQuarterDataClient:
    """Enough financial-metrics history to clear InsufficientData."""

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        from v2.data.models import FinancialMetrics

        quarters = ["2024-12-31", "2024-09-30", "2024-06-30", "2024-03-31",
                    "2023-12-31", "2023-09-30", "2023-06-30", "2023-03-31"]
        return [
            FinancialMetrics(ticker=ticker, report_period=q, period="ttm", filing_date=q,
                             return_on_equity=0.2, gross_margin=0.4,
                             book_value_per_share=10.0, market_cap=1e9)
            for q in quarters
        ]

    def get_company_facts(self, ticker):
        return None


def _abstained_view(agent_cls, tmp_path):
    """A run_daily-shaped view entry, built via the real agent.predict() /
    LLMAgent._abstain() path — mirrors bridge/run_daily.py's view construction."""
    from v2.llm import PromptCache

    agent = agent_cls(llm=_RaisingLLM(), cache=PromptCache(tmp_path / agent_cls.__name__))
    sig = agent.predict("TEST", "2025-01-15", _EightQuarterDataClient())
    assert sig.metadata["abstained"] is True  # sanity: the real abstain path fired
    return {"value": sig.value, "reasoning": sig.reasoning,
            "abstained": bool(sig.metadata.get("abstained", False))}


def test_llm_failure_ratio_all_abstained_via_real_abstain_path(tmp_path):
    from v2.signals import BuffettAgent, DamodaranAgent, MungerAgent

    per_ticker = {
        "TEST": {
            "buffett": _abstained_view(BuffettAgent, tmp_path),
            "damodaran": _abstained_view(DamodaranAgent, tmp_path),
            "munger": _abstained_view(MungerAgent, tmp_path),
        }
    }

    assert llm_failure_ratio(per_ticker) == 1.0
    assert ticker_failure_ratios(per_ticker) == {"TEST": 1.0}


def test_llm_failure_ratio_still_counts_error_prefixed_reasoning():
    """run_daily's own per-agent except catch never sets `abstained` (it
    isn't a Signal at all) — the 'ERROR:' reasoning prefix must still count."""
    per_ticker = {"TEST": {"pead": {"value": 0.0, "reasoning": "ERROR: boom",
                                    "abstained": False}}}
    assert llm_failure_ratio(per_ticker) == 1.0


def test_ticker_failure_ratios_isolates_a_single_dead_ticker():
    """2026-07-22: LLY's committee was 100% dead (FMP 402) while the global
    ratio read 8.75% — a healthy-looking global average can hide one dead
    ticker. Per-ticker ratios must isolate it."""
    per_ticker = {
        "LLY": {"buffett": {"value": 0.0, "reasoning": "abstained: insufficient data",
                            "abstained": True}},
        "AAPL": {"buffett": {"value": 0.6, "reasoning": "Wonderful business.",
                             "abstained": False}},
    }
    ratios = ticker_failure_ratios(per_ticker)
    assert ratios == {"LLY": 1.0, "AAPL": 0.0}
