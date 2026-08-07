import json

import pytest

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


def test_build_universe_adds_fresh_keeps_core_and_held():
    from bridge.run_daily import build_universe
    core = ["AAPL", "MSFT"]
    held = ["LLY", "AAPL"]  # AAPL overlaps core -> dedupe
    fresh = [("NVDA", 120.0), ("AAPL", 220.0), ("PLTR", 30.0),
             ("PENNY", 2.0), ("LLY", 900.0), ("AMD", 150.0)]
    u = build_universe(core, held, fresh, k_fresh=2, price_floor=5.0)
    # core, then held-not-in-core, order-stable and deduped
    assert u[:3] == ["AAPL", "MSFT", "LLY"]
    assert u.count("AAPL") == 1
    # fresh: NVDA (new) + PLTR (new) fill k=2; AAPL/LLY skipped (already in),
    # PENNY below the price floor, AMD never reached (cap hit).
    assert "NVDA" in u and "PLTR" in u
    assert "PENNY" not in u
    assert "AMD" not in u


def test_rank_movers_orders_by_absolute_move_and_carries_price():
    # 2026-07-31 (user: "large cap movers and AI and memory stocks"): fresh
    # candidates come from a curated quality pool ranked by today's move
    # magnitude, not Alpaca volume-most-actives (which surfaced leveraged ETFs
    # and pennies). Biggest movers first; a down move ranks by magnitude, not
    # sign; missing snapshots drop out.
    from bridge.run_daily import rank_movers
    pool = ["MU", "NVDA", "AMD", "GHOST"]
    snaps = {
        "MU": (838.0, -0.042),   # -4.2%
        "NVDA": (180.0, 0.011),  # +1.1%
        "AMD": (150.0, 0.070),   # +7.0% -> biggest
        # GHOST absent
    }
    ranked = rank_movers(pool, snaps)
    assert [s for s, _ in ranked] == ["AMD", "MU", "NVDA"]  # |7| > |4.2| > |1.1|
    assert dict(ranked)["MU"] == 838.0  # price carried through for the floor
    assert "GHOST" not in dict(ranked)  # no snapshot -> excluded


def test_build_universe_never_orphans_a_held_name():
    from bridge.run_daily import build_universe
    u = build_universe(["MSFT"], ["ZZZ"], [], k_fresh=5)
    assert "ZZZ" in u and "MSFT" in u  # held kept even with no fresh candidates


def test_build_universe_drops_an_option_leg_held_by_the_hedge_sleeve():
    # 2026-08-05. The daily email carried this line every day:
    #   "excluded (dead committee, held as-is): XSP260904P00726000 100% failed"
    # XSP260904P00726000 is the index-hedge sleeve's put. `held` comes from
    # broker.positions(), which returns options alongside equities, so an OSI
    # leg reached a committee of stock pickers that cannot value an option.
    # All 8 abstained, the ratio hit 1.0, and a real committee outage would
    # have looked identical to this. A red that fires every day trains the eye
    # to ignore red.
    from bridge.run_daily import build_universe
    u = build_universe(["AAPL"], ["LLY", "XSP260904P00726000"], [])
    assert u == ["AAPL", "LLY"]


def test_build_universe_keeps_a_ticker_that_merely_looks_option_ish():
    # The filter must key on the OSI shape, not on length or digits. A real
    # equity ticker never carries a 6-digit date + right + 8-digit strike.
    from bridge.run_daily import build_universe
    u = build_universe([], ["BRK.B", "GOOGL", "PBR.A"], [])
    assert u == ["BRK.B", "GOOGL", "PBR.A"]


def test_rebalance_never_emits_an_order_against_an_option_symbol():
    # Defense in depth. Exclusion is what stops the picks rebalancer from
    # trading the hedge leg today, and exclusion is a side effect of every
    # persona abstaining — not a rule. Once the leg leaves the universe it is
    # no longer excluded, so `current_mv` alone would make it sellable: a
    # notional equity sell against an OSI symbol, on a leg another sleeve owns.
    from bridge.run_daily import rebalance_orders
    orders = rebalance_orders(
        targets={"AAPL": 0.10},
        current_mv={"AAPL": 5_000.0, "XSP260904P00726000": 1_240.0},
        equity=100_000.0,
    )
    assert [o["symbol"] for o in orders] == ["AAPL"]


def test_llm_failure_ratio_gate_restricts_to_core_tickers():
    # A fresh name with no fundamentals goes dead; it must NOT trip the global
    # dead-committee HALT gate — that gate detects credit/provider death, which
    # would kill the core anchors too. Per-ticker exclusion still drops it.
    from bridge.run_daily import llm_failure_ratio
    per_ticker = {
        "AAPL": {"buffett": {"value": 0.6, "reasoning": "ok", "abstained": False}},
        "FRESHX": {"buffett": {"value": 0.0, "reasoning": "abstained: insufficient data",
                               "abstained": True}},
    }
    assert llm_failure_ratio(per_ticker) == 0.5                      # unrestricted
    assert llm_failure_ratio(per_ticker, gate_tickers={"AAPL"}) == 0.0  # core-gated


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


# --- daily_closes: batched history for the learning loop ---------------------


class _FakeResp:
    def __init__(self, payload, ok=True, status=200, text=""):
        self._payload, self.ok, self.status_code = payload, ok, status
        self.text = text or json.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    """Replays a queue of responses and records every request."""

    def __init__(self, responses):
        self._queue, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self._queue.pop(0)


def _client_with(session):
    from bridge.alpaca import AlpacaPaper

    client = AlpacaPaper.__new__(AlpacaPaper)  # skip env/endpoint checks
    client.session = session
    return client


def _bar(day, close):
    return {"t": f"{day}T04:00:00Z", "c": close}


def test_daily_closes_keys_each_close_by_its_session_date():
    session = _FakeSession([_FakeResp(
        {"bars": {"MU": [_bar("2026-07-27", 100.0), _bar("2026-07-28", 101.5)]}})])

    out = _client_with(session).daily_closes(["MU"], "2026-07-27", "2026-08-01")

    assert out == {"MU": {"2026-07-27": 100.0, "2026-07-28": 101.5}}


def test_daily_closes_asks_for_split_adjusted_daily_bars():
    """Unadjusted, a 2-for-1 split inside the window reads as a 50% drop."""
    session = _FakeSession([_FakeResp({"bars": {}})])

    _client_with(session).daily_closes(["MU"], "2026-07-27", "2026-08-01")

    _, params = session.calls[0]
    assert params["adjustment"] == "all"
    assert params["timeframe"] == "1Day"
    assert params["symbols"] == "MU"


def test_daily_closes_follows_pagination_and_merges_the_pages():
    session = _FakeSession([
        _FakeResp({"bars": {"MU": [_bar("2026-07-27", 100.0)]},
                   "next_page_token": "p2"}),
        _FakeResp({"bars": {"MU": [_bar("2026-07-28", 101.0)]}}),
    ])

    out = _client_with(session).daily_closes(["MU"], "2026-07-27", "2026-08-01")

    assert out["MU"] == {"2026-07-27": 100.0, "2026-07-28": 101.0}
    assert session.calls[1][1]["page_token"] == "p2"


def test_daily_closes_raises_rather_than_returning_a_silent_empty():
    from bridge.alpaca import AlpacaPaperError

    session = _FakeSession([_FakeResp({}, ok=False, status=429, text="slow down")])

    with pytest.raises(AlpacaPaperError):
        _client_with(session).daily_closes(["MU"], "2026-07-27", "2026-08-01")


def test_daily_closes_asks_for_nothing_when_no_name_was_called():
    session = _FakeSession([])

    assert _client_with(session).daily_closes([], "2026-07-27", "2026-08-01") == {}
    assert session.calls == []


def test_daily_closes_refuses_to_page_forever():
    from bridge.alpaca import AlpacaPaperError

    session = _FakeSession([_FakeResp({"bars": {}, "next_page_token": "p"})] * 3)

    with pytest.raises(AlpacaPaperError):
        _client_with(session).daily_closes(["MU"], "2026-07-27", "2026-08-01",
                                           page_limit=3)


def test_daily_closes_pins_the_feed_the_key_can_actually_read():
    """A SIP request reaching the last 15 minutes 403s on this plan, and the
    403 costs the whole window — one live run scored 0 of 98 picks."""
    session = _FakeSession([_FakeResp({"bars": {}})])

    _client_with(session).daily_closes(["MU"], "2026-07-27", "2026-08-01")

    assert session.calls[0][1]["feed"] == "iex"
