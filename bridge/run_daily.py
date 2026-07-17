"""Daily signal→Alpaca-paper bridge.

Runs the v2 alpha models (LLM investor agents + quant) over a fixed universe,
converts composite convictions into target portfolio weights, and rebalances
the Alpaca PAPER account toward them. Every trade is written to a JSONL ledger
with the full per-agent reasoning ("every call is explained").

Usage:
    poetry run python -m bridge.run_daily            # trade
    poetry run python -m bridge.run_daily --dry-run  # decide, print, no orders

Quarantine: bridge/alpaca.py refuses non-paper endpoints. This experiment is
isolated from the autopilot books entirely (user directive 2026-07-10;
eval 2026-08-10 before any fold-in).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

UNIVERSE = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "JPM", "XOM", "LLY"]
AGENTS = ["buffett", "damodaran", "munger", "burry", "wood", "lynch", "graham", "pead"]
MAX_WEIGHT = 0.10          # per-name cap, long or short
GROSS_CAP = 1.0            # total |weights| <= 100% of equity
MIN_TRADE_USD = 200        # ignore rebalance dust
LEDGER = Path(__file__).parent / "ledger.jsonl"


def composite(signals: list[float]) -> float:
    live = [s for s in signals if s != 0.0]
    return sum(live) / len(live) if live else 0.0


def target_weights(convictions: dict[str, float]) -> dict[str, float]:
    """Conviction-proportional weights, per-name cap, gross cap. Pure fn."""
    raw = {t: v for t, v in convictions.items() if v != 0.0}
    if not raw:
        return {}
    gross = sum(abs(v) for v in raw.values())
    scale = GROSS_CAP / gross if gross > GROSS_CAP else 1.0
    return {
        t: max(-MAX_WEIGHT, min(MAX_WEIGHT, v * scale))
        for t, v in raw.items()
    }


def rebalance_orders(
    targets: dict[str, float], current_mv: dict[str, float], equity: float
) -> list[dict]:
    """Diff target dollar exposure vs current. Pure fn. Returns order intents."""
    orders = []
    for symbol in sorted(set(targets) | set(current_mv)):
        want = targets.get(symbol, 0.0) * equity
        have = current_mv.get(symbol, 0.0)
        delta = want - have
        if abs(delta) < MIN_TRADE_USD:
            continue
        orders.append({"symbol": symbol, "delta_usd": round(delta, 2),
                       "side": "buy" if delta > 0 else "sell"})
    return orders


def llm_failure_ratio(per_ticker: dict[str, dict]) -> float:
    """Fraction of committee signals that are failures rather than opinions.

    A failed signal is an abstain/error produced by the plumbing, not the
    model's judgment: reasoning starts with 'LLM call failed' (llm_agent
    abstain path) or 'ERROR:' (run_daily per-agent catch). 0.0 on an empty
    committee (nothing to judge — the no-credentials path exits earlier)."""
    total = failed = 0
    for views in per_ticker.values():
        for v in views.values():
            total += 1
            reason = str(v.get("reasoning") or "")
            if reason.startswith("LLM call failed") or reason.startswith("ERROR:"):
                failed += 1
    return (failed / total) if total else 0.0


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from v2.data import CachedDataClient, FDClient
    from v2.signals import ALPHA_MODEL_REGISTRY

    asof = date.today().isoformat()
    per_ticker: dict[str, dict] = {}
    if os.environ.get("UW_TOKEN"):
        from v2.data.home_client import HomeDataClient
        raw_client = HomeDataClient()
        print("data plane: HomeDataClient (Alpaca+UW+FMP)")
    elif os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        raw_client = FDClient()
        print("data plane: financialdatasets.ai")
    else:
        print("BLOCKED: no data credentials (UW_TOKEN or FINANCIAL_DATASETS_API_KEY). Exiting clean.")
        return
    with raw_client as raw:
        fd = CachedDataClient(raw)
        for ticker in UNIVERSE:
            views = {}
            for agent in AGENTS:
                try:
                    sig = ALPHA_MODEL_REGISTRY[agent]().predict(ticker, asof, fd)
                    views[agent] = {"value": sig.value, "reasoning": sig.reasoning}
                except Exception as e:  # one agent failing must not kill the run
                    views[agent] = {"value": 0.0, "reasoning": f"ERROR: {e}"}
            per_ticker[ticker] = views
            print(f"{ticker}: " + " ".join(f"{a}={v['value']:+.2f}" for a, v in views.items()))

    # 2026-07-17 (dead-committee incident): the Anthropic key ran out of
    # credits on 7/16-17 — all 7 LLM personas abstained on every ticker,
    # the run stayed GREEN, and the book rebalanced for 2 days on the PEAD
    # quant alone. Dead personas must HALT rebalancing and fail the run
    # loudly, not dilute silently into neutral.
    fail_ratio = llm_failure_ratio(per_ticker)
    if fail_ratio > 0.5:
        print(f"FATAL: {fail_ratio:.0%} of committee signals are LLM/agent "
              "failures — refusing to rebalance on a dead committee. "
              "Check the ANTHROPIC_API_KEY credit balance / provider status.")
        sys.exit(2)

    convictions = {t: composite([v["value"] for v in views.values()])
                   for t, views in per_ticker.items()}
    targets = target_weights(convictions)

    from bridge.alpaca import AlpacaPaper
    broker = AlpacaPaper()
    acct = broker.account()
    equity = float(acct["equity"])
    current_mv = broker.positions()
    orders = rebalance_orders(targets, current_mv, equity)

    print(f"\nequity=${equity:,.0f} targets={ {t: round(w,3) for t,w in targets.items()} }")
    print(f"orders ({len(orders)}): {orders}")

    placed = []
    if not args.dry_run:
        for o in orders:
            try:
                body = {"notional": abs(o["delta_usd"]), "side": o["side"]}
                if o["side"] == "sell":
                    # Alpaca 422 42210000: fractional (notional) orders cannot
                    # sell short — shorts must be whole-share qty orders.
                    held = current_mv.get(o["symbol"], 0.0)
                    if held <= 0 or abs(o["delta_usd"]) > held:
                        px = broker.latest_price(o["symbol"])
                        if not px:
                            raise RuntimeError("no price for whole-share short sizing")
                        qty = int(abs(o["delta_usd"]) // px)
                        if qty < 1:
                            placed.append({**o, "skipped": "short_below_one_share"})
                            continue
                        body = {"qty": str(qty), "side": "sell"}
                res = broker.submit_market_order(o["symbol"], body)
                placed.append({**o, "order_id": res.get("id"), "status": res.get("status")})
                print(f"  placed {o['side']} ${abs(o['delta_usd'])} {o['symbol']} -> {res.get('status')}")
            except Exception as e:
                placed.append({**o, "error": str(e)})
                print(f"  FAILED {o['symbol']}: {e}")

    with LEDGER.open("a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "asof": asof, "equity": equity,
            "signals": per_ticker, "convictions": convictions,
            "targets": targets, "orders": placed if placed else orders,
            "dry_run": args.dry_run,
        }) + "\n")
    print(f"\nledger appended -> {LEDGER}")


if __name__ == "__main__":
    main()
