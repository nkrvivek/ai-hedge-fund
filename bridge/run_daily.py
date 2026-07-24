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
FAIL_THRESHOLD = 0.5       # committee failure ratio that halts a run (global) or excludes a ticker
LEDGER = Path(__file__).parent / "ledger.jsonl"


def composite(signals: list[float]) -> float:
    live = [s for s in signals if s != 0.0]
    return sum(live) / len(live) if live else 0.0


def target_weights(convictions: dict[str, float]) -> dict[str, float]:
    """Conviction-proportional LONG-ONLY weights, per-name cap, gross cap.
    Pure fn.

    2026-07-23: the Alpaca paper account refuses shorts (POST /orders 403
    code 40310000 "account is not allowed to short"), so every bearish
    target errored daily and the ledger diverged from the real book.
    Negative conviction now maps to NO position — sell to zero, hold cash —
    the closest expressible portfolio inside the account's limits. The
    conviction table in the ledger/email still shows the raw negative
    values, so the committee's bearish view stays visible."""
    raw = {t: v for t, v in convictions.items() if v > 0.0}
    if not raw:
        return {}
    gross = sum(raw.values())
    scale = GROSS_CAP / gross if gross > GROSS_CAP else 1.0
    return {t: min(MAX_WEIGHT, v * scale) for t, v in raw.items()}


def rebalance_orders(
    targets: dict[str, float], current_mv: dict[str, float], equity: float,
    excluded: frozenset[str] = frozenset(),
) -> list[dict]:
    """Diff target dollar exposure vs current. Pure fn. Returns order intents.

    `excluded` tickers (dead committee — see ticker_failure_ratios) are held
    exactly as-is: no buy, no sell, regardless of target or current value.
    A data outage is not a reason to force-close a position."""
    orders = []
    for symbol in sorted(set(targets) | set(current_mv)):
        if symbol in excluded:
            continue
        want = targets.get(symbol, 0.0) * equity
        have = current_mv.get(symbol, 0.0)
        delta = want - have
        if abs(delta) < MIN_TRADE_USD:
            continue
        orders.append({"symbol": symbol, "delta_usd": round(delta, 2),
                       "side": "buy" if delta > 0 else "sell"})
    return orders


def _is_failed_signal(v: dict) -> bool:
    """A signal counts as a committee failure: plumbing, not judgment.

    2026-07-22 audit: the abstain path (v2/signals/llm_agent.py
    `_abstain()`) writes reasoning prefixed 'abstained: ...', not
    'LLM call failed' — the old string-match here missed every abstain
    and undercounted a dead committee (87.5% true failure read as
    8.75%). Read the `abstained` metadata flag directly instead; still
    count the run_daily per-agent except catch ('ERROR:' reasoning,
    v.get('abstained') is never set there)."""
    return bool(v.get("abstained")) or str(v.get("reasoning") or "").startswith("ERROR:")


def llm_failure_ratio(per_ticker: dict[str, dict]) -> float:
    """Fraction of committee signals (all tickers, all agents) that are
    failures rather than opinions. 0.0 on an empty committee (nothing to
    judge — the no-credentials path exits earlier)."""
    total = failed = 0
    for views in per_ticker.values():
        for v in views.values():
            total += 1
            if _is_failed_signal(v):
                failed += 1
    return (failed / total) if total else 0.0


def ticker_failure_ratios(per_ticker: dict[str, dict]) -> dict[str, float]:
    """Per-ticker committee failure ratio, same rule as llm_failure_ratio.

    2026-07-22 audit: LLY's committee was 100% dead (FMP 402) the same day
    the global ratio read 8.75% — a single dead endpoint for one ticker
    hides inside a healthy-looking global average. Callers exclude any
    ticker at/above FAIL_THRESHOLD from target_weights."""
    ratios = {}
    for ticker, views in per_ticker.items():
        total = len(views)
        failed = sum(1 for v in views.values() if _is_failed_signal(v))
        ratios[ticker] = (failed / total) if total else 0.0
    return ratios


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
                    views[agent] = {"value": sig.value, "reasoning": sig.reasoning,
                                    "abstained": bool(sig.metadata.get("abstained", False))}
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
    if fail_ratio >= FAIL_THRESHOLD:
        msg = (f"FATAL: {fail_ratio:.0%} of committee signals are LLM/agent "
               "failures — refusing to rebalance on a dead committee. "
               "Check the ANTHROPIC_API_KEY credit balance / provider status.")
        print(msg)
        try:
            _send_daily_email(asof=asof, equity=0.0, convictions={},
                              targets={}, placed=[],
                              fail_ratio=fail_ratio, dry_run=True,
                              halt_reason=msg)
        except Exception:  # noqa: BLE001
            pass
        sys.exit(2)

    # 2026-07-22 (per-ticker audit finding): a single ticker's committee can
    # be wiped out (e.g. LLY, FMP 402) while the global ratio stays low.
    # Exclude that ticker from target_weights entirely — no new position —
    # and hold whatever is already there untouched (no forced close on a
    # data outage). The global halt above still fires on top of this.
    ticker_ratios = ticker_failure_ratios(per_ticker)
    excluded = {t: r for t, r in ticker_ratios.items() if r >= FAIL_THRESHOLD}
    if excluded:
        print("excluded (dead committee, held as-is): "
              + ", ".join(f"{t} {r:.0%}" for t, r in sorted(excluded.items())))

    convictions = {t: composite([v["value"] for v in views.values()])
                   for t, views in per_ticker.items() if t not in excluded}
    targets = target_weights(convictions)

    from bridge.alpaca import AlpacaPaper
    broker = AlpacaPaper()
    acct = broker.account()
    equity = float(acct["equity"])
    current_mv = broker.positions()
    orders = rebalance_orders(targets, current_mv, equity, excluded=frozenset(excluded))

    print(f"\nequity=${equity:,.0f} targets={ {t: round(w,3) for t,w in targets.items()} }")
    print(f"orders ({len(orders)}): {orders}")

    placed = []
    if not args.dry_run:
        for o in orders:
            try:
                body = {"notional": abs(o["delta_usd"]), "side": o["side"]}
                if o["side"] == "sell":
                    # Long-only account (403 40310000): sells only reduce a
                    # held long, never open a short. Full exits go through
                    # the broker-sized close-position endpoint so a stale
                    # local MV can't oversell (2026-07-23 META
                    # insufficient-qty class).
                    held = current_mv.get(o["symbol"], 0.0)
                    if held <= 0:
                        placed.append({**o, "skipped": "nothing_held_long_only"})
                        continue
                    if targets.get(o["symbol"], 0.0) <= 0.0 or abs(o["delta_usd"]) >= held:
                        res = broker.close_position(o["symbol"]) or {}
                        placed.append({**o, "order_id": res.get("id"),
                                       "status": res.get("status", "closed"),
                                       "liquidated": True})
                        print(f"  liquidated {o['symbol']} (long-only full exit) -> {res.get('status', 'closed')}")
                        continue
                    body = {"notional": round(min(abs(o["delta_usd"]), held), 2), "side": "sell"}
                res = broker.submit_market_order(o["symbol"], body)
                placed.append({**o, "order_id": res.get("id"), "status": res.get("status")})
                print(f"  placed {o['side']} ${abs(o['delta_usd'])} {o['symbol']} -> {res.get('status')}")
            except Exception as e:
                placed.append({**o, "error": str(e)})
                print(f"  FAILED {o['symbol']}: {e}")

    # Index-hedge sleeve (INDEX_HEDGE_RULE.md, approved 2026-07-24): the only
    # place bearish net conviction becomes a position — long XSP puts,
    # defined risk. Runs after the stock rebalance; never raises. Ledger
    # attribution sleeve="index_hedge" keeps the 8/10 eval clean.
    from bridge.index_hedge import run_index_hedge
    hedge = run_index_hedge(broker, convictions=convictions, equity=equity,
                            dry_run=args.dry_run)
    print(f"index_hedge: {hedge.get('action')} — {hedge.get('reason', '')}")

    with LEDGER.open("a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "asof": asof, "equity": equity,
            "signals": per_ticker, "convictions": convictions,
            "targets": targets, "orders": placed if placed else orders,
            "excluded": excluded,
            "long_only": True,
            "index_hedge": hedge,
            "dry_run": args.dry_run,
        }) + "\n")
    print(f"\nledger appended -> {LEDGER}")

    _send_daily_email(asof=asof, equity=equity, convictions=convictions,
                      targets=targets, placed=placed if placed else orders,
                      fail_ratio=fail_ratio, dry_run=args.dry_run,
                      excluded=excluded, hedge=hedge)


def _hedge_email_line(hedge: dict | None) -> str:
    """One line splitting the hedge from the picks — a trough print must read
    as volatility, not committee failure (2026-07-23 lesson, $5K book)."""
    if not hedge:
        return ""
    puts = hedge.get("open_puts") or []
    if puts:
        cost = sum(abs(float(p.get("cost_basis") or 0)) for p in puts)
        mark = sum(abs(float(p.get("market_value") or 0)) for p in puts)
        detail = (f"{len(puts)} XSP put leg(s) · mark ${mark:,.0f} "
                  f"(cost ${cost:,.0f}, unrealized {mark - cost:+,.0f})")
    else:
        detail = "no open hedge"
    return (f"<p style='margin:4px 0;font-size:13px'><b>Index hedge</b> "
            f"(sleeve, excluded from picks eval): {detail} · today: "
            f"{hedge.get('action')} — {hedge.get('reason', '')}</p>")


def _send_daily_email(*, asof: str, equity: float, convictions: dict,
                      targets: dict, placed: list, fail_ratio: float,
                      dry_run: bool, excluded: dict[str, float] | None = None,
                      halt_reason: str | None = None,
                      hedge: dict | None = None) -> None:
    """Daily digest via Resend (2026-07-17 user directive: 'I'm not getting
    any email for AI hedge fund, enable that'). Best-effort — an email
    failure never fails the run. Requires RESEND_API_KEY/FROM/TO env
    (GH secrets on the fork; absent locally → silent skip).

    `halt_reason` set (global gate tripped, bridge/run_daily.py FATAL path)
    sends a bare HALTED notice instead of the normal digest table — there
    is no digest to show, only the reason no trades happened."""
    import urllib.request
    key = os.getenv("RESEND_API_KEY")
    frm = os.getenv("RESEND_FROM")
    to = os.getenv("RESEND_TO")
    if not (key and frm and to):
        print("email skip: RESEND_* not configured")
        return
    excluded = excluded or {}

    if halt_reason:
        subject = f"[ai-hedge-fund] {asof} — HALTED — no trades: {halt_reason}"
        html = (
            f"<div style='font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif'>"
            f"<h3 style='margin:0 0 8px'>[ai-hedge-fund] {asof} — HALTED — no trades</h3>"
            f"<p style='margin:4px 0'>🔴 {halt_reason}</p>"
            f"</div>")
    else:
        ranked = sorted(convictions.items(), key=lambda kv: -kv[1])
        conv_lines = "".join(
            f"<tr><td style='padding:2px 10px'>{t}</td>"
            f"<td style='padding:2px 10px;text-align:right'>{c:+.2f}</td>"
            f"<td style='padding:2px 10px;text-align:right'>{targets.get(t, 0):.1%}</td></tr>"
            for t, c in ranked)
        order_lines = "".join(
            f"<li>{o.get('side','?')} ${abs(o.get('delta_usd', 0)):,.0f} "
            f"{o.get('symbol','?')} — {o.get('status') or o.get('error') or o.get('skipped') or 'staged'}</li>"
            for o in placed) or "<li>no rebalance orders</li>"
        if excluded:
            health = ("🔴 excluded (dead committee, held as-is): "
                      + ", ".join(f"{t} {r:.0%} failed" for t, r in sorted(excluded.items())))
        elif fail_ratio >= FAIL_THRESHOLD:
            health = f"🔴 {fail_ratio:.0%} committee signals failed"
        elif fail_ratio > 0:
            health = f"🟡 {fail_ratio:.0%} committee signals failed"
        else:
            health = "🟢 committee healthy"
        html = (
            f"<div style='font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif'>"
            f"<h3 style='margin:0 0 8px'>[ai-hedge-fund] {asof} — equity ${equity:,.0f}"
            f"{' (DRY RUN)' if dry_run else ''}</h3>"
            f"<p style='margin:4px 0'>{health}</p>"
            f"<table style='border-collapse:collapse;font-size:13px'>"
            f"<tr><th style='padding:2px 10px;text-align:left'>ticker</th>"
            f"<th style='padding:2px 10px'>conviction</th>"
            f"<th style='padding:2px 10px'>target</th></tr>{conv_lines}</table>"
            f"<p style='margin:8px 0 4px'><b>Orders</b></p><ul style='margin:0;font-size:13px'>{order_lines}</ul>"
            f"{_hedge_email_line(hedge)}"
            f"</div>")
        subject = f"[ai-hedge-fund] daily — equity ${equity:,.0f} · {asof}"

    payload = json.dumps({"from": frm,
                          "to": [t.strip() for t in to.split(",") if t.strip()],
                          "subject": subject,
                          "html": html}).encode()
    # Cloudflare bot-fight 403s python-urllib's default UA (error 1010) —
    # same gotcha as the 2026-07-15 cloud IBKR screen port. Real UA required.
    req = urllib.request.Request("https://api.resend.com/emails", data=payload,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json",
                                          "User-Agent": "ai-hedge-fund-bridge/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print(f"email sent: {r.status}")
    except Exception as e:  # noqa: BLE001
        print(f"email FAILED (non-fatal): {e}")


if __name__ == "__main__":
    main()
