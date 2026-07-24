"""Index-hedge sleeve — XSP puts expressing the committee's net-bearish view.

Rule: bridge/INDEX_HEDGE_RULE.md, approved 2026-07-24. Start decision (user,
2026-07-24): live immediately, but until the 8/10 eval ends the arm threshold
is −5.0 net conviction; after 8/10 the rule's normal −1.5 applies. The
long-only clamp (`target_weights`) stays untouched — this sleeve is the only
place bearish conviction becomes a position, always as long puts (max loss =
premium paid).

Ledger rows carry sleeve="index_hedge" so the 8/10 stock-picking eval can
exclude them.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Callable

ARM_THRESHOLD = -1.5           # rule-doc arm level (post-eval)
PRE_EVAL_ARM_THRESHOLD = -5.0  # user decision 2026-07-24: gate until eval end
EVAL_END = date(2026, 8, 10)
STAND_DOWN = -0.5              # hysteresis: close at/above this
SIZE_PER_CONVICTION = 2.0      # 1 contract per full −2.0 net conviction
MAX_CONTRACTS = 2
DTE_MIN, DTE_MAX = 30, 45
OTM_MIN, OTM_MAX = 0.03, 0.05
TRANCHE_PREMIUM_CAP = 0.015    # single entry ≤ 1.5% of equity
TOTAL_PREMIUM_CAP = 0.03       # all open hedge premium ≤ 3% of equity
MAX_SPREAD_FRAC = 0.10         # skip if bid/ask spread > 10% of mid
TIME_EXIT_DTE = 21
UNDERLYING = "XSP"

_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def net_conviction(convictions: dict[str, float]) -> float:
    """Sum of composite convictions across the scored universe (both signs)."""
    return sum(convictions.values()) if convictions else 0.0


def parse_occ(symbol: str) -> tuple[str, date, str, float]:
    """OCC compact symbol -> (root, expiry, right, strike)."""
    m = _OCC_RE.match(symbol.strip())
    if not m:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    root, ymd, right, strike = m.groups()
    expiry = datetime.strptime(ymd, "%y%m%d").date()
    return root, expiry, right, int(strike) / 1000.0


def arm_threshold(today: date) -> float:
    return PRE_EVAL_ARM_THRESHOLD if today <= EVAL_END else ARM_THRESHOLD


def contracts_within_tranche_cap(contracts: int, *, mid: float, equity: float) -> int:
    """Trim contract count until premium fits the single-tranche cap."""
    cap = TRANCHE_PREMIUM_CAP * equity
    while contracts > 0 and contracts * mid * 100 > cap:
        contracts -= 1
    return contracts


def within_total_premium_cap(open_puts: list[dict], *, new_premium: float,
                             equity: float) -> bool:
    open_premium = sum(abs(float(p.get("cost_basis") or 0.0)) for p in open_puts)
    return open_premium + new_premium <= TOTAL_PREMIUM_CAP * equity


def hedge_action(net_conv: float, today: date, *, open_puts: list[dict],
                 equity: float) -> dict[str, Any]:
    """Pure decision: what the hedge sleeve does today.

    Returns {"action": "none"|"open"|"close"|"hold", ...}. "open" carries the
    signal-sized contract count BEFORE premium caps (caps need a live quote,
    applied by the caller via contracts_within_tranche_cap /
    within_total_premium_cap).
    """
    # Time exit first: never ride decay inside TIME_EXIT_DTE, signal or not.
    expiring = [p["symbol"] for p in open_puts
                if (parse_occ(p["symbol"])[1] - today).days <= TIME_EXIT_DTE]
    if expiring:
        return {"action": "close", "symbols": expiring,
                "reason": f"dte<={TIME_EXIT_DTE}"}

    if open_puts:
        if net_conv >= STAND_DOWN:
            return {"action": "close",
                    "symbols": [p["symbol"] for p in open_puts],
                    "reason": f"stand_down net_conviction={net_conv:+.2f}"}
        # No rolls, no adds — every entry is a fresh signal-day decision on a
        # flat book. Existing hedge just rides until stand-down or time exit.
        return {"action": "hold", "reason": f"hedge open, net_conviction={net_conv:+.2f}"}

    thr = arm_threshold(today)
    armed = net_conv < thr if today <= EVAL_END else net_conv <= thr
    if not armed:
        return {"action": "none",
                "reason": f"net_conviction={net_conv:+.2f} vs arm {thr:+.1f}"}
    contracts = min(MAX_CONTRACTS, max(1, int(abs(net_conv) / SIZE_PER_CONVICTION)))
    return {"action": "open", "contracts": contracts,
            "reason": f"net_conviction={net_conv:+.2f} armed (thr {thr:+.1f})"}


def select_contract(contracts: list[dict], *, spot: float, today: date,
                    quote_fn: Callable[[str], tuple[float, float] | None],
                    ) -> dict[str, Any] | None:
    """Pick the rule's contract: 30-45 DTE, 3-5% OTM put, liquid quote.

    Prefers the strike closest to 4% OTM among eligible candidates. Returns
    {"symbol", "mid", "bid", "ask"} or None when nothing qualifies.
    """
    lo, hi = spot * (1 - OTM_MAX), spot * (1 - OTM_MIN)
    best: dict[str, Any] | None = None
    best_dist = float("inf")
    target = spot * (1 - (OTM_MIN + OTM_MAX) / 2)
    for c in contracts:
        strike = float(c["strike_price"])
        expiry = date.fromisoformat(c["expiration_date"])
        dte = (expiry - today).days
        if not (DTE_MIN <= dte <= DTE_MAX and lo <= strike <= hi):
            continue
        q = quote_fn(c["symbol"])
        if not q:
            continue
        bid, ask = q
        mid = (bid + ask) / 2
        if mid <= 0 or bid <= 0 or (ask - bid) > MAX_SPREAD_FRAC * mid:
            continue
        if abs(strike - target) < best_dist:
            best_dist = abs(strike - target)
            best = {"symbol": c["symbol"], "mid": round(mid, 2),
                    "bid": bid, "ask": ask}
    return best


def run_index_hedge(broker: Any, *, convictions: dict[str, float],
                    equity: float, today: date | None = None,
                    dry_run: bool = False) -> dict[str, Any]:
    """Orchestrate one hedge decision against the live paper account.

    Never raises — the hedge sleeve must not break the stock rebalance run.
    Returns the ledger record for the day's hedge activity.
    """
    today = today or date.today()
    record: dict[str, Any] = {"sleeve": "index_hedge",
                              "net_conviction": round(net_conviction(convictions), 3)}
    try:
        open_puts = [p for p in broker.positions_full()
                     if p["symbol"].startswith(UNDERLYING)
                     and _OCC_RE.match(p["symbol"])]
        record["open_puts"] = [
            {"symbol": p["symbol"], "qty": p.get("qty"),
             "cost_basis": p.get("cost_basis"),
             "market_value": p.get("market_value")} for p in open_puts]

        act = hedge_action(record["net_conviction"], today,
                           open_puts=open_puts, equity=equity)
        record.update(act)

        if act["action"] == "close" and not dry_run:
            record["closed"] = []
            for sym in act["symbols"]:
                try:
                    res = broker.close_position(sym) or {}
                    record["closed"].append({"symbol": sym,
                                             "status": res.get("status", "closed")})
                except Exception as e:  # noqa: BLE001
                    record["closed"].append({"symbol": sym, "error": str(e)})
            return record

        if act["action"] != "open":
            return record

        # XSP spot proxy: SPY tracks SPX/10 (≈XSP) within ~0.5% — good enough
        # for picking a 3-5% OTM band. Fail-closed if unavailable.
        spot = broker.latest_price("SPY")
        if not spot:
            record.update(action="none", reason="no spot for XSP (SPY proxy failed)")
            return record
        chain = broker.option_contracts(
            UNDERLYING, type_="put", today=today,
            dte_min=DTE_MIN, dte_max=DTE_MAX,
            strike_min=spot * (1 - OTM_MAX) - 1, strike_max=spot * (1 - OTM_MIN) + 1)
        pick = select_contract(chain, spot=spot, today=today,
                               quote_fn=broker.option_quote_latest)
        if not pick:
            record.update(action="none", reason="no eligible contract (dte/otm/liquidity)")
            return record

        n = contracts_within_tranche_cap(act["contracts"], mid=pick["mid"], equity=equity)
        if n == 0:
            record.update(action="none", reason=f"tranche premium cap (mid {pick['mid']})")
            return record
        if not within_total_premium_cap(open_puts, new_premium=n * pick["mid"] * 100,
                                        equity=equity):
            record.update(action="none", reason="total premium cap")
            return record

        record.update(contracts=n, contract=pick)
        if dry_run:
            record["status"] = "dry_run"
            return record
        res = broker.submit_limit_order(pick["symbol"], qty=n, side="buy",
                                        limit_price=pick["mid"])
        record["order_id"] = res.get("id")
        record["status"] = res.get("status")
        return record
    except Exception as e:  # noqa: BLE001
        record.update(action="error", error=str(e))
        return record
