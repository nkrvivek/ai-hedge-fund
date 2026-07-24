# Index-hedge sleeve — XSP puts (v1, approved 2026-07-24)

Status: APPROVED + WIRED (bridge/index_hedge.py, runs in run_daily after the
stock rebalance). Start decision (user, 2026-07-24): live now, but the arm
threshold is −5.0 until the 8/10 eval ends; the −1.5 rule below applies
after 8/10. Reference: 7/23 read −3.75 does NOT arm before 8/10.

## Why

The committee's bearish views currently express as "sell to zero, hold
cash" (long-only clamp, `0b9b056`) because the Alpaca paper account is a
cash account — `shorting_enabled: False`, no API flip. The account DOES
carry `options_trading_level: 3`, and Alpaca added index options (XSP,
SPX, VIX…) to paper in July 2026. XSP is cash-settled, European-style,
1/10th SPX — no assignment paths, defined risk. Long XSP puts are the
cleanest expressible short.

## Signal

`net_conviction` = sum of per-ticker composite convictions across the
scored universe, AFTER dead-committee exclusions (same inputs
`target_weights` already receives). Range ≈ −N..+N for N tickers.

Reference point: 2026-07-23 read = −3.75 across 9 names (8 bearish).

## Rule

- **Arm** when `net_conviction ≤ −1.5` on a rebalance day.
- **Stand down** (close hedge, and do not renew) when `net_conviction ≥ −0.5`.
  The −1.5/−0.5 gap is hysteresis — no flapping on small daily drift.
- **Size**: 1 XSP put per full −2.0 of net conviction, capped at 2
  contracts. (−3.75 today → 1 contract.)
- **Contract**: 30-45 DTE, 3-5% OTM, LIMIT at mid.
- **Premium caps**: single tranche ≤ 1.5% of equity (~$1.5K today); total
  open hedge premium ≤ 3% of equity. Max loss = premium paid, always.
- **Liquidity guard**: skip if bid/ask spread > 10% of mid.
- **Time exit**: close at 21 DTE regardless of signal (no riding decay
  into expiry); re-enter next rebalance if the signal still arms.
- **No rolls** — every entry is a fresh signal-day decision.

## Accounting

- Ledger rows carry `sleeve: "index_hedge"` — excluded from the
  stock-picking eval metrics (the 8/10 fold-in decision judges the
  committee's picks, not the hedge).
- Daily email splits the hedge like the autopilot PM brief now does:
  mark, cost, unrealized, % of equity — a trough print must read as
  volatility, not committee failure (2026-07-23 lesson, $5K book).

## Start decision (resolved 2026-07-24)

User picked the escape hatch as the rule: live immediately, arm only if
net conviction drops below −5.0 before the 8/10 eval ends (the unexpressed
view becomes the bigger distortion at that depth); normal −1.5 arm after
8/10. Implemented as `PRE_EVAL_ARM_THRESHOLD` / `EVAL_END` in
bridge/index_hedge.py.
