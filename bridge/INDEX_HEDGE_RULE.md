# Index-hedge sleeve — XSP puts (DRAFT v1, 2026-07-23)

Status: DRAFT — user review pending. No code exists for this yet; nothing
trades until the user approves the rule AND the start date.

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

## Open decision (user)

**Start date**: ① now — hedge live during the eval window, ledger-separated
so the eval stays clean on paper; ② after the 8/10 eval — zero
contamination risk, but the committee's strongest bearish read to date
goes unexpressed for 2.5 more weeks. Drafter's lean: ② unless the
committee's net conviction deepens below −5, which would make the
unexpressed view itself the bigger eval distortion.
