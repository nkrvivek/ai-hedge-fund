# AGENTS.md — ai-hedge-fund

Pointer into the common stack card. Canonical home:
`~/Library/Mobile Documents/iCloud~md~obsidian/Documents/obsidian/wiki/trading/stack-card.md`
(Obsidian: [[wiki/trading/stack-card]]). Read that first.

## This repo, in one screen

- **Role:** 7-persona committee book (~$97K). Daily digest email
  `[ai-hedge-fund]` from hello@sibt.ai around 9:10 AM PT — its absence or a
  🟡/🔴 status is a finding to surface. Dead-committee incident 7/16–17 is why.
- **Probation (DJ-20260810-05, review 2026-09-10):** month-1 eval read −2.64%
  vs SPY +2.43% w/ hit rate 45.9%. Extended one month, learning-only, three
  conditions: any committee failure = no trades that day
  (`probation_no_trade_reason`), daily turnover capped at 25% of equity
  (`cap_churn`), and the scored hit rate must clear 0.50 by the review or the
  book dies. Guarded by `bridge/test_probation.py`.
- **GitHub:** nkrvivek / **SSH** always (`git@github.com:nkrvivek/...`).
- **Deploy:** `.github/workflows/bridge-daily.yml` — **cron `35 14 * * 1-5`
  (10:35 ET) plus manual `workflow_dispatch`. NOT push-triggered.** The job
  runs the bridge and commits the day's ledger back to the repo (Poetry-based).
- **Ledger:** today a 1-row-per-day `bridge/ledger.jsonl` carrying an `orders`
  list (`symbol`, `delta_usd`, `side`, `order_id`, `status`). Target: per-fill
  `bridge/trades.jsonl` in the canonical shape. Daily rows hold submitted
  orders, not fill qty/price — true fills need Alpaca confirmations.
- **Secrets:** Alpaca keys + FMP/FD in `.env` and GitHub Actions secrets.
  Never commit, echo, or log.
- **Gates:** NO MARGIN DEBIT · income-tilted, no naked options, 10% single-name
  max on the bildof side.
