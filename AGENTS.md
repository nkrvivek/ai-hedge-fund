# AGENTS.md — ai-hedge-fund

Pointer into the common stack card. Canonical home:
`~/Library/Mobile Documents/iCloud~md~obsidian/Documents/obsidian/wiki/trading/stack-card.md`
(Obsidian: [[wiki/trading/stack-card]]). Read that first.

## RETIRED 2026-09-11

Probation review (DJ-20260810-05, due 2026-09-10) never cleared the 0.50 hit-rate
bar — last scored read (2026-09-11 `bridge/learning_log.jsonl`) was hit rate
0.476 across 698 scored picks over 42 matured days, equity $92,542.16 vs the
$100K start (last ledger row before retirement, 2026-09-11 — a −7.46% total
return since inception on 2026-07-10, across 47 traded days and 284 orders
placed). No continuously-computed SPY comparison exists in the ledger for
this extended-probation window (only the original month-1 eval, −2.64% vs
SPY +2.43%, is recorded, in code comments/tests). User directive 2026-09-11:
"we can retire ai-hedge-fund book fully." Retirement actions taken the same day: Cloudflare
dispatch worker `ai-hedge-fund-dispatch` redeployed with `crons = []` (worker
kept for `/status` parity, same pattern as the hackathon book); GitHub
workflow `bridge-daily.yml` disabled (`gh workflow disable bridge-daily`);
Alpaca PAPER account PA3GFFYS3PYL flattened via bulk `DELETE /v2/positions`
(5 of 6 legs queued for Monday's open since the market was closed Saturday;
the XSP put option leg 422'd — "options market orders only allowed during
market hours" — and needs a manual close during market hours). A `retired`
event row is appended to `bridge/ledger.jsonl`. History below kept as-is.

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
