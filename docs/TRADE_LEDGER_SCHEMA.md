# Trade Ledger — canonical record schema (v1)

Every book records each fill in this shape. The shape is shared; the data is
not. Each book writes its own ledger to its own store and never reads another
book's. Consistency comes from the format, not from a shared file.

Books that use this schema, each in its own store:

| Book | Store (this book only) |
|---|---|
| autopilot $5K live | R2 `autopilot-state` → `ledger/{YYYY-MM-DD}/trades.jsonl` |
| autopilot $200K paper | R2 `autopilot-paper-state` → `ledger/{YYYY-MM-DD}/trades.jsonl` |
| ai-hedge-fund (~$98K) | repo `bridge/trades.jsonl` |
| sibt.ai | Supabase `paper_fills` (+ canonical columns) |

This file is copied into each repo as-is. No repo imports it from another.

## Record shape

One JSON object per fill, append-only. Fields absent from a book are `null` —
never a fabricated value.

```jsonc
{
  "schema_version": 1,
  "record_id": "autopilot_5k_live-20260729-0007", // unique + ordered WITHIN this book
  "book": "autopilot_5k_live",       // fixed per book; enum below
  "account": {"broker": "public", "account_id": "…"},
  "ts_utc": "2026-07-29T20:31:05Z",  // RFC3339, always Z
  "ts_pt":  "2026-07-29T13:31:05-07:00", // America/Los_Angeles, same instant
  "kind": "open",                    // open|close|roll|fill|expiry|assignment
  "instrument": {
    "asset_class": "OPTION",         // OPTION|EQUITY
    "symbol": "SPY260828P00701000",  // OCC compact for options, ticker for equity
    "underlying": "SPY",
    "right": "P",                    // C|P|null (equity)
    "strike": 701.0,                 // null for equity
    "expiry": "2026-08-28"           // null for equity
  },
  "side": "buy_to_open",             // buy_to_open|sell_to_open|buy_to_close|sell_to_close|buy|sell
  "qty": 2,                          // contracts (options) or shares (equity), always positive
  "price": 3.00,                     // per-contract / per-share fill price
  "mult": 100,                       // 100 options, 1 equity
  "gross_usd": -600.00,              // signed cash effect: debit negative, credit positive
  "fees_usd": 0.0,
  "sleeve": "hedge",                 // wheel|banger|directional|micro|pm_momentum|earnings_pead|hedge|committee|…
  "thesis_ref": null,                // vault wikilink or null
  "proposal_id": null,               // upstream proposal/order id if any
  "broker_order_id": "…",
  "realized_pnl_usd": null,          // set on close/expiry/assignment; null on open
  "dj_ref": null,                    // "DJ-YYYYMMDD-NN" link to the decision-journal, or null
  "source": "runner.py:_record_trade" // emitter provenance
}
```

## `book` enum (never shared across stores)

`autopilot_5k_live` · `autopilot_200k_paper` · `ai_hedge_fund` · `sibt_paper` ·
`sibt_live`. A book writes only its own value.

## Rules

- **Append-only.** Never edit or delete a row. A correction is a new row
  (`kind` reflects it; `gross_usd` nets out).
- **Per-fill.** One row per fill, not per day and not per order. A multi-leg
  order emits one row per leg fill.
- **Both timestamps.** `ts_utc` and `ts_pt` carry the same instant. Surfacing
  code shows PT first (house rule).
- **No fabricated numbers.** A field the book can't source is `null`.
- **`record_id` is ordered within the book** so rows sort deterministically;
  it carries no meaning across books.
- **Signed `gross_usd`.** Cash leaving the account is negative. Lets a book sum
  its own ledger to a cash-flow total without re-deriving sign from `side`.

## Autopilot emitter (this repo)

`trade_ledger.py` builds every row; `AuditLog.record_trade` writes it (local
`trades.jsonl` + R2 `ledger/{day}/trades.jsonl`), fully guarded so a ledger
write never breaks a fill. Three fill paths feed it:

- **`records_for_fill`** — the 6-sleeve `_apply_fill_to_state` choke point
  (wheel / banger / directional / micro / pm_momentum / earnings_pead), opens
  and closes. Only the wheel sells premium to open (`sell_to_open`); every long
  sleeve buys to open. When the broker returns no synchronous `fill_price` (Public
  and TS rest the order), the row falls back to the booked mid/bid the state
  itself used and tags `source` with `:booked_est` — an estimate, flagged as one.
- **`record_for_hedge_leg`** — one row per confirmed leg of the SPY hedge, at
  open-confirm (full or partial), keyed by the leg's stamped `entry_px`.
- **`record_for_bracket_close`** — bracket exits, which bypass
  `_apply_fill_to_state`; carries the realized P&L the close computed.

**Rolls** emit one row (`kind: roll`) against the resulting leg, priced at the
net credit — the real cash moved. Per-leg close/open prices are not split from a
net spread, so they are not invented.

**Known gap:** the partial-roll-close repair path
(`_apply_partial_roll_close`) does not emit yet; those fills still land in
`audit.jsonl` and are backfillable. Same for any exit the bracket path misses.

## Backfill

Best-effort, per book, from whatever that book already persisted (autopilot
`audit.jsonl`, ai-hedge-fund `ledger.jsonl` rows, sibt `paper_fills`). Backfilled
rows set `source` to the projector that built them and leave unknown fields
`null`. Gaps stay gaps — no invented history.
