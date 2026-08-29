# AI Hedge Fund Cloudflare dispatcher

This Worker replaces GitHub Actions' drifting `schedule` event. Cloudflare runs
at 14:35 UTC Monday-Friday and sends a `workflow_dispatch` request for
`bridge-daily.yml` on the fork's default branch, `alpaca-bridge`.

Cloudflare weekday numbering is `1=Sunday`, so the Wrangler cron is
`35 14 * * 2-6` for Monday-Friday. That is 10:35 AM EDT or 9:35 AM EST.

`GITHUB_TOKEN` is a Worker secret. The initial value comes from the authenticated
`gh` CLI token; replace it with a fine-grained PAT limited to Actions write on
`nkrvivek/ai-hedge-fund` when convenient.
