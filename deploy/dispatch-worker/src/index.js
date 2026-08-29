const WORKFLOW_DISPATCH_URL =
  "https://api.github.com/repos/nkrvivek/ai-hedge-fund/actions/workflows/bridge-daily.yml/dispatches";

async function dispatchWorkflow(env) {
  if (!env.GITHUB_TOKEN) {
    console.error("ai-hedge-fund dispatch failed: GITHUB_TOKEN is not configured");
    return;
  }

  const response = await fetch(WORKFLOW_DISPATCH_URL, {
    method: "POST",
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      "Content-Type": "application/json",
      "User-Agent": "ai-hedge-fund-cloudflare-dispatcher",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    body: JSON.stringify({ ref: "alpaca-bridge" }),
  });

  if (!response.ok) {
    const detail = (await response.text()).slice(0, 500);
    console.error(
      `ai-hedge-fund dispatch failed: GitHub HTTP ${response.status}: ${detail}`,
    );
    return;
  }

  console.log("bridge-daily workflow_dispatch accepted by GitHub");
}

export default {
  async scheduled(_controller, env, ctx) {
    ctx.waitUntil(dispatchWorkflow(env));
  },

  async fetch() {
    return Response.json({ ok: true, service: "ai-hedge-fund-dispatch" });
  },
};
