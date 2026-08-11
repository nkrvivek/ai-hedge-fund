"""LLM provider protocol + the Anthropic implementation.

Mirrors the DataClient pattern (v2/data/protocol.py): agents depend on the
`LLMClient` protocol, never a concrete provider. Any class with a
`complete(system, user) -> str` method plugs in — community providers welcome.

We deliberately do NOT use langchain's structured-output machinery: its
forced-tool mode breaks on Anthropic reasoning models (v1 carries the same
workaround). We ask for JSON in the prompt and parse it ourselves.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol, runtime_checkable

DEFAULT_MODEL = "claude-sonnet-5"


class LLMParseError(ValueError):
    """The model's response did not contain parseable JSON."""


class LLMTruncatedError(LLMParseError):
    """The model never finished the object — it ran into the token cap, or
    came back empty. Distinct from LLMParseError because the remedy is
    different: a truncated response is a cap problem, not a prompt problem,
    and reading one as the other cost a trading day on 2026-08-11."""


# Appended to the user prompt on the second attempt. The personas already
# ask for 2-4 sentences; a response that overran the cap is one ignoring
# that, so the retry narrows it rather than repeating the same request.
RETRY_SUFFIX = (
    "\n\nYour previous answer did not finish. Reply with the JSON object "
    "alone — no preamble, no code fence — and keep `reasoning` to one "
    "sentence."
)


@runtime_checkable
class LLMClient(Protocol):
    """Protocol all LLM providers must satisfy.

    complete() returns the model's raw text. Providers should raise on
    transport failure — the LLMAgent layer decides to abstain, not the
    provider.
    """

    model: str

    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    """Anthropic provider via the existing langchain-anthropic dependency
    (transport only — no structured-output magic)."""

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 60.0,
        # 1024 was not enough headroom for a persona that runs past the
        # 2-4 sentences its prompt asks for: seven of ~170 calls came back
        # unfinished on 2026-08-11. The retry above covers the rest.
        max_tokens: int = 2048,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in your .env to use LLM agents."
            )
        from langchain_anthropic import ChatAnthropic

        self.model = model or os.getenv("V2_LLM_MODEL", DEFAULT_MODEL)
        self._chat = ChatAnthropic(
            model=self.model,
            api_key=api_key,
            timeout=timeout,
            max_retries=1,
            max_tokens=max_tokens,
        )

    def complete(self, system: str, user: str) -> str:
        text = self._invoke(system, user)
        if text is not None:
            return text
        # Cut off, or empty. Ask once more for the JSON alone before giving
        # up — on 2026-08-11 seven personas hit this and each abstention
        # counted toward the probation gate that blocked the whole day.
        text = self._invoke(system, user + RETRY_SUFFIX)
        if text is None:
            raise LLMTruncatedError(
                f"{self.model} did not finish the response twice "
                "(stop_reason max_tokens, or an empty body)"
            )
        return text

    def _invoke(self, system: str, user: str) -> str | None:
        """One call. Returns the text, or None when it did not finish."""
        result = self._chat.invoke([("system", system), ("human", user)])
        content = result.content
        # Anthropic reasoning models return a list of content blocks.
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        stop_reason = getattr(result, "response_metadata", {}).get("stop_reason")
        if stop_reason == "max_tokens" or not content.strip():
            return None
        return content


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response.

    Tries: ```json fence -> whole string -> first balanced {...} block.
    Raises LLMParseError if nothing parses.
    """
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise LLMParseError(f"no JSON object found in response: {text[:200]!r}")
