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
import logging
import os
import re
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"

# Anthropic's ordinary endings. `max_tokens` is handled above as a failure;
# anything outside this set is unexpected and gets logged. A provider that
# reports nothing at all (None) counts as normal — the 2026-08-11 fix already
# established that a missing stop_reason must not be read as trouble.
_NORMAL_STOP_REASONS = frozenset({None, "end_turn", "stop_sequence", "tool_use"})


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
        if stop_reason not in _NORMAL_STOP_REASONS:
            # Not a failure — the text is usable, so return it. But write the
            # value down: on 2026-08-12 five responses arrived cut off mid-object
            # and none of them said max_tokens, so what Anthropic reports on
            # those calls is unknown and this line is how we find out. Length
            # goes with it — it is what separates hitting the cap from
            # stopping short of it.
            logger.warning(
                "%s finished with stop_reason=%r (%d chars)",
                self.model, stop_reason, len(content),
            )
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
        end = _end_of_first_object(text, start)
        if end is not None:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        else:
            # An opening brace that never closes: the response stopped
            # mid-object. Say so, because the remedy is to ask again rather
            # than to rewrite the prompt. On 2026-08-12 five personas hit
            # this and `stop_reason` reported a normal finish for every one,
            # so the text is the only reliable witness. The length goes in
            # the message: it is what tells a postmortem whether the model
            # ran into the token cap or stopped short of it.
            raise LLMTruncatedError(
                f"response stopped inside an unclosed JSON object "
                f"({len(text)} chars): {text[:200]!r}"
            )

    raise LLMParseError(f"no JSON object found in response: {text[:200]!r}")


def _end_of_first_object(text: str, start: int) -> int | None:
    """Index of the `}` closing the object that opens at `start`, or None if
    it never closes.

    Counts braces outside string literals only. A persona quoting a brace
    ("guidance was {redacted}") would otherwise skew the depth and read as a
    truncated response — the opposite error from the one this module already
    made once.
    """
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None
