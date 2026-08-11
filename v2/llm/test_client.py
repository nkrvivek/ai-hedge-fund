"""AnthropicLLM.complete() tests — fake chat transport, no network.

These cover the 2026-08-11 incident: seven persona calls came back cut off
mid-`reasoning`, `extract_json` found an unclosed `{`, every one of them
abstained, and the 4% failure ratio tripped the probation gate so the book
did not trade at all that day.
"""

import pytest

from v2.llm.client import AnthropicLLM, LLMTruncatedError


class FakeChat:
    """Stands in for ChatAnthropic. Returns each queued (text, stop_reason)."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        text, stop_reason = self._replies.pop(0)
        return FakeMessage(text, stop_reason)


class FakeMessage:
    def __init__(self, content, stop_reason):
        self.content = content
        self.response_metadata = {"stop_reason": stop_reason}


def make_llm(replies):
    """Build an AnthropicLLM without touching the API key or langchain."""
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.model = "fake-model"
    llm._chat = FakeChat(replies)
    return llm


GOOD = '{"signal": "bullish", "confidence": 70, "reasoning": "Wide moat."}'
CUT = '{"signal": "bullish", "confidence": 70, "reasoning": "Wide moat and a'


def test_returns_a_complete_response_without_a_second_call():
    # Arrange
    llm = make_llm([(GOOD, "end_turn")])

    # Act
    result = llm.complete("system", "user")

    # Assert
    assert result == GOOD
    assert len(llm._chat.prompts) == 1


def test_asks_again_when_the_first_response_was_cut_off_at_max_tokens():
    # Arrange
    llm = make_llm([(CUT, "max_tokens"), (GOOD, "end_turn")])

    # Act
    result = llm.complete("system", "user")

    # Assert
    assert result == GOOD
    assert len(llm._chat.prompts) == 2


def test_the_retry_asks_for_the_json_alone():
    # Arrange
    llm = make_llm([(CUT, "max_tokens"), (GOOD, "end_turn")])

    # Act
    llm.complete("system", "the snapshot")

    # Assert — original prompt kept, terse instruction appended
    retry_user = llm._chat.prompts[1][1][1]
    assert retry_user.startswith("the snapshot")
    assert "one sentence" in retry_user


def test_asks_again_when_the_response_came_back_empty():
    # An empty body is the extreme of the same failure — lynch/MSFT on
    # 2026-08-11 returned '' and read as an unparseable view.
    # Arrange
    llm = make_llm([("   ", "end_turn"), (GOOD, "end_turn")])

    # Act
    result = llm.complete("system", "user")

    # Assert
    assert result == GOOD
    assert len(llm._chat.prompts) == 2


def test_raises_a_truncation_error_when_the_retry_is_also_cut_off():
    # Naming the cause beats "no JSON object found": the second one sent the
    # session hunting the parser instead of the token cap.
    # Arrange
    llm = make_llm([(CUT, "max_tokens"), (CUT, "max_tokens")])

    # Act / Assert
    with pytest.raises(LLMTruncatedError) as exc:
        llm.complete("system", "user")
    assert "max_tokens" in str(exc.value)


def test_raises_when_both_responses_are_empty():
    # Arrange
    llm = make_llm([("", "end_turn"), ("", "end_turn")])

    # Act / Assert
    with pytest.raises(LLMTruncatedError):
        llm.complete("system", "user")


def test_reads_a_list_of_content_blocks():
    # Arrange
    blocks = [{"type": "text", "text": GOOD}]
    llm = make_llm([(blocks, "end_turn")])

    # Act
    result = llm.complete("system", "user")

    # Assert
    assert result == GOOD


def test_a_missing_stop_reason_is_not_read_as_truncation():
    # A provider that does not report stop_reason must not send every call
    # twice.
    # Arrange
    llm = make_llm([(GOOD, None)])

    # Act
    result = llm.complete("system", "user")

    # Assert
    assert result == GOOD
    assert len(llm._chat.prompts) == 1
