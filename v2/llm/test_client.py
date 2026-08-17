"""AnthropicLLM.complete() tests — fake chat transport, no network.

These cover the 2026-08-11 incident: seven persona calls came back cut off
mid-`reasoning`, `extract_json` found an unclosed `{`, every one of them
abstained, and the 4% failure ratio tripped the probation gate so the book
did not trade at all that day.
"""

import logging

import pytest

from v2.llm.client import (
    AnthropicLLM,
    LLMMalformedError,
    LLMParseError,
    LLMTruncatedError,
    extract_json,
)


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


# ---------------------------------------------------------------------------
# extract_json truncation detection (2026-08-12)
#
# The 2026-08-11 fix trusted `stop_reason` to say a response was cut off. The
# 2026-08-12 run proved it does not: five personas came back with an unclosed
# `{`, `complete()` returned the text without raising, and every one abstained
# as a plain parse error. Detection has to read the text itself.
# ---------------------------------------------------------------------------


def test_an_unclosed_object_is_reported_as_truncation_not_a_parse_error():
    # Arrange — exactly the 2026-08-12 shape: valid JSON that stops mid-string.
    cut = '{"signal": "bullish", "confidence": 74, "reasoning": "Meta is a wonderful bus'

    # Act / Assert
    with pytest.raises(LLMTruncatedError):
        extract_json(cut)


def test_prose_with_no_object_at_all_stays_a_plain_parse_error():
    # Arrange
    prose = "I cannot answer that."

    # Act / Assert
    with pytest.raises(LLMParseError) as exc:
        extract_json(prose)
    assert not isinstance(exc.value, LLMTruncatedError)
    assert not isinstance(exc.value, LLMMalformedError)


def test_a_complete_object_the_decoder_refuses_is_malformed_not_missing():
    # The AMAT abstention of 2026-08-14. The object is whole and the braces
    # balance; a raw newline inside the string is what json rejects.
    # Arrange
    reply = '{"signal": "neutral", "confidence": 45, "reasoning": "AMAT is\nsolid"}'

    # Act / Assert
    with pytest.raises(LLMMalformedError) as exc:
        extract_json(reply)
    assert "no JSON object found" not in str(exc.value)


def test_a_malformed_object_is_not_reported_as_truncation():
    # The two retryable failures have different remedies at the prompt, so
    # neither may answer to the other's name.
    # Arrange
    reply = '{"signal": "bullish", "confidence": 70,}'

    # Act / Assert
    with pytest.raises(LLMMalformedError) as exc:
        extract_json(reply)
    assert not isinstance(exc.value, LLMTruncatedError)


def test_the_malformed_message_carries_the_decoder_complaint_and_the_length():
    # Arrange
    reply = '{"signal": "bearish", "confidence": 30, "reasoning": "runs "hot""}'

    # Act
    with pytest.raises(LLMMalformedError) as exc:
        extract_json(reply)

    # Assert — the position the decoder named, and how long the answer ran
    assert "char" in str(exc.value)
    assert f"({len(reply)} chars)" in str(exc.value)


def test_the_truncation_message_carries_the_response_length():
    # The diagnostic the 2026-08-12 postmortem could not get: was the response
    # near the token cap, or short?
    # Arrange
    cut = '{"signal": "bullish", "confidence": 74, "reasoning": "' + "x" * 300

    # Act / Assert
    with pytest.raises(LLMTruncatedError) as exc:
        extract_json(cut)
    assert str(len(cut)) in str(exc.value)


def test_a_brace_inside_a_string_does_not_count_toward_depth():
    # Arrange — a persona quoting a brace must not read as unbalanced.
    text = ('Here is my view:\n'
            '{"signal": "neutral", "confidence": 40, "reasoning": "guidance was {redacted}"}')

    # Act
    data = extract_json(text)

    # Assert
    assert data["signal"] == "neutral"


def test_an_unclosed_brace_inside_a_string_is_not_truncation():
    # Arrange — the object closes; only a quoted brace is unbalanced.
    text = ('Here is my view:\n'
            '{"signal": "bearish", "confidence": 60, "reasoning": "margin { pressure"}')

    # Act
    data = extract_json(text)

    # Assert
    assert data["signal"] == "bearish"


def test_an_escaped_quote_does_not_end_the_string_scan():
    # Arrange — CRWD/wood on 2026-08-11 contained an escaped apostrophe.
    text = ('Here is my view:\n'
            '{"signal": "bearish", "confidence": 55, "reasoning": "CrowdStrike\\"s growth"}')

    # Act
    data = extract_json(text)

    # Assert
    assert data["confidence"] == 55


# Instrumentation. On 2026-08-12 five responses arrived cut off and every one
# of them reported a normal finish, so the only way to learn what Anthropic
# actually says on those calls is to write it down when it is not `end_turn`.


def test_an_unusual_stop_reason_is_logged_even_when_the_text_looks_fine(caplog):
    # Arrange
    llm = make_llm([(GOOD, "refusal")])

    # Act
    with caplog.at_level(logging.WARNING, logger="v2.llm.client"):
        text = llm.complete("sys", "user")

    # Assert — the call still succeeds; the oddity is recorded, not raised.
    assert text == GOOD
    assert "refusal" in caplog.text


def test_a_normal_finish_logs_nothing(caplog):
    # Arrange
    llm = make_llm([(GOOD, "end_turn")])

    # Act
    with caplog.at_level(logging.WARNING, logger="v2.llm.client"):
        llm.complete("sys", "user")

    # Assert
    assert caplog.text == ""


def test_the_log_line_carries_the_response_length(caplog):
    # Arrange — length is what tells a postmortem whether the cap was hit.
    llm = make_llm([(GOOD, "pause_turn")])

    # Act
    with caplog.at_level(logging.WARNING, logger="v2.llm.client"):
        llm.complete("sys", "user")

    # Assert
    assert str(len(GOOD)) in caplog.text
