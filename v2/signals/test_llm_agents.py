"""LLMAgent + BuffettAgent tests — fake LLM and data client, no network."""

import json

import pytest

from v2.data.client import FDClientError
from v2.data.models import FinancialMetrics
from v2.llm import PromptCache, extract_json
from v2.llm.client import LLMParseError
from v2.models import Signal
from v2.signals import BuffettAgent


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeLLM:
    """Canned-response LLM; counts calls; can raise instead."""

    model = "fake-model"

    def __init__(self, response="", error=None):
        self._response = response
        self._error = error
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._response


class MockDataClient:
    def __init__(self, metrics=None, error=None):
        self._metrics = metrics or []
        self._error = error

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        if self._error is not None:
            raise self._error
        return self._metrics

    def get_company_facts(self, ticker):
        return None


def _history(n=8):
    quarters = ["2024-12-31", "2024-09-30", "2024-06-30", "2024-03-31",
                "2023-12-31", "2023-09-30", "2023-06-30", "2023-03-31"]
    return [
        FinancialMetrics(
            ticker="TEST", report_period=q, period="ttm", filing_date=q,
            return_on_equity=0.2, gross_margin=0.4, book_value_per_share=10.0,
            market_cap=1e9,
        )
        for q in quarters[:n]
    ]


BULLISH = json.dumps({"signal": "bullish", "confidence": 80, "reasoning": "Wonderful business."})


def _agent(tmp_path, llm):
    return BuffettAgent(llm=llm, cache=PromptCache(tmp_path / "llm"))


# ---------------------------------------------------------------------------
# Signal folding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("signal,confidence,expected", [
    ("bullish", 80, 0.8),
    ("bearish", 60, -0.6),
    ("neutral", 90, 0.0),
])
def test_value_folding(tmp_path, signal, confidence, expected):
    response = json.dumps({"signal": signal, "confidence": confidence, "reasoning": "r"})
    agent = _agent(tmp_path, FakeLLM(response))

    sig = agent.predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))

    assert isinstance(sig, Signal)
    assert sig.model_name == "buffett"
    assert sig.value == pytest.approx(expected)
    assert sig.metadata["abstained"] is False


# ---------------------------------------------------------------------------
# Failure contract
# ---------------------------------------------------------------------------

def test_malformed_json_abstains(tmp_path):
    agent = _agent(tmp_path, FakeLLM("I am bullish, trust me."))
    sig = agent.predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))
    assert sig.value == 0.0
    assert sig.metadata["abstained"] is True


def test_llm_error_abstains(tmp_path):
    agent = _agent(tmp_path, FakeLLM(error=TimeoutError("llm timed out")))
    sig = agent.predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))
    assert sig.value == 0.0
    assert sig.metadata["abstained"] is True
    assert "timed out" in sig.metadata["abstain_reason"]


def test_insufficient_data_abstains(tmp_path):
    agent = _agent(tmp_path, FakeLLM(BULLISH))
    sig = agent.predict("TEST", "2025-01-15", MockDataClient(metrics=_history(2)))
    assert sig.value == 0.0
    assert sig.metadata["abstained"] is True


def test_data_layer_error_propagates(tmp_path):
    """Fail loud: an infrastructure failure must NOT become a neutral view."""
    client = MockDataClient(error=FDClientError("API down", status_code=500))
    agent = _agent(tmp_path, FakeLLM(BULLISH))
    with pytest.raises(FDClientError):
        agent.predict("TEST", "2025-01-15", client)


# ---------------------------------------------------------------------------
# Cache = persistence
# ---------------------------------------------------------------------------

def test_cache_hit_skips_llm_call(tmp_path):
    llm = FakeLLM(BULLISH)
    client = MockDataClient(metrics=_history())
    agent = _agent(tmp_path, llm)

    first = agent.predict("TEST", "2025-01-15", client)
    second = agent.predict("TEST", "2025-01-15", client)

    assert llm.calls == 1  # second predict served from cache
    assert first.value == second.value
    assert first.metadata["cached"] is False
    assert second.metadata["cached"] is True


def test_prompt_and_response_persisted(tmp_path):
    agent = _agent(tmp_path, FakeLLM(BULLISH))
    agent.predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))

    records = list((tmp_path / "llm").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["agent"] == "buffett"
    assert "You are Warren Buffett" in record["system"]
    assert "2024-12-31" in record["user"]  # the rendered snapshot
    assert record["response"] == BULLISH
    assert record["parsed"]["signal"] == "bullish"


def test_failed_parse_still_persists_response(tmp_path):
    agent = _agent(tmp_path, FakeLLM("garbage"))
    agent.predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))

    records = list((tmp_path / "llm").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["response"] == "garbage"
    assert "parse_error" in record


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

def test_extract_json_fenced():
    assert extract_json('here:\n```json\n{"a": 1}\n```\ndone') == {"a": 1}


def test_extract_json_bare():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_embedded():
    assert extract_json('Sure! {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}


def test_extract_json_raises_on_garbage():
    with pytest.raises(LLMParseError):
        extract_json("no json here at all")


# ---------------------------------------------------------------------------
# Truncated-response retry (2026-08-12)
#
# `stop_reason` did not identify the cut-off responses that blocked trading on
# 2026-08-11 and 2026-08-12, so the client's own retry never fired. The agent
# sees the parse, so the agent is where the second ask belongs.
# ---------------------------------------------------------------------------

CUT = '{"signal": "bullish", "confidence": 74, "reasoning": "Meta is a wonderful bus'

# The AMAT response of 2026-08-14: whole object, balanced braces, and a raw
# newline inside the string that json refuses. It abstained on the first
# attempt because this read as "no JSON object found", which is not retryable.
MALFORMED = '{"signal": "neutral", "confidence": 45, "reasoning": "AMAT is\nsolid"}'


class SequenceLLM:
    """Returns each queued response in turn; records the prompts it got."""

    model = "fake-model"

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return self._responses.pop(0)


def test_a_truncated_response_is_asked_again_instead_of_abstaining(tmp_path):
    # Arrange
    llm = SequenceLLM([CUT, BULLISH])
    agent = _agent(tmp_path, llm)

    # Act
    sig = agent.predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))

    # Assert
    assert sig.metadata["abstained"] is False
    assert sig.value == 0.8
    assert len(llm.prompts) == 2


def test_the_second_ask_keeps_the_original_prompt_and_narrows_the_answer(tmp_path):
    # The retry cannot tell the model which failure it hit, so it names
    # neither: a truncated answer did not finish and a malformed one did.
    # Arrange
    llm = SequenceLLM([CUT, BULLISH])
    agent = _agent(tmp_path, llm)

    # Act
    agent.predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))

    # Assert
    assert "could not be read as JSON" in llm.prompts[1]
    assert "one sentence" in llm.prompts[1]
    assert llm.prompts[1].startswith(llm.prompts[0])


def test_a_malformed_object_is_asked_again_instead_of_abstaining(tmp_path):
    # Arrange
    llm = SequenceLLM([MALFORMED, BULLISH])
    agent = _agent(tmp_path, llm)

    # Act
    sig = agent.predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))

    # Assert
    assert sig.metadata["abstained"] is False
    assert sig.value == 0.8
    assert len(llm.prompts) == 2


def test_abstains_when_the_second_object_is_also_malformed(tmp_path):
    # Arrange
    llm = SequenceLLM([MALFORMED, MALFORMED])
    agent = _agent(tmp_path, llm)

    # Act
    sig = agent.predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))

    # Assert — and the reason says what was wrong, not that nothing was found
    assert sig.metadata["abstained"] is True
    assert "would not parse" in sig.metadata["abstain_reason"]
    assert "no JSON object found" not in sig.metadata["abstain_reason"]
    assert len(llm.prompts) == 2


def test_abstains_when_the_second_response_is_also_truncated(tmp_path):
    # Arrange
    llm = SequenceLLM([CUT, CUT])
    agent = _agent(tmp_path, llm)

    # Act
    sig = agent.predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))

    # Assert
    assert sig.metadata["abstained"] is True
    assert "unclosed" in sig.metadata["abstain_reason"]
    assert len(llm.prompts) == 2


def test_a_plain_parse_failure_is_not_retried(tmp_path):
    # Prose with no object at all is a prompt problem; asking twice just costs
    # a second call.
    # Arrange
    llm = SequenceLLM(["I am bullish, trust me.", BULLISH])
    agent = _agent(tmp_path, llm)

    # Act
    sig = agent.predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))

    # Assert
    assert sig.metadata["abstained"] is True
    assert len(llm.prompts) == 1


# ---------------------------------------------------------------------------
# The insufficient-data abstain has to say so out loud
#
# 2026-08-21. The [ai-hedge-fund] digest read "🔴 excluded (dead committee,
# held as-is): ALOY 88% failed" and the whole run logged one warning, for a
# different ticker. Seven personas had abstained through this path and it was
# the only abstain path in the file with no logger call, so the cause left no
# trace anywhere an operator looks.
# ---------------------------------------------------------------------------

def test_insufficient_data_abstain_is_logged(caplog):
    agent = BuffettAgent(llm=FakeLLM(response="{}"))
    client = MockDataClient(metrics=_history(2))  # under MIN_PERIODS
    with caplog.at_level("WARNING"):
        sig = agent.predict("ALOY", "2026-08-21", client)
    assert sig.metadata["abstained"] is True
    assert "insufficient data" in sig.reasoning
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "ALOY" in logged and "insufficient data" in logged
