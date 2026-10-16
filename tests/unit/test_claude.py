import json
from typing import Any

import anthropic
import httpx2
import pytest

from noc_noc.application.rerank import RankedCode
from noc_noc.domain.classification import Classification
from noc_noc.infrastructure.claude import MODEL, ClaudeRankModel, RerankError


def message(text: str, stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1_500, "output_tokens": 100},
    }


def model_answering(
    body: dict[str, Any], requests: list[httpx2.Request] | None = None
) -> ClaudeRankModel:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if requests is not None:
            requests.append(request)
        return httpx2.Response(200, json=body)

    client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeRankModel(client)


RANKING = {"ranking": [{"code": "21232", "matching_duty_ids": ["21232-d1"]}]}


def test_sends_candidates_with_duty_ids_and_a_strict_schema(noc: Classification) -> None:
    requests: list[httpx2.Request] = []
    model = model_answering(message(json.dumps(RANKING)), requests)
    candidates = [o for o in noc if o.code in {"21231", "21232"}]

    ranked = model.rank("I write web apps in Python", candidates)

    assert ranked == [RankedCode("21232", ("21232-d1",))]
    (request,) = requests
    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    prompt = body["messages"][0]["content"]
    assert "I write web apps in Python" in prompt
    assert "21232-d1: Design, write, read, test, and correct code for new software" in prompt


def test_tracks_usage_and_cost(noc: Classification) -> None:
    model = model_answering(message(json.dumps(RANKING)))
    model.rank("programmer", list(noc))
    model.rank("programmer", list(noc))

    assert (model.usage.requests, model.usage.input_tokens, model.usage.output_tokens) == (
        2,
        3_000,
        200,
    )
    assert model.usage.cost_usd == pytest.approx((3_000 * 4 + 200 * 20) / 1_000_000)


@pytest.mark.parametrize(
    ("stop_reason", "error"), [("refusal", "declined"), ("max_tokens", "cut off")]
)
def test_raises_instead_of_reading_an_unusable_answer(
    noc: Classification, stop_reason: str, error: str
) -> None:
    model = model_answering(message("", stop_reason))
    with pytest.raises(RerankError, match=error):
        model.rank("programmer", list(noc))


def test_surfaces_api_errors(noc: Classification) -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            401,
            json={"type": "error", "error": {"type": "authentication_error", "message": "bad key"}},
        )

    client = anthropic.Anthropic(
        api_key="bad",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    with pytest.raises(anthropic.AuthenticationError):
        ClaudeRankModel(client).rank("programmer", list(noc))
