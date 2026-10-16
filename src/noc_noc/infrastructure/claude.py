"""Claude as the reranking model."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import anthropic
from anthropic.types.beta import BetaMessageParam, BetaOutputConfigParam

from noc_noc.application.rerank import RankedCode, duty_id
from noc_noc.domain.occupation import Occupation

MODEL = "claude-opus-5-5"
# USD per million tokens for claude-opus-5-5, to report what an evaluation run cost
PRICE_PER_MTOK = {"input": 4.00, "output": 20.00}

INSTRUCTIONS = """\
You match job titles and job descriptions to Canada's National Occupational Classification \
(NOC 2021). You receive what a person wrote about their job and a numbered list of candidate \
unit groups, each with its official title, example job titles and main duties. Each duty has an \
id such as 21232-d3.

Rank the candidates from most to least likely to be this person's unit group. Judge by the work \
described, not by shared words: a "film developer" is not a software developer. Follow the NOC's \
own logic: when a candidate's example titles include the person's title or a close variant, that \
is strong evidence; managers belong in management unit groups only when the title says they \
manage.

For each candidate, cite the ids of the duties that match what the person wrote; cite none when \
nothing matches. Use only codes and duty ids from the list."""

SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "matching_duty_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["code", "matching_duty_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ranking"],
    "additionalProperties": False,
}


class RerankError(RuntimeError):
    pass


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_PER_MTOK["input"]
            + self.output_tokens * PRICE_PER_MTOK["output"]
        ) / 1_000_000


def describe(candidates: Sequence[Occupation]) -> str:
    blocks = []
    for n, o in enumerate(candidates, start=1):
        duties = "\n".join(f"  {duty_id(o.code, i)}: {d}" for i, d in enumerate(o.main_duties))
        examples = ", ".join(o.example_titles[:12])
        blocks.append(f"{n}. {o.code} {o.title}\n  Example titles: {examples}\n  Duties:\n{duties}")
    return "\n\n".join(blocks)


Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ClaudeRankModel:
    def __init__(self, client: anthropic.Anthropic, effort: Effort = "low") -> None:
        self._client = client
        self._effort = effort
        self.usage = Usage()

    def rank(self, query: str, candidates: Sequence[Occupation]) -> list[RankedCode]:
        message: BetaMessageParam = {
            "role": "user",
            "content": f"What the person wrote:\n{query}\n\nCandidates:\n\n{describe(candidates)}",
        }
        output_config: BetaOutputConfigParam = {
            "effort": self._effort,  # reranking is classification: low effort is enough
            "format": {"type": "json_schema", "schema": SCHEMA},
        }
        response = self._client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=INSTRUCTIONS,
            messages=[message],
            output_config=output_config,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # rerun a declined request on Anthropic's recommended model
        )
        self.usage.requests += 1
        self.usage.input_tokens += response.usage.input_tokens
        self.usage.output_tokens += response.usage.output_tokens

        if response.stop_reason == "refusal":
            raise RerankError(f"The model declined to rank ({response.stop_details})")
        if response.stop_reason == "max_tokens":
            raise RerankError("The ranking was cut off by max_tokens")
        text = next(block.text for block in response.content if block.type == "text")
        return [
            RankedCode(item["code"], tuple(item["matching_duty_ids"]))
            for item in json.loads(text)["ranking"]
        ]
