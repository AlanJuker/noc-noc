"""A benchmark built from Statistics Canada's own job titles.

Every example title belongs to exactly one unit group. A fifth of them, picked by a hash of the text
so the split never changes, are held out of the index and used as queries.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from noc_noc.domain.classification import Classification

HOLD_OUT_ONE_IN = 5


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[str]:
        """The `k` most likely unit group codes, best first."""
        ...


RetrieverFactory = Callable[[Classification], Retriever]


@dataclass(frozen=True, slots=True)
class Case:
    query: str
    code: str


def _held_out(title: str) -> bool:
    return int(hashlib.sha256(title.encode()).hexdigest(), 16) % HOLD_OUT_ONE_IN == 0


def split(classification: Classification) -> tuple[Classification, list[Case]]:
    """The classification without the held-out titles, and those titles as test cases.

    Every occupation keeps at least one example title in the index.
    """
    kept, cases = [], []
    for occupation in classification:
        held = [t for t in occupation.example_titles if _held_out(t)]
        if len(held) == len(occupation.example_titles):
            held = held[1:]
        kept.append(
            replace(
                occupation,
                example_titles=tuple(t for t in occupation.example_titles if t not in held),
            )
        )
        cases += [Case(title, occupation.code) for title in held]
    return Classification(kept), cases


@dataclass(frozen=True, slots=True)
class Result:
    cases: int
    top1: float
    top5: float
    mrr: float  # mean reciprocal rank over the top 10
    confusable_error_share: (
        float  # wrong top-1 answers that Statistics Canada lists as easy to confuse
    )


def evaluate(retriever: Retriever, cases: Sequence[Case], classification: Classification) -> Result:
    top1 = top5 = 0
    reciprocal_ranks = 0.0
    errors = confusable = 0
    for case in cases:
        ranking = retriever.search(case.query, 10)
        if ranking[:1] == [case.code]:
            top1 += 1
        else:
            errors += 1
            confusable += bool(ranking) and ranking[0] in classification.confused_with(case.code)
        top5 += case.code in ranking[:5]
        if case.code in ranking:
            reciprocal_ranks += 1 / (ranking.index(case.code) + 1)
    n = len(cases) or 1
    return Result(
        cases=len(cases),
        top1=top1 / n,
        top5=top5 / n,
        mrr=reciprocal_ranks / n,
        confusable_error_share=confusable / errors if errors else 0.0,
    )
