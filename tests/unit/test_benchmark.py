from dataclasses import replace

import pytest

from noc_noc.application.benchmark import Case, evaluate, split
from noc_noc.domain.classification import Classification


class FakeRetriever:
    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = answers

    def search(self, query: str, k: int) -> list[str]:
        return self.answers.get(query, [])[:k]


def with_titles(noc: Classification, code: str, titles: list[str]) -> Classification:
    return Classification(
        replace(o, example_titles=tuple(titles)) if o.code == code else o for o in noc
    )


def test_split_holds_titles_out_of_the_index(noc: Classification) -> None:
    titles = [f"title {i}" for i in range(40)]
    index, cases = split(with_titles(noc, "21232", titles))

    kept = index.get("21232")
    assert kept is not None
    held = [c for c in cases if c.code == "21232"]
    assert held, "some titles should be held out"
    assert not {c.query for c in held} & set(kept.example_titles)
    assert len(held) + len(kept.example_titles) == 40


def test_split_is_deterministic(noc: Classification) -> None:
    titles = [f"title {i}" for i in range(40)]
    first = split(with_titles(noc, "21232", titles))[1]
    second = split(with_titles(noc, "21232", list(reversed(titles))))[1]
    assert {c.query for c in first} == {c.query for c in second}


def test_split_keeps_at_least_one_title_per_occupation(noc: Classification) -> None:
    # A single title that would be held out by the hash stays in the index
    held_by_hash = next(
        t
        for t in (f"title {i}" for i in range(100))
        if split(with_titles(noc, "65201", [t, "x"]))[1]
    )
    index, cases = split(with_titles(noc, "65201", [held_by_hash]))
    kept = index.get("65201")
    assert kept is not None
    assert kept.example_titles == (held_by_hash,)
    assert not [c for c in cases if c.code == "65201"]


def test_evaluate(noc: Classification) -> None:
    cases = [
        Case("application programmer", "21232"),  # right at rank 1
        Case("web programmer", "21232"),  # rank 2, top-1 wrong with a confusable code
        Case("barista", "65201"),  # rank 3, top-1 wrong with an unrelated code
        Case("dishwasher", "65201"),  # not found
    ]
    retriever = FakeRetriever(
        {
            "application programmer": ["21232", "21231"],
            "web programmer": ["21231", "21232"],
            "barista": ["21232", "21231", "65201"],
            "dishwasher": [],
        }
    )

    result = evaluate(retriever, cases, noc)

    assert result.cases == 4
    assert result.top1 == 0.25
    assert result.top5 == 0.75
    assert result.mrr == pytest.approx((1 + 1 / 2 + 1 / 3) / 4)
    assert result.confusable_error_share == pytest.approx(1 / 3)


def test_evaluate_without_cases(noc: Classification) -> None:
    result = evaluate(FakeRetriever({}), [], noc)
    assert (result.cases, result.top1, result.confusable_error_share) == (0, 0.0, 0.0)
