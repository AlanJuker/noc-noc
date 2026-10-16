from collections.abc import Sequence

from noc_noc.application.rerank import RankedCode, Reranker, duty_id
from noc_noc.domain.classification import Classification
from noc_noc.domain.occupation import Occupation


class FixedRetriever:
    def __init__(self, ranking: list[str]) -> None:
        self.ranking = ranking

    def search(self, query: str, k: int) -> list[str]:
        return self.ranking[:k]


class ScriptedModel:
    def __init__(self, answer: list[RankedCode]) -> None:
        self.answer = answer
        self.seen: list[str] = []

    def rank(self, query: str, candidates: Sequence[Occupation]) -> list[RankedCode]:
        self.seen = [c.code for c in candidates]
        return self.answer


def reranker(noc: Classification, answer: list[RankedCode], depth: int = 10) -> Reranker:
    base = FixedRetriever(["21231", "21232", "65201"])
    return Reranker(base, noc, ScriptedModel(answer), depth)


def test_follows_the_model_order(noc: Classification) -> None:
    answer = [RankedCode("65201", ()), RankedCode("21232", ()), RankedCode("21231", ())]
    assert reranker(noc, answer).search("barista", 3) == ["65201", "21232", "21231"]


def test_keeps_verified_duty_citations_only(noc: Classification) -> None:
    answer = [
        RankedCode("21232", (duty_id("21232", 0), "21232-d9", duty_id("21231", 0), "21232-d1"))
    ]
    (top, *_) = reranker(noc, answer).explain("programmer", 3)
    # d9 doesn't exist, 21231-d1 belongs to another code, and the repeat is dropped
    assert top.occupation.code == "21232"
    assert top.matching_duties == ("Design, write, read, test, and correct code for new software",)


def test_ignores_invented_and_repeated_codes(noc: Classification) -> None:
    answer = [RankedCode("99999", ()), RankedCode("21232", ()), RankedCode("21232", ())]
    assert reranker(noc, answer).search("programmer", 3) == ["21232", "21231", "65201"]


def test_candidates_left_out_keep_their_order_after_the_ranked_ones(noc: Classification) -> None:
    assert reranker(noc, [RankedCode("65201", ())]).search("barista", 3) == [
        "65201",
        "21231",
        "21232",
    ]


def test_only_the_top_candidates_are_reranked(noc: Classification) -> None:
    model = ScriptedModel([RankedCode("21232", ())])
    reranked = Reranker(FixedRetriever(["21231", "21232", "65201"]), noc, model, depth=2)

    assert reranked.search("programmer", 3) == ["21232", "21231", "65201"]
    assert model.seen == ["21231", "21232"]  # 65201 stays below, untouched


def test_respects_k(noc: Classification) -> None:
    assert len(reranker(noc, []).search("anything", 2)) == 2
