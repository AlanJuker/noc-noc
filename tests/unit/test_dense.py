import pytest

from noc_noc.domain.classification import Classification
from noc_noc.infrastructure.dense import DenseRetriever, HybridRetriever
from tests.fakes import WordHashEmbedder


class FixedRetriever:
    def __init__(self, ranking: list[str]) -> None:
        self.ranking = ranking

    def search(self, query: str, k: int) -> list[str]:
        return self.ranking[:k]


@pytest.mark.parametrize("granularity", ["document", "passages"])
def test_ranks_the_closest_unit_group_first(noc: Classification, granularity: str) -> None:
    retriever = DenseRetriever(noc, WordHashEmbedder(), granularity)  # type: ignore[arg-type]
    assert retriever.search("barista", 3)[0] == "65201"
    assert retriever.search("computer game developer", 3)[0] == "21232"


def test_passages_score_a_group_by_its_best_passage(noc: Classification) -> None:
    retriever = DenseRetriever(noc, WordHashEmbedder(), "passages")
    # "software architect" is one example title of 21231, among many other passages
    assert retriever.search("software architect", 1) == ["21231"]


def test_returns_every_group_ranked_up_to_k(noc: Classification) -> None:
    ranking = DenseRetriever(noc, WordHashEmbedder()).search("software", 5)
    assert sorted(ranking) == ["21231", "21232", "65201"]


def test_hybrid_fuses_rankings() -> None:
    hybrid = HybridRetriever(
        [FixedRetriever(["a", "b", "c"]), FixedRetriever(["b", "c", "a"]), FixedRetriever(["b"])]
    )
    assert hybrid.search("anything", 2) == ["b", "a"]


def test_hybrid_keeps_codes_found_by_only_one_retriever() -> None:
    hybrid = HybridRetriever([FixedRetriever(["a"]), FixedRetriever(["z"])])
    assert set(hybrid.search("anything", 5)) == {"a", "z"}
