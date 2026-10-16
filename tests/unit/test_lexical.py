from noc_noc.domain.classification import Classification
from noc_noc.infrastructure.lexical import TfidfRetriever, document


def test_document_uses_title_examples_definition_and_duties(noc: Classification) -> None:
    occupation = noc.get("65201")
    assert occupation is not None
    text = document(occupation)
    for part in (
        "Food counter attendants",
        "barista",
        "take customers' orders",
        "Take customers' orders",
    ):
        assert part.lower() in text.lower()


def test_ranks_the_matching_occupation_first(noc: Classification) -> None:
    retriever = TfidfRetriever(noc)
    assert retriever.search("barista", 3)[0] == "65201"
    assert retriever.search("game developer", 3)[0] == "21232"


def test_character_ngrams_tolerate_typos(noc: Classification) -> None:
    assert TfidfRetriever(noc, "char_wb").search("barrista", 1) == ["65201"]


def test_returns_nothing_for_unknown_words(noc: Classification) -> None:
    assert TfidfRetriever(noc).search("zzz", 5) == []


def test_respects_k(noc: Classification) -> None:
    assert len(TfidfRetriever(noc).search("software", 1)) == 1
