from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from noc_noc.api.app import create_app
from noc_noc.application.rerank import RankedCode, Reranker, duty_id
from noc_noc.domain.classification import Classification
from noc_noc.domain.occupation import Occupation
from noc_noc.infrastructure.dense import DenseRetriever, HybridRetriever
from noc_noc.infrastructure.lexical import TfidfRetriever
from tests.fakes import WordHashEmbedder


class CitesFirstDuty:
    def rank(self, query: str, candidates: Sequence[Occupation]) -> list[RankedCode]:
        return [RankedCode(c.code, (duty_id(c.code, 0),)) for c in candidates]


def hybrid(noc: Classification) -> HybridRetriever:
    return HybridRetriever([TfidfRetriever(noc), DenseRetriever(noc, WordHashEmbedder())])


@pytest.fixture
def client(noc: Classification) -> TestClient:
    retriever = hybrid(noc)
    return TestClient(create_app(noc, retriever, Reranker(retriever, noc, CitesFirstDuty())))


@pytest.fixture
def client_without_llm(noc: Classification) -> TestClient:
    return TestClient(create_app(noc, hybrid(noc)))


def test_search(client: TestClient) -> None:
    results = client.get("/api/search", params={"q": "barista", "k": 2}).json()
    assert len(results) == 2
    assert results[0] == {
        "code": "65201",
        "title": "Food counter attendants and kitchen helpers",
        "teer": 5,
        "teer_description": "Short work demonstration and no formal education",
        "matching_duties": [],
    }


def test_search_with_explanations(client: TestClient) -> None:
    top = client.get("/api/search", params={"q": "barista", "explain": True}).json()[0]
    assert top["matching_duties"] == ["Take customers' orders"]


def test_explain_falls_back_to_the_hybrid_without_an_llm(client_without_llm: TestClient) -> None:
    top = client_without_llm.get("/api/search", params={"q": "barista", "explain": True}).json()[0]
    assert (top["code"], top["matching_duties"]) == ("65201", [])


@pytest.mark.parametrize(
    "params", [{"q": "x"}, {"q": "barista", "k": 0}, {"q": "barista", "k": 11}, {"q": "a" * 2001}]
)
def test_rejects_invalid_searches(client: TestClient, params: dict[str, object]) -> None:
    assert client.get("/api/search", params=params).status_code == 422


def test_occupation(client: TestClient) -> None:
    body = client.get("/api/occupations/21232").json()
    assert body["title"] == "Software developers and programmers"
    assert body["teer"] == 1
    assert "web programmer" in body["example_titles"]
    assert client.get("/api/occupations/99999").status_code == 404


def test_home_page(client: TestClient, client_without_llm: TestClient) -> None:
    page = client.get("/").text
    assert "Find my code" in page
    assert 'name="explain"' in page
    assert "Not immigration advice" in page
    assert 'name="explain"' not in client_without_llm.get("/").text


def test_search_page_returns_a_fragment_to_htmx(client: TestClient) -> None:
    fragment = client.post("/search", data={"q": "barista"}, headers={"HX-Request": "true"}).text
    assert "<html" not in fragment
    assert "65201" in fragment


def test_search_page_works_without_javascript(client: TestClient) -> None:
    page = client.post("/search", data={"q": "barista", "explain": "on"}).text
    assert "<html" in page
    assert "Take customers&#39; orders" in page
    assert ">barista</textarea>" in page


class FindsNothing:
    def search(self, query: str, k: int) -> list[str]:
        return []


def test_search_page_without_matches(noc: Classification) -> None:
    client = TestClient(create_app(noc, FindsNothing()))
    fragment = client.post("/search", data={"q": "zzz"}, headers={"HX-Request": "true"}).text
    assert "No unit group matched" in fragment


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
