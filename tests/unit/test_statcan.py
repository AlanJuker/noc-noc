from pathlib import Path

import httpx
import pytest

from noc_noc.domain.classification import Classification
from noc_noc.domain.occupation import Teer
from noc_noc.infrastructure import statcan
from tests.conftest import FIXTURES


def test_parses_unit_groups_only(noc: Classification) -> None:
    assert len(noc) == 3
    assert {o.code for o in noc} == {"21231", "21232", "65201"}


def test_parses_every_element_type(noc: Classification) -> None:
    developer = noc.get("21232")
    assert developer is not None
    assert developer.title == "Software developers and programmers"
    assert developer.definition.startswith("Software developers and programmers design")
    assert developer.example_titles == (
        "application programmer",
        "web programmer",
        "computer game developer",
    )
    assert developer.main_duties == (
        "Design, write, read, test, and correct code for new software",
    )
    assert developer.requirements == (
        "A bachelor's degree in computer science is usually required.",
    )
    assert developer.teer is Teer.UNIVERSITY_DEGREE
    assert developer.broad_category == 2


def test_skips_the_line_that_introduces_the_duties(noc: Classification) -> None:
    engineer = noc.get("21231")
    assert engineer is not None
    assert engineer.main_duties == ("Design software architecture for large systems",)


def test_confused_with_follows_exclusions_both_ways(noc: Classification) -> None:
    assert noc.confused_with("21232") == {"21231"}  # 21231 sends programmers to 21232
    assert noc.confused_with("21231") == {"21232"}
    assert noc.confused_with("65201") == set()


def test_unknown_code(noc: Classification) -> None:
    assert noc.get("99999") is None


def test_downloads_once_then_reads_the_cache(tmp_path: Path) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        name = request.url.path.rsplit("/", 1)[-1]
        fixture = "structure.csv" if name == statcan.STRUCTURE else "elements.csv"
        return httpx.Response(200, content=(FIXTURES / fixture).read_bytes())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert len(statcan.load(tmp_path, client)) == 3
        assert len(statcan.load(tmp_path, client)) == 3

    assert len(requests) == 2  # one per file, the second load used the cache
    assert (tmp_path / statcan.ELEMENTS).exists()


def test_surfaces_download_errors(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(404))
    with httpx.Client(transport=transport) as client, pytest.raises(httpx.HTTPStatusError):
        statcan.load(tmp_path, client)
