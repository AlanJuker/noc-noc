import shutil
from pathlib import Path

import pytest

from noc_noc import cli
from noc_noc.infrastructure import statcan
from tests.conftest import FIXTURES
from tests.fakes import WordHashEmbedder


@pytest.fixture
def cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copy(FIXTURES / "structure.csv", tmp_path / statcan.STRUCTURE)
    shutil.copy(FIXTURES / "elements.csv", tmp_path / statcan.ELEMENTS)
    monkeypatch.setattr(cli, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cli, "embedder", WordHashEmbedder)  # no model download in tests
    return tmp_path


def test_search(cached: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["search", "barista", "-k", "2", "--method", "tfidf"])
    assert capsys.readouterr().out.startswith(
        "1. 65201  Food counter attendants and kitchen helpers  (TEER 5)"
    )


def test_search_without_a_match(cached: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["search", "zzz", "--method", "tfidf"])
    assert capsys.readouterr().out.strip() == "No match."


def test_eval_prints_a_markdown_table(cached: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["eval"])
    out = capsys.readouterr().out
    assert "| Method | top-1 | top-5 | MRR | Confusable errors |" in out
    for method in ("tfidf", "dense", "hybrid"):
        assert f"| {method} |" in out


def test_search_defaults_to_the_hybrid_method(
    cached: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["search", "barista"])
    assert capsys.readouterr().out.startswith("1. 65201")
