import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest

from noc_noc import cli
from noc_noc.application.benchmark import Case
from noc_noc.application.rerank import RankedCode, duty_id
from noc_noc.domain.occupation import Occupation
from noc_noc.infrastructure import statcan
from noc_noc.infrastructure.claude import Usage
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


class FakeClaude:
    """Ranks the candidates in reverse and cites each one's first duty."""

    def __init__(self) -> None:
        self.usage = Usage(requests=0)

    def rank(self, query: str, candidates: Sequence[Occupation]) -> list[RankedCode]:
        self.usage.requests += 1
        return [RankedCode(c.code, (duty_id(c.code, 0),)) for c in reversed(candidates)]


@pytest.fixture
def fake_claude(monkeypatch: pytest.MonkeyPatch) -> FakeClaude:
    fake = FakeClaude()
    monkeypatch.setattr(cli, "claude", lambda: fake)
    return fake


def test_search_with_rerank_shows_the_matching_duties(
    cached: Path, fake_claude: FakeClaude, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["search", "barista", "--method", "rerank", "-k", "3"])
    out = capsys.readouterr().out
    assert "     ✓ " in out
    assert fake_claude.usage.requests == 1


def test_eval_skips_paid_methods_unless_asked(
    cached: Path, fake_claude: FakeClaude, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["eval"])
    assert "| rerank |" not in capsys.readouterr().out
    assert fake_claude.usage.requests == 0


def test_eval_of_a_paid_method_uses_a_fixed_sample_and_reports_cost(
    cached: Path,
    fake_claude: FakeClaude,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "DEFAULT_PAID_SAMPLE", 1)
    cli.main(["eval", "--method", "rerank"])
    out = capsys.readouterr().out
    assert out.startswith("1 held-out titles (fixed sample of")
    assert "Claude: 1 requests" in out


def test_sample_is_stable() -> None:
    cases = [Case(f"title {i}", "00010") for i in range(50)]
    assert cli.sample_cases(cases, 5) == cli.sample_cases(list(reversed(cases)), 5)
    assert cli.sample_cases(cases, None) == cases


def test_explains_how_to_authenticate_for_paid_methods(
    cached: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoCredentials(FakeClaude):
        def rank(self, query: str, candidates: Sequence[Occupation]) -> list[RankedCode]:
            raise TypeError("Could not resolve authentication method.")

    monkeypatch.setattr(cli, "claude", NoCredentials)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        cli.main(["search", "barista", "--method", "rerank"])


def test_other_type_errors_are_not_disguised(cached: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: object) -> None:
        raise TypeError("a real bug")

    monkeypatch.setattr(cli, "run_search", broken)
    with pytest.raises(TypeError, match="a real bug"):
        cli.main(["search", "barista", "--method", "tfidf"])


def test_build_and_evaluate_the_job_bank_benchmark(
    cached: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postings = tmp_path / "postings.csv"
    postings.write_text(
        "Original Job Title\tNOC21 Code\tOfficial Language\n"
        "Espresso Bar Attendant\t65201\tEnglish\nBackend Developer\t21232\tEnglish\n",
        encoding="utf-16",
    )
    out = tmp_path / "job_bank.tsv"
    monkeypatch.setattr(cli, "JOB_BANK_CASES", out)

    cli.main(["build-job-bank", str(postings), "--out", str(out)])
    assert capsys.readouterr().out.startswith("2 titles over 2 codes")

    cli.main(["eval", "--benchmark", "job-bank", "--method", "tfidf"])
    assert capsys.readouterr().out.startswith("2 Job Bank titles")
