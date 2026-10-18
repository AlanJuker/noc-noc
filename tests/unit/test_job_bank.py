from pathlib import Path

from noc_noc.application.benchmark import Case
from noc_noc.domain.classification import Classification
from noc_noc.infrastructure import job_bank


def posting(title: str, code: str, language: str = "English") -> dict[str, str]:
    return {"Original Job Title": title, "NOC21 Code": code, "Official Language": language}


def test_keeps_real_titles_that_no_method_can_look_up(noc: Classification) -> None:
    postings = [
        posting("Senior Barista &amp;amp; Trainer", "65201"),  # double-escaped by the source
        posting("senior   barista & trainer", "65201"),  # same title, other spelling
        posting("Full Stack Developer", "21232"),
        posting("NA", "21232"),  # no title
        posting("Développeur web", "21232", language="French"),
        posting("Data Wrangler", "99999"),  # unknown code
        posting("Web Programmer", "21232"),  # an official example title
        posting("Kitchen Lead", "65201"),
        posting("Kitchen Lead", "21232"),  # filed under two codes: ambiguous
    ]

    cases = job_bank.build_cases(postings, noc)

    assert cases == [
        Case("Full Stack Developer", "21232"),
        Case("Senior Barista & Trainer", "65201"),
    ]


def test_round_trips_through_a_tsv_file(tmp_path: Path) -> None:
    cases = [Case("Full Stack Developer", "21232"), Case("Barista\tlead", "65201")]
    path = tmp_path / "cases.tsv"
    job_bank.write_cases(cases, path)
    assert job_bank.read_cases(path) == cases


def test_reads_the_utf16_tab_separated_files_job_bank_publishes(tmp_path: Path) -> None:
    path = tmp_path / "postings.csv"
    path.write_text(
        "Original Job Title\tNOC21 Code\tOfficial Language\nBarista\t65201\tEnglish\n",
        encoding="utf-16",
    )
    assert list(job_bank.read_postings(path)) == [posting("Barista", "65201")]


COMMITTED = Path(job_bank.__file__).parent.parent / "data" / "job_bank_2026_07_09.tsv"


def test_the_committed_benchmark_is_well_formed() -> None:
    cases = job_bank.read_cases(COMMITTED)
    assert len(cases) > 4_000
    assert all(len(c.code) == 5 and c.code.isdigit() for c in cases)
    assert len({c.query.lower() for c in cases}) == len(cases)
