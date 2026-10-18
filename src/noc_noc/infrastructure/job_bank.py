"""A second benchmark from real job postings on Canada's Job Bank.

Each posting has the title as the employer wrote it and the NOC 2021 code it was filed under. Only
titles that no method could look up are kept: they aren't official example titles, and every posting
with that title was filed under the same code.
"""

import csv
import html
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from noc_noc.application.benchmark import Case
from noc_noc.domain.classification import Classification

SOURCE = "https://open.canada.ca/data/en/dataset/ea639e28-c0fc-48bf-b5dd-b8899bd43072"


def _normalize(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()


def build_cases(postings: Iterable[dict[str, str]], classification: Classification) -> list[Case]:
    official = {_normalize(t) for o in classification for t in (o.title, *o.example_titles)}
    codes: dict[str, set[str]] = defaultdict(set)
    first_spelling: dict[str, str] = {}
    for posting in postings:
        if posting["Official Language"] != "English":
            continue
        # Titles come double-escaped ("&amp;amp;")
        title = html.unescape(html.unescape(posting["Original Job Title"])).strip()
        code = posting["NOC21 Code"].strip()
        key = _normalize(title)
        if not key or key == "na" or classification.get(code) is None or key in official:
            continue
        codes[key].add(code)
        first_spelling.setdefault(key, re.sub(r"\s+", " ", title))
    return sorted(
        (
            Case(first_spelling[key], next(iter(found)))
            for key, found in codes.items()
            if len(found) == 1
        ),
        key=lambda case: case.query.lower(),
    )


def read_postings(path: Path) -> Iterable[dict[str, str]]:
    """Job Bank publishes tab-separated UTF-16 files."""
    with path.open(encoding="utf-16", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def write_cases(cases: list[Case], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["title", "code"])
        writer.writerows((c.query, c.code) for c in cases)


def read_cases(path: Path) -> list[Case]:
    with path.open(encoding="utf-8", newline="") as f:
        return [Case(row["title"], row["code"]) for row in csv.DictReader(f, delimiter="\t")]
