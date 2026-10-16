"""NOC 2021 Version 1.0 as published by Statistics Canada."""

import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import httpx

from noc_noc.domain.classification import Classification
from noc_noc.domain.occupation import Exclusion, Occupation

BASE_URL = "https://www.statcan.gc.ca/en/subjects/standard/noc/2021/indexV1"
STRUCTURE = "noc-2021-v1.0-classification-structure.csv"
ELEMENTS = "noc-2021-v1.0-elements.csv"
CODE = "Code - NOC 2021 V1.0"


def parse(structure: Iterable[str], elements: Iterable[str]) -> Classification:
    definitions = {
        row[CODE]: (row["Class title"].strip(), row["Class definition"].strip())
        for row in csv.DictReader(structure)
        if row["Hierarchical structure"] == "Unit Group"
    }
    details: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in csv.DictReader(elements):
        text = row["Element Description English"].strip()
        # "This group performs some or all of the following duties:" introduces the list
        if text and not text.endswith(":"):
            details[(row[CODE], row["Element Type Label English"])].append(text)

    return Classification(
        Occupation(
            code=code,
            title=title,
            definition=definition,
            example_titles=tuple(details[(code, "All examples")]),
            main_duties=tuple(details[(code, "Main duties")]),
            requirements=tuple(details[(code, "Employment requirements")]),
            exclusions=tuple(Exclusion.parse(t) for t in details[(code, "Exclusion(s)")]),
        )
        for code, (title, definition) in definitions.items()
    )


def load(cache_dir: Path, client: httpx.Client) -> Classification:
    """Parse the published CSVs, downloading them into `cache_dir` the first time."""
    paths = [_cached(cache_dir, name, client) for name in (STRUCTURE, ELEMENTS)]
    with paths[0].open(encoding="utf-8-sig") as s, paths[1].open(encoding="utf-8-sig") as e:
        return parse(s, e)


def _cached(cache_dir: Path, name: str, client: httpx.Client) -> Path:
    path = cache_dir / name
    if not path.exists():
        response = client.get(f"{BASE_URL}/{name}", follow_redirects=True)
        response.raise_for_status()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return path
