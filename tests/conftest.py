from pathlib import Path

import pytest

from noc_noc.domain.classification import Classification
from noc_noc.infrastructure import statcan

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def noc() -> Classification:
    structure = (FIXTURES / "structure.csv").open(encoding="utf-8-sig")
    elements = (FIXTURES / "elements.csv").open(encoding="utf-8-sig")
    with structure, elements:
        return statcan.parse(structure, elements)
