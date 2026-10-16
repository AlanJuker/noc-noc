import pytest

from noc_noc.domain.classification import Classification
from noc_noc.domain.occupation import Exclusion, Teer


@pytest.mark.parametrize(
    ("code", "teer"),
    [("21231", Teer.UNIVERSITY_DEGREE), ("65201", Teer.NO_FORMAL_EDUCATION)],
)
def test_teer_comes_from_the_second_digit(noc: Classification, code: str, teer: Teer) -> None:
    occupation = noc.get(code)
    assert occupation is not None
    assert occupation.teer is teer


def test_every_teer_has_a_description() -> None:
    assert all(teer.description for teer in Teer)


def test_exclusion_keeps_the_code_it_points_to() -> None:
    exclusion = Exclusion.parse(
        "Computer programmers  (See 21232 Software developers and programmers)"
    )
    assert exclusion == Exclusion(
        "Computer programmers (See 21232 Software developers and programmers)", "21232"
    )
    assert Exclusion.parse("Clerks in other offices").code is None
