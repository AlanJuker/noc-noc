import re
from dataclasses import dataclass
from enum import IntEnum


class Teer(IntEnum):
    """Training, Education, Experience and Responsibilities: the second digit of a NOC 2021 code."""

    MANAGEMENT = 0
    UNIVERSITY_DEGREE = 1
    COLLEGE_OR_APPRENTICESHIP_2_YEARS = 2
    COLLEGE_OR_APPRENTICESHIP_UNDER_2_YEARS = 3
    HIGH_SCHOOL = 4
    NO_FORMAL_EDUCATION = 5

    @property
    def description(self) -> str:
        return _TEER_DESCRIPTIONS[self]


_TEER_DESCRIPTIONS = {
    Teer.MANAGEMENT: "Management responsibilities",
    Teer.UNIVERSITY_DEGREE: "Usually a university degree",
    Teer.COLLEGE_OR_APPRENTICESHIP_2_YEARS: (
        "College diploma or apprenticeship of 2 or more years, or supervisory duties"
    ),
    Teer.COLLEGE_OR_APPRENTICESHIP_UNDER_2_YEARS: (
        "College diploma or apprenticeship of less than 2 years, or more than 6 months of training"
    ),
    Teer.HIGH_SCHOOL: "High school diploma or several weeks of on-the-job training",
    Teer.NO_FORMAL_EDUCATION: "Short work demonstration and no formal education",
}

_SEE_CODE = re.compile(r"\(See (\d{5})")


@dataclass(frozen=True, slots=True)
class Exclusion:
    """An occupation that belongs elsewhere: "Computer applications engineer (See 21231 …)"."""

    text: str
    code: str | None  # the unit group it points to, when the text names one

    @classmethod
    def parse(cls, text: str) -> "Exclusion":
        match = _SEE_CODE.search(text)
        return cls(text=" ".join(text.split()), code=match.group(1) if match else None)


@dataclass(frozen=True, slots=True)
class Occupation:
    """A NOC 2021 unit group, the five-digit level people pick for immigration programs."""

    code: str
    title: str
    definition: str
    example_titles: tuple[str, ...]
    main_duties: tuple[str, ...]
    requirements: tuple[str, ...]
    exclusions: tuple[Exclusion, ...]

    @property
    def teer(self) -> Teer:
        return Teer(int(self.code[1]))

    @property
    def broad_category(self) -> int:
        return int(self.code[0])
