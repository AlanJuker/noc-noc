from collections.abc import Iterable, Iterator

from noc_noc.domain.occupation import Occupation


class Classification:
    """The 516 unit groups of NOC 2021, by code."""

    def __init__(self, occupations: Iterable[Occupation]) -> None:
        self._by_code = {o.code: o for o in occupations}

    def __len__(self) -> int:
        return len(self._by_code)

    def __iter__(self) -> Iterator[Occupation]:
        return iter(self._by_code.values())

    def get(self, code: str) -> Occupation | None:
        return self._by_code.get(code)

    def confused_with(self, code: str) -> set[str]:
        """Codes whose exclusions send people to `code`, or that `code`'s exclusions point to.

        Statistics Canada lists these because they're easy to mix up.
        """
        pointed_to = {e.code for e in self._by_code[code].exclusions if e.code}
        pointing_here = {
            o.code for o in self if any(e.code == code for e in o.exclusions) and o.code != code
        }
        return (pointed_to | pointing_here) - {code}
