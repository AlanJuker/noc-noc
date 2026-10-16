"""Reranking the top candidates with a language model, and keeping it honest.

The model only reorders codes it was given and cites duties by id; anything else it returns is
dropped, and candidates it leaves out keep their original order after the ones it ranked.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from noc_noc.application.benchmark import Retriever
from noc_noc.domain.classification import Classification
from noc_noc.domain.occupation import Occupation


def duty_id(code: str, index: int) -> str:
    return f"{code}-d{index + 1}"


@dataclass(frozen=True, slots=True)
class RankedCode:
    code: str
    duty_ids: tuple[str, ...]  # the duties the model says match, as "21232-d3"


@dataclass(frozen=True, slots=True)
class Match:
    occupation: Occupation
    matching_duties: tuple[str, ...]  # duty texts, verified to belong to the occupation


class RankModel(Protocol):
    def rank(self, query: str, candidates: Sequence[Occupation]) -> list[RankedCode]: ...


class Reranker:
    def __init__(
        self, base: Retriever, classification: Classification, model: RankModel, depth: int = 10
    ) -> None:
        self._base = base
        self._classification = classification
        self._model = model
        self._depth = depth

    def explain(self, query: str, k: int) -> list[Match]:
        codes = self._base.search(query, max(k, self._depth))
        head, tail = codes[: self._depth], codes[self._depth :]
        candidates = [o for c in head if (o := self._classification.get(c))]
        by_code = {o.code: o for o in candidates}

        matches: list[Match] = []
        for ranked in self._model.rank(query, candidates):
            occupation = by_code.pop(ranked.code, None)
            if occupation is None:  # not a candidate, or ranked twice
                continue
            duties = {duty_id(occupation.code, i): d for i, d in enumerate(occupation.main_duties)}
            cited = tuple(duties[i] for i in dict.fromkeys(ranked.duty_ids) if i in duties)
            matches.append(Match(occupation, cited))
        # Candidates the model left out keep their original order
        matches += [Match(o, ()) for o in candidates if o.code in by_code]
        matches += [Match(o, ()) for c in tail if (o := self._classification.get(c))]
        return matches[:k]

    def search(self, query: str, k: int) -> list[str]:
        return [m.occupation.code for m in self.explain(query, k)]
