"""TF-IDF retrieval: the baseline every other method has to beat."""

from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from noc_noc.domain.classification import Classification
from noc_noc.domain.occupation import Occupation


def document(occupation: Occupation) -> str:
    """The text a unit group is found by: its title, example job titles, definition and duties."""
    return "\n".join(
        [
            occupation.title,
            *occupation.example_titles,
            occupation.definition,
            *occupation.main_duties,
        ]
    )


class TfidfRetriever:
    def __init__(
        self, classification: Classification, analyzer: Literal["word", "char_wb"] = "word"
    ) -> None:
        self._codes = [o.code for o in classification]
        ngrams = (1, 2) if analyzer == "word" else (3, 5)
        self._vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=ngrams, sublinear_tf=True, strip_accents="unicode"
        )
        self._matrix = self._vectorizer.fit_transform([document(o) for o in classification])

    def search(self, query: str, k: int) -> list[str]:
        scores = (self._matrix @ self._vectorizer.transform([query]).T).toarray().ravel()
        best = np.argsort(-scores, kind="stable")[:k]
        return [self._codes[i] for i in best if scores[i] > 0]
