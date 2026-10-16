"""Embedding retrieval and its fusion with the lexical ranking."""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from noc_noc.application.benchmark import Retriever
from noc_noc.domain.classification import Classification
from noc_noc.domain.occupation import Occupation
from noc_noc.infrastructure.lexical import document

Vectors = NDArray[np.float32]


class Embedder(Protocol):
    def passages(self, texts: Sequence[str]) -> Vectors:
        """Unit-length vectors, one row per text."""
        ...

    def query(self, text: str) -> Vectors: ...


class FastEmbedder:
    """BAAI/bge-small-en-v1.5 through ONNX Runtime: about 130 MB, CPU only, no PyTorch.

    Passage vectors are cached on disk by a hash of the model and the texts: embedding the ~28,000
    passages takes a couple of minutes, reading them back takes milliseconds.
    """

    MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, cache_dir: Path) -> None:
        from fastembed import TextEmbedding  # noqa: PLC0415 (loading the model is slow)

        self._cache_dir = cache_dir
        self._model = TextEmbedding(self.MODEL, cache_dir=str(cache_dir / "models"))

    def passages(self, texts: Sequence[str]) -> Vectors:
        key = hashlib.sha256("\0".join([self.MODEL, *texts]).encode()).hexdigest()[:16]
        path = self._cache_dir / "vectors" / f"{key}.npy"
        if path.exists():
            cached: Vectors = np.load(path)
            return cached
        vectors = np.array(list(self._model.passage_embed(texts)), dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def query(self, text: str) -> Vectors:
        return np.array(next(iter(self._model.query_embed(text))), dtype=np.float32)


def _passages(occupation: Occupation) -> list[str]:
    return [occupation.title, *occupation.example_titles, *occupation.main_duties]


class DenseRetriever:
    """Cosine similarity between the query and each unit group.

    "document" embeds each unit group's whole text once. "passages" embeds its title, every example
    title and every duty separately and scores the group by its best passage.
    """

    def __init__(
        self,
        classification: Classification,
        embedder: Embedder,
        granularity: Literal["document", "passages"] = "passages",
    ) -> None:
        self._embedder = embedder
        self._codes = [o.code for o in classification]
        texts = [
            [document(o)] if granularity == "document" else _passages(o) for o in classification
        ]
        # Passages of the same unit group are contiguous: `starts` marks where each group begins.
        self._starts = np.cumsum([0, *(len(t) for t in texts[:-1])])
        self._vectors = embedder.passages([text for group in texts for text in group])

    def search(self, query: str, k: int) -> list[str]:
        similarity = self._vectors @ self._embedder.query(query)
        best_per_group = np.maximum.reduceat(similarity, self._starts)
        return [self._codes[i] for i in np.argsort(-best_per_group, kind="stable")[:k]]


class HybridRetriever:
    """Reciprocal rank fusion: each ranking adds 1 / (60 + rank) to a code's score."""

    RRF_K = 60
    DEPTH = 50

    def __init__(self, retrievers: Sequence[Retriever]) -> None:
        self._retrievers = retrievers

    def search(self, query: str, k: int) -> list[str]:
        scores: dict[str, float] = {}
        for retriever in self._retrievers:
            for rank, code in enumerate(retriever.search(query, self.DEPTH)):
                scores[code] = scores.get(code, 0.0) + 1 / (self.RRF_K + rank + 1)
        return sorted(scores, key=lambda code: -scores[code])[:k]
