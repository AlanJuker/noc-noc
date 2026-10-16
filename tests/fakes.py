import hashlib
import re
from collections.abc import Sequence

import numpy as np

from noc_noc.infrastructure.dense import Vectors


class WordHashEmbedder:
    """Deterministic stand-in for a sentence model: a normalized bag of hashed words.

    Texts that share words point in similar directions, which is all the retrievers need in tests.
    """

    DIM = 64

    def __init__(self) -> None:
        self.passage_calls = 0

    def _vector(self, text: str) -> Vectors:
        vector = np.zeros(self.DIM, dtype=np.float32)
        for word in re.findall(r"[a-z]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def passages(self, texts: Sequence[str]) -> Vectors:
        self.passage_calls += 1
        return np.array([self._vector(t) for t in texts], dtype=np.float32)

    def query(self, text: str) -> Vectors:
        return self._vector(text)
