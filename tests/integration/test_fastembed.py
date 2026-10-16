"""Runs the real model. Skipped unless it's already downloaded (`noc-noc eval` downloads it)."""

from pathlib import Path

import numpy as np
import pytest

from noc_noc.infrastructure.dense import FastEmbedder

CACHE = Path("data/cache")
pytestmark = pytest.mark.skipif(
    not (CACHE / "models").exists(), reason="embedding model not downloaded"
)


def test_vectors_are_unit_length_and_cached(tmp_path: Path) -> None:
    embedder = FastEmbedder(CACHE)
    embedder._cache_dir = tmp_path  # keep the test's vectors out of the real cache
    first = embedder.passages(["taxi driver", "film developer"])
    assert first.shape == (2, 384)
    assert np.allclose(np.linalg.norm(first, axis=1), 1, atol=1e-3)
    assert list(tmp_path.glob("vectors/*.npy"))
    assert np.array_equal(embedder.passages(["taxi driver", "film developer"]), first)


def test_meaning_beats_shared_words() -> None:
    embedder = FastEmbedder(CACHE)
    taxi, film = embedder.passages(
        ["Taxi and limousine drivers", "Photographic and film processors"]
    )
    query = embedder.query("uber driver")
    assert query @ taxi > query @ film
