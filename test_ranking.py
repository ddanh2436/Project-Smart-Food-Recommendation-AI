# -*- coding: utf-8 -*-
"""Offline self-checks for the radius fallback and the embedding cache.

No server, database or model needed:  python test_ranking.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import search
from config import settings
from data_store import RestaurantStore


def check_radius() -> None:
    def frame(*km):
        return pd.DataFrame({"distance_km": list(km)})

    relaxed: list[str] = []
    # Enough within 15 km: keep only those.
    kept = search._apply_radius(frame(1, 2, 3, 4, 5, 300), (15, 50), relaxed)
    assert kept["distance_km"].max() == 5 and not relaxed
    # Too few within 15 km, enough within 50: widen to 50.
    kept = search._apply_radius(frame(1, 20, 30, 40, 45, 300), (15, 50), relaxed)
    assert len(kept) == 5 and not relaxed
    # A few local beat a list led by another province.
    kept = search._apply_radius(frame(3, 22, 131, 132), (15, 50), relaxed)
    assert kept["distance_km"].tolist() == [3, 22] and not relaxed
    # Nothing local at all: give up on the radius and say so.
    kept = search._apply_radius(frame(300, 600), (15, 50), relaxed)
    assert len(kept) == 2 and relaxed == ["radius"]


class FakeEmbedder:
    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, texts, **_):
        self.encoded += texts
        return np.array([[len(t), 1.0] for t in texts], dtype="float32")


def check_embedding_cache() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        object.__setattr__(settings, "embedding_cache_path", str(Path(tmp) / "c.npz"))
        store = RestaurantStore()
        store._embedder = FakeEmbedder()

        first = store._encode_cached(["phở", "bún bò", "phở"])
        assert store._embedder.encoded == ["bún bò", "phở"]  # duplicates once
        assert first.shape == (3, 2) and first[0, 0] == len("phở")

        # A fresh process with one new row encodes only that row.
        store._embedder = FakeEmbedder()
        second = store._encode_cached(["bún bò", "phở", "lẩu mắm"])
        assert store._embedder.encoded == ["lẩu mắm"]
        assert second[2, 0] == len("lẩu mắm")

        # A different model never reuses another model's vectors.
        object.__setattr__(settings, "embedding_model", "other/model")
        store._embedder = FakeEmbedder()
        store._encode_cached(["phở"])
        assert store._embedder.encoded == ["phở"]


if __name__ == "__main__":
    check_radius()
    check_embedding_cache()
    print("ranking self-checks passed")
