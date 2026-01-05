"""Wrapper around FAISS index and document id mapping."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, *, index_path: str | Path, doc_ids_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.doc_ids_path = Path(doc_ids_path)

        self.index = faiss.read_index(str(self.index_path))
        self.doc_ids = np.load(self.doc_ids_path, allow_pickle=True)

        if self.index.ntotal != len(self.doc_ids):
            raise RuntimeError(
                "FAISS index vector count does not match doc_ids length"
            )

    def search(self, vector: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        scores, indices = self.index.search(vector.astype("float32"), top_k)
        matched_ids: List[int] = []
        matched_scores: List[float] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            matched_ids.append(self.doc_ids[idx])
            matched_scores.append(float(score))
        return matched_ids, matched_scores


__all__ = ["FaissIndex"]
