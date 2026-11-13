"""Nearest-neighbour retrieval over item embeddings with FAISS."""

import faiss
import numpy as np
import scipy.sparse as sp


class EmbeddingRetriever:
    """Recommend by inner product between user vectors and an indexed set of item vectors.

    Both ALS and the two-tower model produce user and item vectors, so they share this class.
    The index is exact (flat). With 25k items an exact search is fast, and it keeps retrieval
    quality free of approximation error, so differences between retrievers come from the model.
    """

    def __init__(self, user_vectors: np.ndarray, item_vectors: np.ndarray):
        self.user_vectors = np.ascontiguousarray(user_vectors, dtype=np.float32)
        self.n_items = len(item_vectors)
        self.index = faiss.IndexFlatIP(item_vectors.shape[1])
        self.index.add(np.ascontiguousarray(item_vectors, dtype=np.float32))

    def recommend(self, users: np.ndarray, k: int, seen: sp.csr_matrix) -> np.ndarray:
        """Top-k items per user, skipping items in `seen`.

        FAISS cannot filter, so we ask for extra neighbours (as many as the busiest user in
        the batch has seen) and drop seen items afterwards.
        """
        seen_rows = seen[users]
        max_seen = int(np.diff(seen_rows.indptr).max(initial=0))
        search_size = min(self.n_items, k + max_seen)
        _, neighbours = self.index.search(self.user_vectors[users], search_size)

        recommended = np.empty((len(users), k), dtype=np.int64)
        for row, candidates in enumerate(neighbours):
            seen_items = set(seen_rows[row].indices)
            kept = [item for item in candidates if item not in seen_items][:k]
            recommended[row] = kept
        return recommended
