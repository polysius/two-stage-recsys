"""Non-personalised and neighbourhood baselines.

All recommenders share one interface:
    fit(train)                  learn from the train interactions only
    recommend(users, k, seen)   top-k item ids per user, never an item the user has seen
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

MS_PER_DAY = 24 * 60 * 60 * 1000


def build_user_item_matrix(table: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    """Binary implicit-feedback matrix: 1 when the user interacted with the item."""
    values = np.ones(len(table), dtype=np.float32)
    matrix = sp.csr_matrix((values, (table["user_id"], table["item_id"])), shape=(n_users, n_items))
    matrix.data[:] = 1.0  # collapse duplicate (user, item) pairs
    return matrix


def top_k_unseen(scores: np.ndarray, seen: sp.csr_matrix, k: int) -> np.ndarray:
    """Top-k columns of `scores` per row after masking items listed in `seen`."""
    scores = scores.copy()
    rows, cols = seen.nonzero()
    scores[rows, cols] = -np.inf
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


class PopularityRecommender:
    """Recommend the items with the most train interactions, identical for every user."""

    def __init__(self, n_items: int):
        self.n_items = n_items
        self.scores = np.zeros(n_items)

    def _score_items(self, train: pd.DataFrame) -> np.ndarray:
        return np.bincount(train["item_id"], minlength=self.n_items).astype(float)

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        self.scores = self._score_items(train)
        return self

    def recommend(self, users: np.ndarray, k: int, seen: sp.csr_matrix) -> np.ndarray:
        all_scores = np.tile(self.scores, (len(users), 1))
        return top_k_unseen(all_scores, seen[users], k)


class RecentPopularityRecommender(PopularityRecommender):
    """Popularity counted only over the last `window_days` of train."""

    def __init__(self, n_items: int, window_days: int):
        super().__init__(n_items)
        self.window_days = window_days

    def _score_items(self, train: pd.DataFrame) -> np.ndarray:
        cutoff = train["timestamp"].max() - self.window_days * MS_PER_DAY
        return super()._score_items(train[train["timestamp"] >= cutoff])


class ItemItemCF:
    """Cosine item-item collaborative filtering keeping the strongest neighbours per item."""

    BLOCK_SIZE = 2000

    def __init__(self, n_users: int, n_items: int, n_neighbors: int):
        self.n_users, self.n_items, self.n_neighbors = n_users, n_items, n_neighbors
        self.similarity = sp.csr_matrix((n_items, n_items), dtype=np.float32)
        self.history = sp.csr_matrix((n_users, n_items), dtype=np.float32)

    def fit(self, train: pd.DataFrame) -> "ItemItemCF":
        self.history = build_user_item_matrix(train, self.n_users, self.n_items)
        item_users = self.history.T.tocsr()
        norms = np.sqrt(np.asarray(item_users.sum(axis=1)).ravel()).clip(min=1.0)
        normalised = (sp.diags(1.0 / norms) @ item_users).tocsr()

        # Build the similarity matrix in row blocks so memory stays bounded.
        blocks = []
        for start in range(0, self.n_items, self.BLOCK_SIZE):
            block = (normalised[start : start + self.BLOCK_SIZE] @ normalised.T).toarray()
            rows = np.arange(block.shape[0])
            block[rows, start + rows] = 0.0  # an item is not its own neighbour
            blocks.append(self._keep_top_neighbors(block))
        self.similarity = sp.vstack(blocks).tocsr()
        return self

    def _keep_top_neighbors(self, block: np.ndarray) -> sp.csr_matrix:
        keep = min(self.n_neighbors, block.shape[1] - 1)
        weakest = np.argpartition(block, -keep, axis=1)[:, :-keep]
        np.put_along_axis(block, weakest, 0.0, axis=1)
        return sp.csr_matrix(block)

    def recommend(self, users: np.ndarray, k: int, seen: sp.csr_matrix) -> np.ndarray:
        scores = (self.history[users] @ self.similarity.T).toarray()
        return top_k_unseen(scores, seen[users], k)
