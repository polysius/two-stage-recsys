"""Alternating least squares matrix factorisation for implicit feedback."""

import pandas as pd
from implicit.als import AlternatingLeastSquares

from recsys.baselines import build_user_item_matrix
from recsys.retrieval.faiss_index import EmbeddingRetriever


class ALSRetriever:
    def __init__(self, n_users: int, n_items: int, factors: int, regularization: float,
                 iterations: int, alpha: float = 40.0, seed: int = 0):
        # alpha scales the confidence of every observed interaction. Without it, ALS treats the
        # single observed rating as weak evidence and ranks poorly on this dataset.
        self.alpha = alpha
        self.n_users, self.n_items = n_users, n_items
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=seed,
            use_gpu=False,
        )
        self.retriever: EmbeddingRetriever | None = None

    def fit(self, train: pd.DataFrame) -> "ALSRetriever":
        user_items = build_user_item_matrix(train, self.n_users, self.n_items)
        self.model.fit((user_items * self.alpha).tocsr(), show_progress=False)
        self.retriever = EmbeddingRetriever(self.model.user_factors, self.model.item_factors)
        return self

    def recommend(self, users, k, seen):
        return self.retriever.recommend(users, k, seen)

    def score_pairs(self, users, items):
        return self.retriever.score_pairs(users, items)
