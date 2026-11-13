import numpy as np
import pandas as pd

from recsys.baselines import PopularityRecommender, build_user_item_matrix
from recsys.evaluation import score_recommender
from recsys.retrieval.als import ALSRetriever
from recsys.retrieval.faiss_index import EmbeddingRetriever
from recsys.retrieval.two_tower import TwoTowerRetriever
from recsys.split import time_split

N_USERS, N_ITEMS = 200, 60


def _split(interactions):
    return time_split(interactions, val_fraction=0.1, test_fraction=0.1)


def test_faiss_retrieval_matches_brute_force_and_skips_seen_items():
    rng = np.random.default_rng(0)
    users, items = rng.normal(size=(20, 8)), rng.normal(size=(50, 8))
    seen_table = {"user_id": np.arange(20), "item_id": rng.integers(0, 50, 20)}

    seen = build_user_item_matrix(pd.DataFrame(seen_table), 20, 50)

    recommended = EmbeddingRetriever(users, items).recommend(np.arange(20), 5, seen)

    scores = users @ items.T
    scores[seen.nonzero()] = -np.inf
    expected = np.argsort(-scores, axis=1)[:, :5]
    assert np.array_equal(recommended, expected)


def test_als_beats_popularity_on_grouped_taste_data(interactions):
    split = _split(interactions)
    als = ALSRetriever(N_USERS, N_ITEMS, factors=8, regularization=0.01, iterations=10)
    als.fit(split.train)
    popularity = PopularityRecommender(N_ITEMS).fit(split.train)

    def recall(model):
        scores = score_recommender(model, split.train, split.val, N_USERS, N_ITEMS, [10])
        return scores["recall@10"]

    assert recall(als) > recall(popularity)


def test_two_tower_loss_falls_and_beats_popularity(interactions):
    split = _split(interactions)
    model = TwoTowerRetriever(
        N_USERS, N_ITEMS, embedding_dim=16, output_dim=16, epochs=30, batch_size=64,
        learning_rate=0.01, temperature=0.1,
    ).fit(split.train)
    popularity = PopularityRecommender(N_ITEMS).fit(split.train)

    assert model.loss_history[-1] < model.loss_history[0]

    def recall(recommender):
        scores = score_recommender(recommender, split.train, split.val, N_USERS, N_ITEMS, [10])
        return scores["recall@10"]

    assert recall(model) > recall(popularity)
