import numpy as np

from recsys.baselines import (
    ItemItemCF,
    PopularityRecommender,
    RecentPopularityRecommender,
    build_user_item_matrix,
)
from recsys.evaluation import score_recommender
from recsys.split import time_split

N_USERS, N_ITEMS = 200, 60


def _split(interactions):
    return time_split(interactions, val_fraction=0.1, test_fraction=0.1)


def test_recommendations_never_include_seen_items(interactions):
    split = _split(interactions)
    seen = build_user_item_matrix(split.train, N_USERS, N_ITEMS)
    users = np.arange(20)
    for recommender in (
        PopularityRecommender(N_ITEMS).fit(split.train),
        ItemItemCF(N_USERS, N_ITEMS, n_neighbors=10).fit(split.train),
    ):
        recommended = recommender.recommend(users, 10, seen)
        for user, row in zip(users, recommended):
            assert not set(row) & set(seen[user].indices)


def test_recent_popularity_prefers_items_popular_in_the_window(interactions):
    train = interactions.copy()
    cutoff = train["timestamp"].max()
    recent_hit = 59
    extra = train.tail(30).copy()
    extra["item_id"] = recent_hit
    extra["timestamp"] = cutoff
    recommender = RecentPopularityRecommender(N_ITEMS, window_days=1).fit(extra)
    assert recommender.scores.argmax() == recent_hit


def test_item_cf_beats_popularity_on_grouped_taste_data(interactions):
    split = _split(interactions)
    k_values = [10]
    popularity = PopularityRecommender(N_ITEMS).fit(split.train)
    item_cf = ItemItemCF(N_USERS, N_ITEMS, n_neighbors=20).fit(split.train)
    popularity_scores = score_recommender(
        popularity, split.train, split.val, N_USERS, N_ITEMS, k_values
    )
    cf_scores = score_recommender(item_cf, split.train, split.val, N_USERS, N_ITEMS, k_values)
    assert cf_scores["recall@10"] > popularity_scores["recall@10"]
