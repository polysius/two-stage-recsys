import numpy as np
import pandas as pd

from recsys.baselines import PopularityRecommender, build_user_item_matrix
from recsys.evaluation import relevant_items_by_user, score_recommender
from recsys.metrics import evaluate
from recsys.ranking.candidates import generate_candidates, label_candidates
from recsys.ranking.features import build_features, feature_columns
from recsys.ranking.ranker import top_k_per_user, train_ranker
from recsys.retrieval.als import ALSRetriever
from recsys.split import time_split

N_USERS, N_ITEMS = 200, 60


def _stage(interactions, history_part, eval_part, retrievers):
    users = np.array(sorted(relevant_items_by_user(eval_part)))
    seen = build_user_item_matrix(history_part, N_USERS, N_ITEMS)
    candidates = generate_candidates(retrievers, users, seen, per_retriever=20)
    features = build_features(candidates, history_part)
    return users, features, label_candidates(candidates, eval_part)


def _fitted_retrievers(train):
    return {
        "popularity": PopularityRecommender(N_ITEMS).fit(train),
        "als": ALSRetriever(N_USERS, N_ITEMS, factors=8, regularization=0.01, iterations=10).fit(
            train
        ),
    }


def test_candidates_are_the_union_of_retrievers_without_seen_items(interactions):
    split = time_split(interactions, 0.1, 0.1)
    retrievers = _fitted_retrievers(split.train)
    users = np.array(sorted(relevant_items_by_user(split.val)))
    seen = build_user_item_matrix(split.train, N_USERS, N_ITEMS)

    candidates = generate_candidates(retrievers, users, seen, per_retriever=10)

    assert not candidates.duplicated(["user_id", "item_id"]).any()
    assert candidates[["rank_popularity", "rank_als"]].notna().any(axis=1).all()
    assert candidates["rank_als"].dropna().between(1, 10).all()
    assert "score_als" in candidates and "score_popularity" not in candidates
    seen_pairs = set(zip(*seen.nonzero()))
    assert not any(pair in seen_pairs for pair in zip(candidates["user_id"], candidates["item_id"]))


def test_labels_mark_exactly_the_evaluation_interactions():
    candidates = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [5, 6, 5]})
    eval_table = pd.DataFrame({"user_id": [0, 1, 1], "item_id": [6, 5, 5]})

    assert label_candidates(candidates, eval_table).tolist() == [0, 1, 1]


def test_features_only_use_history():
    history = pd.DataFrame({
        "user_id": [0, 0, 1], "item_id": [1, 2, 1], "rating": [4.0, 2.0, 5.0],
        "timestamp": [0, 86_400_000, 2 * 86_400_000],
    })
    candidates = pd.DataFrame({"user_id": [0, 1], "item_id": [1, 2]})

    features = build_features(candidates, history).set_index("user_id")

    assert features.loc[0, "user_mean_rating"] == 3.0
    assert features.loc[1, "item_mean_rating"] == 2.0
    assert features.loc[0, "item_log_count"] == np.log1p(2)
    # Days are measured back from the newest history row, so none can be negative.
    day_columns = ["item_age_days", "item_days_since_last", "user_days_since_last"]
    assert (features[day_columns] >= 0).all().all()
    assert features.loc[1, "user_days_since_last"] == 0


def test_feature_groups_split_columns_by_prefix():
    frame = pd.DataFrame(
        columns=["user_id", "item_id", "rank_als", "score_als", "user_x", "item_x", "pair_x", "label"]
    )

    assert feature_columns(frame, ["retriever"]) == ["rank_als", "score_als"]
    assert feature_columns(frame, ["user", "pair"]) == ["user_x", "pair_x"]
    assert feature_columns(frame) == ["rank_als", "score_als", "user_x", "item_x", "pair_x"]


def test_top_k_per_user_orders_by_score():
    features = pd.DataFrame({"user_id": [0, 0, 0, 1, 1, 1], "item_id": [1, 2, 3, 4, 5, 6]})
    scores = np.array([0.1, 0.9, 0.5, 0.3, 0.2, 0.8])

    assert top_k_per_user(features, scores, 2).tolist() == [[2, 3], [6, 4]]


def test_ranker_trained_on_validation_beats_popularity_on_test(interactions):
    split = time_split(interactions, 0.2, 0.2)
    retrievers = _fitted_retrievers(split.train)
    _, train_x, train_y = _stage(interactions, split.train, split.val, retrievers)
    test_history = pd.concat([split.train, split.val])
    users, test_x, _ = _stage(interactions, test_history, split.test, retrievers)

    columns = feature_columns(train_x)
    ranker = train_ranker(
        train_x, train_y, columns, {"n_estimators": 30, "min_child_samples": 5},
        holdout_fraction=0.2, early_stopping_rounds=10,
    )
    recommended = top_k_per_user(test_x, ranker.predict(test_x[columns]), 10)
    relevant = relevant_items_by_user(split.test)
    popularity = np.bincount(test_history["item_id"], minlength=N_ITEMS).astype(float)
    ranked = evaluate(recommended, [relevant[u] for u in users], popularity, [10])

    baseline = score_recommender(
        PopularityRecommender(N_ITEMS).fit(split.train), test_history, split.test,
        N_USERS, N_ITEMS, [10],
    )
    assert ranked["recall@10"] > baseline["recall@10"]
