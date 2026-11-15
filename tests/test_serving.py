from pathlib import Path

import fakeredis
import numpy as np
import pandas as pd
import pytest
import redis
import yaml
from fastapi.testclient import TestClient

from recsys.ranking.features import build_features
from recsys.serving.api import create_app
from recsys.serving.bundle import build_bundle
from recsys.serving.cache import RedisCandidateCache
from recsys.serving.recommender import TwoStageRecommender
from recsys.split import time_split

N_USERS, N_ITEMS = 200, 60


@pytest.fixture(scope="module")
def serving_setup():
    """A bundle trained on the small synthetic log, shared by every test in this file."""
    rng = np.random.default_rng(0)
    rows = []
    for user in range(N_USERS):
        group = user % 4
        preferred = np.arange(N_ITEMS)[np.arange(N_ITEMS) % 4 == group]
        for step, item in enumerate(rng.choice(preferred, size=12, replace=False)):
            rows.append((user, int(item), 5.0, 1_000 * (step * N_USERS + user)))
    table = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    table = table.sort_values("timestamp").reset_index(drop=True)

    config = {
        "baselines": {"recent_window_days": 90, "item_cf_neighbors": 10},
        "als": {"factors": 8, "regularization": 0.01, "iterations": 5, "alpha": 40},
        "two_tower": {
            "embedding_dim": 8, "output_dim": 8, "epochs": 3, "batch_size": 64,
            "learning_rate": 0.03, "temperature": 0.3,
        },
        "ranking": {
            "candidates_per_retriever": 20, "holdout_fraction": 0.2, "early_stopping_rounds": 5,
            "lightgbm": {"n_estimators": 20, "min_child_samples": 5},
        },
    }
    split = time_split(table, 0.2, 0.2)
    bundle = build_bundle(config, split, N_USERS, N_ITEMS)
    history = pd.concat([split.train, split.val])
    return bundle, history


def _client(bundle, cache=None) -> TestClient:
    return TestClient(create_app(TwoStageRecommender(bundle, cache)))


def _redis_cache(bundle, client=None) -> RedisCandidateCache:
    return RedisCandidateCache(
        client or fakeredis.FakeRedis(), bundle.version, bundle.candidate_columns, 60
    )


def test_recommend_returns_ranked_unseen_items(serving_setup):
    bundle, history = serving_setup
    user = 3

    response = _client(bundle).get("/recommend", params={"user_id": user, "k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "ranked" and body["cache"] == "off"
    items = body["items"]
    assert len(items) == 5
    scores = [item["score"] for item in items]
    assert scores == sorted(scores, reverse=True)
    seen = set(history.loc[history["user_id"] == user, "item_id"])
    assert not seen & {item["item_id"] for item in items}


def test_unknown_user_gets_the_cold_start_fallback(serving_setup):
    bundle, _ = serving_setup

    response = _client(bundle).get("/recommend", params={"user_id": 999_999, "k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "cold_start"
    assert [item["item_id"] for item in body["items"]] == bundle.fallback_items[:5].tolist()
    assert all(item["score"] is None for item in body["items"])


def test_second_request_is_served_from_the_cache_with_the_same_items(serving_setup):
    bundle, _ = serving_setup
    client = _client(bundle, _redis_cache(bundle))

    first = client.get("/recommend", params={"user_id": 7, "k": 10}).json()
    second = client.get("/recommend", params={"user_id": 7, "k": 10}).json()

    assert (first["cache"], second["cache"]) == ("miss", "hit")
    assert [i["item_id"] for i in first["items"]] == [i["item_id"] for i in second["items"]]
    assert [i["score"] for i in first["items"]] == pytest.approx([i["score"] for i in second["items"]])


def test_cache_round_trip_preserves_candidates_and_missing_ranks(serving_setup):
    bundle, _ = serving_setup
    recommender = TwoStageRecommender(bundle)
    candidates = recommender._retrieve(11)
    cache = _redis_cache(bundle)

    cache.put(11, candidates)
    restored = cache.get(11)

    assert restored["item_id"].tolist() == candidates["item_id"].tolist()
    expected = candidates[bundle.candidate_columns].to_numpy(dtype=np.float32)
    assert np.array_equal(restored[bundle.candidate_columns].to_numpy(), expected, equal_nan=True)
    assert candidates.filter(like="rank_").isna().any().any()  # the round trip covered NaN ranks


def test_a_redis_outage_slows_requests_but_does_not_fail_them(serving_setup):
    bundle, _ = serving_setup

    class BrokenRedis:
        def get(self, key):
            raise redis.ConnectionError("down")

        def set(self, key, value, ex=None):
            raise redis.ConnectionError("down")

    response = _client(bundle, _redis_cache(bundle, BrokenRedis())).get(
        "/recommend", params={"user_id": 5}
    )

    assert response.status_code == 200
    assert response.json()["cache"] == "miss"
    assert len(response.json()["items"]) == 10


def test_online_ranking_matches_the_offline_feature_pipeline(serving_setup):
    """Guards against train/serve skew: precomputed statistics must equal recomputed ones."""
    bundle, history = serving_setup
    recommender = TwoStageRecommender(bundle)
    user = 9
    candidates = recommender._retrieve(user)

    offline = build_features(candidates, history)
    offline_scores = bundle.ranker.predict(offline[bundle.feature_names])
    expected = candidates["item_id"].iloc[np.argsort(-offline_scores, kind="stable")[:10]]

    served = recommender.recommend(user, 10)

    assert [item for item, _ in served.items] == expected.tolist()


@pytest.mark.parametrize("params", [{"user_id": -1}, {"user_id": 1, "k": 0}, {"user_id": 1, "k": 101}])
def test_invalid_parameters_are_rejected(serving_setup, params):
    bundle, _ = serving_setup

    assert _client(bundle).get("/recommend", params=params).status_code == 422


def test_health_reports_the_bundle_version(serving_setup):
    bundle, _ = serving_setup

    body = _client(bundle).get("/health").json()

    assert body == {"status": "ok", "bundle_version": bundle.version}


def test_bundle_config_is_the_default_yaml_shape():
    config = yaml.safe_load(Path("configs/default.yaml").read_text())

    assert {"cache_hit", "cache_miss", "cold_start"} <= set(config["serving"]["latency_budget_ms"])
