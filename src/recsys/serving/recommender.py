"""The online two-stage recommendation for a single user."""

from dataclasses import dataclass

import numpy as np

from recsys.ranking.candidates import generate_candidates
from recsys.ranking.features import join_features
from recsys.serving.bundle import ServingBundle
from recsys.serving.cache import NullCache


@dataclass(frozen=True)
class Recommendation:
    user_id: int
    source: str  # "ranked" for users with history, "cold_start" for everyone else
    cache: str  # "hit", "miss" or "off"
    items: list[tuple[int, float | None]]  # (item id, ranker score), best first


class TwoStageRecommender:
    def __init__(self, bundle: ServingBundle, cache=None):
        self.bundle = bundle
        self.cache = cache if cache is not None else NullCache()

    def is_known(self, user_id: int) -> bool:
        """A user is known when the bundle has statistics for them, which needs some history."""
        return user_id in self.bundle.user_stats.index

    def recommend(self, user_id: int, k: int) -> Recommendation:
        if not self.is_known(user_id):
            return self._cold_start(user_id, k)

        candidates = self.cache.get(user_id)
        cache_state = "off" if not self.cache.enabled else "hit"
        if candidates is None:
            candidates = self._retrieve(user_id)
            self.cache.put(user_id, candidates)
            cache_state = "off" if not self.cache.enabled else "miss"
        return Recommendation(user_id, "ranked", cache_state, self._rank(candidates, k))

    def _retrieve(self, user_id: int):
        bundle = self.bundle
        return generate_candidates(
            bundle.retrievers, np.array([user_id]), bundle.seen, bundle.per_retriever
        )

    def _rank(self, candidates, k: int) -> list[tuple[int, float]]:
        bundle = self.bundle
        features = join_features(candidates, bundle.user_stats, bundle.item_stats)
        scores = bundle.ranker.predict(features[bundle.feature_names])
        best = np.argsort(-scores, kind="stable")[:k]
        return [(int(candidates["item_id"].iloc[i]), float(scores[i])) for i in best]

    def _cold_start(self, user_id: int, k: int) -> Recommendation:
        """Recent popularity: all we can say about someone with no history."""
        items = [(int(item), None) for item in self.bundle.fallback_items[:k]]
        return Recommendation(user_id, "cold_start", "off", items)
