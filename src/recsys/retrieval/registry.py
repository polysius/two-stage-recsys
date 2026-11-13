"""Build every retriever and baseline from the config so scripts compare the same models."""

from recsys.baselines import ItemItemCF, PopularityRecommender, RecentPopularityRecommender
from recsys.retrieval.als import ALSRetriever
from recsys.retrieval.two_tower import TwoTowerRetriever


def build_retrievers(config: dict, n_users: int, n_items: int) -> dict:
    return {
        "popularity": PopularityRecommender(n_items),
        "recent_popularity": RecentPopularityRecommender(
            n_items, config["baselines"]["recent_window_days"]
        ),
        "item_item_cf": ItemItemCF(n_users, n_items, config["baselines"]["item_cf_neighbors"]),
        "als": ALSRetriever(n_users, n_items, **config["als"]),
        "two_tower": TwoTowerRetriever(n_users, n_items, **config["two_tower"]),
    }
