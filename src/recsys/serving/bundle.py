"""Everything the online service needs, built offline and saved as one file.

The bundle holds the retrievers fit on all history, the ranker trained on the validation window,
and the precomputed user and item statistics, so a request never touches the raw interaction log.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.baselines import build_user_item_matrix
from recsys.ranking.features import feature_columns, item_statistics, user_statistics
from recsys.ranking.pipeline import build_stage, fit_retrievers
from recsys.ranking.ranker import train_ranker
from recsys.split import Split

FALLBACK_LENGTH = 100


@dataclass
class ServingBundle:
    version: str
    n_users: int
    n_items: int
    per_retriever: int
    retrievers: dict
    ranker: object
    feature_names: list[str]
    candidate_columns: list[str]  # columns of a candidate frame after user_id, in cache order
    user_stats: pd.DataFrame
    item_stats: pd.DataFrame
    seen: sp.csr_matrix
    fallback_items: np.ndarray  # recent-popularity ranking, best first, for users with no history


def build_bundle(config: dict, split: Split, n_users: int, n_items: int) -> ServingBundle:
    """Train the ranker the way `make ranking` does, then refit retrievers for serving."""
    ranking = config["ranking"]
    per_retriever = ranking["candidates_per_retriever"]

    train_retrievers = fit_retrievers(config, n_users, n_items, split.train, "train")
    features, labels, *_ = build_stage(
        train_retrievers, split.train, split.val, n_users, n_items, per_retriever
    )
    columns = feature_columns(features)
    ranker = train_ranker(
        features, labels, columns, ranking["lightgbm"],
        ranking["holdout_fraction"], ranking["early_stopping_rounds"],
    )

    history = pd.concat([split.train, split.val])
    retrievers = fit_retrievers(config, n_users, n_items, history, "train + validation")
    fallback_scores = retrievers["recent_popularity"].scores
    candidate_columns = [
        column for column in features.columns
        if column.startswith(("item_id", "rank_", "score_"))
    ]
    return ServingBundle(
        version=format(int(time.time()), "x"),
        n_users=n_users,
        n_items=n_items,
        per_retriever=per_retriever,
        retrievers=retrievers,
        ranker=ranker,
        feature_names=columns,
        candidate_columns=candidate_columns,
        user_stats=user_statistics(history),
        item_stats=item_statistics(history),
        seen=build_user_item_matrix(history, n_users, n_items),
        fallback_items=np.argsort(-fallback_scores, kind="stable")[:FALLBACK_LENGTH],
    )


def save_bundle(bundle: ServingBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> ServingBundle:
    return joblib.load(path)
