"""LightGBM LambdaMART ranker over retrieved candidates."""

import lightgbm as lgb
import numpy as np
import pandas as pd


def _group_sizes(user_ids: pd.Series) -> np.ndarray:
    """Rows per user for a frame sorted by user, in the order LightGBM expects."""
    return user_ids.groupby(user_ids, sort=False).size().to_numpy()


def train_ranker(
    features: pd.DataFrame,
    labels: np.ndarray,
    columns: list[str],
    params: dict,
    holdout_fraction: float,
    early_stopping_rounds: int,
    seed: int = 0,
) -> lgb.LGBMRanker:
    """Fit on most users and stop early on a held-out group of users.

    The split is by user, never by row, because LambdaMART compares items inside one user's list.
    `features` must be sorted by `user_id`.
    """
    users = features["user_id"].unique()
    holdout_users = np.random.default_rng(seed).choice(
        users, size=int(len(users) * holdout_fraction), replace=False
    )
    is_holdout = features["user_id"].isin(holdout_users).to_numpy()
    fit, hold = features[~is_holdout], features[is_holdout]

    ranker = lgb.LGBMRanker(objective="lambdarank", random_state=seed, verbose=-1, **params)
    ranker.fit(
        fit[columns],
        labels[~is_holdout],
        group=_group_sizes(fit["user_id"]),
        eval_set=[(hold[columns], labels[is_holdout])],
        eval_group=[_group_sizes(hold["user_id"])],
        eval_at=[10],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return ranker


def top_k_per_user(features: pd.DataFrame, scores: np.ndarray, k: int) -> np.ndarray:
    """Item ids of the k highest-scoring candidates per user, with users in ascending order."""
    ordered = features[["user_id", "item_id"]].assign(score=scores).sort_values(
        ["user_id", "score"], ascending=[True, False], kind="stable"  # ties keep row order
    )
    top = ordered.groupby("user_id").head(k)
    if top.groupby("user_id").size().min() < k:
        raise ValueError(f"some users have fewer than {k} candidates")
    return top["item_id"].to_numpy().reshape(-1, k)
