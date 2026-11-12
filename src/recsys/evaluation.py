"""Shared helpers to turn a split into evaluation inputs and score a recommender."""

import numpy as np
import pandas as pd

from recsys.baselines import build_user_item_matrix
from recsys.metrics import evaluate


def relevant_items_by_user(eval_table: pd.DataFrame) -> dict[int, set[int]]:
    return eval_table.groupby("user_id")["item_id"].apply(set).to_dict()


def score_recommender(
    recommender,
    history: pd.DataFrame,
    eval_table: pd.DataFrame,
    n_users: int,
    n_items: int,
    k_values: list[int],
    batch_size: int = 2000,
) -> dict[str, float]:
    """Evaluate `recommender` on `eval_table`, masking everything in `history`.

    For validation, history is train. For test, history is train plus validation, because a
    user could have seen validation items before the test period starts.
    """
    seen = build_user_item_matrix(history, n_users, n_items)
    relevant = relevant_items_by_user(eval_table)
    users = np.array(sorted(relevant))
    item_popularity = np.bincount(history["item_id"], minlength=n_items).astype(float)

    batches = [users[i : i + batch_size] for i in range(0, len(users), batch_size)]
    recommended = np.vstack(
        [recommender.recommend(batch, max(k_values), seen) for batch in batches]
    )
    return evaluate(recommended, [relevant[u] for u in users], item_popularity, k_values)
