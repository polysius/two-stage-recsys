"""Merge the candidates proposed by several retrievers into one frame per (user, item) pair."""

import numpy as np
import pandas as pd
import scipy.sparse as sp


def generate_candidates(
    retrievers: dict,
    users: np.ndarray,
    seen: sp.csr_matrix,
    per_retriever: int,
    batch_size: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Return one row per distinct (user, item) proposed by any retriever.

    Columns: `user_id`, `item_id`, `rank_<name>` (1 is best, NaN when that retriever did not
    propose the item) and `score_<name>` for retrievers that can score arbitrary pairs.
    """
    candidates = None
    for name, retriever in retrievers.items():
        proposed = np.vstack([
            retriever.recommend(users[start : start + batch_size], per_retriever, seen)
            for start in range(0, len(users), batch_size)
        ])
        ranks = pd.DataFrame({
            "user_id": np.repeat(users, per_retriever),
            "item_id": proposed.ravel(),
            f"rank_{name}": np.tile(np.arange(1, per_retriever + 1), len(users)),
        })
        candidates = (
            ranks if candidates is None
            else candidates.merge(ranks, on=["user_id", "item_id"], how="outer")
        )

    # A pair one retriever proposed can still be scored by another, so scores need no fill value.
    for name, retriever in retrievers.items():
        if hasattr(retriever, "score_pairs"):
            candidates[f"score_{name}"] = retriever.score_pairs(
                candidates["user_id"].to_numpy(), candidates["item_id"].to_numpy()
            )
    # Order candidates randomly inside each user. Tied scores are then broken by chance, not by
    # item id, and item ids track popularity over the whole log, including the evaluation window.
    shuffle_key = np.random.default_rng(seed).random(len(candidates))
    ordered = candidates.assign(_shuffle=shuffle_key).sort_values(["user_id", "_shuffle"])
    return ordered.drop(columns="_shuffle").reset_index(drop=True)


def label_candidates(candidates: pd.DataFrame, eval_table: pd.DataFrame) -> np.ndarray:
    """1 when the user interacted with the item in the evaluation window, else 0."""
    positives = eval_table[["user_id", "item_id"]].drop_duplicates().assign(label=1)
    pairs = candidates[["user_id", "item_id"]]
    merged = pairs.merge(positives, how="left", on=["user_id", "item_id"])
    return merged["label"].fillna(0).to_numpy(dtype=np.int8)
