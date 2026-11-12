"""Ranking metrics.

`recommended` is an (n_users, k) integer array, best item first.
`relevant` is a list of item-id sets, one per user, in the same row order.
"""

import numpy as np


def recall_at_k(recommended: np.ndarray, relevant: list[set[int]], k: int) -> float:
    scores = [len(set(row[:k]) & truth) / len(truth) for row, truth in zip(recommended, relevant)]
    return float(np.mean(scores))


def ndcg_at_k(recommended: np.ndarray, relevant: list[set[int]], k: int) -> float:
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    scores = []
    for row, truth in zip(recommended, relevant):
        gains = np.array([item in truth for item in row[:k]], dtype=float)
        ideal_hits = min(len(truth), k)
        scores.append((gains * discounts[: len(gains)]).sum() / discounts[:ideal_hits].sum())
    return float(np.mean(scores))


def mrr(recommended: np.ndarray, relevant: list[set[int]]) -> float:
    """Mean reciprocal rank of the first relevant item (0 when none is recommended)."""
    reciprocal_ranks = []
    for row, truth in zip(recommended, relevant):
        hits = [rank for rank, item in enumerate(row, start=1) if item in truth]
        reciprocal_ranks.append(1.0 / hits[0] if hits else 0.0)
    return float(np.mean(reciprocal_ranks))


def catalogue_coverage(recommended: np.ndarray, n_items: int, k: int) -> float:
    """Share of the catalogue that appears in at least one user's top-k."""
    return len(np.unique(recommended[:, :k])) / n_items


def popularity_bias(recommended: np.ndarray, item_popularity: np.ndarray, k: int) -> float:
    """Average train popularity of recommended items, divided by the catalogue average.

    1.0 means recommendations are as popular as a random item; larger means more popular.
    """
    return float(item_popularity[recommended[:, :k]].mean() / item_popularity.mean())


def evaluate(
    recommended: np.ndarray,
    relevant: list[set[int]],
    item_popularity: np.ndarray,
    k_values: list[int],
) -> dict[str, float]:
    n_items = len(item_popularity)
    results = {"mrr": mrr(recommended, relevant)}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(recommended, relevant, k)
        results[f"ndcg@{k}"] = ndcg_at_k(recommended, relevant, k)
        results[f"coverage@{k}"] = catalogue_coverage(recommended, n_items, k)
        results[f"pop_bias@{k}"] = popularity_bias(recommended, item_popularity, k)
    return results
