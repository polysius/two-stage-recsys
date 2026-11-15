"""Steps shared by the ranking experiment script and the serving bundle builder."""

import time

import numpy as np

from recsys.baselines import build_user_item_matrix
from recsys.evaluation import relevant_items_by_user
from recsys.ranking.candidates import generate_candidates, label_candidates
from recsys.ranking.features import build_features
from recsys.retrieval.registry import build_retrievers


def fit_retrievers(config, n_users, n_items, table, label):
    retrievers = build_retrievers(config, n_users, n_items)
    for name, retriever in retrievers.items():
        started = time.perf_counter()
        retriever.fit(table)
        print(f"fitted {name} on {label} in {time.perf_counter() - started:.0f}s", flush=True)
    return retrievers


def build_stage(retrievers, history, eval_table, n_users, n_items, per_retriever):
    """Candidates, features and labels for every user who has interactions in `eval_table`.

    Returns (features, labels, users, candidate_recall, candidates_per_user).
    """
    users = np.array(sorted(relevant_items_by_user(eval_table)))
    seen = build_user_item_matrix(history, n_users, n_items)
    candidates = generate_candidates(retrievers, users, seen, per_retriever)
    features = build_features(candidates, history)
    labels = label_candidates(candidates, eval_table)
    n_relevant = len(eval_table[["user_id", "item_id"]].drop_duplicates())
    candidate_recall = labels.sum() / n_relevant
    print(f"{len(users)} users, {len(features)} candidates, "
          f"candidate recall {candidate_recall:.4f}", flush=True)
    return features, labels, users, candidate_recall, len(features) / len(users)
