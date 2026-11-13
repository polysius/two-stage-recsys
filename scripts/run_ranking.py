"""Train the LightGBM ranker and compare retrieval-only against retrieval plus ranking.

Protocol:
  1. Fit every retriever on train.
  2. Build candidates and features for validation users from train history, label them with the
     validation window, and train the ranker on them (early stopping on held-out users).
  3. Build candidates and features for test users from train + validation history and score
     them with the ranker. The test window is never used before this step.

Usage: python scripts/run_ranking.py --data-dir ~/recsys-data --output artifacts/ranking.md
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recsys.baselines import build_user_item_matrix
from recsys.data import load_interactions
from recsys.evaluation import relevant_items_by_user, score_recommender
from recsys.metrics import evaluate
from recsys.ranking.candidates import generate_candidates, label_candidates
from recsys.ranking.features import GROUPS, build_features, feature_columns
from recsys.ranking.ranker import top_k_per_user, train_ranker
from recsys.retrieval.registry import build_retrievers
from recsys.split import assert_no_time_leakage, time_split

REPORT_METRICS = ["recall@10", "ndcg@10", "mrr", "recall@100"]


def build_stage(retrievers, history, eval_table, data, per_retriever):
    """Candidates, features and labels for every user who has interactions in `eval_table`."""
    users = np.array(sorted(relevant_items_by_user(eval_table)))
    seen = build_user_item_matrix(history, data.n_users, data.n_items)
    candidates = generate_candidates(retrievers, users, seen, per_retriever)
    features = build_features(candidates, history)
    labels = label_candidates(candidates, eval_table)
    n_relevant = len(eval_table[["user_id", "item_id"]].drop_duplicates())
    print(f"{len(users)} users, {len(features)} candidates, "
          f"candidate recall {labels.sum() / n_relevant:.4f}", flush=True)
    return features, labels, users, labels.sum() / n_relevant, len(features) / len(users)


def score_ranker(ranker, columns, features, users, history, eval_table, data, k_values):
    scores = ranker.predict(features[columns])
    recommended = top_k_per_user(features, scores, max(k_values))
    relevant = relevant_items_by_user(eval_table)
    popularity = np.bincount(history["item_id"], minlength=data.n_items).astype(float)
    return evaluate(recommended, [relevant[u] for u in users], popularity, k_values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="artifacts/ranking.md")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    ranking = config["ranking"]
    k_values = config["eval"]["k_values"]
    data = load_interactions(Path(args.data_dir) / "raw" / f"{config['data']['category']}.csv")
    split = time_split(data.table, config["split"]["val_fraction"], config["split"]["test_fraction"])
    assert_no_time_leakage(split)

    retrievers = build_retrievers(config, data.n_users, data.n_items)
    for name, retriever in retrievers.items():
        started = time.perf_counter()
        retriever.fit(split.train)
        print(f"fitted {name} in {time.perf_counter() - started:.0f}s", flush=True)

    test_history = pd.concat([split.train, split.val])
    train_x, train_y, _, val_recall, val_per_user = build_stage(
        retrievers, split.train, split.val, data, ranking["candidates_per_retriever"]
    )
    test_x, _, test_users, test_recall, test_per_user = build_stage(
        retrievers, test_history, split.test, data, ranking["candidates_per_retriever"]
    )

    def fit_and_score(groups):
        columns = feature_columns(train_x, groups)
        ranker = train_ranker(
            train_x, train_y, columns, ranking["lightgbm"],
            ranking["holdout_fraction"], ranking["early_stopping_rounds"],
        )
        metrics = score_ranker(
            ranker, columns, test_x, test_users, test_history, split.test, data, k_values
        )
        return ranker, columns, metrics

    rows = []
    for name, retriever in retrievers.items():
        metrics = score_recommender(
            retriever, test_history, split.test, data.n_users, data.n_items, k_values
        )
        rows.append({"model": f"{name} only", **metrics})

    ranker, columns, metrics = fit_and_score(None)
    print("ranker trees", ranker.best_iteration_, {k: round(metrics[k], 4) for k in REPORT_METRICS})
    rows.append({"model": "candidates + ranker", **metrics})
    comparison = pd.DataFrame(rows)[["model", *REPORT_METRICS]]

    importance = pd.Series(ranker.booster_.feature_importance("gain"), index=columns)
    importance = (100 * importance / importance.sum()).sort_values(ascending=False).round(1)

    variants = {"all features": None}
    variants.update({f"without {group}": [g for g in GROUPS if g != group] for group in GROUPS})
    variants.update({f"only {group}": [group] for group in GROUPS})
    ablation_rows = []
    for label, groups in variants.items():
        variant_metrics = metrics if groups is None else fit_and_score(groups)[2]
        ablation_rows.append({"features": label, **{k: variant_metrics[k] for k in REPORT_METRICS}})
        print(label, {k: round(variant_metrics[k], 4) for k in REPORT_METRICS}, flush=True)
    ablation = pd.DataFrame(ablation_rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "## Retrieval only vs retrieval + ranking (test window)\n\n"
        + comparison.round(4).to_markdown(index=False)
        + f"\n\nCandidate recall (share of relevant pairs inside the candidate set): "
        f"validation {val_recall:.4f}, test {test_recall:.4f}\n\n"
        f"Candidates per user: validation {val_per_user:.0f}, test {test_per_user:.0f}\n\n"
        f"Trees used: {ranker.best_iteration_}\n\n## Feature importance (gain, %)\n\n"
        + importance.to_frame("gain %").to_markdown() + "\n\n## Feature-group ablation (test window)\n\n"
        + ablation.round(4).to_markdown(index=False) + "\n"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
