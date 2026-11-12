"""Evaluate the baselines on the validation and test windows and write a results table.

Usage: python scripts/run_baselines.py --data-dir ~/recsys-data --output artifacts/baselines.md
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from recsys.baselines import ItemItemCF, PopularityRecommender, RecentPopularityRecommender
from recsys.data import load_interactions
from recsys.evaluation import score_recommender
from recsys.split import assert_no_time_leakage, time_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="artifacts/baselines.md")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    csv_path = Path(args.data_dir) / "raw" / f"{config['data']['category']}.csv"
    data = load_interactions(csv_path)
    split = time_split(data.table, config["split"]["val_fraction"], config["split"]["test_fraction"])
    assert_no_time_leakage(split)
    print(
        f"train={len(split.train)} val={len(split.val)} test={len(split.test)} "
        f"users={data.n_users} items={data.n_items} "
        f"dropped_unseen_share={split.dropped_unseen_share:.3f} "
        f"unknown_user_share={split.unknown_user_share:.3f} "
        f"unknown_item_share={split.unknown_item_share:.3f}"
    )

    k_values = config["eval"]["k_values"]
    baseline_config = config["baselines"]
    recommenders = {
        "popularity": PopularityRecommender(data.n_items),
        "recent_popularity": RecentPopularityRecommender(
            data.n_items, baseline_config["recent_window_days"]
        ),
        "item_item_cf": ItemItemCF(data.n_users, data.n_items, baseline_config["item_cf_neighbors"]),
    }

    rows = []
    for name, recommender in recommenders.items():
        recommender.fit(split.train)
        val_history = split.train
        test_history = pd.concat([split.train, split.val])
        for window, history, eval_table in [
            ("val", val_history, split.val),
            ("test", test_history, split.test),
        ]:
            metrics = score_recommender(
                recommender, history, eval_table, data.n_users, data.n_items, k_values
            )
            rows.append({"model": name, "window": window, **metrics})
            print(name, window, {key: round(value, 4) for key, value in metrics.items()})

    table = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table.round(4).to_markdown(index=False) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
