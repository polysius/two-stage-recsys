"""Small hyperparameter sweep for ALS and the two-tower model, scored on validation only.

The test window is never used here, so choosing a configuration cannot leak into the test
numbers. Usage: python scripts/tune_retrieval.py --model als --data-dir ~/recsys-data
"""

import argparse
import itertools
from pathlib import Path

import yaml

from recsys.data import load_interactions
from recsys.evaluation import score_recommender
from recsys.retrieval.als import ALSRetriever
from recsys.retrieval.two_tower import TwoTowerRetriever
from recsys.split import time_split

GRIDS = {
    "als": {"factors": [64, 128], "regularization": [0.01, 0.1], "alpha": [10, 40, 100]},
    "two_tower": {
        "learning_rate": [0.03, 0.1],
        "epochs": [20],
        "batch_size": [1024],
        "temperature": [0.1, 0.3],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=GRIDS, required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    csv_path = Path(args.data_dir) / "raw" / f"{config['data']['category']}.csv"
    data = load_interactions(csv_path)
    split = time_split(data.table, config["split"]["val_fraction"], config["split"]["test_fraction"])

    base = config[args.model]
    grid = GRIDS[args.model]
    for values in itertools.product(*grid.values()):
        settings = {**base, **dict(zip(grid, values))}
        if args.model == "als":
            model = ALSRetriever(data.n_users, data.n_items, **settings)
        else:
            model = TwoTowerRetriever(data.n_users, data.n_items, **settings)
        model.fit(split.train)
        scores = score_recommender(model, split.train, split.val, data.n_users, data.n_items, [100])
        losses = getattr(model, "loss_history", [])
        print(dict(zip(grid, values)), f"val recall@100={scores['recall@100']:.4f}",
              f"final loss={losses[-1]:.3f}" if losses else "", flush=True)


if __name__ == "__main__":
    main()
