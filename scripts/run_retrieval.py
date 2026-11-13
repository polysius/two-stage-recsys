"""Compare the retrievers against the baselines on the validation and test windows.

Usage: python scripts/run_retrieval.py --data-dir ~/recsys-data --output artifacts/retrieval.md
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from recsys.data import load_interactions
from recsys.evaluation import score_recommender
from recsys.retrieval.registry import build_retrievers
from recsys.split import assert_no_time_leakage, time_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="artifacts/retrieval.md")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    csv_path = Path(args.data_dir) / "raw" / f"{config['data']['category']}.csv"
    data = load_interactions(csv_path)
    split = time_split(data.table, config["split"]["val_fraction"], config["split"]["test_fraction"])
    assert_no_time_leakage(split)

    k_values = config["eval"]["k_values"]
    windows = [
        ("val", split.train, split.val),
        ("test", pd.concat([split.train, split.val]), split.test),
    ]
    rows = []
    for name, recommender in build_retrievers(config, data.n_users, data.n_items).items():
        started = time.perf_counter()
        recommender.fit(split.train)
        fit_seconds = time.perf_counter() - started
        for window, history, eval_table in windows:
            metrics = score_recommender(
                recommender, history, eval_table, data.n_users, data.n_items, k_values
            )
            rows.append({"model": name, "window": window, "fit_seconds": fit_seconds, **metrics})
            print(name, window, {key: round(value, 4) for key, value in metrics.items()})
        losses = getattr(recommender, "loss_history", None)
        if losses:
            print(name, "loss per epoch", [round(loss, 3) for loss in losses])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(pd.DataFrame(rows).round(4).to_markdown(index=False) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
