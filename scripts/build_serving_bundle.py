"""Train everything the online service needs and save it as one file.

Takes about as long as `make ranking` because it fits the retrievers twice (once for training the
ranker, once on all history for serving).

Usage: python scripts/build_serving_bundle.py --data-dir ~/recsys-data --output artifacts/serving_bundle.joblib
"""

import argparse
from pathlib import Path

import yaml

from recsys.data import load_interactions
from recsys.serving.bundle import build_bundle, save_bundle
from recsys.split import assert_no_time_leakage, time_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="artifacts/serving_bundle.joblib")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    data = load_interactions(Path(args.data_dir) / "raw" / f"{config['data']['category']}.csv")
    split = time_split(data.table, config["split"]["val_fraction"], config["split"]["test_fraction"])
    assert_no_time_leakage(split)

    bundle = build_bundle(config, split, data.n_users, data.n_items)
    save_bundle(bundle, Path(args.output))
    print(f"wrote {args.output} (version {bundle.version})")


if __name__ == "__main__":
    main()
