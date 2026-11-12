"""Time-based train / validation / test split.

Every interaction before the validation cutoff goes to train, every interaction before the
test cutoff goes to validation, and the rest goes to test. No future interaction can leak
into an earlier split, which `assert_no_time_leakage` checks.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    dropped_unseen_share: float  # eval rows removed because the user or item is not in train
    unknown_user_share: float  # share of eval rows whose user is not in train
    unknown_item_share: float  # share of eval rows whose item is not in train


def time_split(table: pd.DataFrame, val_fraction: float, test_fraction: float) -> Split:
    """Split by timestamp quantile.

    Evaluation rows are kept only for users and items that appear in train. A collaborative
    model cannot score an unseen id, so those rows would measure cold start, not ranking.
    The share removed is reported so the filter is visible.
    """
    ordered = table.sort_values("timestamp", kind="stable")
    val_start = ordered["timestamp"].quantile(1 - val_fraction - test_fraction)
    test_start = ordered["timestamp"].quantile(1 - test_fraction)

    train = ordered[ordered["timestamp"] < val_start]
    val = ordered[(ordered["timestamp"] >= val_start) & (ordered["timestamp"] < test_start)]
    test = ordered[ordered["timestamp"] >= test_start]

    known_users, known_items = set(train["user_id"]), set(train["item_id"])
    eval_rows = pd.concat([val, test])
    user_known = eval_rows["user_id"].isin(known_users)
    item_known = eval_rows["item_id"].isin(known_items)

    def keep_known(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["user_id"].isin(known_users) & frame["item_id"].isin(known_items)]

    val, test = keep_known(val), keep_known(test)
    return Split(
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        dropped_unseen_share=1 - (user_known & item_known).mean(),
        unknown_user_share=1 - user_known.mean(),
        unknown_item_share=1 - item_known.mean(),
    )


def assert_no_time_leakage(split: Split) -> None:
    """Raise if any split contains an interaction earlier than the previous split's latest."""
    for name, part in (("train", split.train), ("val", split.val), ("test", split.test)):
        if part.empty:
            raise AssertionError(f"{name} split is empty, so ordering cannot be checked")
    if split.train["timestamp"].max() > split.val["timestamp"].min():
        raise AssertionError("validation contains interactions older than the end of train")
    if split.val["timestamp"].max() > split.test["timestamp"].min():
        raise AssertionError("test contains interactions older than the end of validation")
