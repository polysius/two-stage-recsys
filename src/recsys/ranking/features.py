"""Features for candidate (user, item) pairs, computed from interaction history only.

`history` is every interaction the system could have seen when making the recommendation: train
for the validation window, train plus validation for the test window. Nothing from the window
being predicted is used, so a feature cannot leak the label.
"""

import numpy as np
import pandas as pd

MS_PER_DAY = 24 * 60 * 60 * 1000
RECENT_WINDOW_DAYS = 90

ID_COLUMNS = {"user_id", "item_id"}

# A column belongs to the group named by its prefix. The ablation switches groups on and off.
GROUPS = {
    "retriever": ("rank_", "score_"),
    "user": ("user_",),
    "item": ("item_",),
    "pair": ("pair_",),
}


def feature_columns(frame: pd.DataFrame, groups: list[str] | None = None) -> list[str]:
    wanted = groups if groups is not None else list(GROUPS)
    prefixes = tuple(prefix for group in wanted for prefix in GROUPS[group])
    # The id columns match the user_ and item_ prefixes but are not features: ids are assigned in
    # order of first appearance over the whole log, so they would leak when an item or user
    # first shows up, including in the evaluation window.
    return [
        column for column in frame.columns
        if column.startswith(prefixes) and column not in ID_COLUMNS
    ]


def build_features(candidates: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    reference_time = history["timestamp"].max()

    def days_ago(timestamps: pd.Series) -> pd.Series:
        return (reference_time - timestamps) / MS_PER_DAY

    item_count = history.groupby("item_id").size()
    recent = history[history["timestamp"] >= reference_time - RECENT_WINDOW_DAYS * MS_PER_DAY]
    items = pd.DataFrame({
        "item_log_count": np.log1p(item_count),
        "item_log_recent_count": np.log1p(recent.groupby("item_id").size()),
        "item_mean_rating": history.groupby("item_id")["rating"].mean(),
        "item_age_days": days_ago(history.groupby("item_id")["timestamp"].min()),
        "item_days_since_last": days_ago(history.groupby("item_id")["timestamp"].max()),
    })
    items["item_log_recent_count"] = items["item_log_recent_count"].fillna(0.0)

    by_user = history.assign(log_popularity=np.log1p(history["item_id"].map(item_count)))
    by_user = by_user.groupby("user_id")
    users = pd.DataFrame({
        "user_log_count": np.log1p(by_user.size()),
        "user_mean_rating": by_user["rating"].mean(),
        "user_days_since_last": days_ago(by_user["timestamp"].max()),
        "user_mean_log_item_count": by_user["log_popularity"].mean(),
    })

    features = candidates.join(users, on="user_id").join(items, on="item_id")
    # Pair features compare the item with what this user usually consumes.
    features["pair_popularity_gap"] = (
        features["item_log_count"] - features["user_mean_log_item_count"]
    )
    features["pair_rating_gap"] = features["item_mean_rating"] - features["user_mean_rating"]
    return features
