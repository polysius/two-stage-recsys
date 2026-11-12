"""Load raw interactions and map string ids to dense integer ids."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

INTERACTION_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


@dataclass(frozen=True)
class Interactions:
    """Interactions with dense ids. `timestamp` is in milliseconds since the epoch."""

    table: pd.DataFrame
    n_users: int
    n_items: int


def load_interactions(csv_path: Path) -> Interactions:
    raw = pd.read_csv(csv_path).rename(columns={"parent_asin": "item_id"})
    # factorize assigns ids 0..n-1 in order of first appearance
    raw["user_id"], _ = pd.factorize(raw["user_id"])
    raw["item_id"], _ = pd.factorize(raw["item_id"])
    table = raw[INTERACTION_COLUMNS].sort_values("timestamp", kind="stable").reset_index(drop=True)
    return Interactions(
        table=table, n_users=table["user_id"].nunique(), n_items=table["item_id"].nunique()
    )
