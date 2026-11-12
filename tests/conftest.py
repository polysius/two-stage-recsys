"""Small synthetic interaction log shared by the tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def interactions() -> pd.DataFrame:
    """200 users x 60 items. Users prefer items in their own taste group, so CF has signal."""
    rng = np.random.default_rng(0)
    n_users, n_items, n_groups = 200, 60, 4
    rows = []
    for user in range(n_users):
        group = user % n_groups
        preferred = np.arange(n_items)[np.arange(n_items) % n_groups == group]
        chosen = rng.choice(preferred, size=12, replace=False)
        for step, item in enumerate(chosen):
            # Round-robin over users so every user has interactions early and late in time.
            rows.append((user, int(item), 5.0, 1_000 * (step * n_users + user)))
    table = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    return table.sort_values("timestamp").reset_index(drop=True)
