import pytest

from recsys.split import Split, assert_no_time_leakage, time_split


def test_split_is_chronological(interactions):
    split = time_split(interactions, val_fraction=0.1, test_fraction=0.1)
    assert_no_time_leakage(split)


def test_eval_rows_only_use_known_users_and_items(interactions):
    split = time_split(interactions, val_fraction=0.1, test_fraction=0.1)
    known_users, known_items = set(split.train["user_id"]), set(split.train["item_id"])
    for evaluation in (split.val, split.test):
        assert set(evaluation["user_id"]) <= known_users
        assert set(evaluation["item_id"]) <= known_items


def test_leakage_check_rejects_an_empty_split(interactions):
    split = time_split(interactions, val_fraction=0.1, test_fraction=0.1)
    empty_test = Split(split.train, split.val, split.test.iloc[0:0], 0.0, 0.0, 0.0)
    with pytest.raises(AssertionError, match="empty"):
        assert_no_time_leakage(empty_test)


def test_leakage_check_catches_a_shuffled_split(interactions):
    split = time_split(interactions, val_fraction=0.1, test_fraction=0.1)
    leaky = Split(split.test, split.val, split.train, 0.0, 0.0, 0.0)
    with pytest.raises(AssertionError):
        assert_no_time_leakage(leaky)
