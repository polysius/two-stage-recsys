import numpy as np
import pytest

from recsys.metrics import catalogue_coverage, mrr, ndcg_at_k, popularity_bias, recall_at_k


def test_recall_counts_the_share_of_relevant_items_found():
    recommended = np.array([[1, 2, 3], [4, 5, 6]])
    relevant = [{1, 9}, {7}]
    assert recall_at_k(recommended, relevant, k=3) == pytest.approx((0.5 + 0.0) / 2)


def test_ndcg_is_one_for_a_perfect_ranking_and_lower_when_hit_is_late():
    relevant = [{1}]
    assert ndcg_at_k(np.array([[1, 2, 3]]), relevant, k=3) == pytest.approx(1.0)
    late = ndcg_at_k(np.array([[2, 3, 1]]), relevant, k=3)
    assert late == pytest.approx(1 / np.log2(4))


def test_mrr_uses_the_first_relevant_rank():
    recommended = np.array([[9, 1, 2], [5, 6, 7]])
    assert mrr(recommended, [{1, 2}, {0}]) == pytest.approx((1 / 2 + 0) / 2)


def test_coverage_is_the_share_of_distinct_items_recommended():
    recommended = np.array([[0, 1], [1, 2]])
    assert catalogue_coverage(recommended, n_items=10, k=2) == pytest.approx(0.3)


def test_popularity_bias_is_one_for_average_items():
    popularity = np.array([1.0, 3.0])
    assert popularity_bias(np.array([[0, 1]]), popularity, k=2) == pytest.approx(1.0)
    assert popularity_bias(np.array([[1]]), popularity, k=1) == pytest.approx(1.5)
