import numpy as np
import pytest

from recsys.experiment import (
    assign_arms,
    compare_means,
    compare_proportions,
    minimum_detectable_lift,
    repeat_experiment,
    sample_ratio_mismatch_p_value,
    sample_size_per_arm,
)
from recsys.metrics import hit_per_user


def test_sample_size_matches_the_textbook_formula():
    # 10% -> 12% at alpha 0.05 and power 0.8 needs about 3,800 users per arm (Cohen's h).
    assert sample_size_per_arm(0.10, 0.02, alpha=0.05, power=0.8) == pytest.approx(3841, abs=60)


def test_minimum_detectable_lift_is_the_inverse_of_sample_size():
    n = sample_size_per_arm(0.10, 0.02, alpha=0.05, power=0.8)

    assert minimum_detectable_lift(0.10, n, alpha=0.05, power=0.8) == pytest.approx(0.02, abs=1e-3)


def test_assignment_has_the_exact_share_and_is_reproducible():
    first = assign_arms(1_000, 0.3, seed=1)

    assert first.sum() == 300
    assert np.array_equal(first, assign_arms(1_000, 0.3, seed=1))
    assert not np.array_equal(first, assign_arms(1_000, 0.3, seed=2))


def test_sample_ratio_mismatch_flags_a_skewed_split_only():
    assert sample_ratio_mismatch_p_value(5_020, 4_980, 0.5) > 0.5
    assert sample_ratio_mismatch_p_value(4_500, 5_500, 0.5) < 1e-6


def test_proportion_test_detects_a_real_lift_and_interval_contains_it():
    rng = np.random.default_rng(0)
    control = rng.random(20_000) < 0.10
    treatment = rng.random(20_000) < 0.13

    result = compare_proportions(control, treatment, alpha=0.05)

    assert result.p_value < 0.001
    assert result.difference_ci[0] < 0.03 < result.difference_ci[1]
    assert result.relative_lift_ci[0] < 0.3 < result.relative_lift_ci[1]


def test_mean_comparison_reports_the_difference_and_a_matching_interval():
    rng = np.random.default_rng(0)
    control = rng.normal(1.0, 1.0, 5_000)
    treatment = rng.normal(1.1, 2.0, 5_000)

    result = compare_means(control, treatment, alpha=0.05)

    low, high = result.difference_ci
    assert low < result.difference < high
    assert low < 0.1 < high


def test_a_a_test_flags_about_alpha_of_the_runs():
    outcome = np.random.default_rng(0).random(4_000) < 0.1

    result = repeat_experiment(outcome, outcome, 0.5, alpha=0.05, n_repeats=300, seed=0)

    assert 0.02 < result["significant_share"] < 0.09


def test_hit_per_user_marks_users_with_any_relevant_item_in_top_k():
    recommended = np.array([[1, 2, 3], [4, 5, 6]])

    hits = hit_per_user(recommended, [{3}, {9}], k=3)

    assert hits.tolist() == [True, False]
    assert hit_per_user(recommended, [{3}, {9}], k=2).tolist() == [False, False]


def test_proportion_comparison_survives_counts_that_broke_the_score_interval():
    control = np.array([1] * 148 + [0] * (4447 - 148))
    treatment = np.array([1] * 183 + [0] * (4448 - 183))

    result = compare_proportions(control, treatment, alpha=0.05)

    assert result.difference_ci[0] < result.difference < result.difference_ci[1]


def test_proportion_comparison_handles_a_control_with_no_hits():
    zeros, some = np.zeros(500, dtype=bool), np.arange(500) < 25

    with_lift = compare_proportions(zeros, some, alpha=0.05)
    nothing = compare_proportions(zeros, zeros, alpha=0.05)

    assert np.isnan(with_lift.relative_lift)
    assert with_lift.p_value < 0.001
    assert nothing.p_value == 1.0


def test_minimum_detectable_lift_rejects_a_sample_too_small_for_any_lift():
    with pytest.raises(ValueError):
        minimum_detectable_lift(0.99, 5, alpha=0.05, power=0.8)
