"""Statistics for a two-arm online experiment: power, assignment, SRM check and analysis.

Everything here works on plain arrays so it can be tested without the recommender.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_effectsize


def sample_size_per_arm(baseline: float, absolute_lift: float, alpha: float, power: float) -> int:
    """Users needed in each arm to detect `baseline` -> `baseline + absolute_lift`."""
    effect = proportion_effectsize(baseline + absolute_lift, baseline)
    return math.ceil(NormalIndPower().solve_power(effect, alpha=alpha, power=power, ratio=1.0))


def minimum_detectable_lift(baseline: float, n_per_arm: int, alpha: float, power: float) -> float:
    """Smallest absolute lift over `baseline` that `n_per_arm` users per arm can detect."""
    effect = NormalIndPower().solve_power(
        nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0
    )
    # Invert Cohen's h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1)) for p2.
    treated = math.sin(math.asin(math.sqrt(baseline)) + effect / 2) ** 2
    return treated - baseline


def assign_arms(n_users: int, treatment_share: float, seed: int) -> np.ndarray:
    """Random assignment: True means the treatment arm. Exact share, random who."""
    n_treated = round(n_users * treatment_share)
    in_treatment = np.zeros(n_users, dtype=bool)
    in_treatment[np.random.default_rng(seed).permutation(n_users)[:n_treated]] = True
    return in_treatment


def sample_ratio_mismatch_p_value(n_control: int, n_treatment: int, treatment_share: float) -> float:
    """Chi-square test that the observed arm sizes match the planned split.

    A tiny p-value (the usual alarm is below 0.001) means assignment or logging is broken and the
    experiment result cannot be trusted.
    """
    total = n_control + n_treatment
    expected = [total * (1 - treatment_share), total * treatment_share]
    return float(stats.chisquare([n_control, n_treatment], expected).pvalue)


@dataclass(frozen=True)
class ProportionResult:
    rate_control: float
    rate_treatment: float
    difference: float
    difference_ci: tuple[float, float]
    relative_lift: float
    relative_lift_ci: tuple[float, float]
    p_value: float


def compare_proportions(control: np.ndarray, treatment: np.ndarray, alpha: float) -> ProportionResult:
    """Difference and ratio of two success rates with score intervals and a z-test."""
    hits_c, n_c = int(control.sum()), len(control)
    hits_t, n_t = int(treatment.sum()), len(treatment)
    rate_c, rate_t = hits_c / n_c, hits_t / n_t

    def interval(compare: str) -> tuple[float, float]:
        low, high = confint_proportions_2indep(
            hits_t, n_t, hits_c, n_c, compare=compare, method="score", alpha=alpha
        )
        return float(low), float(high)

    # Pooled-variance z-test, the same test the sample-size formula assumes.
    pooled = (hits_c + hits_t) / (n_c + n_t)
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    z = (rate_t - rate_c) / standard_error
    ratio_low, ratio_high = interval("ratio")
    return ProportionResult(
        rate_control=rate_c,
        rate_treatment=rate_t,
        difference=rate_t - rate_c,
        difference_ci=interval("diff"),
        relative_lift=rate_t / rate_c - 1,
        relative_lift_ci=(ratio_low - 1, ratio_high - 1),
        p_value=float(2 * stats.norm.sf(abs(z))),
    )


@dataclass(frozen=True)
class MeanResult:
    mean_control: float
    mean_treatment: float
    difference: float
    difference_ci: tuple[float, float]
    p_value: float


def compare_means(control: np.ndarray, treatment: np.ndarray, alpha: float) -> MeanResult:
    """Welch's t-test, which does not assume equal variance in the two arms."""
    test = stats.ttest_ind(treatment, control, equal_var=False)
    interval = test.confidence_interval(confidence_level=1 - alpha)
    return MeanResult(
        mean_control=float(control.mean()),
        mean_treatment=float(treatment.mean()),
        difference=float(treatment.mean() - control.mean()),
        difference_ci=(float(interval.low), float(interval.high)),
        p_value=float(test.pvalue),
    )


def repeat_experiment(
    outcome_control: np.ndarray,
    outcome_treatment: np.ndarray,
    treatment_share: float,
    alpha: float,
    n_repeats: int,
    seed: int,
) -> dict[str, float]:
    """Re-run the random assignment many times on the same users.

    Each user has an outcome under both recommenders. Replay lets us read either one, but a real
    user sees only the arm they are assigned to, so each repeat reads one outcome per user.
    Passing the same array for both arms gives an A/A test, whose significant share should be
    close to `alpha`.
    """
    differences, significant = [], 0
    for repeat in range(n_repeats):
        in_treatment = assign_arms(len(outcome_control), treatment_share, seed + repeat)
        control = outcome_control[~in_treatment]
        treatment = outcome_treatment[in_treatment]
        result = compare_proportions(control, treatment, alpha)
        differences.append(result.difference)
        significant += result.p_value < alpha
    return {
        "significant_share": significant / n_repeats,
        "mean_difference": float(np.mean(differences)),
        "difference_sd": float(np.std(differences, ddof=1)),
    }
