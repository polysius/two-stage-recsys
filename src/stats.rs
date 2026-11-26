//! Statistics for a two-arm online experiment: power, assignment, SRM check and analysis.
//!
//! Port of `recsys/experiment.py`. Everything works on plain slices so it can be tested without
//! the recommender. The formulas follow statsmodels and scipy so results can be compared directly.

use anyhow::{bail, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal, StudentsT};

/// Standard normal distribution, used for critical values and tail probabilities.
fn standard_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("standard normal parameters are valid")
}

/// Critical value z such that P(Z > z) = `tail`.
fn normal_upper_quantile(tail: f64) -> f64 {
    standard_normal().inverse_cdf(1.0 - tail)
}

// ---------------------------------------------------------------------------
// Power and sample size
// ---------------------------------------------------------------------------

/// Power of a two-sided z-test for two equal-sized arms (statsmodels `NormalIndPower`, ratio 1).
///
/// `effect_size` is Cohen's h. With equal arms the effective sample size is `n_per_arm / 2`.
fn two_sided_power(effect_size: f64, n_per_arm: f64, alpha: f64) -> f64 {
    let normal = standard_normal();
    let critical = normal_upper_quantile(alpha / 2.0);
    let shift = effect_size * (n_per_arm / 2.0).sqrt();
    // Upper tail plus the (tiny) lower tail, exactly as statsmodels adds them.
    normal.sf(critical - shift) + normal.cdf(-critical - shift)
}

/// Finds the smallest `x >= 0` where the increasing function `power_at` reaches `target`.
///
/// Bracket by doubling, then bisect. Power is increasing in both sample size and effect size,
/// which is why a plain bisection is enough.
fn solve_increasing(power_at: impl Fn(f64) -> f64, target: f64) -> f64 {
    let mut low = 0.0;
    let mut high = 1.0;
    while power_at(high) < target {
        low = high;
        high *= 2.0;
    }
    for _ in 0..200 {
        let middle = (low + high) / 2.0;
        if power_at(middle) < target {
            low = middle;
        } else {
            high = middle;
        }
    }
    high
}

/// Cohen's h for two proportions: `2 * (asin(sqrt(p1)) - asin(sqrt(p2)))`.
fn cohens_h(p1: f64, p2: f64) -> f64 {
    2.0 * (p1.sqrt().asin() - p2.sqrt().asin())
}

/// Users needed in each arm to detect `baseline` -> `baseline + absolute_lift`.
pub fn sample_size_per_arm(baseline: f64, absolute_lift: f64, alpha: f64, power: f64) -> u64 {
    // The power curve is symmetric in the sign of the effect, so only its size matters.
    let effect = cohens_h(baseline + absolute_lift, baseline).abs();
    let users = solve_increasing(|n| two_sided_power(effect, n, alpha), power);
    users.ceil() as u64
}

/// Smallest absolute lift over `baseline` that `n_per_arm` users per arm can detect.
pub fn minimum_detectable_lift(
    baseline: f64,
    n_per_arm: u64,
    alpha: f64,
    power: f64,
) -> Result<f64> {
    let n = n_per_arm as f64;
    let effect = solve_increasing(|h| two_sided_power(h, n, alpha), power);
    // Invert Cohen's h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1)) for p2.
    let angle = baseline.sqrt().asin() + effect / 2.0;
    if angle > std::f64::consts::FRAC_PI_2 {
        bail!("this sample size cannot detect any lift above the baseline");
    }
    Ok(angle.sin().powi(2) - baseline)
}

// ---------------------------------------------------------------------------
// Assignment and sample ratio mismatch
// ---------------------------------------------------------------------------

/// Random assignment: `true` means the treatment arm. Exact share, random who.
pub fn assign_arms(n_users: usize, treatment_share: f64, seed: u64) -> Vec<bool> {
    // Round half to even like Python's round(), so exact halves match the original.
    let n_treated = (n_users as f64 * treatment_share).round_ties_even() as usize;

    let mut user_order: Vec<usize> = (0..n_users).collect();
    user_order.shuffle(&mut ChaCha12Rng::seed_from_u64(seed));

    let mut in_treatment = vec![false; n_users];
    for &user in &user_order[..n_treated] {
        in_treatment[user] = true;
    }
    in_treatment
}

/// Chi-square test that the observed arm sizes match the planned split.
///
/// A tiny p-value (the usual alarm is below 0.001) means assignment or logging is broken and the
/// experiment result cannot be trusted.
pub fn sample_ratio_mismatch_p_value(
    n_control: u64,
    n_treatment: u64,
    treatment_share: f64,
) -> f64 {
    let total = (n_control + n_treatment) as f64;
    let expected_control = total * (1.0 - treatment_share);
    let expected_treatment = total * treatment_share;

    let statistic = (n_control as f64 - expected_control).powi(2) / expected_control
        + (n_treatment as f64 - expected_treatment).powi(2) / expected_treatment;

    // Two arms leave one degree of freedom.
    ChiSquared::new(1.0)
        .expect("one degree of freedom is valid")
        .sf(statistic)
}

// ---------------------------------------------------------------------------
// Comparing success rates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProportionResult {
    pub rate_control: f64,
    pub rate_treatment: f64,
    pub difference: f64,
    pub difference_ci: (f64, f64),
    pub relative_lift: f64,
    pub relative_lift_ci: (f64, f64),
    pub p_value: f64,
}

fn count_hits(outcomes: &[bool]) -> u64 {
    outcomes.iter().filter(|&&hit| hit).count() as u64
}

/// Difference (Newcombe interval) and ratio of two success rates, with a pooled z-test.
///
/// The relative lift is NaN when either arm has no successes, since the ratio interval is
/// undefined there.
pub fn compare_proportions(control: &[bool], treatment: &[bool], alpha: f64) -> ProportionResult {
    let hits_c = count_hits(control);
    let hits_t = count_hits(treatment);
    let n_c = control.len() as u64;
    let n_t = treatment.len() as u64;
    let rate_c = hits_c as f64 / n_c as f64;
    let rate_t = hits_t as f64 / n_t as f64;

    let (relative_lift, relative_lift_ci) = if hits_c > 0 && hits_t > 0 {
        // The interval is for treatment / control, so subtract 1 to get a relative lift.
        let (ratio_low, ratio_high) = ratio_score_interval(hits_t, n_t, hits_c, n_c, alpha);
        (rate_t / rate_c - 1.0, (ratio_low - 1.0, ratio_high - 1.0))
    } else {
        (f64::NAN, (f64::NAN, f64::NAN))
    };

    ProportionResult {
        rate_control: rate_c,
        rate_treatment: rate_t,
        difference: rate_t - rate_c,
        difference_ci: newcombe_difference_interval(hits_t, n_t, hits_c, n_c, alpha),
        relative_lift,
        relative_lift_ci,
        p_value: two_proportion_p_value(hits_c, n_c, hits_t, n_t),
    }
}

/// Wilson score interval for one proportion, clipped to [0, 1].
fn wilson_interval(hits: u64, n: u64, alpha: f64) -> (f64, f64) {
    let n = n as f64;
    let rate = hits as f64 / n;
    let z = normal_upper_quantile(alpha / 2.0);
    let z_squared = z * z;

    let denominator = 1.0 + z_squared / n;
    let center = (rate + z_squared / (2.0 * n)) / denominator;
    let half_width = z * (rate * (1.0 - rate) / n + z_squared / (4.0 * n * n)).sqrt() / denominator;

    (
        (center - half_width).clamp(0.0, 1.0),
        (center + half_width).clamp(0.0, 1.0),
    )
}

/// Newcombe hybrid score interval for `p1 - p2`, built from the two Wilson intervals.
fn newcombe_difference_interval(
    hits_1: u64,
    n_1: u64,
    hits_2: u64,
    n_2: u64,
    alpha: f64,
) -> (f64, f64) {
    let rate_1 = hits_1 as f64 / n_1 as f64;
    let rate_2 = hits_2 as f64 / n_2 as f64;
    let (low_1, high_1) = wilson_interval(hits_1, n_1, alpha);
    let (low_2, high_2) = wilson_interval(hits_2, n_2, alpha);

    let distance_down = ((rate_1 - low_1).powi(2) + (high_2 - rate_2).powi(2)).sqrt();
    let distance_up = ((rate_2 - low_2).powi(2) + (high_1 - rate_1).powi(2)).sqrt();

    let difference = rate_1 - rate_2;
    (difference - distance_down, difference + distance_up)
}

/// Score interval for the risk ratio `p1 / p2` (Koopman, in Nam's 1995 cubic form).
///
/// This is what statsmodels runs for `compare="ratio", method="score"`: the limits are roots of
/// a cubic in the restricted MLE of `p2`, so no root finding on the p-value is needed. It uses
/// the Miettinen-Nurminen `n / (n - 1)` variance correction, which statsmodels turns on by default.
fn ratio_score_interval(hits_1: u64, n_1: u64, hits_2: u64, n_2: u64, alpha: f64) -> (f64, f64) {
    // Names follow Nam: sample 0 is the denominator arm.
    let (x0, x1) = (hits_2 as f64, hits_1 as f64);
    let (n0, n1) = (n_2 as f64, n_1 as f64);
    let x = x0 + x1;
    let n = n0 + n1;

    let z = normal_upper_quantile(alpha / 2.0).powi(2) * n / (n - 1.0);

    // Equation 6 in Nam (1995): a1 p^3 + a2 p^2 + a3 p + a4 = 0.
    let a1 = n0 * (n0 * n * x1 + n1 * (n0 + x1) * z);
    let a2 = -n0 * (n0 * n1 * x + 2.0 * n * x0 * x1 + n1 * (n0 + x0 + 2.0 * x1) * z);
    let a3 = 2.0 * n0 * n1 * x0 * x + n * x0 * x0 * x1 + n0 * n1 * x * z;
    let a4 = -n1 * x0 * x0 * x;

    // The two smallest roots give the limits; the larger of them is the lower limit.
    let roots = real_cubic_roots_ascending(a1, a2, a3, a4);
    let ratio_at = |p: f64| (1.0 - (n1 - x1) * (1.0 - p) / (x0 + n1 - n * p)) / p;
    (ratio_at(roots[1]), ratio_at(roots[0]))
}

/// Real roots of `a p^3 + b p^2 + c p + d` in ascending order, assuming all three are real.
///
/// The trigonometric method gives the roots, then Newton steps on the original polynomial remove
/// the rounding error that the large coefficients above would otherwise leave behind.
fn real_cubic_roots_ascending(a: f64, b: f64, c: f64, d: f64) -> [f64; 3] {
    // Depressed form t^3 + p t + q with p = shift - b / (3a).
    let (b, c, d) = (b / a, c / a, d / a);
    let p = c - b * b / 3.0;
    let q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;

    let radius = (-p / 3.0).max(0.0).sqrt();
    let cosine = if radius > 0.0 {
        (-q / (2.0 * radius.powi(3))).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let angle = cosine.acos() / 3.0;

    let mut roots = [0.0; 3];
    for (k, root) in roots.iter_mut().enumerate() {
        let turn = 2.0 * std::f64::consts::PI * k as f64 / 3.0;
        let t = 2.0 * radius * (angle - turn).cos();
        *root = polish_cubic_root(t - b / 3.0, b, c, d);
    }
    roots.sort_by(f64::total_cmp);
    roots
}

/// A few Newton steps on the monic cubic `p^3 + b p^2 + c p + d`.
fn polish_cubic_root(mut p: f64, b: f64, c: f64, d: f64) -> f64 {
    for _ in 0..5 {
        let value = ((p + b) * p + c) * p + d;
        let slope = (3.0 * p + 2.0 * b) * p + c;
        if slope == 0.0 {
            break;
        }
        p -= value / slope;
    }
    p
}

/// Two-sided pooled z-test. Returns 1.0 when both arms have the same all-or-nothing outcome.
pub fn two_proportion_p_value(hits_c: u64, n_c: u64, hits_t: u64, n_t: u64) -> f64 {
    let (n_c, n_t) = (n_c as f64, n_t as f64);
    let pooled = (hits_c + hits_t) as f64 / (n_c + n_t);
    let variance = pooled * (1.0 - pooled) * (1.0 / n_c + 1.0 / n_t);
    if variance == 0.0 {
        return 1.0;
    }
    let z = (hits_t as f64 / n_t - hits_c as f64 / n_c) / variance.sqrt();
    2.0 * standard_normal().sf(z.abs())
}

// ---------------------------------------------------------------------------
// Comparing means
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanResult {
    pub mean_control: f64,
    pub mean_treatment: f64,
    pub difference: f64,
    pub difference_ci: (f64, f64),
    pub p_value: f64,
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample variance with `n - 1` in the denominator.
fn sample_variance(values: &[f64]) -> f64 {
    let center = mean(values);
    let squares: f64 = values.iter().map(|v| (v - center).powi(2)).sum();
    squares / (values.len() - 1) as f64
}

/// Welch's t-test, which does not assume equal variance in the two arms.
pub fn compare_means(control: &[f64], treatment: &[f64], alpha: f64) -> MeanResult {
    let mean_c = mean(control);
    let mean_t = mean(treatment);
    let difference = mean_t - mean_c;

    // Variance of each arm's mean; their sum is the squared standard error of the difference.
    let spread_c = sample_variance(control) / control.len() as f64;
    let spread_t = sample_variance(treatment) / treatment.len() as f64;
    let standard_error = (spread_c + spread_t).sqrt();

    // Welch-Satterthwaite degrees of freedom.
    let degrees_of_freedom = (spread_c + spread_t).powi(2)
        / (spread_c.powi(2) / (control.len() - 1) as f64
            + spread_t.powi(2) / (treatment.len() - 1) as f64);
    let t_distribution = StudentsT::new(0.0, 1.0, degrees_of_freedom)
        .expect("Welch degrees of freedom are positive");

    let t_statistic = difference / standard_error;
    let critical = t_distribution.inverse_cdf(1.0 - alpha / 2.0);

    MeanResult {
        mean_control: mean_c,
        mean_treatment: mean_t,
        difference,
        difference_ci: (
            difference - critical * standard_error,
            difference + critical * standard_error,
        ),
        p_value: 2.0 * t_distribution.sf(t_statistic.abs()),
    }
}

// ---------------------------------------------------------------------------
// Repeated random assignment
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepeatResult {
    pub significant_share: f64,
    pub mean_difference: f64,
    pub difference_sd: f64,
}

/// Re-run the random assignment many times on the same users.
///
/// Each user has an outcome under both recommenders. Replay lets us read either one, but a real
/// user sees only the arm they are assigned to, so each repeat reads one outcome per user.
/// Passing the same slice for both arms gives an A/A test, whose significant share should be
/// close to `alpha`.
pub fn repeat_experiment(
    outcome_control: &[bool],
    outcome_treatment: &[bool],
    treatment_share: f64,
    alpha: f64,
    n_repeats: usize,
    seed: u64,
) -> RepeatResult {
    let mut differences = Vec::with_capacity(n_repeats);
    let mut significant = 0;

    for repeat in 0..n_repeats {
        let in_treatment = assign_arms(
            outcome_control.len(),
            treatment_share,
            seed.wrapping_add(repeat as u64),
        );

        let (mut hits_c, mut n_c, mut hits_t, mut n_t) = (0u64, 0u64, 0u64, 0u64);
        for (user, &treated) in in_treatment.iter().enumerate() {
            if treated {
                n_t += 1;
                hits_t += u64::from(outcome_treatment[user]);
            } else {
                n_c += 1;
                hits_c += u64::from(outcome_control[user]);
            }
        }

        let p_value = two_proportion_p_value(hits_c, n_c, hits_t, n_t);
        differences.push(hits_t as f64 / n_t as f64 - hits_c as f64 / n_c as f64);
        if p_value < alpha {
            significant += 1;
        }
    }

    RepeatResult {
        significant_share: significant as f64 / n_repeats as f64,
        mean_difference: mean(&differences),
        difference_sd: sample_variance(&differences).sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;
    // Looser bound for values found by root finding, where the solver tolerance dominates.
    const ROOT_TOLERANCE: f64 = 1e-4;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    /// An arm of `n` users where the first `hits` have a hit.
    fn arm(hits: usize, n: usize) -> Vec<bool> {
        (0..n).map(|user| user < hits).collect()
    }

    // Golden values below were produced by statsmodels 0.15.0 and scipy in the recsys venv,
    // calling the original recsys.experiment functions.

    #[test]
    fn sample_size_matches_statsmodels() {
        assert_eq!(sample_size_per_arm(0.10, 0.02, 0.05, 0.8), 3835);
        assert_eq!(sample_size_per_arm(0.05, 0.01, 0.05, 0.9), 10901);
        assert_eq!(sample_size_per_arm(0.2, 0.05, 0.01, 0.8), 1625);
    }

    #[test]
    fn minimum_detectable_lift_matches_statsmodels() {
        let lift = minimum_detectable_lift(0.10, 5000, 0.05, 0.8).unwrap();
        assert_close(lift, 0.017427859652555278, ROOT_TOLERANCE);

        let lift = minimum_detectable_lift(0.3, 800, 0.01, 0.9).unwrap();
        assert_close(lift, 0.09154422999262368, ROOT_TOLERANCE);
    }

    #[test]
    fn minimum_detectable_lift_is_the_inverse_of_sample_size() {
        let n = sample_size_per_arm(0.10, 0.02, 0.05, 0.8);
        let lift = minimum_detectable_lift(0.10, n, 0.05, 0.8).unwrap();
        assert_close(lift, 0.02, 1e-3);
    }

    #[test]
    fn minimum_detectable_lift_rejects_a_sample_too_small_for_any_lift() {
        assert!(minimum_detectable_lift(0.99, 5, 0.05, 0.8).is_err());
    }

    #[test]
    fn assignment_has_the_exact_share_and_is_reproducible() {
        let first = assign_arms(1_000, 0.3, 1);

        assert_eq!(first.iter().filter(|&&t| t).count(), 300);
        assert_eq!(first, assign_arms(1_000, 0.3, 1));
        assert_ne!(first, assign_arms(1_000, 0.3, 2));
    }

    #[test]
    fn sample_ratio_mismatch_flags_a_skewed_split_only() {
        assert_close(
            sample_ratio_mismatch_p_value(5_020, 4_980, 0.5),
            0.6891565167793516,
            TOLERANCE,
        );
        assert_close(
            sample_ratio_mismatch_p_value(4_500, 5_500, 0.5),
            1.5239706048320995e-23,
            1e-30,
        );
        // A planted 55/45 split on 5,000 users.
        assert_close(
            sample_ratio_mismatch_p_value(2_750, 2_250, 0.5),
            1.537459794428033e-12,
            1e-18,
        );
    }

    #[test]
    fn compare_proportions_matches_statsmodels_on_small_arms() {
        let result = compare_proportions(&arm(12, 80), &arm(21, 90), 0.05);

        assert_close(result.rate_control, 0.15, TOLERANCE);
        assert_close(result.rate_treatment, 0.23333333333333334, TOLERANCE);
        assert_close(result.difference, 0.08333333333333334, TOLERANCE);
        assert_close(result.difference_ci.0, -0.037243277552849174, TOLERANCE);
        assert_close(result.difference_ci.1, 0.1986419165531229, TOLERANCE);
        assert_close(result.relative_lift, 0.5555555555555556, TOLERANCE);
        assert_close(
            result.relative_lift_ci.0,
            -0.16859305624191867,
            ROOT_TOLERANCE,
        );
        assert_close(
            result.relative_lift_ci.1,
            1.9542890255087433,
            ROOT_TOLERANCE,
        );
        assert_close(result.p_value, 0.17032107567750698, TOLERANCE);
    }

    #[test]
    fn compare_proportions_survives_counts_that_broke_the_score_interval() {
        let result = compare_proportions(&arm(148, 4447), &arm(183, 4448), 0.05);

        assert_close(result.difference_ci.0, -1.747164545660637e-05, TOLERANCE);
        assert_close(result.difference_ci.1, 0.01578713280381004, TOLERANCE);
        assert_close(result.relative_lift, 0.23620849941668287, TOLERANCE);
        assert_close(
            result.relative_lift_ci.0,
            -0.00016822502857305466,
            ROOT_TOLERANCE,
        );
        assert_close(
            result.relative_lift_ci.1,
            0.5286118894059939,
            ROOT_TOLERANCE,
        );
        assert_close(result.p_value, 0.050169713905633725, TOLERANCE);
        assert!(result.difference_ci.0 < result.difference);
        assert!(result.difference < result.difference_ci.1);
    }

    #[test]
    fn compare_proportions_handles_a_control_with_no_hits() {
        let zeros = arm(0, 500);
        let with_lift = compare_proportions(&zeros, &arm(25, 500), 0.05);
        let nothing = compare_proportions(&zeros, &zeros, 0.05);

        assert!(with_lift.relative_lift.is_nan());
        assert!(with_lift.relative_lift_ci.0.is_nan());
        assert_close(with_lift.difference_ci.0, 0.03236085297296876, TOLERANCE);
        assert_close(with_lift.difference_ci.1, 0.07276816117154761, TOLERANCE);
        assert_close(with_lift.p_value, 4.1120338951105705e-07, 1e-12);
        assert_eq!(nothing.p_value, 1.0);

        // A different alpha exercises the interval's dependence on the critical value.
        let strict = compare_proportions(&zeros, &arm(25, 500), 0.01);
        assert_close(strict.difference_ci.0, 0.026315308681219358, TOLERANCE);
        assert_close(strict.difference_ci.1, 0.08152112820374981, TOLERANCE);
    }

    #[test]
    fn compare_proportions_with_equal_arms_centers_the_intervals() {
        let result = compare_proportions(&arm(30, 60), &arm(30, 60), 0.05);

        assert_close(result.difference_ci.0, -0.17345295059769072, TOLERANCE);
        assert_close(result.difference_ci.1, 0.17345295059769072, TOLERANCE);
        assert_close(
            result.relative_lift_ci.0,
            -0.3041346065175823,
            ROOT_TOLERANCE,
        );
        assert_close(
            result.relative_lift_ci.1,
            0.43705953675261267,
            ROOT_TOLERANCE,
        );
        assert_close(result.p_value, 1.0, TOLERANCE);
    }

    #[test]
    fn compare_means_matches_scipy_welch_test() {
        let control = [1.2, 0.5, 2.3, 1.9, 0.7, 1.1, 3.0, 0.2];
        let treatment = [1.8, 2.9, 0.4, 3.5, 2.2, 1.7, 4.1, 0.9, 2.6, 3.3];

        let result = compare_means(&control, &treatment, 0.05);
        assert_close(result.mean_control, 1.3624999999999998, TOLERANCE);
        assert_close(result.mean_treatment, 2.3400000000000003, TOLERANCE);
        assert_close(result.difference, 0.9775000000000005, TOLERANCE);
        assert_close(result.difference_ci.0, -0.0876878585374401, TOLERANCE);
        assert_close(result.difference_ci.1, 2.042687858537441, TOLERANCE);
        assert_close(result.p_value, 0.06951115594475026, TOLERANCE);

        let strict = compare_means(&control, &treatment, 0.01);
        assert_close(strict.difference_ci.0, -0.49023589488609587, TOLERANCE);
        assert_close(strict.difference_ci.1, 2.445235894886097, TOLERANCE);
    }

    #[test]
    fn a_a_test_flags_about_alpha_of_the_runs() {
        // Deterministic stand-in for a 10% hit rate over 4,000 users.
        let outcome: Vec<bool> = (0..4_000).map(|user| user % 10 == 0).collect();

        let result = repeat_experiment(&outcome, &outcome, 0.5, 0.05, 300, 0);

        assert!(result.significant_share > 0.02 && result.significant_share < 0.09);
        assert!(result.mean_difference.abs() < 0.01);
        assert!(result.difference_sd > 0.0);
    }
}
