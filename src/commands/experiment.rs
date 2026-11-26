//! Simulate an online A/B test between the recent-popularity control and the ranker treatment.
//!
//! Needs the recommendations that the `ranking` command writes. Order of work mirrors a real
//! experiment:
//!   1. Plan: estimate the baseline rate on the validation window and size the test. The
//!      baseline uses train and validation data only. The user count comes from who is active
//!      in the test window, which is a head count and not an outcome, so the plan cannot depend
//!      on the result.
//!   2. Run: randomly assign the test users to arms and compare hit rate at 10.
//!   3. Check: sample-ratio mismatch, an A/A test, and repeated assignments.

use std::path::PathBuf;

use anyhow::{Context, Result};

use super::{prepare, CommonArgs};
use crate::baselines::Popularity;
use crate::evaluation::relevant_items_by_user;
use crate::metrics::{hit_per_user, ndcg_per_user};
use crate::recommender::Recommender;
use crate::report::{format_general, markdown_table, write_report, Cell};
use crate::sparse::BinaryCsr;
use crate::stats::{
    assign_arms, compare_means, compare_proportions, minimum_detectable_lift, repeat_experiment,
    sample_ratio_mismatch_p_value, sample_size_per_arm,
};
use crate::store::{load, TestRecommendations};

#[derive(clap::Args)]
pub struct Args {
    #[command(flatten)]
    pub common: CommonArgs,
    #[arg(long, default_value = "artifacts/test_recommendations.bin")]
    pub recommendations: PathBuf,
    #[arg(long, default_value = "artifacts/experiment.md")]
    pub output: PathBuf,
}

fn format_interval(low: f64, high: f64, digits: usize) -> String {
    format!("[{low:.digits$}, {high:.digits$}]")
}

/// Keeps the values at the positions where `in_treatment` equals `want`.
fn arm_of<T: Copy>(values: &[T], in_treatment: &[bool], want: bool) -> Vec<T> {
    values
        .iter()
        .zip(in_treatment)
        .filter(|(_, &arm)| arm == want)
        .map(|(&v, _)| v)
        .collect()
}

pub fn run(args: &Args) -> Result<()> {
    let prepared = prepare(&args.common)?;
    let (config, split) = (&prepared.config, &prepared.split);
    let plan = &config.experiment;
    let (k, alpha) = (plan.top_k, plan.alpha);

    let recommendations: TestRecommendations = load(&args.recommendations)
        .context("run the ranking command first, it writes the recommendations")?;
    let users = &recommendations.users;
    let width = recommendations.width;
    let n_users = users.len();
    let treatment_share = plan.treatment_share;
    let n_per_arm = (n_users as f64 * treatment_share.min(1.0 - treatment_share)) as u64;

    // 1. Plan
    let baseline = validation_baseline_rate(&prepared, k)?;
    let mdl = minimum_detectable_lift(baseline, n_per_arm, alpha, plan.power)?;
    let planning_rows: Vec<Vec<Cell>> = plan
        .planning_relative_lifts
        .iter()
        .map(|&lift| {
            let users_per_arm = sample_size_per_arm(baseline, baseline * lift, alpha, plan.power);
            vec![
                format!("{:.0}%", lift * 100.0).into(),
                (baseline * lift).into(),
                (users_per_arm as f64).into(),
            ]
        })
        .collect();
    println!(
        "baseline hit@{k} on validation {baseline:.4}; {n_users} users available; \
         minimum detectable lift {mdl:.4} ({:.0}% relative)",
        mdl / baseline * 100.0
    );

    // 2. Run
    let (test_users, truth) = relevant_items_by_user(&split.test);
    anyhow::ensure!(
        &test_users == users,
        "the saved recommendations were made for a different set of test users"
    );
    let outcome = |model: &str| {
        let items = &recommendations.by_model[model];
        let hits = hit_per_user(items, width, &truth, k);
        let ndcg = ndcg_per_user(items, width, &truth, k);
        (hits, ndcg)
    };
    let (hit_control, ndcg_control) = outcome("recent_popularity");
    let (hit_treatment, ndcg_treatment) = outcome("ranker");

    let in_treatment = assign_arms(n_users, treatment_share, plan.seed);
    let n_treated = in_treatment.iter().filter(|&&arm| arm).count();
    let n_control = n_users - n_treated;
    let srm_p = sample_ratio_mismatch_p_value(n_control as u64, n_treated as u64, treatment_share);

    let hit = compare_proportions(
        &arm_of(&hit_control, &in_treatment, false),
        &arm_of(&hit_treatment, &in_treatment, true),
        alpha,
    );
    let ndcg = compare_means(
        &arm_of(&ndcg_control, &in_treatment, false),
        &arm_of(&ndcg_treatment, &in_treatment, true),
        alpha,
    );
    println!("hit {hit:?}");
    println!("ndcg {ndcg:?}");

    // 3. Checks
    let planted = (n_users as f64 * 0.55).round() as u64;
    let planted_srm_p =
        sample_ratio_mismatch_p_value(n_users as u64 - planted, planted, treatment_share);
    let aa = repeat_experiment(
        &hit_control,
        &hit_control,
        treatment_share,
        alpha,
        plan.n_repeats,
        plan.seed,
    );
    let ab = repeat_experiment(
        &hit_control,
        &hit_treatment,
        treatment_share,
        alpha,
        plan.n_repeats,
        plan.seed,
    );
    let mean =
        |flags: &[bool]| flags.iter().filter(|&&flag| flag).count() as f64 / flags.len() as f64;
    let full_difference = mean(&hit_treatment) - mean(&hit_control);

    let planning = markdown_table(
        &["relative lift", "absolute lift", "users per arm"],
        &planning_rows,
    );
    let hit_row = format!(
        "| hit@{k} | {:.4} | {:.4} | {:+.4} | {} | {:+.1}% ({}) | {} |",
        hit.rate_control,
        hit.rate_treatment,
        hit.difference,
        format_interval(hit.difference_ci.0, hit.difference_ci.1, 4),
        hit.relative_lift * 100.0,
        format_interval(hit.relative_lift_ci.0, hit.relative_lift_ci.1, 2),
        format_general(hit.p_value, 2),
    );
    let ndcg_row = format!(
        "| ndcg@{k} | {:.4} | {:.4} | {:+.4} | {} | {:+.1}% | {} |",
        ndcg.mean_control,
        ndcg.mean_treatment,
        ndcg.difference,
        format_interval(ndcg.difference_ci.0, ndcg.difference_ci.1, 4),
        ndcg.difference / ndcg.mean_control * 100.0,
        format_general(ndcg.p_value, 2),
    );
    let text = format!(
        "## Plan\n\n\
         Metric: hit rate at {k} (a user counts as a hit when any of the {k} recommended items is \
         one they interact with in the test window). Control hit rate on validation: \
         {baseline:.4}. alpha {alpha}, power {}, {:.0}% to treatment.\n\n\
         {planning}\n\n\
         With {n_users} users ({n_per_arm} per arm) the minimum detectable lift is {mdl:.4} \
         absolute, {:.0}% relative.\n\n\
         ## Result\n\n\
         Arms: {n_control} control, {n_treated} treatment. Sample-ratio mismatch p-value \
         {srm_p:.3}.\n\n\
         | metric | control | treatment | difference | 95% CI | relative lift | p-value |\n\
         |:--|--:|--:|--:|:--|--:|--:|\n\
         {hit_row}\n{ndcg_row}\n\n\
         ## Checks\n\n\
         - Sample-ratio mismatch: the real assignment gives p = {srm_p:.3}. A planted 55/45 \
         assignment gives p = {}, so the check does fire.\n\
         - A/A test, {} random assignments with the control in both arms: {:.1}% came out \
         significant (expected about {:.0}%).\n\
         - A/B, {} random assignments: {:.1}% significant, mean difference {:+.4}, standard \
         deviation {:.4}. Reading both recommenders for every user gives {full_difference:+.4}.\n",
        plan.power,
        treatment_share * 100.0,
        mdl / baseline * 100.0,
        format_general(planted_srm_p, 2),
        plan.n_repeats,
        aa.significant_share * 100.0,
        alpha * 100.0,
        plan.n_repeats,
        ab.significant_share * 100.0,
        ab.mean_difference,
        ab.difference_sd,
    );
    write_report(&args.output, &text)
}

/// Hit rate at k of the control on the validation window, fit on train only.
fn validation_baseline_rate(prepared: &super::Prepared, k: usize) -> Result<f64> {
    let (config, data, split) = (&prepared.config, &prepared.data, &prepared.split);
    let control = Popularity::fit(
        data.n_items,
        &split.train,
        Some(config.baselines.recent_window_days),
    );
    let (users, relevant) = relevant_items_by_user(&split.val);
    let seen = BinaryCsr::user_item(&split.train, data.n_users, data.n_items);
    let recommended = control.recommend(&users, k, &seen)?;
    let hits = hit_per_user(&recommended, k, &relevant, k);
    Ok(hits.iter().filter(|&&hit| hit).count() as f64 / hits.len() as f64)
}
