"""Simulate an online A/B test between the recent-popularity control and the ranker treatment.

Needs `artifacts/test_recommendations.npz`, which `make ranking` writes. Usage:
    python scripts/run_experiment.py --data-dir ~/recsys-data --output artifacts/experiment.md

Order of work, which mirrors a real experiment:
  1. Plan: estimate the baseline rate on the validation window and size the test. The baseline
     uses train and validation data only. The user count comes from who is active in the test
     window, which is a head count and not an outcome, so the plan cannot depend on the result.
  2. Run: randomly assign the test users to arms and compare hit rate at 10.
  3. Check: sample-ratio mismatch, an A/A test, and repeated assignments.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recsys.baselines import RecentPopularityRecommender, build_user_item_matrix
from recsys.data import load_interactions
from recsys.evaluation import relevant_items_by_user
from recsys.experiment import (
    assign_arms,
    compare_means,
    compare_proportions,
    minimum_detectable_lift,
    repeat_experiment,
    sample_ratio_mismatch_p_value,
    sample_size_per_arm,
)
from recsys.metrics import hit_per_user, ndcg_per_user
from recsys.split import time_split


def validation_baseline_rate(config, split, data, k) -> float:
    """Hit rate at k of the control on the validation window, fit on train only."""
    control = RecentPopularityRecommender(data.n_items, config["baselines"]["recent_window_days"])
    control.fit(split.train)
    relevant = relevant_items_by_user(split.val)
    users = np.array(sorted(relevant))
    seen = build_user_item_matrix(split.train, data.n_users, data.n_items)
    recommended = np.vstack([
        control.recommend(users[start : start + 2000], k, seen)
        for start in range(0, len(users), 2000)
    ])
    return float(hit_per_user(recommended, [relevant[u] for u in users], k).mean())


def format_interval(low: float, high: float, digits: int = 4) -> str:
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--recommendations", default="artifacts/test_recommendations.npz")
    parser.add_argument("--output", default="artifacts/experiment.md")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    plan = config["experiment"]
    k, alpha = plan["top_k"], plan["alpha"]
    data = load_interactions(Path(args.data_dir) / "raw" / f"{config['data']['category']}.csv")
    split = time_split(data.table, config["split"]["val_fraction"], config["split"]["test_fraction"])

    recommendations = np.load(args.recommendations)
    users = recommendations["users"]
    n_users = len(users)
    treatment_share = plan["treatment_share"]
    n_per_arm = int(n_users * min(treatment_share, 1 - treatment_share))

    # 1. Plan
    baseline = validation_baseline_rate(config, split, data, k)
    planning = pd.DataFrame([
        {
            "relative lift": f"{lift:.0%}",
            "absolute lift": baseline * lift,
            "users per arm": sample_size_per_arm(baseline, baseline * lift, alpha, plan["power"]),
        }
        for lift in plan["planning_relative_lifts"]
    ])
    mdl = minimum_detectable_lift(baseline, n_per_arm, alpha, plan["power"])
    print(f"baseline hit@{k} on validation {baseline:.4f}; {n_users} users available; "
          f"minimum detectable lift {mdl:.4f} ({mdl / baseline:.0%} relative)", flush=True)

    # 2. Run
    relevant = relevant_items_by_user(split.test)
    truth = [relevant[user] for user in users]
    outcomes = {
        arm: {
            "hit": hit_per_user(recommendations[model][:, :k], truth, k),
            "ndcg": ndcg_per_user(recommendations[model], truth, k),
        }
        for arm, model in (("control", "recent_popularity"), ("treatment", "ranker"))
    }
    in_treatment = assign_arms(n_users, treatment_share, plan["seed"])
    n_treated = int(in_treatment.sum())
    srm_p = sample_ratio_mismatch_p_value(n_users - n_treated, n_treated, treatment_share)

    def observed(metric):
        return outcomes["control"][metric][~in_treatment], outcomes["treatment"][metric][in_treatment]

    hit = compare_proportions(*observed("hit"), alpha)
    ndcg = compare_means(*observed("ndcg"), alpha)
    print("hit", hit, flush=True)
    print("ndcg", ndcg, flush=True)

    # 3. Checks
    planted = round(n_users * 0.55)
    planted_srm_p = sample_ratio_mismatch_p_value(n_users - planted, planted, treatment_share)
    hit_control, hit_treatment = outcomes["control"]["hit"], outcomes["treatment"]["hit"]
    aa = repeat_experiment(hit_control, hit_control, treatment_share, alpha,
                           plan["n_repeats"], plan["seed"])
    ab = repeat_experiment(hit_control, hit_treatment, treatment_share, alpha,
                           plan["n_repeats"], plan["seed"])
    full_difference = float(hit_treatment.mean() - hit_control.mean())

    lines = [
        "## Plan",
        "",
        (f"Metric: hit rate at {k} (a user counts as a hit when any of the {k} recommended items "
          f"is one they interact with in the test window). Control hit rate on validation: "
          f"{baseline:.4f}. alpha {alpha}, power {plan['power']}, {treatment_share:.0%} to treatment."),
        "",
        planning.round(4).to_markdown(index=False),
        "",
        (f"With {n_users} users ({n_per_arm} per arm) the minimum detectable lift is "
          f"{mdl:.4f} absolute, {mdl / baseline:.0%} relative."),
        "",
        "## Result",
        "",
        (f"Arms: {n_users - n_treated} control, {n_treated} treatment. "
          f"Sample-ratio mismatch p-value {srm_p:.3f}."),
        "",
        "| metric | control | treatment | difference | 95% CI | relative lift | p-value |",
        "|:--|--:|--:|--:|:--|--:|--:|",
        (f"| hit@{k} | {hit.rate_control:.4f} | {hit.rate_treatment:.4f} | {hit.difference:+.4f} "
          f"| {format_interval(*hit.difference_ci)} | {hit.relative_lift:+.1%} "
          f"({format_interval(*hit.relative_lift_ci, 2)}) | {hit.p_value:.2g} |"),
        (f"| ndcg@{k} | {ndcg.mean_control:.4f} | {ndcg.mean_treatment:.4f} "
          f"| {ndcg.difference:+.4f} | {format_interval(*ndcg.difference_ci)} | "
          f"{ndcg.difference / ndcg.mean_control:+.1%} | {ndcg.p_value:.2g} |"),
        "",
        "## Checks",
        "",
        (f"- Sample-ratio mismatch: the real assignment gives p = {srm_p:.3f}. A planted 55/45 "
          f"assignment gives p = {planted_srm_p:.2g}, so the check does fire."),
        (f"- A/A test, {plan['n_repeats']} random assignments with the control in both arms: "
          f"{aa['significant_share']:.1%} came out significant (expected about {alpha:.0%})."),
        (f"- A/B, {plan['n_repeats']} random assignments: {ab['significant_share']:.1%} "
          f"significant, mean difference {ab['mean_difference']:+.4f}, standard deviation "
          f"{ab['difference_sd']:.4f}. Reading both recommenders for every user gives "
          f"{full_difference:+.4f}."),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
