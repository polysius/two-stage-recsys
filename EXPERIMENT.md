# Simulated online experiment (M4)

This document describes a simulated A/B test of the M3 ranker against the strongest M1 baseline.
`make ranking` then `make experiment` regenerate every number in it. The raw output is written to
`artifacts/experiment.md`.

## Question

Does serving the two-stage ranker (treatment) raise the hit rate at 10 over serving recent
popularity (control)?

- Unit of randomisation: the user. Each user sees one arm for the whole experiment.
- Population: the 8,895 test-window users who are warm (the user and the items appear in
  train). 66% of raw validation and test rows are removed by this filter, so the result says nothing about cold
  users.
- Primary metric: hit rate at 10. A user is a hit when any of the 10 recommended items is an item
  they interact with in the test window. It is one binary outcome per user, which keeps the
  statistics simple and matches the sample-size formula.
- Secondary metric: ndcg at 10 per user, compared with Welch's t-test.
- Control: recent popularity over the last 90 days. Treatment: candidates from five retrievers
  ranked by LightGBM LambdaMART. Both exclude items the user has already interacted with.
- Test: two-sided, alpha 0.05, power 0.8, 50/50 split.

## Design and power analysis

The plan uses only train and validation data, so it cannot depend on the result. The control hit
rate on the validation window is 0.0757. Sample sizes for a two-proportion z-test:

| relative lift | absolute lift | users per arm |
|:--|--:|--:|
| 10% | 0.0076 | 20,019 |
| 20% | 0.0151 | 5,215 |
| 30% | 0.0227 | 2,408 |
| 50% | 0.0379 | 930 |

The test window has 4,447 users per arm, so the minimum detectable effect is 0.0165 absolute, or
22% relative lift, at 80% power. Anything smaller than that would not be reliably detected here.

The validation baseline turned out to be a poor guide to the test window: the control scored
0.0373 on test, half of its 0.0757 on validation. Demand shifts quickly in this category (see
the M1 findings), so a plan made on one window can mis-state the baseline for the next. The
conclusion below does not depend on it, because the observed effect is far above the detectable
size under either baseline.

## Result

Arms: 4,447 control and 4,448 treatment users. The sample-ratio mismatch test gives p = 0.992,
so the split matches the plan.

| metric | control | treatment | difference | 95% CI | relative lift | p-value |
|:--|--:|--:|--:|:--|--:|--:|
| hit@10 | 0.0373 | 0.0789 | +0.0416 | [0.0320, 0.0513] | +111% (95% CI 77% to 153%) | 5.3e-17 |
| ndcg@10 | 0.0101 | 0.0308 | +0.0207 | [0.0167, 0.0247] | +206% | 1.3e-23 |

The ranker roughly doubles the hit rate. The interval for the absolute difference excludes zero
by a wide margin and sits well above the minimum detectable effect of 0.0165.

## Validity checks

- Sample-ratio mismatch. The real assignment gives p = 0.992. A planted 55/45 assignment gives
  p = 4e-21, so the check does catch a broken split. The usual alarm threshold is p < 0.001.
- A/A test. Repeating the random assignment 1,000 times with the control in both arms gives
  4.0% significant results, close to the 5% expected, so the test does not raise false alarms
  at this sample size.
- Repeated assignment. Over 1,000 random assignments of the same users, 100% of the A/B tests
  were significant, with a mean difference of +0.0405 and a standard deviation of 0.0042. Reading
  both recommenders for every user gives +0.0405, so a single split recovers the full-population
  effect with the spread shown.

## Limits of the simulation

This is a replay of logged behaviour, not a live experiment, and the result should be read that
way.

- The outcome is whether a user later interacted with a recommended item. Nobody saw these
  recommendations, so the metric cannot capture the effect of exposure, position, or the page
  around it. It rewards predicting what users did anyway.
- Both outcomes exist for every user, so the only randomness in the simulation comes from who is
  assigned to which arm. The confidence intervals therefore describe assignment noise over a
  fixed set of users. They do not include the variation in real behaviour that a live test would
  add, so real intervals would be wider.
- The offline result has no new information beyond M3. The hit rate here is a different summary
  of the same recommendations behind the M3 ndcg and recall figures. What this stage adds is the
  procedure: a power calculation done before looking, a mismatch check, an A/A calibration, and
  intervals on the effect size.
- Interference is absent by construction. In a live system users could influence each other
  through shared inventory, ratings or popularity counts.
- The ranker is trained on the validation window and scored on the test window. A live system
  would retrain over time, and the effect would change as the data shifts.
- One run on one time window. No claim is made about other categories or periods.
- The 22% minimum detectable effect rests on a baseline that did not hold on test. A real
  experiment should fix the baseline from the most recent window and re-check it before launch.
