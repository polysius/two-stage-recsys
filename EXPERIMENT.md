# Simulated online experiment (M4)

This document describes a simulated A/B test of the M3 ranker against the strongest M1 baseline.
`make ranking` then `make experiment` regenerate every number in it. The raw output is written to
`artifacts/experiment.md`.

## Question

Does serving the two-stage ranker (treatment) raise the hit rate at 10 over serving recent
popularity (control)?

- Unit of randomisation: the user. Each user sees one arm for the whole experiment.
- Population: the 8,895 test-window users who are warm (the user and the items appear in
  train). 66% of raw validation and test rows are removed by this filter, so the result says
  nothing about cold users.
- Primary metric: hit rate at 10. A user is a hit when any of the 10 recommended items is an item
  they interact with in the test window. It is one binary outcome per user, which keeps the
  statistics simple.
- Secondary metric: ndcg at 10 per user, compared with Welch's t-test.
- Control: recent popularity over the last 90 days, refit on train plus validation. Treatment:
  candidates from five retrievers, also refit on train plus validation, ranked by LightGBM
  LambdaMART. Both exclude items the user has already interacted with.
- Test: two-sided, alpha 0.05, power 0.8, 50/50 split.

## Design and power analysis

The baseline comes from the validation window, using the control fit on train: the hit rate is
0.0757. Sample sizes for a two-proportion test at that baseline:

| relative lift | absolute lift | users per arm |
|:--|--:|--:|
| 10% | 0.0076 | 20,019 |
| 20% | 0.0151 | 5,215 |
| 30% | 0.0227 | 2,408 |
| 50% | 0.0379 | 930 |

The number of users is the head count of the test window, which is not an outcome. With 4,447
users per arm, the minimum detectable effect is 0.0165 absolute, or 22% relative lift, at 80%
power. A true lift smaller than that would usually go undetected here. The control scored 0.0814
on the test window, close to the 0.0757 the plan assumed.

## Result

Arms: 4,447 control and 4,448 treatment users. The sample-ratio mismatch test gives p = 0.992,
so the split matches the plan.

| metric | control | treatment | difference | 95% CI | relative lift | p-value |
|:--|--:|--:|--:|:--|--:|--:|
| hit@10 | 0.0814 | 0.0933 | +0.0119 | [0.0002, 0.0237] | +14.6% (95% CI 0.2% to 31%) | 0.047 |
| ndcg@10 | 0.0305 | 0.0372 | +0.0067 | [0.0014, 0.0120] | +21.9% | 0.013 |

The ranker looks better, but the evidence is weak. The primary metric has p = 0.047 and a
confidence interval whose lower end is almost zero. The observed lift of 14.6% is below the 22%
the experiment was sized to detect.

## Validity checks

- Sample-ratio mismatch. The real assignment gives p = 0.992. A planted 55/45 assignment gives
  p = 4e-21, so the check does catch a broken split. The usual alarm threshold is p < 0.001.
- A/A test. Repeating the random assignment 1,000 times with the control in both arms gives
  4.8% significant results, close to the 5% expected, so the test does not raise false alarms
  at this sample size.
- Repeated assignment. Over 1,000 random assignments of the same users, only 18.2% of the A/B
  tests were significant, with a mean difference of +0.0070 and a standard deviation of 0.0052.
  Reading both recommenders for every user gives +0.0072. The single split above landed on the
  high side of that spread (+0.0119), so its p-value is better than a typical run would give.
  At the full-population effect of 0.0072 (a 9% relative lift), detecting it at 80% power would
  take about 23,500 users per arm, around five times the available test set.

## What went wrong on the first pass

The first version of this experiment used a control that was fit on train only. Its 90-day window
ended about two years before the test window, so it recommended items that were no longer
popular. Against that stale control the ranker appeared to double the hit rate (+111%, p = 5e-17).
Refitting the control on train plus validation, as a real system would, brought the control from
0.0373 to 0.0814 and the effect down to the figures above. The ranking stage had the same flaw
and now refits all retrievers before the test window (see RESULTS.md). The lesson for a real
experiment is to compare against the baseline as it would actually be deployed.

## Limits of the simulation

This is a replay of logged behaviour, not a live experiment, and the result should be read that
way.

- The outcome is whether a user later interacted with a recommended item. Nobody saw these
  recommendations, so the metric cannot capture the effect of exposure, position, or the page
  around it. It rewards predicting what users did anyway.
- Both outcomes exist for every user, so the only randomness in the simulation comes from who is
  assigned to which arm. The reported intervals use the standard formula that treats users as
  independent draws. They are somewhat wider than the assignment noise alone (standard error
  0.0060 against a standard deviation of 0.0052 over repeated assignments), because each user's
  outcomes under the two recommenders are correlated. A live test would add real behavioural
  variation that this simulation does not contain.
- The offline result has no new information beyond M3. The hit rate here is a different summary
  of the same recommendations behind the M3 ndcg and recall figures. What this stage adds is the
  procedure: a power calculation done before looking, a mismatch check, an A/A calibration, and
  intervals on the effect size.
- Interference is absent by construction. In a live system users could influence each other
  through shared inventory, ratings or popularity counts.
- The ranker is trained on the validation window and scored on the test window. A live system
  would retrain over time, and the effect would change as the data shifts.
- One run on one time window. No claim is made about other categories or periods.
