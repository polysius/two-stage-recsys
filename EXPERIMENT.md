# Simulated online experiment (M4)

This document describes a simulated A/B test of the M3 ranker against the strongest M1 baseline.
`make ranking` then `make experiment` regenerate the tables in it. The raw output is written to
`artifacts/experiment.md`. One derived figure (the sample size at the full-population effect) was
computed by hand from the same formula the code uses.

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
  candidates from five retrievers, also refit on train plus validation, ranked by LambdaMART.
  Both exclude items the user has already interacted with.
- Test: two-sided, alpha 0.05, power 0.8, 50/50 split.

## Design and power analysis

The baseline comes from the validation window, using the control fit on train: the hit rate is
0.0758. Sample sizes for a two-proportion test at that baseline:

| relative lift | absolute lift | users per arm |
|:--|--:|--:|
| 10% | 0.0076 | 19,991 |
| 20% | 0.0152 | 5,207 |
| 30% | 0.0228 | 2,405 |
| 50% | 0.0379 | 928 |

The number of users is the head count of the test window, which is not an outcome. With 4,447
users per arm, the minimum detectable effect is 0.0165 absolute, or 22% relative lift, at 80%
power. Smaller true lifts are detected less often. The control scored 0.0850 on the test window,
close to the 0.0758 the plan assumed.

## Result

Arms: 4,447 control and 4,448 treatment users. The sample-ratio mismatch test gives p = 0.992,
so the split matches the plan.

| metric | control | treatment | difference | 95% CI | relative lift | p-value |
|:--|--:|--:|--:|:--|--:|--:|
| hit@10 | 0.0850 | 0.0908 | +0.0058 | [-0.0060, 0.0176] | +6.9% (95% CI -7% to 22%) | 0.33 |
| ndcg@10 | 0.0322 | 0.0372 | +0.0050 | [-0.0004, 0.0104] | +15.6% | 0.069 |

This run does not show a difference. The primary metric has p = 0.33 and an interval that spans
zero, and ndcg@10 is borderline at p = 0.069. The observed lift of 6.9% is well below the 22%
the experiment was sized to detect. The result is not evidence that the ranker is no better: the
repeated assignments below show that this single split is on the low side of what the same data
produces.

## Validity checks

- Sample-ratio mismatch. The real assignment gives p = 0.992. A planted 55/45 assignment gives
  p = 4.3e-21, so the check does catch a broken split. The usual alarm threshold is p < 0.001.
- A/A test. Repeating the random assignment 1,000 times with the control in both arms gives
  4.1% significant results, close to the 5% expected, so the test does not raise false alarms
  at this sample size.
- Repeated assignment. Over 1,000 random assignments of the same users, 33.8% of the A/B
  tests were significant, with a mean difference of +0.0097 and a standard deviation of 0.0052.
  Reading both recommenders for every user gives +0.0100, a 12% relative lift. The single split
  above landed at +0.0058, more than one standard deviation below that mean, so its p-value is
  worse than a typical run would give. At the full-population effect of 0.0100, detecting it at
  80% power would take roughly 13,000 users per arm, about three times the available test set.

## Changes since the Python version

The experiment was first run with the Python implementation, which reported hit@10 +14.6%
(p = 0.047) and ndcg@10 +21.9% (p = 0.013), with a full-population difference of +0.0072. The
Rust port reports a larger full-population difference (+0.0100) but a single split that is not
significant. Two things changed. The ranker was retrained, and its ndcg@10 on the test window
went from 0.0356 to 0.0375. The random assignment also changed, because the Rust random number
generator does not reproduce numpy's stream. The spread over repeated assignments (standard
deviation about 0.005) is larger than the difference between the two single-split results, so
the two runs are consistent. The honest summary is that a single assignment of this population
sits close to the edge of significance either way.

## What went wrong on the first pass

The first version of this experiment, written in Python, used a control that was fit on train
only. Its 90-day window ended about 19 months before the test window, so it recommended items
that were no longer popular. Against that stale control the ranker appeared to double the hit
rate (+111%, p = 5e-17). Refitting the control on train plus validation, which gives the control
the same history the ranker's features use, brought the effect down to the figures above. The
ranking stage had the same flaw and now refits all retrievers before the test window (see
RESULTS.md). The lesson for a real experiment is to compare against the baseline as it would
actually be deployed.

## Limits of the simulation

This is a replay of logged behaviour, not a live experiment, and the result should be read that
way.

- The outcome is whether a user later interacted with a recommended item. Nobody saw these
  recommendations, so the metric cannot capture the effect of exposure, position, or the page
  around it. It rewards predicting what users did anyway.
- Both outcomes exist for every user, so the only randomness in the simulation comes from who is
  assigned to which arm. The reported intervals use the standard formula that treats users as
  independent draws. They are somewhat wider than the assignment noise alone (standard error
  about 0.006 against a standard deviation of 0.0052 over repeated assignments), because the
  formula treats the arms as independent draws from an endless population, while the replay
  assigns a fixed set of users. A live test would add real behavioural variation that this
  simulation does not contain.
- The offline result has no new information beyond M3. The hit rate here is a different summary
  of the same recommendations behind the M3 ndcg and recall figures. What this stage adds is the
  procedure: a power calculation done before looking, a mismatch check, an A/A calibration, and
  intervals on the effect size.
- The ranker is trained on candidates from retrievers fit on train and applied to candidates
  from retrievers refit on train plus validation. Score features such as the ALS and two-tower
  scores come from different fits and have no shared scale, and day counts such as item age shift
  with the history. This is not leakage, but it most likely works against the ranker.
- The refit history is train plus the validation rows of warm users and items, about half of all
  validation rows.
- Interference is absent by construction. In a live system users could influence each other
  through shared inventory, ratings or popularity counts.
- The ranker is trained on the validation window and scored on the test window. A live system
  would retrain over time, and the effect would change as the data shifts.
- One run on one time window. No claim is made about other categories or periods.
