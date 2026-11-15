# Results

All numbers come from `make baselines` and `make retrieval` on Amazon Reviews 2023, Video Games, 5-core ratings
(814,586 interactions, 94,762 users, 25,612 items). Full tables with k=20 and coverage at every
cutoff are written to `artifacts/baselines.md` and `artifacts/retrieval.md` by the scripts.

## M1: baselines

Split by timestamp quantile: 80% train, 10% validation, 10% test. Relevant items for a user are
the items they interacted with in the evaluation window. Items already seen before the window
are never recommended. For the test window, "seen" includes validation.

Evaluation rows are kept only when both the user and the item appear in train. Of all
validation and test rows, 66.0% were removed: 38.3% have a user unseen in train and 47.7% have an
item unseen in train (a row can be both). Remaining sizes: 41,315 validation rows and 14,098
test rows. The numbers below describe warm users and warm items only. Cold start is handled
separately in M5.

Test window:

| model | recall@10 | recall@100 | ndcg@10 | mrr | coverage@10 | popularity bias@10 |
|:--|--:|--:|--:|--:|--:|--:|
| global popularity | 0.0189 | 0.0523 | 0.0103 | 0.0115 | 0.0007 | 69.1 |
| recent popularity (90 days) | 0.0238 | 0.1394 | 0.0102 | 0.0126 | 0.0007 | 23.3 |
| item-item CF (cosine, 50 neighbours) | 0.0074 | 0.0344 | 0.0039 | 0.0051 | 0.6180 | 6.2 |

Validation window:

| model | recall@10 | recall@100 | ndcg@10 | mrr |
|:--|--:|--:|--:|--:|
| global popularity | 0.0195 | 0.0581 | 0.0130 | 0.0197 |
| recent popularity (90 days) | 0.0411 | 0.1721 | 0.0207 | 0.0257 |
| item-item CF (cosine, 50 neighbours) | 0.0121 | 0.0489 | 0.0079 | 0.0117 |

Popularity bias is the mean train popularity of recommended items divided by the catalogue mean,
so 1.0 is a random item.

Observations:

- Recent popularity is the strongest baseline on both windows, which suggests that the demand for
  video games shifts quickly over time.
- Item-item CF scores below both popularity baselines on accuracy and far above them on
  coverage. It was not tuned (neighbour count and similarity are defaults), so treat it as a
  floor for personalised methods, not as a tuned reference.
- Popularity baselines cover under 0.2% of the catalogue at k=10.

Note added during M3: the test-window rows of the M1 and M2 tables score models fit on train
only, which leaves them about one window out of date. That understates every model that learns
from recent behaviour, recent popularity most of all (recall@100 0.139 there, 0.250 when refit
on train plus validation). The validation rows are not affected. The refit test numbers are in the
M3 section, and the observation above that recent popularity wins still holds.

## M2: candidate generation

Produced by `make retrieval`; the full table is written to `artifacts/retrieval.md`. Same split,
filtering and seen-item rules as M1. Items are retrieved with an exact FAISS inner-product index.

Test window:

| model | recall@10 | recall@100 | ndcg@10 | mrr | coverage@100 | popularity bias@100 |
|:--|--:|--:|--:|--:|--:|--:|
| global popularity | 0.0189 | 0.0523 | 0.0103 | 0.0115 | 0.0048 | 30.6 |
| recent popularity (90 days) | 0.0238 | 0.1394 | 0.0102 | 0.0126 | 0.0052 | 12.2 |
| item-item CF | 0.0074 | 0.0344 | 0.0039 | 0.0051 | 0.8013 | 3.8 |
| ALS | 0.0151 | 0.0723 | 0.0090 | 0.0118 | 0.2097 | 10.9 |
| two-tower | 0.0045 | 0.0340 | 0.0027 | 0.0040 | 0.9980 | 1.7 |

Validation recall@100: ALS 0.0956, two-tower 0.0496 (recent popularity 0.1721).

Tuning used the validation window only, so the test window did not influence any choice.

- ALS: `scripts/tune_retrieval.py --model als` sweeps factors, regularisation and `alpha`
  (confidence scaling). `alpha` mattered most. Without it (alpha 1, the earlier run) validation recall@100 was 0.056;
  alpha 40 gives 0.096 and alpha 100 gives 0.097. Factors and regularisation changed it by less
  than 0.01.
- Two-tower: configurations were tried by hand with validation recall@100 as the score.

  | learning rate | temperature | epochs | batch | val recall@100 |
  |--:|--:|--:|--:|--:|
  | 0.001 | 0.1 | 5 | 4096 | 0.004 |
  | 0.01 | 0.1 | 5 | 1024 | 0.016 |
  | 0.03 | 0.1 | 5 | 1024 | 0.025 |
  | 0.01 | 0.5 | 5 | 1024 | 0.023 |
  | 0.1 | 0.1 | 15 | 1024 | 0.039 |
  | 0.03 | 0.1 | 20 | 1024 | 0.039 |
  | 0.03 | 0.3 | 20 | 1024 | 0.051 (0.0496 after the seed fix) |

  The first setting left the loss at ln(batch size), so the model was at random. The last row is
  the committed config. The seed was later fixed to cover weight initialisation, which moved
  its score from 0.051 to 0.0496. The grid in `tune_retrieval.py` is a starting point for reproducing the
  search, not the exact record of it.

Observations:

- ALS beats item-item CF and global popularity on recall@100 and beats item-item CF on ndcg@10
  (0.0090 against 0.0039; global popularity is 0.0103, within noise). It covers 21% of the
  catalogue against 0.5% for global popularity. It does not beat recent popularity.
- The ID-only two-tower model is on par with item-item CF at recall@100 and lower at the top of
  the list. Users have about 9 interactions on average, so id embeddings are weakly trained, and
  in-batch negatives are sampled by popularity, which pushes it toward long-tail items
  (popularity bias 1.7, coverage 99.8%). Item and user side features are the likely fix and are
  not part of M2.
- Recent popularity still wins on recall@100, so the ranker in M3 should take recency and
  popularity features alongside retriever scores, and the final candidate set can be a union of
  retrievers.
- Single seed, no confidence intervals; differences below about 0.005 should not be read as real.

Note added during M3: the test-window rows of the M1 and M2 tables score models fit on train
only, which leaves them about one window out of date. That understates every model that learns
from recent behaviour, recent popularity most of all (recall@100 0.139 there, 0.250 when refit
on train plus validation). The validation rows are not affected. The refit test numbers are in the
M3 section, and the observation above that recent popularity wins still holds.

## M3: ranking

Produced by `make ranking`; the full output, including the feature importance table, is written
to `artifacts/ranking.md`.

Protocol:

1. Fit all five retrievers on train (popularity, recent popularity, item-item CF, ALS, two-tower).
2. Take the top 100 from each and merge them into one candidate set per user (424 candidates per
   user on validation). Compute features from train history only.
3. Label a candidate 1 when the user interacted with it in the validation window. Train the
   LightGBM LambdaMART ranker on these rows. 20% of validation users are held out to stop
   boosting early (49 trees were used). The split is by user.
4. Refit every retriever on train plus validation (the warm validation rows, about half of all
   validation rows) before serving the test window. Build candidates (425 per user) and features from train plus validation
   history and score them with the ranker. The ranker is not refit, and the test window is not
   touched before this step.

Features (18 in total), grouped for the ablation:

- retriever: rank in each retriever's top 100 (empty when absent) and the ALS and two-tower
  scores for every candidate
- user: interaction count, mean rating, days since last interaction, mean popularity of the
  items they interact with
- item: interaction count, count in the last 90 days, mean rating, days since first and last
  interaction
- pair: item popularity minus the user's usual popularity, item mean rating minus user mean rating

Three mistakes were found and fixed while building this stage, so the numbers below are the
ones to trust.

- Id columns were passed to the ranker as features by accident. Ids follow the raw file order,
  which tracks popularity over the whole log, test window included. They are now excluded.
- Candidates were ordered by item id inside each user, and tied scores kept that order. A
  user-only ranker gives every candidate of a user the same score, so its "ranking" was the lowest
  item ids first, and it scored ndcg@10 0.0119 from that alone. Candidates are now shuffled within
  each user with a fixed seed.
- The test-window retrievers were fit on train only, so recent popularity looked at the last 90
  days of train, which ended about 19 months before the test window. The ranker's features used
  train plus validation, so the baseline was staler than the ranker's inputs. The M1 and M2 test
  rows have the same flaw. Retrievers are now
  refit on train plus validation for the test window. This lifts recent popularity recall@100
  from 0.139 to 0.250.

Test window, same 8,895 users as M1 and M2, retrievers refit on train plus validation:

| model | recall@10 | ndcg@10 | mrr | recall@100 |
|:--|--:|--:|--:|--:|
| global popularity | 0.0190 | 0.0106 | 0.0124 | 0.0590 |
| recent popularity | 0.0598 | 0.0318 | 0.0349 | 0.2500 |
| item-item CF | 0.0114 | 0.0063 | 0.0079 | 0.0505 |
| ALS | 0.0251 | 0.0138 | 0.0172 | 0.1174 |
| two-tower | 0.0061 | 0.0029 | 0.0043 | 0.0474 |
| recent popularity, fit on train only (as in M1 and M2) | 0.0238 | 0.0102 | 0.0126 | 0.1394 |
| candidates + ranker | 0.0644 | 0.0356 | 0.0414 | 0.2757 |

Against a baseline that is refit, the ranker is a modest improvement: ndcg@10 0.0356 against
0.0318 (+12%), recall@10 0.0644 against 0.0598, recall@100 0.2757 against 0.2500. With a single
run these differences are small, and the experiment in EXPERIMENT.md finds the lift borderline.
The candidate set contains 33.6% of the relevant test pairs, so the share of all relevant pairs
any ranker can recover from these candidates is capped at 33.6%.

Feature-group ablation, retraining the ranker each time and scoring on test:

| features | recall@10 | ndcg@10 | mrr | recall@100 |
|:--|--:|--:|--:|--:|
| all features | 0.0644 | 0.0356 | 0.0414 | 0.2757 |
| without retriever | 0.0275 | 0.0138 | 0.0193 | 0.2491 |
| without user | 0.0673 | 0.0378 | 0.0436 | 0.2719 |
| without item | 0.0634 | 0.0363 | 0.0419 | 0.2554 |
| without pair | 0.0602 | 0.0348 | 0.0420 | 0.2710 |
| only retriever | 0.0646 | 0.0379 | 0.0440 | 0.2505 |
| only user | 0.0073 | 0.0038 | 0.0066 | 0.0853 |
| only item | 0.0283 | 0.0173 | 0.0254 | 0.2262 |
| only pair | 0.0124 | 0.0065 | 0.0094 | 0.1062 |

Observations:

- The retriever group carries the ranker. Removing it (rank and score features together) cuts
  ndcg@10 from 0.0356 to 0.0138, and the retriever features alone reach 0.0379.
- Adding user, item and pair features on top of the retriever features does not improve
  ndcg@10 or recall@10 here. Dropping the user group scores higher (0.0378) than the full set
  (0.0356), and the differences are inside run-to-run noise. Item features do help recall@100
  (0.2505 with retriever features only, 0.2757 with everything).
- By gain, the single most important feature is the item's interaction count over the last 90
  days (27.3%), then the two-tower score (12.9%) and days since the item was last seen (12.8%).
  The retriever features already carry recency, because the refit recent-popularity rank is one of
  them, which is why the item group adds little on top.
- The two-tower score is second by gain even though the two-tower model is weak on its own.
  That suggests its scores add information the other retrievers lack, but it was not tested
  directly, for example by dropping that one score.
- Single seed and a single split, no confidence intervals. Differences below about 0.003 in
  ndcg@10 should not be read as real. The ranker is trained on one validation window and applied
  one window later. It is also trained on scores from retrievers fit on train and applied to
  scores from retrievers refit on train plus validation, so score and age features shift between
  training and test. This is not leakage and most likely works against the ranker.

## M5: serving

`GET /recommend?user_id=<id>&k=<n>` retrieves candidates from the five retrievers, ranks them with
the M3 ranker and returns the top k. `GET /health` reports the bundle version. Users with no
history get the recent-popularity ranking (`source: "cold_start"`, scores null) instead of an
error. `make bundle` trains everything and saves one file (about 150 MB, about 15 minutes),
`make serve` starts the service, and `make benchmark` produces the table below.

Design:

- The bundle holds the retrievers fit on train plus validation, the ranker trained on the
  validation window (as in M3), and precomputed user and item statistics, so a request never
  reads the interaction log.
- Redis caches each user's merged candidate list (items plus retriever ranks and scores, about
  14 KB per user) for an hour. A hit skips retrieval. Ranking still runs on every request, which
  keeps the cached payload small and independent of k. The statistics come from the bundle and
  do not change between builds, so a hit ranks to the same list the miss produced. The key
  includes the bundle version.
- If Redis fails or times out (100 ms per call), the request is treated as a miss and still
  answers, and the cache is skipped for the next 5 seconds. A miss makes two Redis calls, so
  against an unreachable Redis the first request can pay about 200 ms extra, which on a miss is
  enough to exceed the 250 ms budget. After the first failure the cost is gone until the retry.
  A malformed cache value is also treated as a miss.

Latency budgets were set before measuring: p95 of 50 ms on a cache hit, 250 ms on a miss and 25 ms
for cold start. Requests go over HTTP to one uvicorn worker, one at a time, from the same
machine (8 cores, WSL), 500 users per scenario:

| scenario | requests | p50 ms | p95 ms | p99 ms | p95 budget ms | budget met |
|:--|--:|--:|--:|--:|--:|:--|
| cache miss | 500 | 22.4 | 27.3 | 33.5 | 250 | yes |
| cache hit | 500 | 6.4 | 8.3 | 9.9 | 50 | yes |
| cold start | 500 | 1.3 | 2.0 | 2.6 | 25 | yes |
| cache hit, 8 concurrent clients | 500 | 88.6 | 117.2 | 142.5 | none | |

Observations:

- All three budgets are met with a wide margin. A hit is about 3.5 times faster than a miss at
  the median (6.4 ms against 22.4 ms). The remaining 6 ms on a hit is ranking and HTTP.
- With 8 clients at once, one worker serves about 85 requests per second and p95 rises to 117 ms,
  No budget was set for this case. Throughput is below the 155 requests per second that the 6.4 ms
  sequential median implies. Contention for the Python interpreter lock in the single worker is a
  likely cause, but it was not diagnosed, and more workers were not tried. The load generator is also
  Python threads in one process, so treat the concurrent row as indicative. Concurrent cache misses
  were not benchmarked.
- Cold start is a single array lookup, so it is the cheapest path. Its quality was not evaluated:
  the M1 evaluation removed cold users, and recent popularity is not personalised.

Limits:

- Nothing updates between bundle builds. The seen mask, retrievers and statistics are fixed in
  the bundle, so a user's new interactions do not change their list until the next rebuild, and an
  item they just consumed can be recommended again. The cache only repeats the same list, and
  its keys carry the bundle version, so a rebuild starts with an empty cache. A system fed by
  live interactions would need to invalidate entries.
- User ids are the dense integers used in training. A real service needs a mapping from external
  ids, and ids that are missing from it would fall into the cold-start path.
- One run on one machine. The bundle is a snapshot, so new interactions and new items need a rebuild.
