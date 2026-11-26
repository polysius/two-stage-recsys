# Results

All numbers come from the Rust implementation (`make baselines`, `make retrieval`, `make ranking`,
`make bundle`, `make benchmark`) on Amazon Reviews 2023, Video Games, 5-core ratings
(814,586 interactions, 94,762 users, 25,612 items). Full tables with k=20 and coverage at every
cutoff are written to `artifacts/baselines.md` and `artifacts/retrieval.md` by the commands.
Runs used one seed on an 8-core ARM64 machine under WSL with 9.9 GB of memory.

## The Rust port

The project was first written in Python (pandas, scikit-learn, implicit, FAISS, PyTorch,
LightGBM, FastAPI). It was then rewritten in Rust, and every number below was produced again by
the Rust code. None was carried over.

Wall-clock time of the same step on the same machine and data:

| step | Python | Rust | speedup |
|:--|--:|--:|--:|
| `make baselines` (load, split, three baselines, scoring) | 39.7 s | 6.6 s | 6x |
| item-item CF fit | 12.9 s | 0.08 s | 160x |
| ALS fit (15 iterations) | 7.7 s | 1.9 s | 4x |
| two-tower fit (20 epochs) | 540.6 s | 171.5 s | 3.2x |
| `make retrieval` (all five models, both windows) | 628.8 s | 187.9 s | 3.3x |
| serving, cache miss, p50 | 22.4 ms | 3.4 ms | 6.6x |
| serving, cache hit, p50 | 6.4 ms | 1.4 ms | 4.6x |
| serving, cache hit, 8 concurrent clients | 85 requests/s | 1,917 requests/s | 22x |

The Rust `make ranking` takes 8 minutes 22 seconds with a peak of 3.7 GB, and `make bundle` takes
5 minutes 57 seconds with a peak of 2.4 GB (the file is 76 MB). I did not get a Python timing for
`make ranking`: the run was stopped by the machine running low on memory. The earlier write-up
put the Python bundle at about 15 minutes and 150 MB, a figure that was not re-measured here.

How the pieces map across:

- Item-item CF builds neighbour lists by counting co-occurrences per item, in parallel, instead of
  multiplying dense blocks.
- ALS solves each row with three conjugate gradient steps from the previous value, as `implicit`
  does. Each step is the cost of one matrix-vector product, so the per-row matrix is never built.
- The two-tower model has a hand-written backward pass and a fused Adam update. A test checks the
  analytic gradients against finite differences. Adam still updates every embedding row at every
  step, as `torch.optim.Adam` does with a dense gradient, so the training matches the Python
  version. Most of the remaining 171 s is that dense update, which is limited by memory bandwidth.
- Retrieval is an exact scan of the item matrix. It replaces FAISS `IndexFlatIP`, which was also
  exact, and masks seen items before taking the top k instead of fetching extra neighbours.
- LightGBM is replaced by a LambdaMART written for this project (`src/lambdamart.rs`):
  histogram-based, leaf-wise trees with learned handling of missing values, the LightGBM
  lambdarank gradients, per-query bagging and early stopping on NDCG@10. On synthetic data it
  scored validation NDCG@10 0.852 against 0.855 for LightGBM with the same settings.
- Python's `statsmodels` and `scipy` calls are replaced by `src/stats.rs`. Its tests compare
  against values produced by those libraries.
- Random streams differ from numpy and PyTorch, so a given seed no longer gives the same split of
  users or the same initial weights. Metrics agree to within the noise of one seed.

Parity with the Python numbers, test window, retrievers refit on train plus validation:

| quantity | Python | Rust |
|:--|--:|--:|
| ALS recall@10 (train-only fit) | 0.0151 | 0.0147 |
| ALS recall@100 | 0.0723 | 0.0731 |
| two-tower recall@100 | 0.0340 | 0.0309 |
| recent popularity recall@100 (refit) | 0.2500 | 0.2525 |
| ranker ndcg@10 | 0.0356 | 0.0375 |
| ranker recall@100 | 0.2757 | 0.2749 |
| candidates that are relevant | 33.6% | 33.8% |

The baselines match to within 0.001 on most metrics. Item-item CF matched after ties in
similarity were broken by a shuffled item id instead of the raw id (see the README), which had
lifted its recall@100 by 0.002.

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
| global popularity | 0.0189 | 0.0525 | 0.0103 | 0.0115 | 0.0007 | 69.1 |
| recent popularity (90 days) | 0.0237 | 0.1399 | 0.0102 | 0.0127 | 0.0007 | 23.3 |
| item-item CF (cosine, 50 neighbours) | 0.0074 | 0.0344 | 0.0039 | 0.0051 | 0.6171 | 6.2 |

Validation window:

| model | recall@10 | recall@100 | ndcg@10 | mrr |
|:--|--:|--:|--:|--:|
| global popularity | 0.0195 | 0.0581 | 0.0130 | 0.0197 |
| recent popularity (90 days) | 0.0411 | 0.1722 | 0.0207 | 0.0256 |
| item-item CF (cosine, 50 neighbours) | 0.0122 | 0.0490 | 0.0079 | 0.0117 |

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
from recent behaviour, recent popularity most of all (recall@100 0.140 there, 0.253 when refit
on train plus validation). The validation rows are not affected. The refit test numbers are in the
M3 section, and the observation above that recent popularity wins still holds.

## M2: candidate generation

Produced by `make retrieval`; the full table is written to `artifacts/retrieval.md`. Same split,
filtering and seen-item rules as M1. Items are retrieved with an exact inner-product scan.

Test window:

| model | recall@10 | recall@100 | ndcg@10 | mrr | coverage@100 | popularity bias@100 |
|:--|--:|--:|--:|--:|--:|--:|
| global popularity | 0.0189 | 0.0525 | 0.0103 | 0.0115 | 0.0048 | 30.6 |
| recent popularity (90 days) | 0.0237 | 0.1399 | 0.0102 | 0.0127 | 0.0052 | 12.2 |
| item-item CF | 0.0074 | 0.0344 | 0.0039 | 0.0051 | 0.7996 | 3.9 |
| ALS | 0.0147 | 0.0731 | 0.0091 | 0.0122 | 0.2094 | 11.0 |
| two-tower | 0.0042 | 0.0309 | 0.0022 | 0.0035 | 0.9984 | 1.7 |

Validation recall@100: ALS 0.0952, two-tower 0.0488 (recent popularity 0.1722).

Hyperparameters were chosen on the validation window only, so the test window did not influence
any choice. The search was run with the Python implementation before the port and was not
repeated in Rust, so the Rust settings are the same ones, not a new search.

- ALS: a sweep of factors, regularisation and `alpha` (confidence scaling). `alpha` mattered most.
  Without it (alpha 1) validation recall@100 was 0.056; alpha 40 gave 0.096 and alpha 100 gave
  0.097. Factors and regularisation changed it by less than 0.01.
- Two-tower: configurations were tried by hand with validation recall@100 as the score.

  | learning rate | temperature | epochs | batch | val recall@100 |
  |--:|--:|--:|--:|--:|
  | 0.001 | 0.1 | 5 | 4096 | 0.004 |
  | 0.01 | 0.1 | 5 | 1024 | 0.016 |
  | 0.03 | 0.1 | 5 | 1024 | 0.025 |
  | 0.01 | 0.5 | 5 | 1024 | 0.023 |
  | 0.1 | 0.1 | 15 | 1024 | 0.039 |
  | 0.03 | 0.1 | 20 | 1024 | 0.039 |
  | 0.03 | 0.3 | 20 | 1024 | 0.051 |

  The first setting left the loss at ln(batch size), so the model was at random. The last row is
  the committed config.

Observations:

- ALS beats item-item CF and global popularity on recall@100 and beats item-item CF on ndcg@10
  (0.0091 against 0.0039; global popularity is 0.0103, within noise). It covers 21% of the
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

## M3: ranking

Produced by `make ranking`; the full output, including the feature importance table, is written
to `artifacts/ranking.md`.

Protocol:

1. Fit all five retrievers on train (popularity, recent popularity, item-item CF, ALS, two-tower).
2. Take the top 100 from each and merge them into one candidate set per user (425 candidates per
   user on validation). Compute features from train history only.
3. Label a candidate 1 when the user interacted with it in the validation window. Train the
   LambdaMART ranker on these rows. 20% of validation users are held out to stop
   boosting early (54 trees were used). The split is by user.
4. Refit every retriever on train plus validation (the warm validation rows, about half of all
   validation rows) before serving the test window. Build candidates (425 per user) and features
   from train plus validation history and score them with the ranker. The ranker is not refit,
   and the test window is not touched before this step.

Features (18 in total), grouped for the ablation:

- retriever: rank in each retriever's top 100 (missing when absent) and the ALS and two-tower
  scores for every candidate
- user: interaction count, mean rating, days since last interaction, mean popularity of the
  items they interact with
- item: interaction count, count in the last 90 days, mean rating, days since first and last
  interaction
- pair: item popularity minus the user's usual popularity, item mean rating minus user mean rating

Three mistakes were found and fixed while building this stage in the Python version, and the
Rust version keeps the fixes.

- Id columns were passed to the ranker as features by accident. Ids follow the raw file order,
  which tracks popularity over the whole log, test window included. They are excluded.
- Candidates were ordered by item id inside each user, and tied scores kept that order. A
  user-only ranker gives every candidate of a user the same score, so its "ranking" was the lowest
  item ids first, and it scored ndcg@10 0.0119 from that alone. Candidates are shuffled within
  each user with a fixed seed.
- The test-window retrievers were fit on train only, so recent popularity looked at the last 90
  days of train, which ended about 19 months before the test window. The M1 and M2 test rows have
  the same flaw. Retrievers are refit on train plus validation for the test window. This lifts
  recent popularity recall@100 from 0.140 to 0.253.

Test window, same 8,895 users as M1 and M2, retrievers refit on train plus validation:

| model | recall@10 | ndcg@10 | mrr | recall@100 |
|:--|--:|--:|--:|--:|
| global popularity | 0.0190 | 0.0106 | 0.0124 | 0.0590 |
| recent popularity | 0.0594 | 0.0316 | 0.0349 | 0.2525 |
| item-item CF | 0.0113 | 0.0063 | 0.0078 | 0.0505 |
| ALS | 0.0243 | 0.0139 | 0.0174 | 0.1167 |
| two-tower | 0.0056 | 0.0031 | 0.0049 | 0.0464 |
| recent popularity, fit on train only (as in M1 and M2) | 0.0237 | 0.0102 | 0.0127 | 0.1399 |
| candidates + ranker | 0.0663 | 0.0375 | 0.0436 | 0.2749 |

Against a baseline that is refit, the ranker is a modest improvement: ndcg@10 0.0375 against
0.0316 (+19%), recall@10 0.0663 against 0.0594, recall@100 0.2749 against 0.2525. With a single
run these differences are small, and the experiment in EXPERIMENT.md does not find the hit-rate
lift significant. The candidate set contains 33.8% of the relevant test pairs, so the share of
all relevant pairs any ranker can recover from these candidates is capped at 33.8%.

Feature-group ablation, retraining the ranker each time and scoring on test:

| features | recall@10 | ndcg@10 | mrr | recall@100 |
|:--|--:|--:|--:|--:|
| all features | 0.0663 | 0.0375 | 0.0436 | 0.2749 |
| without retriever | 0.0250 | 0.0118 | 0.0164 | 0.2488 |
| without user | 0.0598 | 0.0317 | 0.0368 | 0.2706 |
| without item | 0.0629 | 0.0359 | 0.0418 | 0.2572 |
| without pair | 0.0644 | 0.0347 | 0.0401 | 0.2747 |
| only retriever | 0.0644 | 0.0363 | 0.0408 | 0.2465 |
| only user | 0.0095 | 0.0048 | 0.0067 | 0.0845 |
| only item | 0.0302 | 0.0146 | 0.0189 | 0.2414 |
| only pair | 0.0150 | 0.0081 | 0.0113 | 0.1167 |

Observations:

- The retriever group carries the ranker. Removing it (rank and score features together) cuts
  ndcg@10 from 0.0375 to 0.0118, and the retriever features alone reach 0.0363.
- The other groups add a little on top of the retriever features. Item features help recall@100
  (0.2465 with retriever features only, 0.2749 with everything). Removing the user group lowers
  ndcg@10 to 0.0317 here, which the Python version did not show (it scored higher without user
  features), so the user group's effect is within run-to-run noise.
- By gain, the most important feature is days since the item was last seen (23.6%), then the
  item's interaction count over the last 90 days (16.7%) and the two-tower score (11.5%). The
  retriever features already carry recency, because the refit recent-popularity rank is one of
  them.
- The two-tower score is third by gain even though the two-tower model is weak on its own.
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
error. `make bundle` trains everything and saves one file (76 MB, about 6 minutes),
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
  answers, and the cache is skipped for the next 5 seconds. The connection is opened on first use,
  so a Redis that is down at startup is a run of misses, not a failed start. A malformed cache
  value is also treated as a miss.
- Retrieval and ranking run on a blocking thread pool, so requests waiting on Redis do not
  stall CPU work.

Latency budgets were set before measuring: p95 of 50 ms on a cache hit, 250 ms on a miss and 25 ms
for cold start. Requests go over HTTP to one process, one at a time, from the same machine
(8 cores, WSL), 500 users per scenario:

| scenario | requests | p50 ms | p95 ms | p99 ms | p95 budget ms | budget met |
|:--|--:|--:|--:|--:|--:|:--|
| cache miss | 500 | 3.4 | 4.3 | 5.3 | 250 | yes |
| cache hit | 500 | 1.4 | 2.5 | 2.9 | 50 | yes |
| cold start | 500 | 0.3 | 0.4 | 0.8 | 25 | yes |
| cache hit, 8 concurrent clients | 500 | 2.9 | 9.7 | 16.0 | none | |

Concurrent throughput: 1,917 requests per second.

Observations:

- All three budgets are met with a wide margin. A hit is about 2.4 times faster than a miss at
  the median (1.4 ms against 3.4 ms). The cache saves less than it did in Python because
  retrieval itself is now cheap, so on this catalogue the cache is worth little. It would matter
  more for a larger catalogue or heavier retrievers.
- The load generator is a Rust client in a separate process. The Python numbers in the port table
  came from Python threads, so the client itself is faster now and part of the speedup comes from
  that. The concurrent row is indicative, and concurrent cache misses were not benchmarked.
- Cold start is a single list lookup, so it is the cheapest path. Its quality was not evaluated:
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
