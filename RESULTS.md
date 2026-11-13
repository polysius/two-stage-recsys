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

## M3: ranking

Produced by `make ranking`; the full output, including the feature importance table, is written
to `artifacts/ranking.md`.

Protocol:

1. Fit all five retrievers on train (popularity, recent popularity, item-item CF, ALS, two-tower).
2. Take the top 100 from each and merge them into one candidate set per user (424 candidates per
   user on validation, 426 on test). Compute features from train history only.
3. Label a candidate 1 when the user interacted with it in the validation window. Train the
   LightGBM LambdaMART ranker on these rows. 20% of validation users are held out to stop
   boosting early (49 trees were used). The split is by user.
4. For the test window, build candidates and features the same way from train plus validation
   history (retrievers are not refit) and score with the ranker. The test window is not touched
   before this step.

Features (18 in total), grouped for the ablation:

- retriever: rank in each retriever's top 100 (empty when absent) and the ALS and two-tower
  scores for every candidate
- user: interaction count, mean rating, days since last interaction, mean popularity of the
  items they interacted with
- item: interaction count, count in the last 90 days, mean rating, days since first and last
  interaction
- pair: item popularity minus the user's usual popularity, item mean rating minus user mean rating

Two leaks were found and removed while building this stage, so the numbers below are the ones to
trust.

- Id columns were passed to the ranker as features by accident. Ids follow the raw file order,
  which tracks popularity over the whole log, test window included. They are now excluded.
- Candidates were ordered by item id inside each user, and tied scores kept that order. A
  user-only ranker gives every candidate of a user the same score, so its "ranking" was the lowest
  item ids first, and it scored ndcg@10 0.0119 from that alone. Candidates are now shuffled within
  each user with a fixed seed, and the same ablation row scores 0.0024.

Test window, same 8,895 users as M1 and M2:

| model | recall@10 | ndcg@10 | mrr | recall@100 |
|:--|--:|--:|--:|--:|
| global popularity | 0.0189 | 0.0103 | 0.0115 | 0.0523 |
| recent popularity | 0.0238 | 0.0102 | 0.0126 | 0.1394 |
| item-item CF | 0.0074 | 0.0039 | 0.0051 | 0.0344 |
| ALS | 0.0151 | 0.0090 | 0.0118 | 0.0723 |
| two-tower | 0.0045 | 0.0027 | 0.0040 | 0.0340 |
| candidates + ranker | 0.0545 | 0.0306 | 0.0349 | 0.1720 |

The ranker triples ndcg@10 over the best single retriever (0.0306 against 0.0103) and lifts
recall@100 from 0.139 to 0.172. The candidate set contains 22.7% of the relevant test pairs, so
the share of all relevant pairs any ranker can recover from these candidates is capped at 22.7%.

Feature-group ablation, retraining the ranker each time and scoring on test:

| features | recall@10 | ndcg@10 | mrr | recall@100 |
|:--|--:|--:|--:|--:|
| all features | 0.0545 | 0.0306 | 0.0349 | 0.1720 |
| without retriever | 0.0277 | 0.0152 | 0.0197 | 0.1611 |
| without user | 0.0530 | 0.0297 | 0.0342 | 0.1703 |
| without item | 0.0448 | 0.0254 | 0.0284 | 0.1446 |
| without pair | 0.0549 | 0.0291 | 0.0328 | 0.1704 |
| only retriever | 0.0483 | 0.0268 | 0.0298 | 0.1454 |
| only user | 0.0047 | 0.0024 | 0.0041 | 0.0540 |
| only item | 0.0224 | 0.0150 | 0.0222 | 0.1492 |
| only pair | 0.0084 | 0.0043 | 0.0064 | 0.0712 |

Observations:

- The retriever group matters most in the ablation: removing it (rank and score features
  together) halves ndcg@10. The item group is next: removing it costs 0.005 ndcg@10 and 0.027
  recall@100.
- By gain, the single most important feature is the item's interaction count over the last 90
  days (27.3%), then the two-tower score (12.9%) and days since the item was last seen (12.8%).
  This matches the M1 finding that recency drives demand in this category.
- The two-tower score ranks second by gain even though the two-tower model is weak on its own.
  That suggests its scores add information the other retrievers lack, but it was not tested
  directly, for example by dropping that one score.
- Removing the user group or the pair group changes ndcg@10 by less than 0.002, which is inside the
  noise of a single run. User and pair features add nothing measurable here. Users average about
  9 interactions, which leaves little to summarise.
- Single seed and a single split, no confidence intervals. Differences below about 0.003 in
  ndcg@10 should not be read as real. The ranker is trained on one validation window with
  features from train only, and applied with features from a longer history on test, so the
  feature distributions shift slightly between training and test.
