# Results

All numbers come from `make baselines` on Amazon Reviews 2023, Video Games, 5-core ratings
(814,586 interactions, 94,762 users, 25,612 items). Full tables with k=20 and coverage at every
cutoff are written to `artifacts/baselines.md` by the script.

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
| two-tower | 0.0044 | 0.0349 | 0.0025 | 0.0039 | 0.9975 | 1.7 |

Validation recall@100: ALS 0.0956, two-tower 0.0489 (recent popularity 0.1721).

Tuning used the validation window only (`scripts/tune_retrieval.py`).

- ALS: confidence scaling (`alpha`) mattered most. Without it validation recall@100 was 0.056;
  alpha 40 gives 0.096. Factors and regularisation changed it by less than 0.01.
- Two-tower: the initial settings (lr 0.001, temperature 0.1) left the loss at ln(batch size),
  so the model was at random. Raising the learning rate to 0.03 and the temperature to 0.3
  lifted validation recall@100 from 0.005 to 0.049.

Observations:

- ALS beats item-item CF and global popularity on recall@100 and ndcg@10, and covers 21% of the
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
