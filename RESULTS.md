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
