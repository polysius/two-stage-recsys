# Two-stage recommender system

A recommender built the way large catalogues are served: a fast retrieval stage proposes
candidates, a learned ranker orders them, and the system is evaluated offline and in a simulated
online experiment.

Dataset: Amazon Reviews 2023, Video Games 5-core ratings.

## Status

| milestone | state |
|:--|:--|
| M1 data and baselines | done, see [RESULTS.md](RESULTS.md) |
| M2 candidate generation (ALS, two-tower, FAISS) | planned |
| M3 ranking (LightGBM LambdaMART) | planned |
| M4 simulated A/B experiment | planned |
| M5 serving (FastAPI, Redis) | planned |

## Setup

LightGBM and implicit do not install on native ARM64 Windows. Run everything in WSL or Linux.

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" tabulate
make data DATA_DIR=~/recsys-data
make test
make baselines DATA_DIR=~/recsys-data
```

`make data` downloads the ratings file and writes its SHA-256 next to it. Data is not committed.

## Layout

```
src/recsys/data.py         load interactions, dense ids
src/recsys/split.py        time-based split and leakage check
src/recsys/metrics.py      recall@k, NDCG@k, MRR, coverage, popularity bias
src/recsys/baselines.py    popularity, recent popularity, item-item CF
src/recsys/evaluation.py   score any recommender on a split
scripts/                   data download, baseline run
tests/                     unit tests on a small synthetic log
```

## Evaluation protocol

Interactions are split by time, never at random. Evaluation keeps only users and items seen in
train, and the removed share is reported. Items a user has already seen are masked from their
recommendations. `tests/test_split.py` fails if any split contains an interaction older than the
end of the previous one.
