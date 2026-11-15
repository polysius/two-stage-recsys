# Two-stage recommender system

A recommender built the way large catalogues are served: a fast retrieval stage proposes
candidates, a learned ranker orders them, and the system is evaluated offline and in a simulated
online experiment.

Dataset: Amazon Reviews 2023, Video Games 5-core ratings.

## Status

| milestone | state |
|:--|:--|
| M1 data and baselines | done, see [RESULTS.md](RESULTS.md) |
| M2 candidate generation (ALS, two-tower, FAISS) | done, see [RESULTS.md](RESULTS.md) |
| M3 ranking (LightGBM LambdaMART) | done, see [RESULTS.md](RESULTS.md) |
| M4 simulated A/B experiment | done, see [EXPERIMENT.md](EXPERIMENT.md) |
| M5 serving (FastAPI, Redis) | done, see [RESULTS.md](RESULTS.md) |

## Setup

LightGBM and implicit do not install on native ARM64 Windows. Run everything in WSL or Linux.

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" tabulate
make data DATA_DIR=~/recsys-data
make test
make baselines DATA_DIR=~/recsys-data
make retrieval DATA_DIR=~/recsys-data
make ranking DATA_DIR=~/recsys-data
make experiment DATA_DIR=~/recsys-data   # needs the output of make ranking
make bundle DATA_DIR=~/recsys-data       # trains and saves everything the service needs
make serve                               # service on port 8000; set REDIS_URL to enable caching
make benchmark                           # latency table, starts its own Redis (needs redis-server)
```

`make data` downloads the ratings file, checks it against the pinned SHA-256 (deleting the file on a
mismatch) and writes the checksum next to it. Data is not committed.

## Layout

```
src/recsys/data.py         load interactions, dense ids
src/recsys/split.py        time-based split and leakage check
src/recsys/metrics.py      recall@k, NDCG@k, MRR, coverage, popularity bias
src/recsys/baselines.py    popularity, recent popularity, item-item CF
src/recsys/evaluation.py   score any recommender on a split
src/recsys/retrieval/      ALS, two-tower and FAISS candidate generation
src/recsys/ranking/        merged candidates, ranking features, LightGBM LambdaMART
src/recsys/experiment.py   simulated A/B experiment, power analysis and statistics
src/recsys/serving/        serving bundle, Redis candidate cache, recommender and FastAPI app
scripts/                   data download, baseline, retrieval, ranking and experiment runs, retrieval tuning, serving bundle build and benchmark
tests/                     unit tests on a small synthetic log
```

## Evaluation protocol

Interactions are split by time, never at random. Evaluation keeps only users and items seen in
train, and the removed share is reported. Items a user has already seen are masked from their
recommendations. `tests/test_split.py` fails if any split contains an interaction older than the
end of the previous one.
