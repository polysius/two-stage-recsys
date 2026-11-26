# Two-stage recommender system

A recommender built the way large catalogues are served: a fast retrieval stage proposes
candidates, a learned ranker orders them, and the system is evaluated offline and in a simulated
online experiment. Everything is written in Rust, with no C or C++ dependencies.

Dataset: Amazon Reviews 2023, Video Games 5-core ratings.

## Status

| milestone | state |
|:--|:--|
| M1 data and baselines | done, see [RESULTS.md](RESULTS.md) |
| M2 candidate generation (ALS, two-tower, exact inner-product search) | done, see [RESULTS.md](RESULTS.md) |
| M3 ranking (LambdaMART) | done, see [RESULTS.md](RESULTS.md) |
| M4 simulated A/B experiment | done, see [EXPERIMENT.md](EXPERIMENT.md) |
| M5 serving (axum, Redis) | done, see [RESULTS.md](RESULTS.md) |
| Rust port | done, timings and parity checks in [RESULTS.md](RESULTS.md) |

## Setup

Install a Rust toolchain with [rustup](https://rustup.rs). The crate uses only Rust
dependencies, so it needs no C compiler or system libraries. On the ARM64 Windows machine it was
developed on the MSVC linker is missing, so it is built and run inside WSL.

```
make test
make data DATA_DIR=~/recsys-data
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
src/table.rs               load interactions, dense ids, time-based split and leakage check
src/metrics.rs             recall@k, NDCG@k, MRR, coverage, popularity bias
src/baselines.rs           popularity, recent popularity, item-item CF
src/evaluation.rs          score any recommender on a split
src/retrieval/             ALS, two-tower (hand-written backward pass) and exact inner-product search
src/lambdamart.rs          gradient boosted trees with the LambdaMART objective
src/ranking/               merged candidates, ranking features, ranker training
src/stats.rs               power analysis, SRM check, proportion and mean tests
src/serving/               serving bundle, Redis candidate cache, recommender and HTTP API
src/commands/              one module per subcommand of the `recsys` binary
configs/default.yaml       every setting
```

## Evaluation protocol

Interactions are split by time, never at random. Evaluation keeps only users and items seen in
train, and the removed share is reported. Items a user has already seen are masked from their
recommendations. A unit test in `src/table.rs` fails if any split contains an interaction older
than the end of the previous one.

Equal scores are ordered by a fixed shuffle of the item id, not by the id itself. Item ids follow
the order of the ratings file, which tracks popularity over the whole log, so breaking ties by id
would give tied items information from the evaluation window.
