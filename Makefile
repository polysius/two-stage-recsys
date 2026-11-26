# Needs a Rust toolchain (rustup). The project has no C or C++ dependencies. On the ARM64 Windows
# machine it was developed on, the MSVC linker is missing, so it is built and run inside WSL.
DATA_DIR ?= data
RECSYS = target/release/recsys --data-dir $(DATA_DIR)

.PHONY: build data test lint baselines retrieval ranking experiment bundle serve benchmark

build:
	cargo build --release

data: build
	target/release/recsys download-data --data-dir $(DATA_DIR)

test:
	cargo test --release

lint:
	cargo clippy --release --all-targets -- -D warnings
	cargo fmt --check

baselines: build
	$(RECSYS) baselines --output artifacts/baselines.md

retrieval: build
	$(RECSYS) retrieval --output artifacts/retrieval.md

ranking: build
	$(RECSYS) ranking --output artifacts/ranking.md

experiment: build
	$(RECSYS) experiment --output artifacts/experiment.md   # needs the output of make ranking

bundle: build
	$(RECSYS) bundle --output artifacts/serving_bundle.bin  # trains and saves everything the service needs

serve: build
	BUNDLE_PATH=artifacts/serving_bundle.bin target/release/recsys serve --port 8000   # set REDIS_URL to enable caching

benchmark: build
	target/release/recsys benchmark --output artifacts/serving.md   # starts its own Redis (needs redis-server)
