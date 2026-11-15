"""Measure request latency of the running service against the budgets in the config.

Starts its own Redis and a single-worker uvicorn process, then sends requests one at a time over
HTTP with a keep-alive client, so each number includes serialisation and the network stack.

Scenarios:
  cache miss   known users, empty cache: retrieve, rank, then store the candidates
  cache hit    the same users again: read candidates from Redis, rank
  cold start   ids with no history: the recent-popularity fallback
  concurrent   cache hits from several clients at once, to see how latency holds up under load

Usage: python scripts/benchmark_serving.py --output artifacts/serving.md (needs the bundle from make bundle)
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import yaml

from recsys.serving.bundle import load_bundle

REDIS_PORT = 6390
SERVER_PORT = 8765


def wait_until_healthy(client: httpx.Client, server: subprocess.Popen, timeout_seconds: float):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"the service exited with code {server.returncode}")
        try:
            if client.get("/health").status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise TimeoutError("the service did not become healthy")


def timed_request(client: httpx.Client, user_id: int, k: int):
    started = time.perf_counter()
    response = client.get("/recommend", params={"user_id": user_id, "k": k})
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return elapsed_ms, response.json()


def run_scenario(client, user_ids, k, expected_source, expected_cache) -> np.ndarray:
    """Latencies in ms. Fails loudly if a request was not served the way the scenario assumes."""
    latencies = []
    for user_id in user_ids:
        elapsed_ms, body = timed_request(client, int(user_id), k)
        if (body["source"], body["cache"]) != (expected_source, expected_cache):
            raise RuntimeError(f"user {user_id}: got {body['source']}/{body['cache']}")
        latencies.append(elapsed_ms)
    return np.array(latencies)


def summarise(name, latencies, budget_ms) -> dict:
    p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
    return {"scenario": name, "requests": len(latencies), "p50": p50, "p95": p95, "p99": p99,
            "budget": budget_ms, "met": budget_ms is None or p95 <= budget_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bundle", default="artifacts/serving_bundle.joblib")
    parser.add_argument("--output", default="artifacts/serving.md")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--clients", type=int, default=8)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    budgets = config["serving"]["latency_budget_ms"]
    bundle = load_bundle(Path(args.bundle))
    rng = np.random.default_rng(0)
    known = rng.permutation(bundle.user_stats.index.to_numpy())
    warmup_users, users = known[:50], known[50 : 50 + args.requests]
    unknown_users = bundle.n_users + np.arange(args.requests)

    redis_server = subprocess.Popen(
        ["redis-server", "--port", str(REDIS_PORT), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
    )
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "recsys.serving.api:create_app_from_env", "--factory",
         "--port", str(SERVER_PORT), "--log-level", "warning"],
        env={**os.environ, "BUNDLE_PATH": args.bundle, "CONFIG_PATH": args.config,
             "REDIS_URL": f"redis://localhost:{REDIS_PORT}/0"},
    )
    try:
        base_url = f"http://localhost:{SERVER_PORT}"
        with httpx.Client(base_url=base_url, timeout=10) as client:
            wait_until_healthy(client, server, timeout_seconds=180)
            run_scenario(client, warmup_users, args.k, "ranked", "miss")  # warm code paths
            run_scenario(client, unknown_users[:20], args.k, "cold_start", "off")

            rows = [
                summarise("cache miss", run_scenario(client, users, args.k, "ranked", "miss"),
                          budgets["cache_miss"]),
                summarise("cache hit", run_scenario(client, users, args.k, "ranked", "hit"),
                          budgets["cache_hit"]),
                summarise("cold start", run_scenario(client, unknown_users, args.k, "cold_start",
                                                     "off"), budgets["cold_start"]),
            ]

        def one_client(user_ids):
            with httpx.Client(base_url=base_url, timeout=10) as local:
                return run_scenario(local, user_ids, args.k, "ranked", "hit")

        started = time.perf_counter()
        with ThreadPoolExecutor(args.clients) as pool:
            chunks = pool.map(one_client, np.array_split(users, args.clients))
            concurrent = np.concatenate(list(chunks))
        throughput = len(concurrent) / (time.perf_counter() - started)
        rows.append(summarise(f"cache hit, {args.clients} concurrent clients", concurrent, None))
    finally:
        server.terminate()
        redis_server.terminate()

    lines = [
        "| scenario | requests | p50 ms | p95 ms | p99 ms | p95 budget ms | budget met |",
        "|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for row in rows:
        budget = "" if row["budget"] is None else str(row["budget"])
        met = "" if row["budget"] is None else ("yes" if row["met"] else "no")
        lines.append(f"| {row['scenario']} | {row['requests']} | {row['p50']:.1f} | "
                     f"{row['p95']:.1f} | {row['p99']:.1f} | {budget} | {met} |")
    lines += ["", f"Concurrent throughput: {throughput:.0f} requests per second.",
              f"CPU count: {os.cpu_count()}, one uvicorn worker, k={args.k}."]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
