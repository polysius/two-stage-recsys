"""FastAPI app: GET /recommend?user_id=...&k=... and GET /health."""

import os
import time
from pathlib import Path

import redis
import yaml
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel

from recsys.serving.bundle import load_bundle
from recsys.serving.cache import RedisCandidateCache
from recsys.serving.recommender import TwoStageRecommender

MAX_K = 100


class ScoredItem(BaseModel):
    item_id: int
    score: float | None  # null for cold-start users, who are not ranked


class RecommendResponse(BaseModel):
    user_id: int
    source: str
    cache: str
    items: list[ScoredItem]


def create_app(recommender: TwoStageRecommender) -> FastAPI:
    app = FastAPI(title="two-stage-recsys")

    @app.middleware("http")
    async def add_process_time(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "bundle_version": recommender.bundle.version}

    # A plain def endpoint runs in FastAPI's thread pool, so the CPU-bound ranking does not
    # block the event loop while another request is waiting on Redis.
    @app.get("/recommend", response_model=RecommendResponse)
    def recommend(user_id: int = Query(ge=0), k: int = Query(default=10, ge=1, le=MAX_K)):
        result = recommender.recommend(user_id, k)
        return RecommendResponse(
            user_id=result.user_id,
            source=result.source,
            cache=result.cache,
            items=[ScoredItem(item_id=item, score=score) for item, score in result.items],
        )

    return app


def create_app_from_env() -> FastAPI:
    """Entry point for uvicorn. BUNDLE_PATH is required; REDIS_URL turns caching on."""
    config = yaml.safe_load(Path(os.environ.get("CONFIG_PATH", "configs/default.yaml")).read_text())
    bundle = load_bundle(Path(os.environ["BUNDLE_PATH"]))
    cache = None
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        cache = RedisCandidateCache(
            redis.Redis.from_url(redis_url, socket_timeout=0.1, socket_connect_timeout=0.1),
            bundle.version,
            bundle.candidate_columns,
            config["serving"]["cache_ttl_seconds"],
        )
    return create_app(TwoStageRecommender(bundle, cache))
