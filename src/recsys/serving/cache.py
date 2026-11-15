"""Cache of each user's merged candidate list, so a repeat request skips retrieval.

Retrieval is the expensive part of a request: five retrievers run against the catalogue. Ranking
needs only the candidate items and their retriever ranks and scores, so that is what is stored.
User and item statistics come from the bundle on every request. They do not change between
bundle builds, so a hit ranks to the same list the miss produced.
"""

import time

import numpy as np
import pandas as pd
import redis

RETRY_AFTER_SECONDS = 5.0


class NullCache:
    """Used when no Redis is configured. Every lookup is a miss and nothing is stored."""

    enabled = False

    def get(self, user_id: int):
        return None

    def put(self, user_id: int, candidates: pd.DataFrame) -> None:
        return None


class RedisCandidateCache:
    """Stores candidate frames in Redis as packed float32 rows with a time to live.

    The key carries the bundle version, so a new bundle never reads candidates produced by an
    old one. Redis failures are treated as a miss: the service keeps answering. After a failure
    the cache stays out of the way for a few seconds, so an outage costs one timeout, not two on
    every request.
    """

    enabled = True

    def __init__(self, client: redis.Redis, bundle_version: str, columns: list[str], ttl_seconds: int):
        self.client = client
        self.version = bundle_version
        self.columns = columns  # candidate columns after user_id, in the stored order
        self.ttl_seconds = ttl_seconds
        self._retry_at = 0.0  # monotonic time before which Redis is not contacted

    def _key(self, user_id: int) -> str:
        return f"candidates:{self.version}:{user_id}"

    def _is_paused(self) -> bool:
        return time.monotonic() < self._retry_at

    def _pause(self) -> None:
        self._retry_at = time.monotonic() + RETRY_AFTER_SECONDS

    def get(self, user_id: int) -> pd.DataFrame | None:
        if self._is_paused():
            return None
        try:
            raw = self.client.get(self._key(user_id))
        except redis.RedisError:
            self._pause()
            return None
        if raw is None:
            return None
        row_width = len(self.columns) * np.dtype(np.float32).itemsize
        if len(raw) % row_width:
            return None  # not something this cache wrote, so recompute instead of failing
        values = np.frombuffer(raw, dtype=np.float32).reshape(-1, len(self.columns))
        candidates = pd.DataFrame(values, columns=self.columns)
        candidates.insert(0, "user_id", user_id)
        candidates["item_id"] = candidates["item_id"].astype(np.int64)
        return candidates

    def put(self, user_id: int, candidates: pd.DataFrame) -> None:
        if self._is_paused():
            return
        packed = np.ascontiguousarray(candidates[self.columns].to_numpy(dtype=np.float32))
        try:
            self.client.set(self._key(user_id), packed.tobytes(), ex=self.ttl_seconds)
        except redis.RedisError:
            self._pause()
