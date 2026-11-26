//! Cache of each user's merged candidate list, so a repeat request skips retrieval.
//!
//! Retrieval is the expensive part of a request: five retrievers run against the catalogue.
//! Ranking needs only the candidate items and their retriever ranks and scores, so that is what
//! is stored. User and item statistics come from the bundle on every request. They do not
//! change between bundle builds, so a hit ranks to the same list the miss produced.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use redis::aio::{ConnectionManager, ConnectionManagerConfig};
use redis::AsyncCommands;
use tokio::sync::OnceCell;

use crate::ranking::candidates::UserCandidates;

const RETRY_AFTER: Duration = Duration::from_secs(5);
const REDIS_TIMEOUT: Duration = Duration::from_millis(100);

/// Stores candidate lists in Redis as packed little-endian f32 rows with a time to live.
///
/// The key carries the bundle version, so a new bundle never reads candidates produced by an
/// old one. Redis failures are treated as a miss: the service keeps answering. After a failure
/// the cache stays out of the way for a few seconds, so an outage costs one timeout, not two on
/// every request.
pub struct RedisCache {
    client: redis::Client,
    connection: OnceCell<ConnectionManager>,
    version: String,
    n_retrievers: usize,
    n_scorers: usize,
    ttl_seconds: u64,
    started: Instant,
    /// Milliseconds since `started` before which Redis is not contacted.
    retry_at_ms: AtomicU64,
}

pub enum Cache {
    /// Used when no Redis is configured. Every lookup is a miss and nothing is stored.
    Off,
    Redis(Box<RedisCache>),
    #[cfg(test)]
    Memory(
        std::sync::Mutex<std::collections::HashMap<u32, Vec<u8>>>,
        usize,
        usize,
    ),
}

impl Cache {
    pub fn redis(
        url: &str,
        version: &str,
        n_retrievers: usize,
        n_scorers: usize,
        ttl_seconds: u64,
    ) -> Result<Cache, redis::RedisError> {
        Ok(Cache::Redis(Box::new(RedisCache {
            client: redis::Client::open(url)?,
            connection: OnceCell::new(),
            version: version.to_string(),
            n_retrievers,
            n_scorers,
            ttl_seconds,
            started: Instant::now(),
            retry_at_ms: AtomicU64::new(0),
        })))
    }

    pub fn is_enabled(&self) -> bool {
        !matches!(self, Cache::Off)
    }

    pub async fn get(&self, user: u32) -> Option<UserCandidates> {
        match self {
            Cache::Off => None,
            Cache::Redis(cache) => cache.get(user).await,
            #[cfg(test)]
            Cache::Memory(entries, n_retrievers, n_scorers) => {
                let bytes = entries.lock().unwrap().get(&user).cloned()?;
                unpack(&bytes, *n_retrievers, *n_scorers)
            }
        }
    }

    pub async fn put(&self, user: u32, candidates: &UserCandidates) {
        match self {
            Cache::Off => {}
            Cache::Redis(cache) => cache.put(user, candidates).await,
            #[cfg(test)]
            Cache::Memory(entries, ..) => {
                entries.lock().unwrap().insert(user, pack(candidates));
            }
        }
    }
}

/// Packs each candidate as `[item, ranks..., scores...]` in f32, which keeps ids exact up to 2^24.
pub fn pack(candidates: &UserCandidates) -> Vec<u8> {
    let n_items = candidates.items.len().max(1);
    let n_ranks = candidates.ranks.len() / n_items;
    let n_scores = candidates.scores.len() / n_items;
    let mut bytes = Vec::with_capacity(candidates.items.len() * (1 + n_ranks + n_scores) * 4);
    for (row, &item) in candidates.items.iter().enumerate() {
        bytes.extend((item as f32).to_le_bytes());
        for value in &candidates.ranks[row * n_ranks..][..n_ranks] {
            bytes.extend(value.to_le_bytes());
        }
        for value in &candidates.scores[row * n_scores..][..n_scores] {
            bytes.extend(value.to_le_bytes());
        }
    }
    bytes
}

/// Reverses `pack`. Returns `None` for bytes this cache did not write, so the caller recomputes.
pub fn unpack(bytes: &[u8], n_retrievers: usize, n_scorers: usize) -> Option<UserCandidates> {
    let width = 1 + n_retrievers + n_scorers;
    if bytes.is_empty() || !bytes.len().is_multiple_of(width * 4) {
        return None;
    }
    let (words, _) = bytes.as_chunks::<4>();
    let values: Vec<f32> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
    let mut candidates = UserCandidates {
        items: Vec::new(),
        ranks: Vec::new(),
        scores: Vec::new(),
    };
    for row in values.chunks_exact(width) {
        candidates.items.push(row[0] as u32);
        candidates
            .ranks
            .extend_from_slice(&row[1..1 + n_retrievers]);
        candidates
            .scores
            .extend_from_slice(&row[1 + n_retrievers..]);
    }
    Some(candidates)
}

impl RedisCache {
    fn key(&self, user: u32) -> String {
        format!("candidates:{}:{user}", self.version)
    }

    fn now_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    fn is_paused(&self) -> bool {
        self.now_ms() < self.retry_at_ms.load(Ordering::Relaxed)
    }

    fn pause(&self) {
        self.retry_at_ms.store(
            self.now_ms() + RETRY_AFTER.as_millis() as u64,
            Ordering::Relaxed,
        );
    }

    /// The shared connection, opened on first use so a Redis that is down at startup is only a
    /// run of misses, not a failed start.
    async fn connection(&self) -> Option<ConnectionManager> {
        let config = ConnectionManagerConfig::new()
            .set_connection_timeout(REDIS_TIMEOUT)
            .set_response_timeout(REDIS_TIMEOUT)
            .set_number_of_retries(0);
        let opened = self
            .connection
            .get_or_try_init(|| ConnectionManager::new_with_config(self.client.clone(), config))
            .await;
        match opened {
            Ok(connection) => Some(connection.clone()),
            Err(_) => {
                self.pause();
                None
            }
        }
    }

    async fn get(&self, user: u32) -> Option<UserCandidates> {
        if self.is_paused() {
            return None;
        }
        let mut connection = self.connection().await?;
        match connection.get::<_, Option<Vec<u8>>>(self.key(user)).await {
            Ok(bytes) => unpack(&bytes?, self.n_retrievers, self.n_scorers),
            Err(_) => {
                self.pause();
                None
            }
        }
    }

    async fn put(&self, user: u32, candidates: &UserCandidates) {
        if self.is_paused() {
            return;
        }
        let Some(mut connection) = self.connection().await else {
            return;
        };
        let stored: redis::RedisResult<()> = connection
            .set_ex(self.key(user), pack(candidates), self.ttl_seconds)
            .await;
        if stored.is_err() {
            self.pause();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates() -> UserCandidates {
        UserCandidates {
            items: vec![7, 3],
            ranks: vec![1.0, f32::NAN, f32::NAN, 2.0],
            scores: vec![0.5, -0.25],
        }
    }

    #[test]
    fn packing_round_trips_candidates_and_missing_ranks() {
        let restored = unpack(&pack(&candidates()), 2, 1).unwrap();
        assert_eq!(restored.items, vec![7, 3]);
        assert_eq!(restored.scores, vec![0.5, -0.25]);
        assert_eq!(restored.ranks[0], 1.0);
        assert!(restored.ranks[1].is_nan() && restored.ranks[2].is_nan());
        assert_eq!(restored.ranks[3], 2.0);
    }

    #[test]
    fn bytes_this_cache_did_not_write_are_a_miss() {
        assert!(unpack(b"not float32 row", 2, 1).is_none()); // 15 bytes, not a whole row
        assert!(unpack(b"", 2, 1).is_none());
    }

    #[tokio::test]
    async fn an_unreachable_redis_is_a_miss_and_pauses_the_cache() {
        let cache = Cache::redis("redis://127.0.0.1:1/0", "v1", 2, 1, 60).unwrap();
        assert!(cache.get(1).await.is_none());
        let Cache::Redis(redis_cache) = &cache else {
            unreachable!()
        };
        assert!(
            redis_cache.is_paused(),
            "a failed lookup should pause the cache"
        );
        cache.put(1, &candidates()).await; // skipped while paused, must not panic or block
    }
}
