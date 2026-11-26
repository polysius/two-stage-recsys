//! The online two-stage recommendation for a single user.

use std::sync::Arc;

use crate::ranking::candidates::{user_candidates, CandidateFrame, UserCandidates};
use crate::ranking::features::build_features;
use crate::serving::bundle::ServingBundle;
use crate::serving::cache::Cache;

pub struct Recommendation {
    pub user_id: u32,
    /// "ranked" for users with history, "cold_start" for everyone else.
    pub source: &'static str,
    /// "hit", "miss" or "off".
    pub cache: &'static str,
    /// Item id and ranker score, best first. The score is `None` for cold-start users.
    pub items: Vec<(u32, Option<f32>)>,
}

pub struct TwoStageRecommender {
    pub bundle: ServingBundle,
    cache: Cache,
}

impl TwoStageRecommender {
    pub fn new(bundle: ServingBundle, cache: Cache) -> TwoStageRecommender {
        TwoStageRecommender { bundle, cache }
    }

    /// A user is known when the bundle has statistics for them, which needs some history.
    pub fn is_known(&self, user_id: u32) -> bool {
        self.bundle.user_stats.is_known(user_id)
    }

    pub async fn recommend(self: &Arc<Self>, user_id: u32, k: usize) -> Recommendation {
        if !self.is_known(user_id) {
            return self.cold_start(user_id, k);
        }

        let (candidates, cache_state) = match self.cache.get(user_id).await {
            Some(candidates) => (
                candidates,
                if self.cache.is_enabled() {
                    "hit"
                } else {
                    "off"
                },
            ),
            None => {
                let candidates = self.run_blocking(move |this| this.retrieve(user_id)).await;
                self.cache.put(user_id, &candidates).await;
                (
                    candidates,
                    if self.cache.is_enabled() {
                        "miss"
                    } else {
                        "off"
                    },
                )
            }
        };
        let items = self
            .run_blocking(move |this| this.rank(user_id, &candidates, k))
            .await;
        Recommendation {
            user_id,
            source: "ranked",
            cache: cache_state,
            items,
        }
    }

    /// Ranking and retrieval are CPU work, so they run off the async threads that wait on Redis.
    async fn run_blocking<T: Send + 'static>(
        self: &Arc<Self>,
        work: impl FnOnce(&TwoStageRecommender) -> T + Send + 'static,
    ) -> T {
        let this = Arc::clone(self);
        tokio::task::spawn_blocking(move || work(&this))
            .await
            .expect("worker thread panicked")
    }

    pub fn retrieve(&self, user_id: u32) -> UserCandidates {
        let bundle = &self.bundle;
        user_candidates(
            &bundle.retrievers,
            user_id,
            bundle.seen.row(user_id as usize),
            bundle.per_retriever,
            0,
        )
    }

    pub fn rank(
        &self,
        user_id: u32,
        candidates: &UserCandidates,
        k: usize,
    ) -> Vec<(u32, Option<f32>)> {
        let bundle = &self.bundle;
        let n_rows = candidates.items.len();
        let frame = CandidateFrame {
            users: vec![user_id; n_rows],
            items: candidates.items.clone(),
            n_retrievers: bundle.retrievers.models.len(),
            n_scorers: bundle.scorer_names.len(),
            ranks: candidates.ranks.clone(),
            scores: candidates.scores.clone(),
            group_sizes: vec![n_rows],
            query_users: vec![user_id],
        };
        let scorers: Vec<&str> = bundle.scorer_names.iter().map(String::as_str).collect();
        let features = build_features(
            &frame,
            &bundle.retrievers.names,
            &scorers,
            &bundle.user_stats,
            &bundle.item_stats,
        );
        let scores = bundle.ranker.predict(&features);

        let mut best: Vec<usize> = (0..n_rows).collect();
        best.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        best.into_iter()
            .take(k)
            .map(|row| (candidates.items[row], Some(scores[row])))
            .collect()
    }

    /// Recent popularity: all we can say about someone with no history.
    fn cold_start(&self, user_id: u32, k: usize) -> Recommendation {
        let items = self
            .bundle
            .fallback_items
            .iter()
            .take(k)
            .map(|&item| (item, None))
            .collect();
        Recommendation {
            user_id,
            source: "cold_start",
            cache: "off",
            items,
        }
    }
}
