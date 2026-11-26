//! Everything the online service needs, built offline and saved as one file.
//!
//! The bundle holds the retrievers fit on all history, the ranker trained on the validation
//! window, and the precomputed user and item statistics, so a request never touches the raw
//! interaction log.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::ranking::build_stage;
use crate::ranking::candidates::scorer_names;
use crate::ranking::features::{item_statistics, user_statistics, ItemStats, UserStats};
use crate::ranking::ranker::{train_ranker, TrainedRanker};
use crate::recommender::Recommender;
use crate::registry::{fit_retrievers, RetrieverSet, RETRIEVER_NAMES};
use crate::sparse::BinaryCsr;
use crate::table::{Split, Table};

/// How many recent-popularity items are kept for users with no history.
pub const FALLBACK_LENGTH: usize = 100;

#[derive(Serialize, Deserialize)]
pub struct ServingBundle {
    pub version: String,
    pub n_users: usize,
    pub n_items: usize,
    pub per_retriever: usize,
    pub retrievers: RetrieverSet,
    pub ranker: TrainedRanker,
    pub scorer_names: Vec<String>,
    pub user_stats: UserStats,
    pub item_stats: ItemStats,
    pub seen: BinaryCsr,
    /// Recent-popularity ranking, best first, for users with no history.
    pub fallback_items: Vec<u32>,
}

impl ServingBundle {
    /// Values stored per candidate: the item id, one rank per retriever, one score per scorer.
    pub fn candidate_width(&self) -> usize {
        1 + self.retrievers.models.len() + self.scorer_names.len()
    }
}

/// Trains the ranker the way the `ranking` command does, then refits retrievers for serving.
pub fn build_bundle(
    config: &Config,
    split: &Split,
    n_users: usize,
    n_items: usize,
) -> Result<ServingBundle> {
    let ranking = &config.ranking;
    let per_retriever = ranking.candidates_per_retriever;

    let train_retrievers = RetrieverSet::from_fitted(fit_retrievers(
        &RETRIEVER_NAMES,
        config,
        n_users,
        n_items,
        &split.train,
        "train",
    ));
    let stage = build_stage(
        &train_retrievers,
        &split.train,
        &split.val,
        n_users,
        n_items,
        per_retriever,
    )?;
    let columns = stage.features.columns_in_groups(None);
    let ranker = train_ranker(
        &stage.features,
        &stage.frame.group_sizes,
        &stage.labels,
        &columns,
        ranking,
        0,
    );

    let history = Table::concat(&split.train, &split.val);
    let retrievers = RetrieverSet::from_fitted(fit_retrievers(
        &RETRIEVER_NAMES,
        config,
        n_users,
        n_items,
        &history,
        "train + validation",
    ));
    let fallback_items =
        retrievers
            .get("recent_popularity")
            .recommend_user(0, FALLBACK_LENGTH, &[]);
    let seconds = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    Ok(ServingBundle {
        version: format!("{seconds:x}"),
        n_users,
        n_items,
        per_retriever,
        scorer_names: scorer_names(&retrievers)
            .into_iter()
            .map(String::from)
            .collect(),
        retrievers,
        ranker,
        user_stats: user_statistics(&history, n_users, n_items),
        item_stats: item_statistics(&history, n_items),
        seen: BinaryCsr::user_item(&history, n_users, n_items),
        fallback_items,
    })
}
