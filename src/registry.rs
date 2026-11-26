//! Builds every retriever and baseline from the config so scripts compare the same models.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::baselines::{ItemItemCf, Popularity};
use crate::config::Config;
use crate::recommender::Recommender;
use crate::retrieval::als::fit_als;
use crate::retrieval::embeddings::EmbeddingRetriever;
use crate::retrieval::two_tower::fit_two_tower;
use crate::table::Table;

pub const SEED: u64 = 0;

/// Every model that can propose candidates, in the order the reports list them.
pub const RETRIEVER_NAMES: [&str; 5] = [
    "popularity",
    "recent_popularity",
    "item_item_cf",
    "als",
    "two_tower",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Retriever {
    Popularity(Popularity),
    ItemItemCf(ItemItemCf),
    Embedding(EmbeddingRetriever),
}

impl Retriever {
    /// True for models that can score any (user, item) pair, not only list their own top items.
    pub fn can_score_pairs(&self) -> bool {
        matches!(self, Retriever::Embedding(_))
    }
}

impl Recommender for Retriever {
    fn recommend_user(&self, user: u32, k: usize, seen: &[u32]) -> Vec<u32> {
        match self {
            Retriever::Popularity(model) => model.recommend_user(user, k, seen),
            Retriever::ItemItemCf(model) => model.recommend_user(user, k, seen),
            Retriever::Embedding(model) => model.recommend_user(user, k, seen),
        }
    }

    fn score_pair(&self, user: u32, item: u32) -> Option<f32> {
        match self {
            Retriever::Embedding(model) => model.score_pair(user, item),
            _ => None,
        }
    }
}

pub struct FittedRetriever {
    pub name: &'static str,
    pub model: Retriever,
    pub fit_seconds: f64,
    /// Training loss per epoch, for models that train with gradient descent.
    pub loss_history: Vec<f32>,
}

fn fit_one(
    name: &'static str,
    config: &Config,
    n_users: usize,
    n_items: usize,
    table: &Table,
) -> FittedRetriever {
    let started = Instant::now();
    let mut loss_history = Vec::new();
    let model = match name {
        "popularity" => Retriever::Popularity(Popularity::fit(n_items, table, None)),
        "recent_popularity" => Retriever::Popularity(Popularity::fit(
            n_items,
            table,
            Some(config.baselines.recent_window_days),
        )),
        "item_item_cf" => Retriever::ItemItemCf(ItemItemCf::fit(
            n_users,
            n_items,
            config.baselines.item_cf_neighbors,
            table,
        )),
        "als" => Retriever::Embedding(fit_als(n_users, n_items, table, &config.als, SEED)),
        "two_tower" => {
            let (retriever, losses) =
                fit_two_tower(n_users, n_items, table, &config.two_tower, SEED);
            loss_history = losses;
            Retriever::Embedding(retriever)
        }
        other => panic!("unknown retriever {other}"),
    };
    FittedRetriever {
        name,
        model,
        fit_seconds: started.elapsed().as_secs_f64(),
        loss_history,
    }
}

/// Fitted models with their names, in a fixed order, as the ranking stage consumes them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrieverSet {
    pub names: Vec<String>,
    pub models: Vec<Retriever>,
}

impl RetrieverSet {
    pub fn from_fitted(fitted: Vec<FittedRetriever>) -> RetrieverSet {
        RetrieverSet {
            names: fitted
                .iter()
                .map(|retriever| retriever.name.to_string())
                .collect(),
            models: fitted
                .into_iter()
                .map(|retriever| retriever.model)
                .collect(),
        }
    }

    pub fn get(&self, name: &str) -> &Retriever {
        let position = self.names.iter().position(|candidate| candidate == name);
        &self.models[position.unwrap_or_else(|| panic!("no retriever named {name}"))]
    }
}

/// Fits the named models on `table` and prints how long each took.
pub fn fit_retrievers(
    names: &[&'static str],
    config: &Config,
    n_users: usize,
    n_items: usize,
    table: &Table,
    label: &str,
) -> Vec<FittedRetriever> {
    names
        .iter()
        .map(|&name| {
            let fitted = fit_one(name, config, n_users, n_items, table);
            println!("fitted {name} on {label} in {:.1}s", fitted.fit_seconds);
            fitted
        })
        .collect()
}
