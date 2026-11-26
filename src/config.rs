//! Experiment settings, read from `configs/default.yaml`.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub data: DataConfig,
    pub split: SplitConfig,
    pub eval: EvalConfig,
    pub als: AlsConfig,
    pub two_tower: TwoTowerConfig,
    pub baselines: BaselineConfig,
    pub ranking: RankingConfig,
    pub experiment: ExperimentConfig,
    pub serving: ServingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataConfig {
    pub category: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SplitConfig {
    pub val_fraction: f64,
    pub test_fraction: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvalConfig {
    pub k_values: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AlsConfig {
    pub factors: usize,
    pub regularization: f64,
    pub iterations: usize,
    pub alpha: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TwoTowerConfig {
    pub embedding_dim: usize,
    pub output_dim: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub temperature: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BaselineConfig {
    pub recent_window_days: i64,
    pub item_cf_neighbors: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RankingConfig {
    pub candidates_per_retriever: usize,
    pub holdout_fraction: f64,
    pub early_stopping_rounds: usize,
    pub lightgbm: BoostingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BoostingConfig {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub num_leaves: usize,
    pub min_child_samples: usize,
    pub subsample: f64,
    pub subsample_freq: usize,
    pub colsample_bytree: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExperimentConfig {
    pub top_k: usize,
    pub alpha: f64,
    pub power: f64,
    pub treatment_share: f64,
    pub n_repeats: usize,
    pub seed: u64,
    pub planning_relative_lifts: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServingConfig {
    pub cache_ttl_seconds: u64,
    pub latency_budget_ms: LatencyBudgets,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LatencyBudgets {
    pub cache_hit: f64,
    pub cache_miss: f64,
    pub cold_start: f64,
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read config {}", path.display()))?;
        serde_yaml::from_str(&text).with_context(|| format!("invalid config {}", path.display()))
    }
}
