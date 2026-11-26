//! LambdaMART ranker over retrieved candidates.

use std::collections::HashSet;

use anyhow::{bail, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::config::{BoostingConfig, RankingConfig};
use crate::lambdamart::{self, Dataset, Model, Params};
use crate::ranking::features::FeatureMatrix;

/// A fitted model together with the feature columns it reads.
#[derive(Serialize, Deserialize)]
pub struct TrainedRanker {
    pub model: Model,
    /// Positions of the model's inputs in the full feature matrix.
    pub columns: Vec<usize>,
    pub column_names: Vec<String>,
}

fn boosting_params(config: &BoostingConfig, seed: u64) -> Params {
    Params {
        n_estimators: config.n_estimators,
        learning_rate: config.learning_rate,
        num_leaves: config.num_leaves,
        min_child_samples: config.min_child_samples,
        subsample: config.subsample,
        subsample_freq: config.subsample_freq,
        colsample_bytree: config.colsample_bytree,
        max_bin: 255,
        truncation_level: 30,
        sigmoid: 1.0,
        min_sum_hessian: 1e-3,
        seed,
    }
}

/// Row ranges of each group, for a matrix whose rows are grouped by user.
fn group_row_ranges(group_sizes: &[usize]) -> Vec<std::ops::Range<usize>> {
    let mut start = 0;
    group_sizes
        .iter()
        .map(|&size| {
            let range = start..start + size;
            start += size;
            range
        })
        .collect()
}

/// Fits on most users and stops early on a held-out group of users.
///
/// The split is by user, never by row, because LambdaMART compares items inside one user's list.
pub fn train_ranker(
    features: &FeatureMatrix,
    group_sizes: &[usize],
    labels: &[u8],
    columns: &[usize],
    ranking: &RankingConfig,
    seed: u64,
) -> TrainedRanker {
    let (config, early_stopping_rounds) = (&ranking.lightgbm, ranking.early_stopping_rounds);
    let ranges = group_row_ranges(group_sizes);
    let mut group_order: Vec<usize> = (0..ranges.len()).collect();
    group_order.shuffle(&mut ChaCha12Rng::seed_from_u64(seed));
    let n_holdout = (ranges.len() as f64 * ranking.holdout_fraction) as usize;
    let holdout: HashSet<usize> = group_order[..n_holdout].iter().copied().collect();

    let gather = |keep_holdout: bool| {
        let groups: Vec<usize> = (0..ranges.len())
            .filter(|group| holdout.contains(group) == keep_holdout)
            .collect();
        let rows: Vec<usize> = groups
            .iter()
            .flat_map(|&group| ranges[group].clone())
            .collect();
        let values = features.select(rows.iter().copied(), columns);
        let part_labels: Vec<u8> = rows.iter().map(|&row| labels[row]).collect();
        let sizes: Vec<usize> = groups.iter().map(|&group| group_sizes[group]).collect();
        (values, part_labels, sizes)
    };
    let (fit_values, fit_labels, fit_sizes) = gather(false);
    let (hold_values, hold_labels, hold_sizes) = gather(true);

    let model = lambdamart::train(
        &Dataset {
            features: &fit_values,
            n_features: columns.len(),
            labels: &fit_labels,
            group_sizes: &fit_sizes,
        },
        &Dataset {
            features: &hold_values,
            n_features: columns.len(),
            labels: &hold_labels,
            group_sizes: &hold_sizes,
        },
        &boosting_params(config, seed),
        early_stopping_rounds,
        10,
    );
    TrainedRanker {
        model,
        columns: columns.to_vec(),
        column_names: columns
            .iter()
            .map(|&col| features.names[col].clone())
            .collect(),
    }
}

impl TrainedRanker {
    pub fn predict(&self, features: &FeatureMatrix) -> Vec<f32> {
        let values = features.select(0..features.n_rows, &self.columns);
        self.model.predict(&values, self.columns.len())
    }

    pub fn best_iteration(&self) -> usize {
        self.model.best_iteration()
    }

    /// Share of total split gain per feature, in percent, largest first.
    pub fn importance_percent(&self) -> Vec<(String, f64)> {
        let gains = self.model.feature_importance_gain();
        let total: f64 = gains.iter().sum();
        let mut named: Vec<(String, f64)> = self
            .column_names
            .iter()
            .cloned()
            .zip(gains.iter().map(|gain| 100.0 * gain / total))
            .collect();
        named.sort_by(|a, b| b.1.total_cmp(&a.1));
        named
    }
}

/// Item ids of the `k` highest-scoring candidates per user, users in the frame's order.
///
/// Equal scores keep their row order, which the candidate shuffle made random.
pub fn top_k_per_user(
    group_sizes: &[usize],
    items: &[u32],
    scores: &[f32],
    k: usize,
) -> Result<Vec<u32>> {
    let mut top = Vec::with_capacity(group_sizes.len() * k);
    for range in group_row_ranges(group_sizes) {
        if range.len() < k {
            bail!("some users have fewer than {k} candidates");
        }
        let mut rows: Vec<usize> = range.collect();
        rows.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        top.extend(rows[..k].iter().map(|&row| items[row]));
    }
    Ok(top)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_orders_by_score_and_keeps_row_order_for_ties() {
        let items = [10, 11, 12, 20, 21];
        let scores = [0.5, 0.9, 0.5, 1.0, 1.0];
        let top = top_k_per_user(&[3, 2], &items, &scores, 2).unwrap();
        assert_eq!(top, vec![11, 10, 20, 21]);
    }

    #[test]
    fn top_k_rejects_short_candidate_lists() {
        assert!(top_k_per_user(&[1], &[7], &[0.1], 2).is_err());
    }
}
