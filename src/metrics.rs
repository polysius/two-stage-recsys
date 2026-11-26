//! Ranking metrics.
//!
//! `recommended` is a row-major `n_users x width` block of item ids, best item first.
//! `relevant` holds one sorted list of relevant item ids per user, in the same row order.

use std::collections::BTreeMap;

fn rows(recommended: &[u32], width: usize) -> impl Iterator<Item = &[u32]> {
    recommended.chunks_exact(width)
}

fn is_relevant(truth: &[u32], item: u32) -> bool {
    truth.binary_search(&item).is_ok()
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0usize), |(sum, count), value| {
        (sum + value, count + 1)
    });
    sum / count as f64
}

pub fn recall_at_k(recommended: &[u32], width: usize, relevant: &[Vec<u32>], k: usize) -> f64 {
    mean(rows(recommended, width).zip(relevant).map(|(row, truth)| {
        let hits = row[..k]
            .iter()
            .filter(|&&item| is_relevant(truth, item))
            .count();
        hits as f64 / truth.len() as f64
    }))
}

fn discount(rank_from_zero: usize) -> f64 {
    1.0 / (rank_from_zero as f64 + 2.0).log2()
}

pub fn ndcg_per_user(
    recommended: &[u32],
    width: usize,
    relevant: &[Vec<u32>],
    k: usize,
) -> Vec<f64> {
    rows(recommended, width)
        .zip(relevant)
        .map(|(row, truth)| {
            let gained: f64 = row[..k]
                .iter()
                .enumerate()
                .filter(|(_, &item)| is_relevant(truth, item))
                .map(|(rank, _)| discount(rank))
                .sum();
            let ideal: f64 = (0..truth.len().min(k)).map(discount).sum();
            gained / ideal
        })
        .collect()
}

pub fn ndcg_at_k(recommended: &[u32], width: usize, relevant: &[Vec<u32>], k: usize) -> f64 {
    mean(ndcg_per_user(recommended, width, relevant, k).into_iter())
}

/// 1 when any of the top-k items is relevant to the user, else 0.
pub fn hit_per_user(
    recommended: &[u32],
    width: usize,
    relevant: &[Vec<u32>],
    k: usize,
) -> Vec<bool> {
    rows(recommended, width)
        .zip(relevant)
        .map(|(row, truth)| row[..k].iter().any(|&item| is_relevant(truth, item)))
        .collect()
}

/// Mean reciprocal rank of the first relevant item (0 when none is recommended).
pub fn mrr(recommended: &[u32], width: usize, relevant: &[Vec<u32>]) -> f64 {
    mean(rows(recommended, width).zip(relevant).map(|(row, truth)| {
        row.iter()
            .position(|&item| is_relevant(truth, item))
            .map_or(0.0, |rank| 1.0 / (rank as f64 + 1.0))
    }))
}

/// Share of the catalogue that appears in at least one user's top-k.
pub fn catalogue_coverage(recommended: &[u32], width: usize, n_items: usize, k: usize) -> f64 {
    let mut shown = vec![false; n_items];
    for row in rows(recommended, width) {
        for &item in &row[..k] {
            shown[item as usize] = true;
        }
    }
    shown.iter().filter(|&&was_shown| was_shown).count() as f64 / n_items as f64
}

/// Average train popularity of recommended items, divided by the catalogue average.
///
/// 1.0 means recommendations are as popular as a random item; larger means more popular.
pub fn popularity_bias(
    recommended: &[u32],
    width: usize,
    item_popularity: &[f64],
    k: usize,
) -> f64 {
    let shown = mean(
        rows(recommended, width)
            .flat_map(|row| row[..k].iter())
            .map(|&item| item_popularity[item as usize]),
    );
    shown / mean(item_popularity.iter().copied())
}

/// Metric name to value, ordered by name. Names look like `recall@10` or `mrr`.
pub type Metrics = BTreeMap<String, f64>;

/// Metric names in the order reports list them: `mrr`, then four metrics for each k.
pub fn metric_names(k_values: &[usize]) -> Vec<String> {
    let mut names = vec!["mrr".to_string()];
    for k in k_values {
        for metric in ["recall", "ndcg", "coverage", "pop_bias"] {
            names.push(format!("{metric}@{k}"));
        }
    }
    names
}

pub fn evaluate(
    recommended: &[u32],
    width: usize,
    relevant: &[Vec<u32>],
    item_popularity: &[f64],
    k_values: &[usize],
) -> Metrics {
    let n_items = item_popularity.len();
    let mut results = Metrics::new();
    results.insert("mrr".into(), mrr(recommended, width, relevant));
    for &k in k_values {
        results.insert(
            format!("recall@{k}"),
            recall_at_k(recommended, width, relevant, k),
        );
        results.insert(
            format!("ndcg@{k}"),
            ndcg_at_k(recommended, width, relevant, k),
        );
        results.insert(
            format!("coverage@{k}"),
            catalogue_coverage(recommended, width, n_items, k),
        );
        results.insert(
            format!("pop_bias@{k}"),
            popularity_bias(recommended, width, item_popularity, k),
        );
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() < 1e-9
    }

    #[test]
    fn recall_counts_the_share_of_relevant_items_found() {
        let recommended = [1, 2, 3, 4, 5, 6];
        let relevant = vec![vec![1, 9], vec![7]];
        assert!(close(recall_at_k(&recommended, 3, &relevant, 3), 0.25));
    }

    #[test]
    fn ndcg_is_one_for_a_perfect_ranking_and_lower_when_hit_is_late() {
        let relevant = vec![vec![1]];
        assert!(close(ndcg_at_k(&[1, 2, 3], 3, &relevant, 3), 1.0));
        assert!(close(
            ndcg_at_k(&[2, 3, 1], 3, &relevant, 3),
            1.0 / 4f64.log2()
        ));
    }

    #[test]
    fn mrr_uses_the_first_relevant_rank() {
        let relevant = vec![vec![1, 2], vec![0]];
        assert!(close(mrr(&[9, 1, 2, 5, 6, 7], 3, &relevant), 0.25));
    }

    #[test]
    fn coverage_is_the_share_of_distinct_items_recommended() {
        assert!(close(catalogue_coverage(&[0, 1, 1, 2], 2, 10, 2), 0.3));
    }

    #[test]
    fn popularity_bias_is_one_for_average_items() {
        let popularity = [1.0, 3.0];
        assert!(close(popularity_bias(&[0, 1], 2, &popularity, 2), 1.0));
        assert!(close(popularity_bias(&[1], 1, &popularity, 1), 1.5));
    }

    #[test]
    fn hits_flag_users_with_any_relevant_item() {
        let relevant = vec![vec![2], vec![8]];
        assert_eq!(
            hit_per_user(&[1, 2, 3, 4], 2, &relevant, 2),
            vec![true, false]
        );
    }
}
