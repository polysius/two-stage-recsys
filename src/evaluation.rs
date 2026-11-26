//! Shared helpers to turn a split into evaluation inputs and score a recommender.

use std::collections::BTreeMap;

use anyhow::Result;

use crate::metrics::{evaluate, Metrics};
use crate::recommender::Recommender;
use crate::sparse::BinaryCsr;
use crate::table::Table;

/// Users with at least one interaction in `eval_table` (ascending) and their sorted item ids.
pub fn relevant_items_by_user(eval_table: &Table) -> (Vec<u32>, Vec<Vec<u32>>) {
    let mut by_user: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (&user, &item) in eval_table.user.iter().zip(&eval_table.item) {
        by_user.entry(user).or_default().push(item);
    }
    let users = by_user.keys().copied().collect();
    let relevant = by_user
        .into_values()
        .map(|mut items| {
            items.sort_unstable();
            items.dedup();
            items
        })
        .collect();
    (users, relevant)
}

/// Number of interactions per item in `history`, as floats for the popularity metric.
pub fn item_popularity(history: &Table, n_items: usize) -> Vec<f64> {
    let mut counts = vec![0f64; n_items];
    for &item in &history.item {
        counts[item as usize] += 1.0;
    }
    counts
}

/// Evaluates `recommender` on `eval_table`, masking everything in `history`.
///
/// For validation, history is train. For test, history is train plus validation, because a
/// user could have seen validation items before the test period starts.
pub fn score_recommender(
    recommender: &dyn Recommender,
    history: &Table,
    eval_table: &Table,
    n_users: usize,
    n_items: usize,
    k_values: &[usize],
) -> Result<Metrics> {
    let seen = BinaryCsr::user_item(history, n_users, n_items);
    let (users, relevant) = relevant_items_by_user(eval_table);
    let width = *k_values.iter().max().expect("at least one k");
    let recommended = recommender.recommend(&users, width, &seen)?;
    let popularity = item_popularity(history, n_items);
    Ok(evaluate(
        &recommended,
        width,
        &relevant,
        &popularity,
        k_values,
    ))
}
