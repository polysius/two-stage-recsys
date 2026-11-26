//! The interface every recommender shares, and top-k selection that skips seen items.

use std::cmp::Ordering;

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::sparse::BinaryCsr;

/// A model that proposes items for a user.
///
/// `fit` is not part of the trait because each model learns from different inputs. After
/// fitting, every model answers `recommend_user` and some can also score an arbitrary pair.
pub trait Recommender: Sync {
    /// The `k` best items for `user`, best first, never an item listed in `seen` (sorted).
    fn recommend_user(&self, user: u32, k: usize, seen: &[u32]) -> Vec<u32>;

    /// A relevance score for one (user, item) pair, for models that can compute one.
    fn score_pair(&self, _user: u32, _item: u32) -> Option<f32> {
        None
    }

    /// Top-k per user for many users, as one row-major `users.len() x k` block.
    fn recommend(&self, users: &[u32], k: usize, seen: &BinaryCsr) -> Result<Vec<u32>> {
        let rows: Vec<Vec<u32>> = users
            .par_iter()
            .map(|&user| self.recommend_user(user, k, seen.row(user as usize)))
            .collect();
        let mut flat = Vec::with_capacity(users.len() * k);
        for (user, row) in users.iter().zip(rows) {
            if row.len() < k {
                bail!("user {user} has fewer than {k} unseen items");
            }
            flat.extend(row);
        }
        Ok(flat)
    }
}

/// A fixed shuffle of item ids, used to order items whose scores are equal.
///
/// Raw item ids follow the order of the ratings file, which tracks popularity over the whole
/// log, evaluation window included. Breaking ties by id would hand tied items that information.
/// Multiplying by an odd constant is a bijection on u32, so distinct ids never collide.
pub fn tie_break_key(item: u32) -> u32 {
    item.wrapping_mul(0x9E37_79B1)
}

/// Descending by score. Equal scores are ordered by `tie_break_key`, so results are repeatable.
fn better_first(left: &(f32, u32), right: &(f32, u32)) -> Ordering {
    right
        .0
        .total_cmp(&left.0)
        .then(tie_break_key(left.1).cmp(&tie_break_key(right.1)))
}

/// Ids of the `k` highest scores among the items not in `seen` (sorted), best first.
///
/// Returns fewer than `k` ids only when fewer than `k` unseen items exist.
pub fn top_k_unseen(scores: &[f32], seen: &[u32], k: usize) -> Vec<u32> {
    let mut seen_items = seen.iter().copied().peekable();
    let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(scores.len());
    for (item, &score) in scores.iter().enumerate() {
        let item = item as u32;
        while seen_items.next_if(|&seen_item| seen_item < item).is_some() {}
        if seen_items.peek() == Some(&item) {
            continue;
        }
        candidates.push((score, item));
    }
    if candidates.len() > k {
        candidates.select_nth_unstable_by(k - 1, better_first);
        candidates.truncate(k);
    }
    candidates.sort_unstable_by(better_first);
    candidates.into_iter().map(|(_, item)| item).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_skips_seen_items_and_breaks_ties_with_the_shuffled_id() {
        let scores = [5.0, 9.0, 9.0, 1.0, 7.0];
        assert_eq!(top_k_unseen(&scores, &[1], 3), vec![2, 4, 0]);
        let tied_pair = top_k_unseen(&scores, &[], 2);
        assert!(tied_pair.contains(&1) && tied_pair.contains(&2));
        let (first, second) = (tied_pair[0], tied_pair[1]);
        assert!(tie_break_key(first) < tie_break_key(second));
    }

    #[test]
    fn top_k_returns_what_exists_when_too_few_are_unseen() {
        assert_eq!(top_k_unseen(&[1.0, 2.0], &[0], 3), vec![1]);
    }
}
