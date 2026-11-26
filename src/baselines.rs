//! Non-personalised and neighbourhood baselines.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::recommender::{tie_break_key, top_k_unseen, Recommender};
use crate::sparse::BinaryCsr;
use crate::table::{Table, MS_PER_DAY};

/// Recommends the items with the most interactions, identical for every user.
///
/// With a window it counts only the last `window_days` of the log it is fit on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Popularity {
    pub scores: Vec<f32>,
    /// Item ids, most popular first, equal counts ordered by item id.
    ranking: Vec<u32>,
}

impl Popularity {
    pub fn fit(n_items: usize, train: &Table, window_days: Option<i64>) -> Popularity {
        let cutoff =
            window_days.map(|days| train.max_timestamp() as f64 - days as f64 * MS_PER_DAY);
        let mut scores = vec![0f32; n_items];
        for row in 0..train.len() {
            let in_window = cutoff.is_none_or(|cutoff| train.timestamp[row] as f64 >= cutoff);
            if in_window {
                scores[train.item[row] as usize] += 1.0;
            }
        }
        let mut ranking: Vec<u32> = (0..n_items as u32).collect();
        ranking.sort_by(|&a, &b| {
            scores[b as usize]
                .total_cmp(&scores[a as usize])
                .then(tie_break_key(a).cmp(&tie_break_key(b)))
        });
        Popularity { scores, ranking }
    }
}

impl Recommender for Popularity {
    fn recommend_user(&self, _user: u32, k: usize, seen: &[u32]) -> Vec<u32> {
        self.ranking
            .iter()
            .copied()
            .filter(|item| seen.binary_search(item).is_err())
            .take(k)
            .collect()
    }
}

/// Cosine item-item collaborative filtering keeping the strongest neighbours per item.
///
/// Stored the way scoring reads it: for each item `h` a user has seen, the items that list `h`
/// among their neighbours, with the similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemItemCf {
    n_items: usize,
    history: BinaryCsr,
    followers_indptr: Vec<usize>,
    followers: Vec<(u32, f32)>,
}

impl ItemItemCf {
    pub fn fit(n_users: usize, n_items: usize, n_neighbors: usize, train: &Table) -> ItemItemCf {
        let history = BinaryCsr::user_item(train, n_users, n_items);
        let item_users = history.transpose();
        let norms: Vec<f32> = (0..n_items)
            .map(|item| (item_users.row(item).len() as f32).sqrt().max(1.0))
            .collect();

        let neighbours: Vec<Vec<(u32, f32)>> = (0..n_items)
            .into_par_iter()
            .map_init(
                || vec![0u32; n_items],
                |counts, item| {
                    strongest_neighbours(item, n_neighbors, &history, &item_users, &norms, counts)
                },
            )
            .collect();

        // Reverse the lists: item `i` has neighbour `h` with similarity `s` becomes (h -> i, s).
        let mut followers_by_item: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_items];
        for (item, list) in neighbours.iter().enumerate() {
            for &(neighbour, similarity) in list {
                followers_by_item[neighbour as usize].push((item as u32, similarity));
            }
        }
        let mut followers_indptr = vec![0];
        let mut followers = Vec::new();
        for list in followers_by_item {
            followers.extend(list);
            followers_indptr.push(followers.len());
        }
        ItemItemCf {
            n_items,
            history,
            followers_indptr,
            followers,
        }
    }
}

/// The `n_neighbors` items that co-occur most with `item`, by cosine similarity.
fn strongest_neighbours(
    item: usize,
    n_neighbors: usize,
    history: &BinaryCsr,
    item_users: &BinaryCsr,
    norms: &[f32],
    counts: &mut [u32],
) -> Vec<(u32, f32)> {
    let mut touched: Vec<u32> = Vec::new();
    for &user in item_users.row(item) {
        for &other in history.row(user as usize) {
            if other as usize == item {
                continue; // an item is not its own neighbour
            }
            if counts[other as usize] == 0 {
                touched.push(other);
            }
            counts[other as usize] += 1;
        }
    }
    let mut scored: Vec<(u32, f32)> = touched
        .iter()
        .map(|&other| {
            let similarity = counts[other as usize] as f32 / (norms[item] * norms[other as usize]);
            counts[other as usize] = 0;
            (other, similarity)
        })
        .collect();
    let strongest_first = |a: &(u32, f32), b: &(u32, f32)| {
        b.1.total_cmp(&a.1)
            .then(tie_break_key(a.0).cmp(&tie_break_key(b.0)))
    };
    if scored.len() > n_neighbors {
        scored.select_nth_unstable_by(n_neighbors - 1, strongest_first);
        scored.truncate(n_neighbors);
    }
    scored
}

impl Recommender for ItemItemCf {
    fn recommend_user(&self, user: u32, k: usize, seen: &[u32]) -> Vec<u32> {
        let mut scores = vec![0f32; self.n_items];
        for &held in self.history.row(user as usize) {
            let range =
                self.followers_indptr[held as usize]..self.followers_indptr[held as usize + 1];
            for &(item, similarity) in &self.followers[range] {
                scores[item as usize] += similarity;
            }
        }
        top_k_unseen(&scores, seen, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(pairs: &[(u32, u32)], times: &[i64]) -> Table {
        Table {
            user: pairs.iter().map(|p| p.0).collect(),
            item: pairs.iter().map(|p| p.1).collect(),
            rating: vec![5.0; pairs.len()],
            timestamp: times.to_vec(),
        }
    }

    #[test]
    fn popularity_ranks_by_count_and_skips_seen_items() {
        let train = table(&[(0, 2), (1, 2), (2, 2), (0, 1), (1, 1), (2, 0)], &[0; 6]);
        let model = Popularity::fit(4, &train, None);
        assert_eq!(model.recommend_user(0, 2, &[]), vec![2, 1]);
        assert_eq!(model.recommend_user(0, 2, &[2]), vec![1, 0]);
    }

    #[test]
    fn recent_popularity_ignores_old_interactions() {
        let day = MS_PER_DAY as i64;
        let train = table(&[(0, 0), (1, 0), (2, 0), (0, 1)], &[0, 0, 0, 100 * day]);
        let model = Popularity::fit(2, &train, Some(90));
        assert_eq!(model.recommend_user(0, 1, &[]), vec![1]);
    }

    #[test]
    fn item_cf_recommends_items_that_co_occur_with_the_history() {
        // Users 0 and 1 both hold items 0 and 1; user 2 holds only item 0.
        let train = table(&[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 2)], &[0; 6]);
        let model = ItemItemCf::fit(4, 3, 5, &train);
        let seen = BinaryCsr::user_item(&train, 4, 3);
        assert_eq!(model.recommend_user(2, 1, seen.row(2)), vec![1]);
    }
}
