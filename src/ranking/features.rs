//! Features for candidate (user, item) pairs, computed from interaction history only.
//!
//! `history` is every interaction the system could have seen when making the recommendation:
//! train for the validation window, train plus validation for the test window. Nothing from the
//! window being predicted is used, so a feature cannot leak the label.

use serde::{Deserialize, Serialize};

use crate::ranking::candidates::CandidateFrame;
use crate::table::{Table, MS_PER_DAY};

const RECENT_WINDOW_DAYS: f64 = 90.0;

/// A column belongs to the group named by its prefix. The ablation switches groups on and off.
pub const GROUPS: [(&str, &[&str]); 4] = [
    ("retriever", &["rank_", "score_"]),
    ("user", &["user_"]),
    ("item", &["item_"]),
    ("pair", &["pair_"]),
];

/// Row-major feature values with their column names.
pub struct FeatureMatrix {
    pub names: Vec<String>,
    pub values: Vec<f32>,
    pub n_rows: usize,
}

impl FeatureMatrix {
    pub fn n_cols(&self) -> usize {
        self.names.len()
    }

    /// Indices of the columns that belong to `groups` (all groups when `None`).
    pub fn columns_in_groups(&self, groups: Option<&[&str]>) -> Vec<usize> {
        let wanted: Vec<&str> = match groups {
            Some(groups) => groups.to_vec(),
            None => GROUPS.iter().map(|(name, _)| *name).collect(),
        };
        let prefixes: Vec<&str> = GROUPS
            .iter()
            .filter(|(name, _)| wanted.contains(name))
            .flat_map(|(_, prefixes)| prefixes.iter().copied())
            .collect();
        (0..self.n_cols())
            .filter(|&col| {
                prefixes
                    .iter()
                    .any(|prefix| self.names[col].starts_with(prefix))
            })
            .collect()
    }

    /// The listed columns of the listed rows, as a new row-major block.
    pub fn select(&self, rows: impl IntoIterator<Item = usize>, columns: &[usize]) -> Vec<f32> {
        let width = self.n_cols();
        let mut selected = Vec::new();
        for row in rows {
            let values = &self.values[row * width..][..width];
            selected.extend(columns.iter().map(|&col| values[col]));
        }
        selected
    }
}

/// Per-item statistics measured on a history. NaN for items the history never saw.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemStats {
    pub log_count: Vec<f32>,
    pub log_recent_count: Vec<f32>,
    pub mean_rating: Vec<f32>,
    pub age_days: Vec<f32>,
    pub days_since_last: Vec<f32>,
}

/// Per-user statistics measured on a history. NaN for users the history never saw.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserStats {
    pub log_count: Vec<f32>,
    pub mean_rating: Vec<f32>,
    pub days_since_last: Vec<f32>,
    pub mean_log_item_count: Vec<f32>,
}

impl UserStats {
    /// A user is known when the history holds at least one interaction of theirs.
    pub fn is_known(&self, user: u32) -> bool {
        self.log_count
            .get(user as usize)
            .is_some_and(|count| !count.is_nan())
    }
}

/// Running totals for one id while scanning the history.
#[derive(Clone, Copy)]
struct Totals {
    count: f64,
    recent_count: f64,
    rating_sum: f64,
    first: i64,
    last: i64,
}

impl Totals {
    const EMPTY: Totals = Totals {
        count: 0.0,
        recent_count: 0.0,
        rating_sum: 0.0,
        first: i64::MAX,
        last: i64::MIN,
    };

    fn add(&mut self, rating: f32, timestamp: i64, is_recent: bool) {
        self.count += 1.0;
        self.recent_count += f64::from(u8::from(is_recent));
        self.rating_sum += f64::from(rating);
        self.first = self.first.min(timestamp);
        self.last = self.last.max(timestamp);
    }
}

fn totals_by(history: &Table, ids: &[u32], n_ids: usize) -> Vec<Totals> {
    let recent_start = history.max_timestamp() as f64 - RECENT_WINDOW_DAYS * MS_PER_DAY;
    let mut totals = vec![Totals::EMPTY; n_ids];
    for row in 0..history.len() {
        let is_recent = history.timestamp[row] as f64 >= recent_start;
        totals[ids[row] as usize].add(history.rating[row], history.timestamp[row], is_recent);
    }
    totals
}

fn days_before(reference: i64, timestamp: i64) -> f32 {
    ((reference - timestamp) as f64 / MS_PER_DAY) as f32
}

/// One entry per item with its popularity, rating and recency, measured on `history`.
pub fn item_statistics(history: &Table, n_items: usize) -> ItemStats {
    let reference = history.max_timestamp();
    let totals = totals_by(history, &history.item, n_items);
    let per_item = |value: &dyn Fn(&Totals) -> f32| -> Vec<f32> {
        totals
            .iter()
            .map(|t| if t.count == 0.0 { f32::NAN } else { value(t) })
            .collect()
    };
    ItemStats {
        log_count: per_item(&|t| t.count.ln_1p() as f32),
        log_recent_count: per_item(&|t| t.recent_count.ln_1p() as f32),
        mean_rating: per_item(&|t| (t.rating_sum / t.count) as f32),
        age_days: per_item(&|t| days_before(reference, t.first)),
        days_since_last: per_item(&|t| days_before(reference, t.last)),
    }
}

/// One entry per user with their activity level, ratings and taste for popular items.
pub fn user_statistics(history: &Table, n_users: usize, n_items: usize) -> UserStats {
    let reference = history.max_timestamp();
    let totals = totals_by(history, &history.user, n_users);

    let mut item_counts = vec![0f64; n_items];
    for &item in &history.item {
        item_counts[item as usize] += 1.0;
    }
    let mut log_popularity_sum = vec![0f64; n_users];
    for (&user, &item) in history.user.iter().zip(&history.item) {
        log_popularity_sum[user as usize] += item_counts[item as usize].ln_1p();
    }

    let per_user = |value: &dyn Fn(usize, &Totals) -> f32| -> Vec<f32> {
        totals
            .iter()
            .enumerate()
            .map(|(user, t)| {
                if t.count == 0.0 {
                    f32::NAN
                } else {
                    value(user, t)
                }
            })
            .collect()
    };
    UserStats {
        log_count: per_user(&|_, t| t.count.ln_1p() as f32),
        mean_rating: per_user(&|_, t| (t.rating_sum / t.count) as f32),
        days_since_last: per_user(&|_, t| days_before(reference, t.last)),
        mean_log_item_count: per_user(&|user, t| (log_popularity_sum[user] / t.count) as f32),
    }
}

/// Retriever columns, then user, item and pair columns, one row per candidate.
pub fn build_features(
    frame: &CandidateFrame,
    retriever_names: &[String],
    scorer_names: &[&str],
    users: &UserStats,
    items: &ItemStats,
) -> FeatureMatrix {
    let mut names: Vec<String> = retriever_names
        .iter()
        .map(|name| format!("rank_{name}"))
        .collect();
    names.extend(scorer_names.iter().map(|name| format!("score_{name}")));
    names.extend(
        [
            "user_log_count",
            "user_mean_rating",
            "user_days_since_last",
            "user_mean_log_item_count",
            "item_log_count",
            "item_log_recent_count",
            "item_mean_rating",
            "item_age_days",
            "item_days_since_last",
            "pair_popularity_gap",
            "pair_rating_gap",
        ]
        .map(String::from),
    );

    let mut values = Vec::with_capacity(frame.len() * names.len());
    for row in 0..frame.len() {
        let (user, item) = (frame.users[row] as usize, frame.items[row] as usize);
        values.extend_from_slice(&frame.ranks[row * frame.n_retrievers..][..frame.n_retrievers]);
        values.extend_from_slice(&frame.scores[row * frame.n_scorers..][..frame.n_scorers]);
        values.extend_from_slice(&[
            users.log_count[user],
            users.mean_rating[user],
            users.days_since_last[user],
            users.mean_log_item_count[user],
            items.log_count[item],
            items.log_recent_count[item],
            items.mean_rating[item],
            items.age_days[item],
            items.days_since_last[item],
            // Pair features compare the item with what this user usually consumes.
            items.log_count[item] - users.mean_log_item_count[user],
            items.mean_rating[item] - users.mean_rating[user],
        ]);
    }
    FeatureMatrix {
        names,
        values,
        n_rows: frame.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history() -> Table {
        let day = MS_PER_DAY as i64;
        Table {
            user: vec![0, 0, 1],
            item: vec![0, 1, 0],
            rating: vec![4.0, 2.0, 5.0],
            timestamp: vec![0, 200 * day, 200 * day],
        }
    }

    #[test]
    fn item_statistics_measure_popularity_rating_and_recency() {
        let stats = item_statistics(&history(), 3);
        assert!((stats.log_count[0] - 2f32.ln_1p()).abs() < 1e-6);
        assert!((stats.mean_rating[0] - 4.5).abs() < 1e-6);
        assert!((stats.age_days[0] - 200.0).abs() < 1e-4);
        assert!((stats.log_recent_count[0] - 1f32.ln_1p()).abs() < 1e-6);
        assert!(
            stats.log_count[2].is_nan(),
            "an unseen item has no statistics"
        );
    }

    #[test]
    fn user_statistics_flag_known_users() {
        let stats = user_statistics(&history(), 3, 3);
        assert!(stats.is_known(0) && stats.is_known(1));
        assert!(!stats.is_known(2) && !stats.is_known(9));
        assert!((stats.mean_rating[0] - 3.0).abs() < 1e-6);
        assert!((stats.days_since_last[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn group_selection_follows_column_prefixes() {
        let matrix = FeatureMatrix {
            names: [
                "rank_als",
                "score_als",
                "user_log_count",
                "item_log_count",
                "pair_rating_gap",
            ]
            .map(String::from)
            .to_vec(),
            values: vec![],
            n_rows: 0,
        };
        assert_eq!(matrix.columns_in_groups(None), vec![0, 1, 2, 3, 4]);
        assert_eq!(matrix.columns_in_groups(Some(&["retriever"])), vec![0, 1]);
        assert_eq!(
            matrix.columns_in_groups(Some(&["user", "pair"])),
            vec![2, 4]
        );
    }
}
