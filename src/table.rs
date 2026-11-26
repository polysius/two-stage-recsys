//! Interaction log, dense ids and the time-based split.
//!
//! Every interaction before the validation cutoff goes to train, every interaction before the
//! test cutoff goes to validation, and the rest goes to test. No future interaction can leak
//! into an earlier split, which `assert_no_time_leakage` checks.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const MS_PER_DAY: f64 = 24.0 * 60.0 * 60.0 * 1000.0;

/// Interactions stored column by column. `timestamp` is in milliseconds since the epoch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Table {
    pub user: Vec<u32>,
    pub item: Vec<u32>,
    pub rating: Vec<f32>,
    pub timestamp: Vec<i64>,
}

impl Table {
    pub fn len(&self) -> usize {
        self.user.len()
    }

    pub fn is_empty(&self) -> bool {
        self.user.is_empty()
    }

    fn push_row_of(&mut self, other: &Table, row: usize) {
        self.user.push(other.user[row]);
        self.item.push(other.item[row]);
        self.rating.push(other.rating[row]);
        self.timestamp.push(other.timestamp[row]);
    }

    /// Rows for which `keep` returns true, in their original order.
    pub fn filter(&self, keep: impl Fn(usize) -> bool) -> Table {
        let mut kept = Table::default();
        for row in (0..self.len()).filter(|&row| keep(row)) {
            kept.push_row_of(self, row);
        }
        kept
    }

    pub fn concat(first: &Table, second: &Table) -> Table {
        let mut joined = first.clone();
        for row in 0..second.len() {
            joined.push_row_of(second, row);
        }
        joined
    }

    pub fn max_timestamp(&self) -> i64 {
        self.timestamp.iter().copied().max().unwrap_or(i64::MIN)
    }

    pub fn min_timestamp(&self) -> i64 {
        self.timestamp.iter().copied().min().unwrap_or(i64::MAX)
    }
}

pub struct Interactions {
    pub table: Table,
    pub n_users: usize,
    pub n_items: usize,
}

/// Assigns ids 0..n-1 in order of first appearance, like `pandas.factorize`.
#[derive(Default)]
struct Factorizer {
    ids: HashMap<String, u32>,
}

impl Factorizer {
    fn id_of(&mut self, key: &str) -> u32 {
        if let Some(&id) = self.ids.get(key) {
            return id;
        }
        let id = self.ids.len() as u32;
        self.ids.insert(key.to_string(), id);
        id
    }
}

/// Reads the ratings CSV and returns interactions sorted by time (ties keep file order).
pub fn load_interactions(csv_path: &Path) -> Result<Interactions> {
    let mut reader = csv::Reader::from_path(csv_path)
        .with_context(|| format!("cannot open {}", csv_path.display()))?;
    let headers = reader.headers()?.clone();
    let column = |name: &str| {
        headers
            .iter()
            .position(|header| header == name)
            .with_context(|| format!("column {name} missing from {}", csv_path.display()))
    };
    let (user_col, item_col) = (column("user_id")?, column("parent_asin")?);
    let (rating_col, time_col) = (column("rating")?, column("timestamp")?);

    let (mut users, mut items) = (Factorizer::default(), Factorizer::default());
    let mut raw = Table::default();
    for record in reader.records() {
        let record = record?;
        raw.user.push(users.id_of(&record[user_col]));
        raw.item.push(items.id_of(&record[item_col]));
        raw.rating.push(record[rating_col].parse()?);
        raw.timestamp.push(record[time_col].parse()?);
    }

    let mut order: Vec<usize> = (0..raw.len()).collect();
    order.sort_by_key(|&row| raw.timestamp[row]); // stable
    let mut table = Table::default();
    for row in order {
        table.push_row_of(&raw, row);
    }
    Ok(Interactions {
        table,
        n_users: users.ids.len(),
        n_items: items.ids.len(),
    })
}

pub struct Split {
    pub train: Table,
    pub val: Table,
    pub test: Table,
    /// Eval rows removed because the user or the item is not in train.
    pub dropped_unseen_share: f64,
    pub unknown_user_share: f64,
    pub unknown_item_share: f64,
}

/// Quantile with linear interpolation between order statistics, as `pandas.Series.quantile`.
fn quantile(sorted: &[i64], q: f64) -> f64 {
    let position = q * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] as f64 + (sorted[upper] as f64 - sorted[lower] as f64) * fraction
}

/// Splits by timestamp quantile.
///
/// Evaluation rows are kept only for users and items that appear in train. A collaborative
/// model cannot score an unseen id, so those rows would measure cold start, not ranking.
/// The share removed is reported so the filter is visible.
pub fn time_split(table: &Table, val_fraction: f64, test_fraction: f64) -> Split {
    let mut sorted = table.timestamp.clone();
    sorted.sort_unstable();
    let val_start = quantile(&sorted, 1.0 - val_fraction - test_fraction);
    let test_start = quantile(&sorted, 1.0 - test_fraction);
    let time_of = |row: usize| table.timestamp[row] as f64;

    let train = table.filter(|row| time_of(row) < val_start);
    let eval = table.filter(|row| time_of(row) >= val_start);

    let known_users: HashSet<u32> = train.user.iter().copied().collect();
    let known_items: HashSet<u32> = train.item.iter().copied().collect();
    let user_known = |row: usize| known_users.contains(&eval.user[row]);
    let item_known = |row: usize| known_items.contains(&eval.item[row]);

    let n_eval = eval.len() as f64;
    let share = |count: usize| count as f64 / n_eval;
    let unknown_users = (0..eval.len()).filter(|&row| !user_known(row)).count();
    let unknown_items = (0..eval.len()).filter(|&row| !item_known(row)).count();
    let dropped = (0..eval.len())
        .filter(|&row| !(user_known(row) && item_known(row)))
        .count();

    let known = |row: usize| user_known(row) && item_known(row);
    let is_val = |row: usize| (eval.timestamp[row] as f64) < test_start;
    Split {
        val: eval.filter(|row| known(row) && is_val(row)),
        test: eval.filter(|row| known(row) && !is_val(row)),
        train,
        dropped_unseen_share: share(dropped),
        unknown_user_share: share(unknown_users),
        unknown_item_share: share(unknown_items),
    }
}

/// Fails if any split contains an interaction earlier than the previous split's latest.
pub fn assert_no_time_leakage(split: &Split) -> Result<()> {
    for (name, part) in [
        ("train", &split.train),
        ("val", &split.val),
        ("test", &split.test),
    ] {
        if part.is_empty() {
            bail!("{name} split is empty, so ordering cannot be checked");
        }
    }
    if split.train.max_timestamp() > split.val.min_timestamp() {
        bail!("validation contains interactions older than the end of train");
    }
    if split.val.max_timestamp() > split.test.min_timestamp() {
        bail!("test contains interactions older than the end of validation");
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A small log: 10 users x 10 items, interaction `n` happens at time `n`.
    pub fn synthetic_table() -> Table {
        let mut table = Table::default();
        for n in 0..100u32 {
            table.user.push(n % 10);
            table.item.push((n * 7 + n / 10) % 10);
            table.rating.push(3.0 + (n % 3) as f32);
            table.timestamp.push(n as i64);
        }
        table
    }

    #[test]
    fn split_is_chronological() {
        let split = time_split(&synthetic_table(), 0.1, 0.1);
        assert_no_time_leakage(&split).unwrap();
    }

    #[test]
    fn eval_rows_only_use_known_users_and_items() {
        let split = time_split(&synthetic_table(), 0.1, 0.1);
        let users: HashSet<u32> = split.train.user.iter().copied().collect();
        let items: HashSet<u32> = split.train.item.iter().copied().collect();
        for part in [&split.val, &split.test] {
            assert!(part.user.iter().all(|user| users.contains(user)));
            assert!(part.item.iter().all(|item| items.contains(item)));
        }
    }

    #[test]
    fn leakage_check_rejects_an_empty_split() {
        let mut split = time_split(&synthetic_table(), 0.1, 0.1);
        split.test = Table::default();
        let error = assert_no_time_leakage(&split).unwrap_err().to_string();
        assert!(error.contains("empty"));
    }

    #[test]
    fn leakage_check_catches_a_shuffled_split() {
        let split = time_split(&synthetic_table(), 0.1, 0.1);
        let leaky = Split {
            train: split.test,
            test: split.train,
            ..split
        };
        assert!(assert_no_time_leakage(&leaky).is_err());
    }

    #[test]
    fn quantile_interpolates_like_pandas() {
        assert_eq!(quantile(&[0, 10, 20, 30], 0.5), 15.0);
        assert_eq!(quantile(&[0, 10, 20, 30], 1.0), 30.0);
    }
}
