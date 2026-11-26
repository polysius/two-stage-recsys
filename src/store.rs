//! Saving and loading fitted objects as compact binary files.

use std::path::Path;

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub fn save<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = bincode::serde::encode_to_vec(value, bincode::config::standard())?;
    std::fs::write(path, bytes).with_context(|| format!("cannot write {}", path.display()))
}

pub fn load<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let (value, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
        .with_context(|| format!("{} is not a file this version wrote", path.display()))?;
    Ok(value)
}

/// Each model's top items for the test users, so the A/B simulation can replay them.
#[derive(Serialize, Deserialize)]
pub struct TestRecommendations {
    pub users: Vec<u32>,
    /// Items per user in each model's block.
    pub width: usize,
    /// Model name to a row-major `users.len() x width` block of item ids.
    pub by_model: BTreeMap<String, Vec<u32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_saved_value_loads_back_unchanged() {
        let path = std::env::temp_dir().join("recsys_store_round_trip.bin");
        let recommendations = TestRecommendations {
            users: vec![3, 5],
            width: 2,
            by_model: BTreeMap::from([("ranker".to_string(), vec![1, 2, 3, 4])]),
        };
        save(&path, &recommendations).unwrap();
        let loaded: TestRecommendations = load(&path).unwrap();
        assert_eq!(loaded.users, vec![3, 5]);
        assert_eq!(loaded.by_model["ranker"], vec![1, 2, 3, 4]);
        std::fs::remove_file(path).ok();
    }
}
