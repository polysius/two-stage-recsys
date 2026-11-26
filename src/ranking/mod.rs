//! Second stage: build features for retrieved candidates and rank them with LambdaMART.

pub mod candidates;
pub mod features;
pub mod ranker;

use anyhow::Result;

use crate::evaluation::relevant_items_by_user;
use crate::ranking::candidates::{
    generate_candidates, label_candidates, scorer_names, CandidateFrame,
};
use crate::ranking::features::{build_features, item_statistics, user_statistics, FeatureMatrix};
use crate::registry::RetrieverSet;
use crate::sparse::BinaryCsr;
use crate::table::Table;

/// Candidates, features and labels for every user with interactions in the evaluation table.
pub struct Stage {
    pub frame: CandidateFrame,
    pub features: FeatureMatrix,
    pub labels: Vec<u8>,
    pub users: Vec<u32>,
    /// Share of relevant (user, item) pairs that appear among the candidates.
    pub candidate_recall: f64,
    pub candidates_per_user: f64,
}

pub fn build_stage(
    retrievers: &RetrieverSet,
    history: &Table,
    eval_table: &Table,
    n_users: usize,
    n_items: usize,
    per_retriever: usize,
) -> Result<Stage> {
    let (users, relevant) = relevant_items_by_user(eval_table);
    let seen = BinaryCsr::user_item(history, n_users, n_items);
    let frame = generate_candidates(retrievers, &users, &seen, per_retriever, 0)?;
    let features = build_features(
        &frame,
        &retrievers.names,
        &scorer_names(retrievers),
        &user_statistics(history, n_users, n_items),
        &item_statistics(history, n_items),
    );
    let labels = label_candidates(&frame, eval_table);

    let n_relevant: usize = relevant.iter().map(Vec::len).sum();
    let hits: usize = labels.iter().map(|&label| usize::from(label)).sum();
    let stage = Stage {
        candidate_recall: hits as f64 / n_relevant as f64,
        candidates_per_user: frame.len() as f64 / users.len() as f64,
        frame,
        features,
        labels,
        users,
    };
    println!(
        "{} users, {} candidates, candidate recall {:.4}",
        stage.users.len(),
        stage.frame.len(),
        stage.candidate_recall
    );
    Ok(stage)
}
