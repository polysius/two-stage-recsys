//! Merge the candidates proposed by several retrievers into one list per user.

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use crate::recommender::Recommender;
use crate::registry::{Retriever, RetrieverSet};
use crate::sparse::BinaryCsr;
use crate::table::Table;

/// One user's distinct candidate items and what each retriever said about them.
///
/// `ranks` holds, for every item, one value per retriever: 1 is best, NaN when that retriever
/// did not propose the item. `scores` holds one value per retriever that can score arbitrary
/// pairs. A pair one retriever proposed can still be scored by another, so scores need no fill.
pub struct UserCandidates {
    pub items: Vec<u32>,
    pub ranks: Vec<f32>,
    pub scores: Vec<f32>,
}

/// Names of the retrievers that provide a score column, in retriever order.
pub fn scorer_names(retrievers: &RetrieverSet) -> Vec<&str> {
    retrievers
        .names
        .iter()
        .zip(&retrievers.models)
        .filter(|(_, model)| model.can_score_pairs())
        .map(|(name, _)| name.as_str())
        .collect()
}

pub fn user_candidates(
    retrievers: &RetrieverSet,
    user: u32,
    seen: &[u32],
    per_retriever: usize,
    seed: u64,
) -> UserCandidates {
    let n_retrievers = retrievers.models.len();
    let mut items: Vec<u32> = Vec::new();
    let mut position_of: HashMap<u32, usize> = HashMap::new();
    let mut ranks: Vec<f32> = Vec::new();

    for (which, model) in retrievers.models.iter().enumerate() {
        for (rank, item) in model
            .recommend_user(user, per_retriever, seen)
            .into_iter()
            .enumerate()
        {
            let position = *position_of.entry(item).or_insert_with(|| {
                items.push(item);
                ranks.extend(std::iter::repeat_n(f32::NAN, n_retrievers));
                items.len() - 1
            });
            ranks[position * n_retrievers + which] = rank as f32 + 1.0;
        }
    }

    // Order candidates randomly inside each user. Tied scores are then broken by chance, not by
    // item id, and item ids track popularity over the whole log, including the evaluation window.
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.shuffle(&mut ChaCha12Rng::seed_from_u64(
        seed ^ (u64::from(user) << 20),
    ));
    let shuffled_items: Vec<u32> = order.iter().map(|&position| items[position]).collect();
    let shuffled_ranks: Vec<f32> = order
        .iter()
        .flat_map(|&position| {
            ranks[position * n_retrievers..][..n_retrievers]
                .iter()
                .copied()
        })
        .collect();

    let scorers: Vec<&Retriever> = retrievers
        .models
        .iter()
        .filter(|model| model.can_score_pairs())
        .collect();
    let scores = shuffled_items
        .iter()
        .flat_map(|&item| {
            scorers
                .iter()
                .map(move |model| model.score_pair(user, item).unwrap())
        })
        .collect();
    UserCandidates {
        items: shuffled_items,
        ranks: shuffled_ranks,
        scores,
    }
}

/// Candidates for many users, laid out user after user in the order of `users`.
pub struct CandidateFrame {
    pub users: Vec<u32>,
    pub items: Vec<u32>,
    pub n_retrievers: usize,
    pub n_scorers: usize,
    /// Row-major `rows x n_retrievers`.
    pub ranks: Vec<f32>,
    /// Row-major `rows x n_scorers`.
    pub scores: Vec<f32>,
    /// Candidate count for each entry of `query_users`.
    pub group_sizes: Vec<usize>,
    pub query_users: Vec<u32>,
}

impl CandidateFrame {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

pub fn generate_candidates(
    retrievers: &RetrieverSet,
    users: &[u32],
    seen: &BinaryCsr,
    per_retriever: usize,
    seed: u64,
) -> Result<CandidateFrame> {
    let per_user: Vec<UserCandidates> = users
        .par_iter()
        .map(|&user| {
            user_candidates(
                retrievers,
                user,
                seen.row(user as usize),
                per_retriever,
                seed,
            )
        })
        .collect();

    let mut frame = CandidateFrame {
        users: Vec::new(),
        items: Vec::new(),
        n_retrievers: retrievers.models.len(),
        n_scorers: scorer_names(retrievers).len(),
        ranks: Vec::new(),
        scores: Vec::new(),
        group_sizes: Vec::with_capacity(users.len()),
        query_users: users.to_vec(),
    };
    for (&user, candidates) in users.iter().zip(per_user) {
        frame
            .users
            .extend(std::iter::repeat_n(user, candidates.items.len()));
        frame.group_sizes.push(candidates.items.len());
        frame.items.extend(candidates.items);
        frame.ranks.extend(candidates.ranks);
        frame.scores.extend(candidates.scores);
    }
    Ok(frame)
}

/// 1 when the user interacted with the item in the evaluation window, else 0.
pub fn label_candidates(frame: &CandidateFrame, eval_table: &Table) -> Vec<u8> {
    let positives: HashSet<(u32, u32)> = eval_table
        .user
        .iter()
        .copied()
        .zip(eval_table.item.iter().copied())
        .collect();
    frame
        .users
        .iter()
        .zip(&frame.items)
        .map(|(&user, &item)| u8::from(positives.contains(&(user, item))))
        .collect()
}
