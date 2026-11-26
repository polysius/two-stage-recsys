//! Exact nearest-neighbour retrieval over item embeddings.

use serde::{Deserialize, Serialize};

use crate::recommender::{top_k_unseen, Recommender};

/// Recommends by inner product between a user vector and every item vector.
///
/// Both ALS and the two-tower model produce user and item vectors, so they share this type.
/// The search is exact. With 25k items a full scan per user is fast, and it keeps retrieval
/// quality free of approximation error, so differences between retrievers come from the model.
/// Seen items are masked before the top-k, so no extra neighbours need to be fetched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRetriever {
    dim: usize,
    user_vectors: Vec<f32>,
    item_vectors: Vec<f32>,
}

impl EmbeddingRetriever {
    pub fn new(dim: usize, user_vectors: Vec<f32>, item_vectors: Vec<f32>) -> EmbeddingRetriever {
        EmbeddingRetriever {
            dim,
            user_vectors,
            item_vectors,
        }
    }

    fn user_vector(&self, user: u32) -> &[f32] {
        &self.user_vectors[user as usize * self.dim..][..self.dim]
    }

    fn item_vector(&self, item: usize) -> &[f32] {
        &self.item_vectors[item * self.dim..][..self.dim]
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

impl Recommender for EmbeddingRetriever {
    fn recommend_user(&self, user: u32, k: usize, seen: &[u32]) -> Vec<u32> {
        let user_vector = self.user_vector(user);
        let scores: Vec<f32> = self
            .item_vectors
            .chunks_exact(self.dim)
            .map(|item_vector| dot(user_vector, item_vector))
            .collect();
        top_k_unseen(&scores, seen, k)
    }

    fn score_pair(&self, user: u32, item: u32) -> Option<f32> {
        Some(dot(self.user_vector(user), self.item_vector(item as usize)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommends_the_highest_inner_products_and_skips_seen() {
        // One user vector (1, 0) and three items with first coordinates 0.5, 2.0, 1.0.
        let retriever =
            EmbeddingRetriever::new(2, vec![1.0, 0.0], vec![0.5, 9.0, 2.0, 9.0, 1.0, 9.0]);
        assert_eq!(retriever.recommend_user(0, 2, &[]), vec![1, 2]);
        assert_eq!(retriever.recommend_user(0, 2, &[1]), vec![2, 0]);
        assert_eq!(retriever.score_pair(0, 2), Some(1.0));
    }
}
