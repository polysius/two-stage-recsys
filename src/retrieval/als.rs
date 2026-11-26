//! Alternating least squares matrix factorisation for implicit feedback.
//!
//! Follows Hu, Koren and Volinsky: every observed pair has confidence `alpha`, every other pair
//! confidence 1, and each side is re-solved in turn while the other side is held fixed.
//! `alpha` scales the confidence of every observed interaction. Without it, ALS treats the
//! single observed rating as weak evidence and ranks poorly on this dataset.
//!
//! Each row is solved with a few conjugate gradient steps started from its previous value, which
//! is what the `implicit` library does. It never forms the per-row matrix, so a step costs
//! `factors^2 + 2 * interactions * factors` instead of the cubic cost of an exact solve.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use crate::config::AlsConfig;
use crate::retrieval::embeddings::EmbeddingRetriever;
use crate::sparse::BinaryCsr;
use crate::table::Table;

const CG_STEPS: usize = 3;

pub struct AlsFactors {
    pub user_factors: Vec<f32>,
    pub item_factors: Vec<f32>,
}

pub fn fit_als(
    n_users: usize,
    n_items: usize,
    train: &Table,
    config: &AlsConfig,
    seed: u64,
) -> EmbeddingRetriever {
    let user_items = BinaryCsr::user_item(train, n_users, n_items);
    let item_users = user_items.transpose();
    let factors = fit_factors(&user_items, &item_users, config, seed);
    EmbeddingRetriever::new(config.factors, factors.user_factors, factors.item_factors)
}

fn fit_factors(
    user_items: &BinaryCsr,
    item_users: &BinaryCsr,
    config: &AlsConfig,
    seed: u64,
) -> AlsFactors {
    let dim = config.factors;
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let mut small_random = |count: usize| -> Vec<f32> {
        (0..count * dim)
            .map(|_| rng.random::<f32>() * 0.01)
            .collect()
    };
    let mut user_factors = small_random(user_items.n_rows());
    let mut item_factors = small_random(item_users.n_rows());

    for _ in 0..config.iterations {
        solve_side(&mut user_factors, &item_factors, user_items, config);
        solve_side(&mut item_factors, &user_factors, item_users, config);
    }
    AlsFactors {
        user_factors,
        item_factors,
    }
}

/// Buffers one conjugate gradient solve needs, reused across rows to avoid allocating.
struct Scratch {
    residual: Vec<f32>,
    direction: Vec<f32>,
    product: Vec<f32>,
}

/// Recomputes every row of `target` from the fixed `other` factors.
///
/// For one row with observed columns `S`, the solution of the weighted ridge problem is
/// `(Y'Y + reg*I + (alpha - 1) * sum_{i in S} y_i y_i') x = alpha * sum_{i in S} y_i`.
fn solve_side(target: &mut [f32], other: &[f32], observed: &BinaryCsr, config: &AlsConfig) {
    let dim = config.factors;
    let gram = regularised_gram(other, dim, config.regularization as f32);
    let extra_weight = (config.alpha - 1.0) as f32;
    let alpha = config.alpha as f32;
    let new_scratch = || Scratch {
        residual: vec![0.0; dim],
        direction: vec![0.0; dim],
        product: vec![0.0; dim],
    };

    target
        .par_chunks_mut(dim)
        .enumerate()
        .for_each_init(new_scratch, |scratch, (row, x)| {
            let columns = observed.row(row);
            if columns.is_empty() {
                x.fill(0.0);
                return;
            }
            let vectors = || {
                columns
                    .iter()
                    .map(|&column| &other[column as usize * dim..][..dim])
            };

            // The right-hand side is alpha * sum(y_i), and the residual starts as b - A x.
            scratch.residual.fill(0.0);
            for vector in vectors() {
                axpy(alpha, vector, &mut scratch.residual);
            }
            apply_matrix(&gram, dim, extra_weight, vectors(), x, &mut scratch.product);
            for (residual, product) in scratch.residual.iter_mut().zip(&scratch.product) {
                *residual -= product;
            }
            scratch.direction.copy_from_slice(&scratch.residual);

            let mut residual_norm = dot(&scratch.residual, &scratch.residual);
            for _ in 0..CG_STEPS {
                apply_matrix(
                    &gram,
                    dim,
                    extra_weight,
                    vectors(),
                    &scratch.direction,
                    &mut scratch.product,
                );
                let curvature = dot(&scratch.direction, &scratch.product);
                if curvature <= 0.0 || residual_norm == 0.0 {
                    break;
                }
                let step = residual_norm / curvature;
                axpy(step, &scratch.direction, x);
                axpy(-step, &scratch.product, &mut scratch.residual);
                let new_norm = dot(&scratch.residual, &scratch.residual);
                let blend = new_norm / residual_norm;
                for (direction, residual) in scratch.direction.iter_mut().zip(&scratch.residual) {
                    *direction = residual + blend * *direction;
                }
                residual_norm = new_norm;
            }
        });
}

/// `out = (gram + extra_weight * sum y y') * input`, without building the sum.
fn apply_matrix<'a>(
    gram: &[f32],
    dim: usize,
    extra_weight: f32,
    vectors: impl Iterator<Item = &'a [f32]>,
    input: &[f32],
    out: &mut [f32],
) {
    for (row, value) in gram.chunks_exact(dim).zip(out.iter_mut()) {
        *value = dot(row, input);
    }
    for vector in vectors {
        axpy(extra_weight * dot(vector, input), vector, out);
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// `target += scale * source`
fn axpy(scale: f32, source: &[f32], target: &mut [f32]) {
    for (out, value) in target.iter_mut().zip(source) {
        *out += scale * value;
    }
}

/// `Y'Y + regularization * I` as a row-major `dim x dim` matrix, summed in parallel.
fn regularised_gram(factors: &[f32], dim: usize, regularization: f32) -> Vec<f32> {
    let mut gram = factors
        .par_chunks(dim * 1024)
        .map(|block| {
            let mut partial = vec![0f32; dim * dim];
            for vector in block.chunks_exact(dim) {
                for (row, &value) in partial.chunks_exact_mut(dim).zip(vector) {
                    axpy(value, vector, row);
                }
            }
            partial
        })
        .reduce(
            || vec![0f32; dim * dim],
            |mut total, partial| {
                total
                    .iter_mut()
                    .zip(&partial)
                    .for_each(|(sum, part)| *sum += part);
                total
            },
        );
    for i in 0..dim {
        gram[i * dim + i] += regularization;
    }
    gram
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recommender::Recommender;

    #[test]
    fn gram_matrix_adds_the_regularisation_to_the_diagonal() {
        // Two vectors (1, 2) and (3, 4): Y'Y = [[10, 14], [14, 20]].
        let gram = regularised_gram(&[1.0, 2.0, 3.0, 4.0], 2, 0.5);
        assert_eq!(gram, vec![10.5, 14.0, 14.0, 20.5]);
    }

    #[test]
    fn matrix_product_matches_the_explicit_sum() {
        // gram = I, one vector y = (1, 2), weight 2, input (1, 1):
        // (I + 2 y y') x = x + 2 * (y.x) * y = (1, 1) + 6 * (1, 2) = (7, 13)
        let mut out = [0.0; 2];
        let y = [1.0f32, 2.0];
        apply_matrix(
            &[1.0, 0.0, 0.0, 1.0],
            2,
            2.0,
            std::iter::once(&y[..]),
            &[1.0, 1.0],
            &mut out,
        );
        assert_eq!(out, [7.0, 13.0]);
    }

    #[test]
    fn two_taste_groups_are_separated() {
        // Users 0-9 like items 0-4 and users 10-19 like items 5-9.
        let mut table = Table::default();
        for user in 0..20u32 {
            for offset in 0..4 {
                table.user.push(user);
                table.item.push((user / 10) * 5 + (user + offset) % 5);
                table.rating.push(5.0);
                table.timestamp.push(0);
            }
        }
        let config = AlsConfig {
            factors: 4,
            regularization: 0.01,
            iterations: 10,
            alpha: 40.0,
        };
        let model = fit_als(20, 10, &table, &config, 0);
        let seen = BinaryCsr::user_item(&table, 20, 10);
        let recommended = model.recommend_user(0, 1, seen.row(0));
        assert!(
            recommended[0] < 5,
            "a first-group user should get a first-group item"
        );
        let recommended = model.recommend_user(15, 1, seen.row(15));
        assert!(
            recommended[0] >= 5,
            "a second-group user should get a second-group item"
        );
    }
}
