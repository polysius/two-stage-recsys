//! Two-tower retrieval model trained with in-batch negatives.
//!
//! Each tower is an id embedding followed by a small MLP, producing a unit-length vector. A
//! (user, item) pair scores as the cosine similarity of the two tower outputs.
//!
//! Training uses in-batch negatives: for a batch of positive pairs, every other item in the
//! batch is a negative for each user, and a softmax over the batch is trained to pick the true
//! item. Items that appear twice in a batch are masked so the same item is never both positive
//! and negative for a row.
//!
//! The backward pass is written by hand. The model is small enough for that, and it lets the
//! embedding gradient stay sparse while Adam still updates every row, as `torch.optim.Adam` does.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::config::TwoTowerConfig;
use crate::linalg::{matmul, MatrixView};
use crate::retrieval::embeddings::EmbeddingRetriever;
use crate::table::Table;

const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPSILON: f32 = 1e-8;
const NORM_EPSILON: f32 = 1e-12;

/// A parameter tensor with its gradient and Adam moments.
struct Param {
    value: Vec<f32>,
    grad: Vec<f32>,
    first_moment: Vec<f32>,
    second_moment: Vec<f32>,
}

impl Param {
    fn new(value: Vec<f32>) -> Param {
        let size = value.len();
        Param {
            value,
            grad: vec![0.0; size],
            first_moment: vec![0.0; size],
            second_moment: vec![0.0; size],
        }
    }

    /// One Adam step over every element, including those whose gradient is zero.
    fn adam_step(&mut self, learning_rate: f32, step: i32) {
        let correction1 = 1.0 - ADAM_BETA1.powi(step);
        let correction2_sqrt = (1.0 - ADAM_BETA2.powi(step)).sqrt();
        let step_size = learning_rate / correction1;
        self.value
            .par_chunks_mut(4096)
            .zip(self.grad.par_chunks(4096))
            .zip(self.first_moment.par_chunks_mut(4096))
            .zip(self.second_moment.par_chunks_mut(4096))
            .for_each(|(((value, grad), first), second)| {
                for i in 0..value.len() {
                    first[i] = ADAM_BETA1 * first[i] + (1.0 - ADAM_BETA1) * grad[i];
                    second[i] = ADAM_BETA2 * second[i] + (1.0 - ADAM_BETA2) * grad[i] * grad[i];
                    let denominator = second[i].sqrt() / correction2_sqrt + ADAM_EPSILON;
                    value[i] -= step_size * first[i] / denominator;
                }
            });
    }
}

/// Activations of one forward pass, kept for the backward pass.
struct Activations {
    ids: Vec<u32>,
    embedded: Vec<f32>,
    pre_activation: Vec<f32>,
    hidden: Vec<f32>,
    norms: Vec<f32>,
    output: Vec<f32>,
}

struct Tower {
    embedding_dim: usize,
    output_dim: usize,
    table: Param,
    w1: Param,
    b1: Param,
    w2: Param,
    b2: Param,
}

impl Tower {
    /// Initial values follow PyTorch's defaults: embeddings are standard normal, linear weights
    /// and biases are uniform within +-1/sqrt(fan_in).
    fn new(n_ids: usize, embedding_dim: usize, output_dim: usize, rng: &mut ChaCha12Rng) -> Tower {
        let normal = |rng: &mut ChaCha12Rng, count: usize| -> Vec<f32> {
            (0..count)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect()
        };
        let uniform = |rng: &mut ChaCha12Rng, count: usize, fan_in: usize| -> Vec<f32> {
            let bound = 1.0 / (fan_in as f32).sqrt();
            (0..count)
                .map(|_| rng.random_range(-bound..bound))
                .collect()
        };
        Tower {
            embedding_dim,
            output_dim,
            table: Param::new(normal(rng, n_ids * embedding_dim)),
            w1: Param::new(uniform(rng, embedding_dim * embedding_dim, embedding_dim)),
            b1: Param::new(uniform(rng, embedding_dim, embedding_dim)),
            w2: Param::new(uniform(rng, output_dim * embedding_dim, embedding_dim)),
            b2: Param::new(uniform(rng, output_dim, embedding_dim)),
        }
    }

    fn forward(&self, ids: &[u32]) -> Activations {
        let (batch, dim, out_dim) = (ids.len(), self.embedding_dim, self.output_dim);
        let mut embedded = Vec::with_capacity(batch * dim);
        for &id in ids {
            embedded.extend_from_slice(&self.table.value[id as usize * dim..][..dim]);
        }

        let mut pre_activation = vec![0.0; batch * dim];
        add_bias(&mut pre_activation, &self.b1.value);
        matmul(
            MatrixView::new(&embedded, batch, dim),
            MatrixView::new(&self.w1.value, dim, dim).t(),
            &mut pre_activation,
            true,
        );
        let hidden: Vec<f32> = pre_activation.iter().map(|&x| x.max(0.0)).collect();

        let mut projected = vec![0.0; batch * out_dim];
        add_bias(&mut projected, &self.b2.value);
        matmul(
            MatrixView::new(&hidden, batch, dim),
            MatrixView::new(&self.w2.value, out_dim, dim).t(),
            &mut projected,
            true,
        );

        let mut norms = vec![0.0; batch];
        let mut output = projected;
        for (row, norm) in output.chunks_exact_mut(out_dim).zip(&mut norms) {
            *norm = row
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(NORM_EPSILON);
            row.iter_mut().for_each(|x| *x /= *norm);
        }
        Activations {
            ids: ids.to_vec(),
            embedded,
            pre_activation,
            hidden,
            norms,
            output,
        }
    }

    /// Adds into the parameter gradients, given the loss gradient with respect to the output.
    fn backward(&mut self, cache: &Activations, output_grad: &[f32]) {
        let (batch, dim, out_dim) = (cache.ids.len(), self.embedding_dim, self.output_dim);

        // Back through the unit-length normalisation: remove the component along the output.
        let mut projected_grad = vec![0.0; batch * out_dim];
        for row in 0..batch {
            let y = &cache.output[row * out_dim..][..out_dim];
            let dy = &output_grad[row * out_dim..][..out_dim];
            let along: f32 = y.iter().zip(dy).map(|(a, b)| a * b).sum();
            for col in 0..out_dim {
                projected_grad[row * out_dim + col] = (dy[col] - y[col] * along) / cache.norms[row];
            }
        }

        let hidden = MatrixView::new(&cache.hidden, batch, dim);
        let grad_view = MatrixView::new(&projected_grad, batch, out_dim);
        matmul(grad_view.t(), hidden, &mut self.w2.grad, true);
        add_column_sums(&mut self.b2.grad, &projected_grad, out_dim);

        let mut hidden_grad = vec![0.0; batch * dim];
        matmul(
            grad_view,
            MatrixView::new(&self.w2.value, out_dim, dim),
            &mut hidden_grad,
            false,
        );
        for (grad, &pre) in hidden_grad.iter_mut().zip(&cache.pre_activation) {
            if pre <= 0.0 {
                *grad = 0.0;
            }
        }

        let embedded = MatrixView::new(&cache.embedded, batch, dim);
        let pre_grad = MatrixView::new(&hidden_grad, batch, dim);
        matmul(pre_grad.t(), embedded, &mut self.w1.grad, true);
        add_column_sums(&mut self.b1.grad, &hidden_grad, dim);

        let mut embedded_grad = vec![0.0; batch * dim];
        matmul(
            pre_grad,
            MatrixView::new(&self.w1.value, dim, dim),
            &mut embedded_grad,
            false,
        );
        for (&id, row) in cache.ids.iter().zip(embedded_grad.chunks_exact(dim)) {
            let target = &mut self.table.grad[id as usize * dim..][..dim];
            target
                .iter_mut()
                .zip(row)
                .for_each(|(sum, grad)| *sum += grad);
        }
    }

    fn zero_gradients(&mut self, touched_ids: &[u32]) {
        let dim = self.embedding_dim;
        for &id in touched_ids {
            self.table.grad[id as usize * dim..][..dim].fill(0.0);
        }
        for param in [&mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2] {
            param.grad.fill(0.0);
        }
    }

    fn adam_step(&mut self, learning_rate: f32, step: i32) {
        for param in [
            &mut self.table,
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
        ] {
            param.adam_step(learning_rate, step);
        }
    }

    /// Output vectors for every id, as a row-major `n_ids x output_dim` block.
    fn embed_all(&self, n_ids: usize) -> Vec<f32> {
        let ids: Vec<u32> = (0..n_ids as u32).collect();
        ids.chunks(4096)
            .flat_map(|chunk| self.forward(chunk).output)
            .collect()
    }
}

fn add_bias(rows: &mut [f32], bias: &[f32]) {
    for row in rows.chunks_exact_mut(bias.len()) {
        row.iter_mut().zip(bias).for_each(|(x, b)| *x += b);
    }
}

fn add_column_sums(sums: &mut [f32], rows: &[f32], width: usize) {
    for row in rows.chunks_exact(width) {
        sums.iter_mut().zip(row).for_each(|(sum, x)| *sum += x);
    }
}

pub struct TwoTower {
    user_tower: Tower,
    item_tower: Tower,
    temperature: f32,
}

impl TwoTower {
    pub fn new(n_users: usize, n_items: usize, config: &TwoTowerConfig, seed: u64) -> TwoTower {
        // Seed before building the towers so initial weights are reproducible.
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        TwoTower {
            user_tower: Tower::new(n_users, config.embedding_dim, config.output_dim, &mut rng),
            item_tower: Tower::new(n_items, config.embedding_dim, config.output_dim, &mut rng),
            temperature: config.temperature as f32,
        }
    }

    /// Mean softmax cross-entropy over the batch. Fills the gradients of both towers.
    fn loss_and_gradients(&mut self, users: &[u32], items: &[u32]) -> f32 {
        let batch = users.len();
        let out_dim = self.user_tower.output_dim;
        let user_cache = self.user_tower.forward(users);
        let item_cache = self.item_tower.forward(items);

        let mut logits = vec![0.0; batch * batch];
        matmul(
            MatrixView::new(&user_cache.output, batch, out_dim),
            MatrixView::new(&item_cache.output, batch, out_dim).t(),
            &mut logits,
            false,
        );
        logits.iter_mut().for_each(|x| *x /= self.temperature);

        // Turn each row of logits into its softmax, masking repeated items, and read the loss.
        let mut loss = 0.0f32;
        logits
            .par_chunks_mut(batch)
            .enumerate()
            .for_each(|(row, logits_row)| {
                for (col, logit) in logits_row.iter_mut().enumerate() {
                    if col != row && items[col] == items[row] {
                        *logit = f32::NEG_INFINITY;
                    }
                }
                let max = logits_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0.0;
                for logit in logits_row.iter_mut() {
                    *logit = (*logit - max).exp();
                    total += *logit;
                }
                logits_row.iter_mut().for_each(|p| *p /= total);
            });
        for row in 0..batch {
            loss -= logits[row * batch + row].ln();
        }
        loss /= batch as f32;

        // Gradient of the mean loss with respect to the logits: (softmax - one-hot) / batch.
        let mut logit_grad = logits;
        for row in 0..batch {
            logit_grad[row * batch + row] -= 1.0;
        }
        let scale = 1.0 / (batch as f32 * self.temperature);
        logit_grad.iter_mut().for_each(|x| *x *= scale);

        let mut user_output_grad = vec![0.0; batch * out_dim];
        let mut item_output_grad = vec![0.0; batch * out_dim];
        let grad_view = MatrixView::new(&logit_grad, batch, batch);
        matmul(
            grad_view,
            MatrixView::new(&item_cache.output, batch, out_dim),
            &mut user_output_grad,
            false,
        );
        matmul(
            grad_view.t(),
            MatrixView::new(&user_cache.output, batch, out_dim),
            &mut item_output_grad,
            false,
        );
        self.user_tower.backward(&user_cache, &user_output_grad);
        self.item_tower.backward(&item_cache, &item_output_grad);
        loss
    }

    /// Trains on the interactions and returns the mean loss of each epoch.
    pub fn fit(&mut self, train: &Table, config: &TwoTowerConfig, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha12Rng::seed_from_u64(seed ^ 0x5eed);
        let mut order: Vec<usize> = (0..train.len()).collect();
        let mut loss_history = Vec::with_capacity(config.epochs);
        let mut step = 0;

        for _ in 0..config.epochs {
            order.shuffle(&mut rng);
            let mut epoch_loss = 0.0f64;
            for batch in order.chunks(config.batch_size) {
                let users: Vec<u32> = batch.iter().map(|&row| train.user[row]).collect();
                let items: Vec<u32> = batch.iter().map(|&row| train.item[row]).collect();
                let loss = self.loss_and_gradients(&users, &items);

                step += 1;
                self.user_tower.adam_step(config.learning_rate as f32, step);
                self.item_tower.adam_step(config.learning_rate as f32, step);
                self.user_tower.zero_gradients(&users);
                self.item_tower.zero_gradients(&items);
                epoch_loss += f64::from(loss) * batch.len() as f64;
            }
            loss_history.push((epoch_loss / train.len() as f64) as f32);
        }
        loss_history
    }

    pub fn into_retriever(self, n_users: usize, n_items: usize) -> EmbeddingRetriever {
        EmbeddingRetriever::new(
            self.user_tower.output_dim,
            self.user_tower.embed_all(n_users),
            self.item_tower.embed_all(n_items),
        )
    }
}

/// Fits a two-tower model and returns its retriever and the loss per epoch.
pub fn fit_two_tower(
    n_users: usize,
    n_items: usize,
    train: &Table,
    config: &TwoTowerConfig,
    seed: u64,
) -> (EmbeddingRetriever, Vec<f32>) {
    let mut model = TwoTower::new(n_users, n_items, config, seed);
    let losses = model.fit(train, config, seed);
    (model.into_retriever(n_users, n_items), losses)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recommender::Recommender;
    use crate::sparse::BinaryCsr;

    fn small_config() -> TwoTowerConfig {
        TwoTowerConfig {
            embedding_dim: 8,
            output_dim: 6,
            epochs: 30,
            batch_size: 16,
            learning_rate: 0.05,
            temperature: 0.5,
        }
    }

    #[test]
    fn analytic_gradients_match_finite_differences() {
        let config = small_config();
        let users = [0u32, 1, 2, 3, 4];
        let items = [0u32, 1, 2, 1, 4]; // item 1 repeats, so the duplicate mask is exercised
        let mut model = TwoTower::new(5, 5, &config, 1);
        model.loss_and_gradients(&users, &items);

        let analytic_w1 = model.user_tower.w1.grad[3];
        let analytic_embedding = model.item_tower.table.grad[2 * 8 + 5];
        let analytic_b2 = model.item_tower.b2.grad[1];

        let loss_with = |perturb: &dyn Fn(&mut TwoTower, f32)| {
            let step = 1e-2f32;
            let mut plus = TwoTower::new(5, 5, &config, 1);
            perturb(&mut plus, step);
            let mut minus = TwoTower::new(5, 5, &config, 1);
            perturb(&mut minus, -step);
            (plus.loss_and_gradients(&users, &items) - minus.loss_and_gradients(&users, &items))
                / (2.0 * step)
        };
        let numeric_w1 = loss_with(&|m, d| m.user_tower.w1.value[3] += d);
        let numeric_embedding = loss_with(&|m, d| m.item_tower.table.value[2 * 8 + 5] += d);
        let numeric_b2 = loss_with(&|m, d| m.item_tower.b2.value[1] += d);

        for (analytic, numeric) in [
            (analytic_w1, numeric_w1),
            (analytic_embedding, numeric_embedding),
            (analytic_b2, numeric_b2),
        ] {
            assert!(
                (analytic - numeric).abs() < 2e-3 + 0.05 * numeric.abs(),
                "analytic {analytic} vs numeric {numeric}"
            );
        }
    }

    #[test]
    fn training_lowers_the_loss_and_learns_the_pairing() {
        // User u always interacts with item u, so the model should rank item u first.
        let mut table = Table::default();
        for _ in 0..4 {
            for user in 0..12u32 {
                table.user.push(user);
                table.item.push(user);
                table.rating.push(5.0);
                table.timestamp.push(0);
            }
        }
        let config = small_config();
        let (retriever, losses) = fit_two_tower(12, 12, &table, &config, 0);
        assert!(
            losses.last().unwrap() < &(losses[0] * 0.5),
            "losses {losses:?}"
        );
        let seen = BinaryCsr::from_pairs(12, 12, &[], &[]);
        let correct = (0..12u32)
            .filter(|&user| retriever.recommend_user(user, 1, seen.row(user as usize))[0] == user)
            .count();
        assert!(
            correct >= 10,
            "only {correct} of 12 users got their own item"
        );
    }
}
