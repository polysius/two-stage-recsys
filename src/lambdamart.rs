//! LambdaMART in pure Rust: gradient boosted trees trained with the lambdarank objective.
//!
//! The design follows LightGBM so results are comparable with `lightgbm.LGBMRanker`:
//! features are binned once into at most 255 bins plus one bin for missing values, trees grow
//! leaf-wise (best split first) on gradient histograms, and the sibling histogram comes from
//! subtracting the smaller child from the parent. Gradients come from LightGBM's
//! `LambdarankNDCG::GetGradientsForOneQuery`.
//!
//! Training is deterministic for a fixed seed. Parallel work is split so that every floating
//! point sum has a fixed order: queries are independent, and each feature histogram is filled
//! by one thread.

use std::ops::Range;
use std::time::Instant;

use rand::seq::index::sample;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Bin index reserved for NaN. Ordinary bins use `0..n_bins` with `n_bins <= MISSING_BIN`.
const MISSING_BIN: usize = 255;
const BINS_PER_FEATURE: usize = MISSING_BIN + 1;
/// Bin boundaries come from a sample of this many rows (LightGBM's `bin_construct_sample_cnt`).
const BIN_SAMPLE_ROWS: usize = 200_000;
/// Below this many rows a histogram is filled on the calling thread; thread hand-off costs more.
const PARALLEL_HISTOGRAM_MIN_ROWS: usize = 20_000;
const HISTOGRAM_ROW_CHUNKS: usize = 4;
const PARALLEL_PARTITION_MIN_ROWS: usize = 100_000;
const PARTITION_CHUNK_ROWS: usize = 32_768;

// ---------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Params {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub num_leaves: usize,
    pub min_child_samples: usize,
    /// Share of queries (not rows) used to grow each tree.
    pub subsample: f64,
    /// Resample the queries every this many iterations; 0 disables bagging.
    pub subsample_freq: usize,
    pub colsample_bytree: f64,
    /// At most 255, because bin indices are stored as `u8` next to the missing bin.
    pub max_bin: usize,
    pub truncation_level: usize,
    pub sigmoid: f64,
    pub min_sum_hessian: f64,
    pub seed: u64,
}

impl Default for Params {
    /// The values from `ranking.lightgbm` in `configs/default.yaml`, plus LightGBM defaults.
    fn default() -> Self {
        Self {
            n_estimators: 500,
            learning_rate: 0.05,
            num_leaves: 31,
            min_child_samples: 50,
            subsample: 0.8,
            subsample_freq: 1,
            colsample_bytree: 0.8,
            max_bin: 255,
            truncation_level: 30,
            sigmoid: 1.0,
            min_sum_hessian: 1e-3,
            seed: 0,
        }
    }
}

/// Row-major feature matrix with binary or graded labels, grouped by query.
pub struct Dataset<'a> {
    pub features: &'a [f32],
    pub n_features: usize,
    pub labels: &'a [u8],
    /// Number of consecutive rows in each query.
    pub group_sizes: &'a [usize],
}

impl Dataset<'_> {
    fn n_rows(&self) -> usize {
        self.labels.len()
    }

    fn check(&self) {
        assert!(self.n_features > 0, "dataset needs at least one feature");
        assert_eq!(
            self.features.len(),
            self.n_rows() * self.n_features,
            "features must hold n_rows * n_features values"
        );
        assert_eq!(
            self.group_sizes.iter().sum::<usize>(),
            self.n_rows(),
            "group sizes must add up to the number of rows"
        );
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    trees: Vec<Tree>,
    n_features: usize,
    best_iteration: usize,
}

impl Model {
    /// Scores for row-major `features`, computed in parallel over rows.
    pub fn predict(&self, features: &[f32], n_features: usize) -> Vec<f32> {
        assert_eq!(
            n_features, self.n_features,
            "feature count differs from training"
        );
        features
            .par_chunks(n_features)
            .map(|row| self.predict_row(row))
            .collect()
    }

    pub fn predict_row(&self, row: &[f32]) -> f32 {
        let total: f64 = self.trees[..self.best_iteration]
            .iter()
            .map(|tree| f64::from(tree.predict_row(row)))
            .sum();
        total as f32
    }

    /// Total split gain per feature over the kept trees.
    pub fn feature_importance_gain(&self) -> Vec<f64> {
        let mut gain = vec![0.0; self.n_features];
        for tree in &self.trees[..self.best_iteration] {
            for node in &tree.nodes {
                gain[node.feature as usize] += node.gain;
            }
        }
        gain
    }

    pub fn best_iteration(&self) -> usize {
        self.best_iteration
    }
}

/// Mean NDCG@k over queries with LightGBM's conventions: gain `2^label - 1`, discount
/// `1 / log2(2 + rank)`, ties in score broken by row order, and a query without any relevant
/// item counts as 1.0.
pub fn ndcg_at_k<S: Copy + Into<f64> + Sync>(
    scores: &[S],
    labels: &[u8],
    group_sizes: &[usize],
    k: usize,
) -> f64 {
    assert_eq!(scores.len(), labels.len());
    let ranges = query_ranges(group_sizes);
    let per_query: Vec<f64> = ranges
        .par_iter()
        .map(|range| query_ndcg(&scores[range.clone()], &labels[range.clone()], k))
        .collect();
    // Summing the collected vector keeps the order, and so the result, fixed.
    per_query.iter().sum::<f64>() / per_query.len().max(1) as f64
}

/// Fits a ranker on `train` and stops when NDCG@`eval_at` on `valid` has not improved for
/// `early_stopping_rounds` iterations (0 turns early stopping off). The returned model keeps
/// only the trees up to the best iteration.
pub fn train(
    train: &Dataset,
    valid: &Dataset,
    params: &Params,
    early_stopping_rounds: usize,
    eval_at: usize,
) -> Model {
    train.check();
    valid.check();
    assert_eq!(
        train.n_features, valid.n_features,
        "train and valid feature counts differ"
    );
    assert!(
        params.max_bin >= 2 && params.max_bin <= MISSING_BIN,
        "max_bin must be in 2..=255"
    );
    assert!(params.num_leaves >= 2, "num_leaves must be at least 2");

    let mut trainer = Trainer::new(train, params);
    let valid_ranges = query_ranges(valid.group_sizes);
    let mut valid_scores = vec![0.0_f64; valid.n_rows()];

    let mut trees: Vec<Tree> = Vec::with_capacity(params.n_estimators);
    let mut best_ndcg = f64::NEG_INFINITY;
    let mut best_iteration = 0;

    for iteration in 0..params.n_estimators {
        let Some(tree) = trainer.boost_one_tree(iteration) else {
            break; // no leaf can be split any more, so further trees would be empty
        };

        valid_scores
            .par_iter_mut()
            .zip(valid.features.par_chunks(valid.n_features))
            .for_each(|(score, row)| *score += f64::from(tree.predict_row(row)));
        trees.push(tree);

        let ndcg = mean_ndcg(&valid_scores, valid.labels, &valid_ranges, eval_at);
        if ndcg > best_ndcg {
            best_ndcg = ndcg;
            best_iteration = trees.len();
        } else if early_stopping_rounds > 0 && trees.len() - best_iteration >= early_stopping_rounds
        {
            break;
        }
    }

    if early_stopping_rounds == 0 || best_iteration == 0 {
        best_iteration = trees.len();
    }
    trees.truncate(best_iteration);
    Model {
        trees,
        n_features: train.n_features,
        best_iteration,
    }
}

// ---------------------------------------------------------------------------------------------
// Trees
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum Child {
    Node(u32),
    Leaf(u32),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Node {
    feature: u32,
    /// Values `<= threshold` go left.
    threshold: f32,
    /// Where NaN goes, learned from the training data at this split.
    default_left: bool,
    left: Child,
    right: Child,
    gain: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Tree {
    /// Node 0 is the root. Empty when the tree is a single leaf.
    nodes: Vec<Node>,
    leaf_values: Vec<f32>,
}

impl Tree {
    fn predict_row(&self, row: &[f32]) -> f32 {
        if self.nodes.is_empty() {
            return self.leaf_values[0];
        }
        let mut current = Child::Node(0);
        loop {
            match current {
                Child::Leaf(leaf) => return self.leaf_values[leaf as usize],
                Child::Node(index) => {
                    let node = &self.nodes[index as usize];
                    let value = row[node.feature as usize];
                    let goes_left = if value.is_nan() {
                        node.default_left
                    } else {
                        value <= node.threshold
                    };
                    current = if goes_left { node.left } else { node.right };
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Binning
// ---------------------------------------------------------------------------------------------

/// Training features as `u8` bin indices, stored one column after another so a histogram pass
/// over one feature reads contiguous memory.
struct BinnedData {
    n_rows: usize,
    columns: Vec<u8>,
    /// Per feature, the upper edge of each ordinary bin. The last edge is `f32::MAX`.
    upper_bounds: Vec<Vec<f32>>,
}

impl BinnedData {
    fn build(data: &Dataset, max_bin: usize) -> Self {
        let n_rows = data.n_rows();
        let n_features = data.n_features;
        let sample_stride = (n_rows / BIN_SAMPLE_ROWS).max(1);

        let upper_bounds: Vec<Vec<f32>> = (0..n_features)
            .into_par_iter()
            .map(|feature| {
                let sample: Vec<f32> = (0..n_rows)
                    .step_by(sample_stride)
                    .map(|row| data.features[row * n_features + feature])
                    .filter(|value| !value.is_nan())
                    .collect();
                compute_upper_bounds(sample, max_bin)
            })
            .collect();

        let mut columns = vec![0u8; n_rows * n_features];
        columns
            .par_chunks_mut(n_rows)
            .zip(upper_bounds.par_iter())
            .enumerate()
            .for_each(|(feature, (column, bounds))| {
                for (row, bin) in column.iter_mut().enumerate() {
                    *bin = bin_of(data.features[row * n_features + feature], bounds);
                }
            });

        Self {
            n_rows,
            columns,
            upper_bounds,
        }
    }

    fn n_features(&self) -> usize {
        self.upper_bounds.len()
    }

    fn column(&self, feature: usize) -> &[u8] {
        &self.columns[feature * self.n_rows..(feature + 1) * self.n_rows]
    }

    fn n_bins(&self, feature: usize) -> usize {
        self.upper_bounds[feature].len()
    }
}

/// Bin edges that put roughly equal numbers of sample rows in each bin. Features with few
/// distinct values get one bin per value.
fn compute_upper_bounds(mut values: Vec<f32>, max_bin: usize) -> Vec<f32> {
    values.sort_unstable_by(|a, b| a.partial_cmp(b).expect("NaN was filtered out"));

    let mut distinct: Vec<(f32, usize)> = Vec::new();
    for value in &values {
        match distinct.last_mut() {
            Some((last, count)) if *last == *value => *count += 1,
            _ => distinct.push((*value, 1)),
        }
    }

    let mut bounds = Vec::new();
    if distinct.len() <= max_bin {
        for pair in distinct.windows(2) {
            bounds.push(midpoint(pair[0].0, pair[1].0));
        }
    } else {
        let target_per_bin = values.len() as f64 / max_bin as f64;
        let mut in_current_bin = 0usize;
        for k in 0..distinct.len() - 1 {
            in_current_bin += distinct[k].1;
            if in_current_bin as f64 >= target_per_bin && bounds.len() < max_bin - 1 {
                bounds.push(midpoint(distinct[k].0, distinct[k + 1].0));
                in_current_bin = 0;
            }
        }
    }
    bounds.push(f32::MAX);
    bounds
}

fn midpoint(low: f32, high: f32) -> f32 {
    let mid = low + (high - low) / 2.0;
    // Adjacent floats can round the midpoint up to `high`, which would merge the two values.
    if mid >= high {
        low
    } else {
        mid
    }
}

/// A value falls in the first bin whose upper edge is not below it.
fn bin_of(value: f32, upper_bounds: &[f32]) -> u8 {
    if value.is_nan() {
        return MISSING_BIN as u8;
    }
    let bin = upper_bounds.partition_point(|&edge| edge < value);
    bin.min(upper_bounds.len() - 1) as u8
}

// ---------------------------------------------------------------------------------------------
// Histograms and split search
// ---------------------------------------------------------------------------------------------

/// Gradient statistics of a set of rows; also one histogram bin.
#[derive(Clone, Copy, Default, Debug)]
struct Stats {
    grad: f64,
    hess: f64,
    count: u32,
}

impl Stats {
    fn plus(self, other: Stats) -> Stats {
        Stats {
            grad: self.grad + other.grad,
            hess: self.hess + other.hess,
            count: self.count + other.count,
        }
    }

    fn minus(self, other: Stats) -> Stats {
        Stats {
            grad: self.grad - other.grad,
            hess: self.hess - other.hess,
            count: self.count - other.count,
        }
    }

    /// `G^2 / H`, the leaf's contribution to the objective (L2 regularisation is 0).
    fn score(self) -> f64 {
        if self.hess > 0.0 {
            self.grad * self.grad / self.hess
        } else {
            0.0
        }
    }
}

/// One histogram holds `BINS_PER_FEATURE` bins for each feature chosen for the current tree.
type Histogram = Vec<Stats>;

struct SplitInfo {
    gain: f64,
    feature: usize,
    bin: usize,
    default_left: bool,
    left: Stats,
    right: Stats,
}

/// A leaf that can still be split. `begin..end` locates its rows in the shared row buffer.
struct Leaf {
    rows: Range<usize>,
    stats: Stats,
    histogram: Histogram,
    best_split: Option<SplitInfo>,
    /// The node that points at this leaf, and whether it is that node's left child.
    parent: Option<(usize, bool)>,
}

struct TreeGrower<'a> {
    binned: &'a BinnedData,
    /// (gradient, hessian) per row, side by side so a histogram pass touches one cache line.
    gradients: &'a [[f32; 2]],
    params: &'a Params,
    /// Features chosen for this tree by `colsample_bytree`.
    features: &'a [usize],
}

impl TreeGrower<'_> {
    /// Grows one tree on `rows` (in-bag rows, ascending). `rows` is reordered so every leaf's
    /// rows are contiguous; the returned ranges say where.
    fn grow(&self, rows: &mut [u32], scratch: &mut Vec<u32>) -> (Tree, Vec<Range<usize>>) {
        let root_histogram = self.build_histogram(rows);
        // Every row falls in exactly one bin of any feature, so one feature's bins sum to the total.
        let root_stats = root_histogram[..BINS_PER_FEATURE]
            .iter()
            .fold(Stats::default(), |sum, bin| sum.plus(*bin));
        let mut leaves = vec![Leaf {
            rows: 0..rows.len(),
            stats: root_stats,
            best_split: self.find_best_split(&root_histogram, root_stats),
            histogram: root_histogram,
            parent: None,
        }];
        let mut nodes: Vec<Node> = Vec::new();

        while leaves.len() < self.params.num_leaves {
            let Some(leaf_id) = best_leaf_to_split(&leaves) else {
                break;
            };
            let split = leaves[leaf_id]
                .best_split
                .take()
                .expect("chosen leaf has a split");

            let node_index = nodes.len();
            let new_leaf_id = leaves.len();
            if let Some((parent_node, is_left)) = leaves[leaf_id].parent {
                let child = Child::Node(node_index as u32);
                if is_left {
                    nodes[parent_node].left = child;
                } else {
                    nodes[parent_node].right = child;
                }
            }
            nodes.push(Node {
                feature: split.feature as u32,
                threshold: self.binned.upper_bounds[split.feature][split.bin],
                default_left: split.default_left,
                left: Child::Leaf(leaf_id as u32),
                right: Child::Leaf(new_leaf_id as u32),
                gain: split.gain,
            });

            // The left child keeps the parent's slot; the right child is appended.
            let parent_range = leaves[leaf_id].rows.clone();
            let n_left = self.partition_rows(&mut rows[parent_range.clone()], scratch, &split);
            let left_range = parent_range.start..parent_range.start + n_left;
            let right_range = parent_range.start + n_left..parent_range.end;

            // Histogram subtraction: build the smaller child, derive the larger from the parent.
            let mut parent_histogram = std::mem::take(&mut leaves[leaf_id].histogram);
            let left_is_smaller = left_range.len() <= right_range.len();
            let small_range = if left_is_smaller {
                left_range.clone()
            } else {
                right_range.clone()
            };
            let small_histogram = self.build_histogram(&rows[small_range]);
            for (parent_bin, small_bin) in parent_histogram.iter_mut().zip(&small_histogram) {
                *parent_bin = parent_bin.minus(*small_bin);
            }
            let large_histogram = parent_histogram;
            let (left_histogram, right_histogram) = if left_is_smaller {
                (small_histogram, large_histogram)
            } else {
                (large_histogram, small_histogram)
            };

            leaves[leaf_id] = Leaf {
                rows: left_range,
                stats: split.left,
                best_split: self.find_best_split(&left_histogram, split.left),
                histogram: left_histogram,
                parent: Some((node_index, true)),
            };
            leaves.push(Leaf {
                rows: right_range,
                stats: split.right,
                best_split: self.find_best_split(&right_histogram, split.right),
                histogram: right_histogram,
                parent: Some((node_index, false)),
            });
        }

        let learning_rate = self.params.learning_rate;
        let leaf_values = leaves
            .iter()
            .map(|leaf| {
                (-leaf.stats.grad / leaf.stats.hess.max(f64::MIN_POSITIVE) * learning_rate) as f32
            })
            .collect();
        let ranges = leaves.iter().map(|leaf| leaf.rows.clone()).collect();
        (Tree { nodes, leaf_values }, ranges)
    }

    fn build_histogram(&self, rows: &[u32]) -> Histogram {
        let n_slots = self.features.len();
        let mut histogram = vec![Stats::default(); n_slots * BINS_PER_FEATURE];

        if rows.len() < PARALLEL_HISTOGRAM_MIN_ROWS {
            for (bins, &feature) in histogram.chunks_mut(BINS_PER_FEATURE).zip(self.features) {
                self.add_rows_to_bins(rows, feature, bins);
            }
            return histogram;
        }

        // Few features cannot keep all cores busy, so each feature is also split by rows. The
        // number of row chunks is a constant, not the thread count, and the partial histograms are
        // added in chunk order, so the float sums do not depend on the machine.
        let chunk_len = rows.len().div_ceil(HISTOGRAM_ROW_CHUNKS);
        let chunks: Vec<&[u32]> = rows.chunks(chunk_len).collect();
        let partials: Vec<Vec<Stats>> = (0..chunks.len() * n_slots)
            .into_par_iter()
            .map(|task| {
                let (chunk, slot) = (task / n_slots, task % n_slots);
                let mut bins = vec![Stats::default(); BINS_PER_FEATURE];
                self.add_rows_to_bins(chunks[chunk], self.features[slot], &mut bins);
                bins
            })
            .collect();
        for (slot, bins) in histogram.chunks_mut(BINS_PER_FEATURE).enumerate() {
            for chunk in 0..chunks.len() {
                for (bin, partial) in bins.iter_mut().zip(&partials[chunk * n_slots + slot]) {
                    *bin = bin.plus(*partial);
                }
            }
        }
        histogram
    }

    fn add_rows_to_bins(&self, rows: &[u32], feature: usize, bins: &mut [Stats]) {
        let column = self.binned.column(feature);
        for &row in rows {
            let row = row as usize;
            let [grad, hess] = self.gradients[row];
            let bin = &mut bins[column[row] as usize];
            bin.grad += f64::from(grad);
            bin.hess += f64::from(hess);
            bin.count += 1;
        }
    }

    /// Scans every feature's histogram and returns the split with the largest gain, if any
    /// split satisfies the child size and hessian limits.
    fn find_best_split(&self, histogram: &Histogram, total: Stats) -> Option<SplitInfo> {
        // A leaf too small to give both children `min_child_samples` rows cannot split.
        if (total.count as usize) < 2 * self.params.min_child_samples {
            return None;
        }
        let parent_score = total.score();
        let mut best: Option<SplitInfo> = None;

        for (slot, &feature) in self.features.iter().enumerate() {
            let bins = &histogram[slot * BINS_PER_FEATURE..(slot + 1) * BINS_PER_FEATURE];
            let missing = bins[MISSING_BIN];
            let mut left_without_missing = Stats::default();

            for (bin, &bin_stats) in bins.iter().enumerate().take(self.binned.n_bins(feature)) {
                left_without_missing = left_without_missing.plus(bin_stats);

                // Missing values go right, then (only if there are any) left. Trying both
                // lets the split learn which side suits NaN.
                let candidates = [
                    (left_without_missing, false),
                    (left_without_missing.plus(missing), true),
                ];
                let n_candidates = if missing.count > 0 { 2 } else { 1 };
                for &(left, default_left) in &candidates[..n_candidates] {
                    let right = total.minus(left);
                    if !self.sides_are_valid(left, right) {
                        continue;
                    }
                    let gain = left.score() + right.score() - parent_score;
                    let beats_best = best.as_ref().map_or(gain > 0.0, |b| gain > b.gain);
                    if beats_best {
                        // With no NaN seen here, send future NaN to the larger side.
                        let default_left = if missing.count > 0 {
                            default_left
                        } else {
                            left.count >= right.count
                        };
                        best = Some(SplitInfo {
                            gain,
                            feature,
                            bin,
                            default_left,
                            left,
                            right,
                        });
                    }
                }
            }
        }
        best
    }

    fn sides_are_valid(&self, left: Stats, right: Stats) -> bool {
        let min_rows = self.params.min_child_samples as u32;
        let min_hess = self.params.min_sum_hessian;
        left.count >= min_rows
            && right.count >= min_rows
            && left.hess >= min_hess
            && right.hess >= min_hess
    }

    /// Moves rows going left to the front, keeping their order, and returns how many there are.
    fn partition_rows(&self, rows: &mut [u32], scratch: &mut Vec<u32>, split: &SplitInfo) -> usize {
        let column = self.binned.column(split.feature);
        let goes_left = |row: u32| {
            let bin = column[row as usize] as usize;
            if bin == MISSING_BIN {
                split.default_left
            } else {
                bin <= split.bin
            }
        };

        let n_left = if rows.len() >= PARALLEL_PARTITION_MIN_ROWS {
            // Split each chunk in parallel, then stitch the pieces together in chunk order so the
            // result equals the serial one.
            let pieces: Vec<(Vec<u32>, Vec<u32>)> = rows
                .par_chunks(PARTITION_CHUNK_ROWS)
                .map(|chunk| chunk.iter().partition(|&&row| goes_left(row)))
                .collect();
            let mut n_left = 0;
            scratch.clear();
            for (left, right) in &pieces {
                rows[n_left..n_left + left.len()].copy_from_slice(left);
                n_left += left.len();
                scratch.extend_from_slice(right);
            }
            n_left
        } else {
            scratch.clear();
            let mut n_left = 0;
            for i in 0..rows.len() {
                let row = rows[i];
                if goes_left(row) {
                    rows[n_left] = row;
                    n_left += 1;
                } else {
                    scratch.push(row);
                }
            }
            n_left
        };
        rows[n_left..].copy_from_slice(scratch);
        debug_assert_eq!(n_left, split.left.count as usize);
        n_left
    }
}

/// The leaf with the largest split gain; the lowest leaf id wins ties so growth is reproducible.
fn best_leaf_to_split(leaves: &[Leaf]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (id, leaf) in leaves.iter().enumerate() {
        if let Some(split) = &leaf.best_split {
            if best.is_none_or(|(_, gain)| split.gain > gain) {
                best = Some((id, split.gain));
            }
        }
    }
    best.map(|(id, _)| id)
}

// ---------------------------------------------------------------------------------------------
// Boosting loop state
// ---------------------------------------------------------------------------------------------

/// The queries (and their rows) used to grow the current tree.
struct Bag {
    in_bag: Vec<bool>,
    rows: Vec<u32>,
}

struct Trainer<'a> {
    data: &'a Dataset<'a>,
    params: &'a Params,
    binned: BinnedData,
    queries: Vec<Range<usize>>,
    inverse_max_dcg: Vec<f64>,
    discounts: Vec<f64>,
    scores: Vec<f64>,
    grad: Vec<f32>,
    hess: Vec<f32>,
    bag: Bag,
    rng: ChaCha8Rng,
    /// Row buffer reused by every tree; leaves own contiguous slices of it.
    row_buffer: Vec<u32>,
    partition_scratch: Vec<u32>,
}

impl<'a> Trainer<'a> {
    fn new(data: &'a Dataset<'a>, params: &'a Params) -> Self {
        let started = Instant::now();
        let binned = BinnedData::build(data, params.max_bin);
        let queries = query_ranges(data.group_sizes);
        let max_query_len = data.group_sizes.iter().copied().max().unwrap_or(0);
        let discounts = (0..max_query_len).map(rank_discount).collect();
        let inverse_max_dcg = queries
            .iter()
            .map(|range| inverse_max_dcg(&data.labels[range.clone()], params.truncation_level))
            .collect();
        let n_rows = data.n_rows();
        let all_in_bag = Bag {
            in_bag: vec![true; queries.len()],
            rows: (0..n_rows as u32).collect(),
        };
        if std::env::var_os("LAMBDAMART_VERBOSE").is_some() {
            eprintln!(
                "lambdamart: binned {} rows in {:.2?}",
                n_rows,
                started.elapsed()
            );
        }
        Self {
            data,
            params,
            binned,
            queries,
            inverse_max_dcg,
            discounts,
            scores: vec![0.0; n_rows],
            grad: vec![0.0; n_rows],
            hess: vec![0.0; n_rows],
            bag: all_in_bag,
            rng: ChaCha8Rng::seed_from_u64(params.seed),
            row_buffer: Vec::new(),
            partition_scratch: Vec::new(),
        }
    }

    /// One boosting round. Returns `None` when no split with positive gain exists.
    fn boost_one_tree(&mut self, iteration: usize) -> Option<Tree> {
        let params = self.params;
        if params.subsample < 1.0
            && params.subsample_freq > 0
            && iteration.is_multiple_of(params.subsample_freq)
        {
            self.resample_bag();
        }
        self.compute_gradients();

        let gradients: Vec<[f32; 2]> = self
            .grad
            .par_iter()
            .zip(&self.hess)
            .map(|(&grad, &hess)| [grad, hess])
            .collect();
        let features = self.sample_features();
        self.row_buffer.clear();
        self.row_buffer.extend_from_slice(&self.bag.rows);

        let grower = TreeGrower {
            binned: &self.binned,
            gradients: &gradients,
            params,
            features: &features,
        };
        let (tree, leaf_ranges) = grower.grow(&mut self.row_buffer, &mut self.partition_scratch);
        if tree.nodes.is_empty() {
            return None;
        }

        // In-bag rows already know their leaf; rows of left-out queries walk the tree.
        for (range, value) in leaf_ranges.iter().zip(&tree.leaf_values) {
            for &row in &self.row_buffer[range.clone()] {
                self.scores[row as usize] += f64::from(*value);
            }
        }
        self.update_out_of_bag_scores(&tree);
        Some(tree)
    }

    fn resample_bag(&mut self) {
        let n_queries = self.queries.len();
        let n_bagged =
            ((n_queries as f64 * self.params.subsample).round() as usize).clamp(1, n_queries);
        let mut chosen = sample(&mut self.rng, n_queries, n_bagged).into_vec();
        chosen.sort_unstable();

        let mut in_bag = vec![false; n_queries];
        let mut rows = Vec::new();
        for &query in &chosen {
            in_bag[query] = true;
            rows.extend(self.queries[query].clone().map(|row| row as u32));
        }
        self.bag = Bag { in_bag, rows };
    }

    fn sample_features(&mut self) -> Vec<usize> {
        let n_features = self.binned.n_features();
        let n_chosen = ((n_features as f64 * self.params.colsample_bytree).round() as usize)
            .clamp(1, n_features);
        let mut chosen = sample(&mut self.rng, n_features, n_chosen).into_vec();
        chosen.sort_unstable();
        chosen
    }

    fn update_out_of_bag_scores(&mut self, tree: &Tree) {
        if self.bag.in_bag.iter().all(|&in_bag| in_bag) {
            return;
        }
        let n_features = self.data.n_features;
        let out_of_bag_rows: Vec<usize> = self
            .queries
            .iter()
            .zip(&self.bag.in_bag)
            .filter(|(_, &in_bag)| !in_bag)
            .flat_map(|(range, _)| range.clone())
            .collect();
        let increments: Vec<f32> = out_of_bag_rows
            .par_iter()
            .map(|&row| {
                tree.predict_row(&self.data.features[row * n_features..(row + 1) * n_features])
            })
            .collect();
        for (&row, increment) in out_of_bag_rows.iter().zip(increments) {
            self.scores[row] += f64::from(increment);
        }
    }

    /// Fills `grad` and `hess` for the bagged queries. Each query is independent, so queries run
    /// in parallel and no float sum crosses query boundaries.
    fn compute_gradients(&mut self) {
        // Hand each query its own mutable slice of the gradient arrays.
        let mut jobs = Vec::with_capacity(self.queries.len());
        let (mut grad_rest, mut hess_rest) = (&mut self.grad[..], &mut self.hess[..]);
        for (query, range) in self.queries.iter().enumerate() {
            let (grad, grad_tail) = grad_rest.split_at_mut(range.len());
            let (hess, hess_tail) = hess_rest.split_at_mut(range.len());
            grad_rest = grad_tail;
            hess_rest = hess_tail;
            if self.bag.in_bag[query] {
                jobs.push((range.clone(), self.inverse_max_dcg[query], grad, hess));
            }
        }

        let (scores, labels) = (&self.scores, self.data.labels);
        let (discounts, params) = (&self.discounts, self.params);
        jobs.into_par_iter().for_each_init(
            QueryScratch::default,
            |scratch, (range, inverse_max_dcg, grad, hess)| {
                lambdarank_gradients_for_query(
                    &scores[range.clone()],
                    &labels[range],
                    inverse_max_dcg,
                    discounts,
                    params,
                    scratch,
                    grad,
                    hess,
                );
            },
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Lambdarank gradients (port of LightGBM's LambdarankNDCG::GetGradientsForOneQuery)
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct QueryScratch {
    /// (score, row within the query), sorted best first.
    ranked: Vec<(f64, u32)>,
    ranked_labels: Vec<u8>,
    lambdas: Vec<f64>,
    hessians: Vec<f64>,
}

fn label_gain(label: u8) -> f64 {
    ((1u64 << label) - 1) as f64
}

fn rank_discount(rank: usize) -> f64 {
    1.0 / (2.0 + rank as f64).log2()
}

/// `1 / maxDCG@truncation`, or 0 when the query has no relevant item (it then gives no gradient).
fn inverse_max_dcg(labels: &[u8], truncation: usize) -> f64 {
    let mut sorted = labels.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let max_dcg: f64 = sorted
        .iter()
        .take(truncation)
        .enumerate()
        .map(|(rank, &label)| label_gain(label) * rank_discount(rank))
        .sum();
    if max_dcg > 0.0 {
        1.0 / max_dcg
    } else {
        0.0
    }
}

#[allow(clippy::too_many_arguments)]
fn lambdarank_gradients_for_query(
    scores: &[f64],
    labels: &[u8],
    inverse_max_dcg: f64,
    discounts: &[f64],
    params: &Params,
    scratch: &mut QueryScratch,
    grad: &mut [f32],
    hess: &mut [f32],
) {
    grad.fill(0.0);
    hess.fill(0.0);
    let n = scores.len();
    if inverse_max_dcg == 0.0 || n < 2 {
        return;
    }

    // Rank items by current score, best first. Ties keep row order, like a stable sort.
    let ranked = &mut scratch.ranked;
    ranked.clear();
    ranked.extend(
        scores
            .iter()
            .enumerate()
            .map(|(row, &score)| (score, row as u32)),
    );
    ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    // Contiguous labels in rank order keep the pair loop below out of random memory.
    let ranked_labels = &mut scratch.ranked_labels;
    ranked_labels.clear();
    ranked_labels.extend(ranked.iter().map(|&(_, row)| labels[row as usize]));
    let lambdas = &mut scratch.lambdas;
    let hessians = &mut scratch.hessians;
    lambdas.clear();
    lambdas.resize(n, 0.0);
    hessians.clear();
    hessians.resize(n, 0.0);

    let best_score = ranked[0].0;
    let worst_score = ranked[n - 1].0;
    let normalize = best_score != worst_score;
    let sigmoid = params.sigmoid;
    let mut sum_lambdas = 0.0;

    // Only pairs whose better-ranked item is inside the truncation level matter for NDCG@trunc.
    for i in 0..(n - 1).min(params.truncation_level) {
        let label_i = ranked_labels[i];
        for j in i + 1..n {
            let label_j = ranked_labels[j];
            if label_i == label_j {
                continue;
            }
            // `high` is the item with the larger label, whichever rank it holds now.
            let (high_rank, low_rank) = if label_i > label_j { (i, j) } else { (j, i) };
            let (high_score, low_score) = (ranked[high_rank].0, ranked[low_rank].0);
            let (high, low) = (ranked[high_rank].1 as usize, ranked[low_rank].1 as usize);

            let score_gap = high_score - low_score;
            let gain_gap =
                label_gain(ranked_labels[high_rank]) - label_gain(ranked_labels[low_rank]);
            let discount_gap = (discounts[high_rank] - discounts[low_rank]).abs();
            let mut delta_ndcg = gain_gap * discount_gap * inverse_max_dcg;
            if normalize {
                // Damp pairs the model already separates widely, as LightGBM does.
                delta_ndcg /= 0.01 + score_gap.abs();
            }

            let p = 1.0 / (1.0 + (sigmoid * score_gap).exp());
            let lambda = -sigmoid * delta_ndcg * p;
            let hessian = sigmoid * sigmoid * delta_ndcg * p * (1.0 - p);
            lambdas[high] += lambda;
            lambdas[low] -= lambda;
            hessians[high] += hessian;
            hessians[low] += hessian;
            sum_lambdas -= 2.0 * lambda;
        }
    }

    // Queries with large total lambda would dominate the trees; LightGBM rescales them.
    let scale = if normalize && sum_lambdas > 0.0 {
        (1.0 + sum_lambdas).log2() / sum_lambdas
    } else {
        1.0
    };
    for row in 0..n {
        grad[row] = (lambdas[row] * scale) as f32;
        hess[row] = (hessians[row] * scale) as f32;
    }
}

// ---------------------------------------------------------------------------------------------
// NDCG
// ---------------------------------------------------------------------------------------------

fn query_ranges(group_sizes: &[usize]) -> Vec<Range<usize>> {
    let mut start = 0;
    group_sizes
        .iter()
        .map(|&size| {
            let range = start..start + size;
            start += size;
            range
        })
        .collect()
}

fn mean_ndcg(scores: &[f64], labels: &[u8], queries: &[Range<usize>], k: usize) -> f64 {
    let per_query: Vec<f64> = queries
        .par_iter()
        .map(|range| query_ndcg(&scores[range.clone()], &labels[range.clone()], k))
        .collect();
    per_query.iter().sum::<f64>() / per_query.len().max(1) as f64
}

fn query_ndcg<S: Copy + Into<f64>>(scores: &[S], labels: &[u8], k: usize) -> f64 {
    let inverse_ideal = inverse_max_dcg(labels, k);
    if inverse_ideal == 0.0 {
        return 1.0; // nothing relevant to find, so any order is perfect (LightGBM's convention)
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].into().total_cmp(&scores[a].into()));
    let dcg: f64 = order
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, &item)| label_gain(labels[item]) * rank_discount(rank))
        .sum();
    dcg * inverse_ideal
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    struct Synthetic {
        features: Vec<f32>,
        labels: Vec<u8>,
        group_sizes: Vec<usize>,
        n_features: usize,
    }

    impl Synthetic {
        fn dataset(&self) -> Dataset<'_> {
            Dataset {
                features: &self.features,
                n_features: self.n_features,
                labels: &self.labels,
                group_sizes: &self.group_sizes,
            }
        }
    }

    fn gaussian(rng: &mut ChaCha8Rng) -> f32 {
        let u1 = 1.0 - rng.random::<f32>();
        let u2 = rng.random::<f32>();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Each query's best `positives` candidates by a latent score are the relevant ones. The
    /// latent score is feature 0 + feature 1 + noise; the other features are pure noise.
    fn make_synthetic(
        n_queries: usize,
        candidates: usize,
        n_features: usize,
        positives: usize,
        seed: u64,
    ) -> Synthetic {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut features = Vec::with_capacity(n_queries * candidates * n_features);
        let mut labels = Vec::with_capacity(n_queries * candidates);
        for _ in 0..n_queries {
            let mut latent = Vec::with_capacity(candidates);
            for _ in 0..candidates {
                let row: Vec<f32> = (0..n_features).map(|_| gaussian(&mut rng)).collect();
                latent.push(row[0] + row[1] + 0.5 * gaussian(&mut rng));
                features.extend(row);
            }
            let mut order: Vec<usize> = (0..candidates).collect();
            order.sort_by(|&a, &b| latent[b].total_cmp(&latent[a]));
            let mut query_labels = vec![0u8; candidates];
            for &item in order.iter().take(positives) {
                query_labels[item] = 1;
            }
            labels.extend(query_labels);
        }
        Synthetic {
            features,
            labels,
            group_sizes: vec![candidates; n_queries],
            n_features,
        }
    }

    fn test_params() -> Params {
        Params {
            n_estimators: 60,
            learning_rate: 0.1,
            num_leaves: 15,
            min_child_samples: 20,
            ..Params::default()
        }
    }

    fn column(data: &Synthetic, feature: usize) -> Vec<f32> {
        data.features
            .chunks(data.n_features)
            .map(|row| row[feature])
            .collect()
    }

    #[test]
    fn ndcg_matches_hand_computation() {
        // Ranked order: item0 (relevant), item1, item2 (relevant).
        // DCG = 1/log2(2) + 1/log2(4) = 1.5; ideal = 1/log2(2) + 1/log2(3).
        let scores = [0.9_f32, 0.8, 0.1];
        let labels = [1u8, 0, 1];
        let expected = 1.5 / (1.0 + 1.0 / 3.0_f64.log2());
        let got = ndcg_at_k(&scores, &labels, &[3], 3);
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");

        // At k = 1 only the first position counts: DCG 1 over ideal 1.
        assert!((ndcg_at_k(&scores, &labels, &[3], 1) - 1.0).abs() < 1e-12);

        // A second query without relevant items counts as 1, so the mean is (expected + 1) / 2.
        let two = ndcg_at_k(&[0.9_f32, 0.8, 0.1, 0.5, 0.4], &[1, 0, 1, 0, 0], &[3, 2], 3);
        assert!((two - (expected + 1.0) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn beats_random_and_single_feature_baseline() {
        let train_set = make_synthetic(400, 50, 6, 2, 1);
        let valid_set = make_synthetic(200, 50, 6, 2, 2);
        let model = train(
            &train_set.dataset(),
            &valid_set.dataset(),
            &test_params(),
            20,
            10,
        );

        let scores = model.predict(&valid_set.features, valid_set.n_features);
        let model_ndcg = ndcg_at_k(&scores, &valid_set.labels, &valid_set.group_sizes, 10);

        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let random_scores: Vec<f32> = (0..valid_set.labels.len()).map(|_| rng.random()).collect();
        let random_ndcg = ndcg_at_k(
            &random_scores,
            &valid_set.labels,
            &valid_set.group_sizes,
            10,
        );
        let feature0_ndcg = ndcg_at_k(
            &column(&valid_set, 0),
            &valid_set.labels,
            &valid_set.group_sizes,
            10,
        );

        println!("model {model_ndcg:.4}, feature0 {feature0_ndcg:.4}, random {random_ndcg:.4}");
        assert!(
            model_ndcg > random_ndcg + 0.3,
            "model {model_ndcg} random {random_ndcg}"
        );
        assert!(
            model_ndcg > feature0_ndcg + 0.05,
            "model {model_ndcg} feature0 {feature0_ndcg}"
        );
    }

    #[test]
    fn missing_values_that_mark_positives_get_large_gain() {
        let mut data = make_synthetic(300, 40, 4, 1, 3);
        // Feature 0 is NaN exactly for the positives, as a retriever rank would be if only
        // the positives were proposed by it.
        let n_features = data.n_features;
        for (row, &label) in data.labels.iter().enumerate() {
            if label == 1 {
                data.features[row * n_features] = f32::NAN;
            }
        }
        let params = Params {
            min_child_samples: 5,
            ..test_params()
        };
        let model = train(&data.dataset(), &data.dataset(), &params, 0, 10);

        let importance = model.feature_importance_gain();
        let share = importance[0] / importance.iter().sum::<f64>();
        let scores = model.predict(&data.features, n_features);
        let ndcg = ndcg_at_k(&scores, &data.labels, &data.group_sizes, 10);
        println!("NaN feature gain share {share:.3}, ndcg {ndcg:.4}");
        assert!(share > 0.8, "gain share of the NaN feature was {share}");
        assert!(ndcg > 0.99, "ndcg {ndcg}");
    }

    #[test]
    fn same_seed_gives_identical_predictions() {
        let train_set = make_synthetic(150, 30, 5, 1, 4);
        let valid_set = make_synthetic(50, 30, 5, 1, 5);
        let params = Params {
            n_estimators: 30,
            ..test_params()
        };
        let first = train(&train_set.dataset(), &valid_set.dataset(), &params, 10, 10);
        let second = train(&train_set.dataset(), &valid_set.dataset(), &params, 10, 10);
        let a = first.predict(&valid_set.features, valid_set.n_features);
        let b = second.predict(&valid_set.features, valid_set.n_features);
        assert_eq!(a, b);

        let other_seed = Params { seed: 7, ..params };
        let third = train(
            &train_set.dataset(),
            &valid_set.dataset(),
            &other_seed,
            10,
            10,
        );
        assert_ne!(a, third.predict(&valid_set.features, valid_set.n_features));
    }

    #[test]
    fn early_stopping_keeps_only_best_iteration_trees() {
        // Labels unrelated to the features: validation NDCG peaks early and then falls.
        let mut rng = ChaCha8Rng::seed_from_u64(6);
        let mut train_set = make_synthetic(100, 30, 4, 1, 7);
        let mut valid_set = make_synthetic(100, 30, 4, 1, 8);
        for set in [&mut train_set, &mut valid_set] {
            for label in set.labels.iter_mut() {
                *label = u8::from(rng.random::<f32>() < 0.05);
            }
        }
        let params = Params {
            n_estimators: 200,
            learning_rate: 0.3,
            ..test_params()
        };
        let model = train(&train_set.dataset(), &valid_set.dataset(), &params, 5, 10);

        assert!(model.best_iteration() <= params.n_estimators);
        assert!(
            model.best_iteration() < params.n_estimators,
            "noise should stop early"
        );
        assert_eq!(model.trees.len(), model.best_iteration());

        let row = &valid_set.features[..valid_set.n_features];
        let manual: f64 = model
            .trees
            .iter()
            .map(|tree| f64::from(tree.predict_row(row)))
            .sum();
        assert_eq!(model.predict_row(row), manual as f32);
    }

    #[test]
    fn serde_round_trip_keeps_predictions() {
        let train_set = make_synthetic(100, 30, 5, 1, 9);
        let valid_set = make_synthetic(40, 30, 5, 1, 10);
        let model = train(
            &train_set.dataset(),
            &valid_set.dataset(),
            &test_params(),
            10,
            10,
        );

        let json = serde_json::to_string(&model).unwrap();
        let restored: Model = serde_json::from_str(&json).unwrap();
        assert_eq!(
            model.predict(&valid_set.features, valid_set.n_features),
            restored.predict(&valid_set.features, valid_set.n_features)
        );
        assert_eq!(model.best_iteration(), restored.best_iteration());
    }

    /// Run with `cargo test --release lambdamart -- --ignored --nocapture throughput`.
    #[test]
    #[ignore]
    fn throughput_on_10k_queries_of_400_candidates() {
        let n_trees: usize = std::env::var("LAMBDAMART_TREES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let started = Instant::now();
        let train_set = make_synthetic(10_000, 400, 15, 1, 11);
        let valid_set = make_synthetic(1_000, 400, 15, 1, 12);
        println!("generated data in {:.1?}", started.elapsed());

        let params = Params {
            n_estimators: n_trees,
            ..Params::default()
        };
        let started = Instant::now();
        let model = train(&train_set.dataset(), &valid_set.dataset(), &params, 0, 10);
        let elapsed = started.elapsed();
        let scores = model.predict(&valid_set.features, valid_set.n_features);
        let ndcg = ndcg_at_k(&scores, &valid_set.labels, &valid_set.group_sizes, 10);
        println!(
            "{} trees on {} rows in {:.1?} ({:.0} ms per tree), valid ndcg@10 {:.4}",
            model.best_iteration(),
            train_set.labels.len(),
            elapsed,
            elapsed.as_secs_f64() * 1000.0 / n_trees as f64,
            ndcg
        );
    }
}
