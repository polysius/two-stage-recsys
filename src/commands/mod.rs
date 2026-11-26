//! One module per command line subcommand, plus the setup they share.

pub mod baselines;
pub mod benchmark;
pub mod bundle;
pub mod download_data;
pub mod experiment;
pub mod ranking;
pub mod retrieval;
pub mod serve;

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::config::Config;
use crate::metrics::{metric_names, Metrics};
use crate::report::Cell;
use crate::table::{
    assert_no_time_leakage, load_interactions, time_split, Interactions, Split, Table,
};

/// Where the config and data live, as given on the command line.
#[derive(clap::Args, Clone)]
pub struct CommonArgs {
    #[arg(long, default_value = "configs/default.yaml")]
    pub config: PathBuf,
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,
}

pub struct Prepared {
    pub config: Config,
    pub data: Interactions,
    pub split: Split,
}

impl Prepared {
    /// Train plus validation: what a deployed system knows before the test window starts.
    pub fn test_history(&self) -> Table {
        Table::concat(&self.split.train, &self.split.val)
    }
}

/// Loads the config and ratings, splits by time and checks that the split leaks nothing.
pub fn prepare(args: &CommonArgs) -> Result<Prepared> {
    let config = Config::load(&args.config)?;
    let csv_path: PathBuf = Path::new(&args.data_dir)
        .join("raw")
        .join(format!("{}.csv", config.data.category));
    let data = load_interactions(&csv_path)?;
    let split = time_split(
        &data.table,
        config.split.val_fraction,
        config.split.test_fraction,
    );
    assert_no_time_leakage(&split)?;
    println!(
        "train={} val={} test={} users={} items={} dropped_unseen_share={:.3} \
         unknown_user_share={:.3} unknown_item_share={:.3}",
        split.train.len(),
        split.val.len(),
        split.test.len(),
        data.n_users,
        data.n_items,
        split.dropped_unseen_share,
        split.unknown_user_share,
        split.unknown_item_share,
    );
    Ok(Prepared {
        config,
        data,
        split,
    })
}

/// Metric cells in report order, for a row that already starts with its label cells.
pub fn metric_cells(metrics: &Metrics, k_values: &[usize]) -> Vec<Cell> {
    metric_names(k_values)
        .iter()
        .map(|name| Cell::Number(metrics[name]))
        .collect()
}

pub fn metric_headers(k_values: &[usize]) -> Vec<String> {
    metric_names(k_values)
}
