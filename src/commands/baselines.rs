//! Evaluate the baselines on the validation and test windows and write a results table.

use std::path::PathBuf;

use anyhow::Result;

use super::{metric_cells, metric_headers, prepare, CommonArgs};
use crate::evaluation::score_recommender;
use crate::recommender::Recommender;
use crate::registry::fit_retrievers;
use crate::report::{markdown_table, write_report, Cell};

#[derive(clap::Args)]
pub struct Args {
    #[command(flatten)]
    pub common: CommonArgs,
    #[arg(long, default_value = "artifacts/baselines.md")]
    pub output: PathBuf,
}

pub fn run(args: &Args) -> Result<()> {
    let prepared = prepare(&args.common)?;
    let (config, data, split) = (&prepared.config, &prepared.data, &prepared.split);
    let k_values = &config.eval.k_values;
    let test_history = prepared.test_history();

    let baselines = ["popularity", "recent_popularity", "item_item_cf"];
    let fitted = fit_retrievers(
        &baselines,
        config,
        data.n_users,
        data.n_items,
        &split.train,
        "train",
    );

    let mut rows = Vec::new();
    for baseline in &fitted {
        for (window, history, eval_table) in [
            ("val", &split.train, &split.val),
            ("test", &test_history, &split.test),
        ] {
            let metrics = score_recommender(
                &baseline.model as &dyn Recommender,
                history,
                eval_table,
                data.n_users,
                data.n_items,
                k_values,
            )?;
            println!("{} {window} {metrics:.4?}", baseline.name);
            let mut row: Vec<Cell> = vec![baseline.name.into(), window.into()];
            row.extend(metric_cells(&metrics, k_values));
            rows.push(row);
        }
    }

    let mut headers = vec!["model".to_string(), "window".to_string()];
    headers.extend(metric_headers(k_values));
    let header_refs: Vec<&str> = headers.iter().map(String::as_str).collect();
    write_report(&args.output, &(markdown_table(&header_refs, &rows) + "\n"))
}
