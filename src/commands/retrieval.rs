//! Compare the retrievers against the baselines on the validation and test windows.

use std::path::PathBuf;

use anyhow::Result;

use super::{metric_cells, metric_headers, prepare, CommonArgs};
use crate::evaluation::score_recommender;
use crate::recommender::Recommender;
use crate::registry::{fit_retrievers, RETRIEVER_NAMES};
use crate::report::{markdown_table, write_report, Cell};

#[derive(clap::Args)]
pub struct Args {
    #[command(flatten)]
    pub common: CommonArgs,
    #[arg(long, default_value = "artifacts/retrieval.md")]
    pub output: PathBuf,
    /// Score only these models, for example `--only als,two_tower`.
    #[arg(long, value_delimiter = ',')]
    pub only: Vec<String>,
}

pub fn run(args: &Args) -> Result<()> {
    let prepared = prepare(&args.common)?;
    let (config, data, split) = (&prepared.config, &prepared.data, &prepared.split);
    let k_values = &config.eval.k_values;
    let test_history = prepared.test_history();

    let names: Vec<&'static str> = RETRIEVER_NAMES
        .into_iter()
        .filter(|name| args.only.is_empty() || args.only.iter().any(|only| only == name))
        .collect();
    let fitted = fit_retrievers(
        &names,
        config,
        data.n_users,
        data.n_items,
        &split.train,
        "train",
    );

    let mut rows = Vec::new();
    for retriever in &fitted {
        for (window, history, eval_table) in [
            ("val", &split.train, &split.val),
            ("test", &test_history, &split.test),
        ] {
            let metrics = score_recommender(
                &retriever.model as &dyn Recommender,
                history,
                eval_table,
                data.n_users,
                data.n_items,
                k_values,
            )?;
            println!("{} {window} {metrics:.4?}", retriever.name);
            let mut row: Vec<Cell> = vec![
                retriever.name.into(),
                window.into(),
                retriever.fit_seconds.into(),
            ];
            row.extend(metric_cells(&metrics, k_values));
            rows.push(row);
        }
        if !retriever.loss_history.is_empty() {
            println!(
                "{} loss per epoch {:.3?}",
                retriever.name, retriever.loss_history
            );
        }
    }

    let mut headers = vec![
        "model".to_string(),
        "window".to_string(),
        "fit_seconds".to_string(),
    ];
    headers.extend(metric_headers(k_values));
    let header_refs: Vec<&str> = headers.iter().map(String::as_str).collect();
    write_report(&args.output, &(markdown_table(&header_refs, &rows) + "\n"))
}
