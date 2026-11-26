use anyhow::Result;
use clap::{Parser, Subcommand};

use recsys::commands::{
    baselines, benchmark, bundle, download_data, experiment, ranking, retrieval, serve,
};

#[derive(Parser)]
#[command(about = "Two-stage recommender: retrieval, ranking, A/B analysis and serving")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download the ratings file and check its checksum.
    DownloadData(download_data::Args),
    /// Score the popularity and item-item baselines.
    Baselines(baselines::Args),
    /// Score every retriever against the baselines.
    Retrieval(retrieval::Args),
    /// Train the ranker and compare it with retrieval alone.
    Ranking(ranking::Args),
    /// Simulate an A/B test between the control and the ranker.
    Experiment(experiment::Args),
    /// Train everything the service needs and save it as one file.
    Bundle(bundle::Args),
    /// Run the recommendation service.
    Serve(serve::Args),
    /// Measure request latency against the budgets in the config.
    Benchmark(benchmark::Args),
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::DownloadData(args) => download_data::run(&args),
        Command::Baselines(args) => baselines::run(&args),
        Command::Retrieval(args) => retrieval::run(&args),
        Command::Ranking(args) => ranking::run(&args),
        Command::Experiment(args) => experiment::run(&args),
        Command::Bundle(args) => bundle::run(&args),
        Command::Serve(args) => serve::run(&args),
        Command::Benchmark(args) => benchmark::run(&args),
    }
}
