//! Train everything the online service needs and save it as one file.
//!
//! Takes about as long as `ranking` because it fits the retrievers twice (once for training the
//! ranker, once on all history for serving).

use std::path::PathBuf;

use anyhow::Result;

use super::{prepare, CommonArgs};
use crate::serving::bundle::build_bundle;
use crate::store::save;

#[derive(clap::Args)]
pub struct Args {
    #[command(flatten)]
    pub common: CommonArgs,
    #[arg(long, default_value = "artifacts/serving_bundle.bin")]
    pub output: PathBuf,
}

pub fn run(args: &Args) -> Result<()> {
    let prepared = prepare(&args.common)?;
    let bundle = build_bundle(
        &prepared.config,
        &prepared.split,
        prepared.data.n_users,
        prepared.data.n_items,
    )?;
    save(&args.output, &bundle)?;
    println!(
        "wrote {} (version {})",
        args.output.display(),
        bundle.version
    );
    Ok(())
}
