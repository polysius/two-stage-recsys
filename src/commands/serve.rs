//! Run the recommendation service. `--redis-url` (or REDIS_URL) turns candidate caching on.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::config::Config;
use crate::serving::api::router;
use crate::serving::bundle::ServingBundle;
use crate::serving::cache::Cache;
use crate::serving::recommender::TwoStageRecommender;
use crate::store::load;

#[derive(clap::Args)]
pub struct Args {
    #[arg(long, env = "CONFIG_PATH", default_value = "configs/default.yaml")]
    pub config: PathBuf,
    #[arg(
        long,
        env = "BUNDLE_PATH",
        default_value = "artifacts/serving_bundle.bin"
    )]
    pub bundle: PathBuf,
    #[arg(long, env = "REDIS_URL")]
    pub redis_url: Option<String>,
    #[arg(long, default_value_t = 8000)]
    pub port: u16,
}

#[tokio::main]
pub async fn run(args: &Args) -> Result<()> {
    let config = Config::load(&args.config)?;
    let bundle: ServingBundle = load(&args.bundle)?;
    let cache = match &args.redis_url {
        Some(url) => Cache::redis(
            url,
            &bundle.version,
            bundle.retrievers.models.len(),
            bundle.scorer_names.len(),
            config.serving.cache_ttl_seconds,
        )
        .context("invalid redis url")?,
        None => Cache::Off,
    };

    let app = router(Arc::new(TwoStageRecommender::new(bundle, cache)));
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", args.port)).await?;
    println!("serving on port {}", args.port);
    axum::serve(listener, app).await?;
    Ok(())
}
