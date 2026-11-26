//! Download the Amazon Reviews 2023 5-core ratings for one category and verify its checksum.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

const BASE_URL: &str = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core/rating_only";

/// Pinned hashes of files this project was developed against. Other categories are only recorded.
const EXPECTED_SHA256: [(&str, &str); 1] = [(
    "Video_Games",
    "2fb8fd6fb8b93a1a555e1aa4d46e3327e2bf345e9fb02a3bea184a9bce2ba4fa",
)];

#[derive(clap::Args)]
pub struct Args {
    #[arg(long, default_value = "Video_Games")]
    pub category: String,
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,
}

fn sha256_of(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(&mut std::fs::File::open(path)?, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Downloads to a temporary name so an interrupted transfer never leaves a partial file.
#[tokio::main]
async fn download(category: &str, target: &Path) -> Result<()> {
    let partial = target.with_extension("part");
    let mut response = reqwest::get(format!("{BASE_URL}/{category}.csv"))
        .await?
        .error_for_status()?;
    let mut file = std::fs::File::create(&partial)?;
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk)?;
    }
    std::fs::rename(partial, target)?;
    Ok(())
}

pub fn run(args: &Args) -> Result<()> {
    let target = args
        .data_dir
        .join("raw")
        .join(format!("{}.csv", args.category));
    std::fs::create_dir_all(target.parent().expect("target has a parent"))?;
    if !target.exists() {
        download(&args.category, &target)?;
    }

    let checksum = sha256_of(&target)?;
    let expected = EXPECTED_SHA256
        .iter()
        .find(|(name, _)| *name == args.category);
    if let Some((_, expected)) = expected {
        if checksum != *expected {
            std::fs::remove_file(&target)?; // remove the bad file so the next run downloads again
            bail!(
                "checksum mismatch for {}: got {checksum}, expected {expected}",
                target.display()
            );
        }
    }
    let file_name = target.file_name().expect("file name").to_string_lossy();
    std::fs::write(
        target.with_extension("sha256"),
        format!("{checksum}  {file_name}\n"),
    )?;
    println!("{} sha256={checksum}", target.display());
    Ok(())
}
