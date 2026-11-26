//! Measure request latency of the running service against the budgets in the config.
//!
//! Starts its own Redis and the service, then sends requests one at a time over HTTP with a
//! keep-alive client, so each number includes serialisation and the network stack.
//!
//! Scenarios:
//!   cache miss   known users, empty cache: retrieve, rank, then store the candidates
//!   cache hit    the same users again: read candidates from Redis, rank
//!   cold start   ids with no history: the recent-popularity fallback
//!   concurrent   cache hits from several clients at once, to see how latency holds up

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use tokio::process::{Child, Command};

use crate::config::Config;
use crate::report::write_report;
use crate::serving::bundle::ServingBundle;
use crate::store::load;

const REDIS_PORT: u16 = 16390;
const SERVER_PORT: u16 = 18765;
const WARMUP_USERS: usize = 50;

#[derive(clap::Args)]
pub struct Args {
    #[arg(long, default_value = "configs/default.yaml")]
    pub config: PathBuf,
    #[arg(long, default_value = "artifacts/serving_bundle.bin")]
    pub bundle: PathBuf,
    #[arg(long, default_value = "artifacts/serving.md")]
    pub output: PathBuf,
    #[arg(long, default_value_t = 500)]
    pub requests: usize,
    #[arg(long, default_value_t = 10)]
    pub k: usize,
    #[arg(long, default_value_t = 8)]
    pub clients: usize,
}

struct Summary {
    scenario: String,
    requests: usize,
    p50: f64,
    p95: f64,
    p99: f64,
    budget_ms: Option<f64>,
}

/// Percentile with linear interpolation, the same definition `numpy.percentile` uses.
fn percentile(sorted: &[f64], percent: f64) -> f64 {
    let position = percent / 100.0 * (sorted.len() - 1) as f64;
    let (lower, upper) = (position.floor() as usize, position.ceil() as usize);
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

fn summarise(scenario: &str, mut latencies: Vec<f64>, budget_ms: Option<f64>) -> Summary {
    latencies.sort_by(f64::total_cmp);
    Summary {
        scenario: scenario.to_string(),
        requests: latencies.len(),
        p50: percentile(&latencies, 50.0),
        p95: percentile(&latencies, 95.0),
        p99: percentile(&latencies, 99.0),
        budget_ms,
    }
}

/// Latencies in ms. Fails loudly if a request was not served the way the scenario assumes.
async fn run_scenario(
    client: &reqwest::Client,
    base_url: &str,
    users: &[u32],
    k: usize,
    expected: (&str, &str),
) -> Result<Vec<f64>> {
    let mut latencies = Vec::with_capacity(users.len());
    for user in users {
        let started = Instant::now();
        let response = client
            .get(format!("{base_url}/recommend"))
            .query(&[("user_id", user.to_string()), ("k", k.to_string())])
            .send()
            .await?
            .error_for_status()?;
        let body: serde_json::Value = response.json().await?;
        latencies.push(started.elapsed().as_secs_f64() * 1000.0);
        let got = (
            body["source"].as_str().unwrap_or(""),
            body["cache"].as_str().unwrap_or(""),
        );
        if got != expected {
            bail!("user {user}: got {}/{}", got.0, got.1);
        }
    }
    Ok(latencies)
}

async fn wait_until_healthy(
    client: &reqwest::Client,
    base_url: &str,
    server: &mut Child,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(180);
    while Instant::now() < deadline {
        if let Some(status) = server.try_wait()? {
            bail!("the service exited with {status}");
        }
        if let Ok(response) = client.get(format!("{base_url}/health")).send().await {
            if response.status().is_success() {
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    bail!("the service did not become healthy")
}

#[tokio::main]
pub async fn run(args: &Args) -> Result<()> {
    let config = Config::load(&args.config)?;
    let budgets = &config.serving.latency_budget_ms;
    let bundle: ServingBundle = load(&args.bundle)?;

    let mut rng = ChaCha12Rng::seed_from_u64(0);
    let mut known: Vec<u32> = (0..bundle.n_users as u32)
        .filter(|&user| bundle.user_stats.is_known(user))
        .collect();
    known.shuffle(&mut rng);
    let warmup_users = &known[..WARMUP_USERS];
    let users: Vec<u32> = known[WARMUP_USERS..WARMUP_USERS + args.requests].to_vec();
    let unknown_users: Vec<u32> = (0..args.requests as u32)
        .map(|offset| bundle.n_users as u32 + offset)
        .collect();
    drop(bundle);

    let mut redis_server = Command::new("redis-server")
        .args([
            "--port",
            &REDIS_PORT.to_string(),
            "--save",
            "",
            "--appendonly",
            "no",
        ])
        .stdout(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;
    let mut server = Command::new(std::env::current_exe()?)
        .arg("serve")
        .args(["--port", &SERVER_PORT.to_string()])
        .arg("--config")
        .arg(&args.config)
        .arg("--bundle")
        .arg(&args.bundle)
        .args(["--redis-url", &format!("redis://127.0.0.1:{REDIS_PORT}/0")])
        .kill_on_drop(true)
        .spawn()?;

    let result = measure(
        args,
        &mut server,
        warmup_users,
        &users,
        &unknown_users,
        budgets,
    )
    .await;
    server.kill().await.ok();
    redis_server.kill().await.ok();
    let (rows, throughput) = result?;

    let mut lines = vec![
        "| scenario | requests | p50 ms | p95 ms | p99 ms | p95 budget ms | budget met |"
            .to_string(),
        "|:--|--:|--:|--:|--:|--:|:--|".to_string(),
    ];
    for row in &rows {
        let (budget, met) = match row.budget_ms {
            None => (String::new(), String::new()),
            Some(budget) => (
                budget.to_string(),
                if row.p95 <= budget { "yes" } else { "no" }.to_string(),
            ),
        };
        lines.push(format!(
            "| {} | {} | {:.1} | {:.1} | {:.1} | {budget} | {met} |",
            row.scenario, row.requests, row.p50, row.p95, row.p99
        ));
    }
    lines.push(String::new());
    lines.push(format!(
        "Concurrent throughput: {throughput:.0} requests per second."
    ));
    lines.push(format!(
        "CPU count: {}, one process, k={}.",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        args.k
    ));
    let text = lines.join("\n") + "\n";
    println!("{text}");
    write_report(&args.output, &text)
}

async fn measure(
    args: &Args,
    server: &mut Child,
    warmup_users: &[u32],
    users: &[u32],
    unknown_users: &[u32],
    budgets: &crate::config::LatencyBudgets,
) -> Result<(Vec<Summary>, f64)> {
    let base_url = format!("http://127.0.0.1:{SERVER_PORT}");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    wait_until_healthy(&client, &base_url, server).await?;
    let k = args.k;

    run_scenario(&client, &base_url, warmup_users, k, ("ranked", "miss")).await?; // warm code paths
    run_scenario(
        &client,
        &base_url,
        &unknown_users[..20],
        k,
        ("cold_start", "off"),
    )
    .await?;

    let miss = run_scenario(&client, &base_url, users, k, ("ranked", "miss")).await?;
    let hit = run_scenario(&client, &base_url, users, k, ("ranked", "hit")).await?;
    let cold = run_scenario(&client, &base_url, unknown_users, k, ("cold_start", "off")).await?;
    let mut rows = vec![
        summarise("cache miss", miss, Some(budgets.cache_miss)),
        summarise("cache hit", hit, Some(budgets.cache_hit)),
        summarise("cold start", cold, Some(budgets.cold_start)),
    ];

    let started = Instant::now();
    let chunk_size = users.len().div_ceil(args.clients);
    let tasks: Vec<_> = users
        .chunks(chunk_size)
        .map(|chunk| {
            let (base_url, chunk) = (base_url.clone(), chunk.to_vec());
            tokio::spawn(async move {
                let client = reqwest::Client::builder()
                    .timeout(Duration::from_secs(10))
                    .build()?;
                run_scenario(&client, &base_url, &chunk, k, ("ranked", "hit")).await
            })
        })
        .collect();
    let mut concurrent = Vec::new();
    for task in tasks {
        concurrent.extend(task.await??);
    }
    let throughput = concurrent.len() as f64 / started.elapsed().as_secs_f64();
    rows.push(summarise(
        &format!("cache hit, {} concurrent clients", args.clients),
        concurrent,
        None,
    ));
    Ok((rows, throughput))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_interpolates_between_neighbours() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&sorted, 50.0), 3.0);
        assert!((percentile(&sorted, 95.0) - 4.8).abs() < 1e-12);
        assert_eq!(percentile(&sorted, 100.0), 5.0);
    }
}
