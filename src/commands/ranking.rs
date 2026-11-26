//! Train the ranker and compare retrieval only against retrieval plus ranking.
//!
//! Protocol:
//!   1. Fit every retriever on train.
//!   2. Build candidates and features for validation users from train history, label them with
//!      the validation window, and train the ranker on them (early stopping on held-out users).
//!   3. Refit every retriever on train + validation, build candidates and features for test
//!      users from that history, and score them with the ranker. The test window is never used
//!      before this step.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;

use super::{prepare, CommonArgs};
use crate::evaluation::{item_popularity, relevant_items_by_user, score_recommender};
use crate::metrics::{evaluate, Metrics};
use crate::ranking::features::GROUPS;
use crate::ranking::ranker::{top_k_per_user, train_ranker, TrainedRanker};
use crate::ranking::{build_stage, Stage};
use crate::recommender::Recommender;
use crate::registry::{fit_retrievers, RetrieverSet, RETRIEVER_NAMES};
use crate::report::{markdown_table, write_report, Cell};
use crate::sparse::BinaryCsr;
use crate::store::{save, TestRecommendations};
use crate::table::Table;

const REPORT_METRICS: [&str; 4] = ["recall@10", "ndcg@10", "mrr", "recall@100"];

#[derive(clap::Args)]
pub struct Args {
    #[command(flatten)]
    pub common: CommonArgs,
    #[arg(long, default_value = "artifacts/ranking.md")]
    pub output: PathBuf,
    #[arg(long, default_value = "artifacts/test_recommendations.bin")]
    pub recommendations: PathBuf,
}

/// A table row: the label, then the reported metrics.
fn report_row(label: &str, metrics: &Metrics) -> Vec<Cell> {
    let mut row: Vec<Cell> = vec![label.into()];
    row.extend(
        REPORT_METRICS
            .iter()
            .map(|name| Cell::Number(metrics[*name])),
    );
    row
}

fn rounded_report(metrics: &Metrics) -> String {
    REPORT_METRICS
        .iter()
        .map(|name| format!("{name}={:.4}", metrics[*name]))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn run(args: &Args) -> Result<()> {
    let prepared = prepare(&args.common)?;
    let (config, data, split) = (&prepared.config, &prepared.data, &prepared.split);
    let ranking = &config.ranking;
    let k_values = &config.eval.k_values;
    let max_k = *k_values.iter().max().expect("at least one k");

    let retrievers = RetrieverSet::from_fitted(fit_retrievers(
        &RETRIEVER_NAMES,
        config,
        data.n_users,
        data.n_items,
        &split.train,
        "train",
    ));
    // Before the test window a deployed system would refit on the history it has, so the test
    // stage uses retrievers fit on train plus validation. The ranker still trains on candidates
    // from the train-only retrievers, which were equally fresh for the validation window.
    let test_history = prepared.test_history();
    let test_retrievers = RetrieverSet::from_fitted(fit_retrievers(
        &RETRIEVER_NAMES,
        config,
        data.n_users,
        data.n_items,
        &test_history,
        "train + validation",
    ));
    let per_retriever = ranking.candidates_per_retriever;
    let train_stage = build_stage(
        &retrievers,
        &split.train,
        &split.val,
        data.n_users,
        data.n_items,
        per_retriever,
    )?;
    let test_stage = build_stage(
        &test_retrievers,
        &test_history,
        &split.test,
        data.n_users,
        data.n_items,
        per_retriever,
    )?;

    let fit_and_score = |groups: Option<&[&str]>| -> Result<(TrainedRanker, Metrics)> {
        let columns = train_stage.features.columns_in_groups(groups);
        let ranker = train_ranker(
            &train_stage.features,
            &train_stage.frame.group_sizes,
            &train_stage.labels,
            &columns,
            ranking,
            0,
        );
        let metrics = score_ranker(
            &ranker,
            &test_stage,
            &test_history,
            &split.test,
            data.n_items,
            k_values,
        )?;
        Ok((ranker, metrics))
    };

    let mut rows: Vec<Vec<Cell>> = Vec::new();
    for (name, model) in test_retrievers.names.iter().zip(&test_retrievers.models) {
        let metrics = score_recommender(
            model as &dyn Recommender,
            &test_history,
            &split.test,
            data.n_users,
            data.n_items,
            k_values,
        )?;
        rows.push(report_row(&format!("{name} only"), &metrics));
    }
    // The M1 and M2 tables score this stale model: recent popularity fit on train only.
    let stale = score_recommender(
        retrievers.get("recent_popularity") as &dyn Recommender,
        &test_history,
        &split.test,
        data.n_users,
        data.n_items,
        k_values,
    )?;
    rows.push(report_row("recent_popularity (fit on train only)", &stale));

    let (ranker, metrics) = fit_and_score(None)?;
    save_test_recommendations(
        &args.recommendations,
        &test_retrievers,
        &ranker,
        &test_stage,
        &BinaryCsr::user_item(&test_history, data.n_users, data.n_items),
        max_k,
    )?;
    println!(
        "ranker trees {} {}",
        ranker.best_iteration(),
        rounded_report(&metrics)
    );
    rows.push(report_row("candidates + ranker", &metrics));

    let mut variants: Vec<(String, Option<Vec<&str>>)> = vec![("all features".into(), None)];
    for (group, _) in GROUPS {
        let others = GROUPS
            .iter()
            .map(|(name, _)| *name)
            .filter(|name| *name != group)
            .collect();
        variants.push((format!("without {group}"), Some(others)));
    }
    for (group, _) in GROUPS {
        variants.push((format!("only {group}"), Some(vec![group])));
    }
    let mut ablation_rows: Vec<Vec<Cell>> = Vec::new();
    for (label, groups) in &variants {
        let variant_metrics = match groups {
            None => metrics.clone(),
            Some(groups) => fit_and_score(Some(groups))?.1,
        };
        println!("{label} {}", rounded_report(&variant_metrics));
        ablation_rows.push(report_row(label, &variant_metrics));
    }

    let mut headers = vec!["model"];
    headers.extend(REPORT_METRICS);
    let comparison = markdown_table(&headers, &rows);
    let mut ablation_headers = vec!["features"];
    ablation_headers.extend(REPORT_METRICS);
    let ablation = markdown_table(&ablation_headers, &ablation_rows);
    let importance_rows: Vec<Vec<Cell>> = ranker
        .importance_percent()
        .into_iter()
        .map(|(name, percent)| vec![name.into(), Cell::Number((percent * 10.0).round() / 10.0)])
        .collect();
    let importance = markdown_table(&["", "gain %"], &importance_rows);

    let text = format!(
        "## Retrieval only vs retrieval + ranking (test window)\n\n{comparison}\n\n\
         Candidate recall (share of relevant pairs inside the candidate set): \
         validation {:.4}, test {:.4}\n\n\
         Candidates per user: validation {:.0}, test {:.0}\n\n\
         Trees used: {}\n\n## Feature importance (gain, %)\n\n{importance}\n\n\
         ## Feature-group ablation (test window)\n\n{ablation}\n",
        train_stage.candidate_recall,
        test_stage.candidate_recall,
        train_stage.candidates_per_user,
        test_stage.candidates_per_user,
        ranker.best_iteration(),
    );
    write_report(&args.output, &text)
}

fn score_ranker(
    ranker: &TrainedRanker,
    stage: &Stage,
    history: &Table,
    eval_table: &Table,
    n_items: usize,
    k_values: &[usize],
) -> Result<Metrics> {
    let max_k = *k_values.iter().max().expect("at least one k");
    let scores = ranker.predict(&stage.features);
    let recommended = top_k_per_user(&stage.frame.group_sizes, &stage.frame.items, &scores, max_k)?;
    let (_, relevant) = relevant_items_by_user(eval_table);
    Ok(evaluate(
        &recommended,
        max_k,
        &relevant,
        &item_popularity(history, n_items),
        k_values,
    ))
}

/// Stores each model's top-k for the test users, so the A/B simulation can replay them.
fn save_test_recommendations(
    path: &std::path::Path,
    retrievers: &RetrieverSet,
    ranker: &TrainedRanker,
    stage: &Stage,
    seen: &BinaryCsr,
    k: usize,
) -> Result<()> {
    let mut by_model = BTreeMap::new();
    for (name, model) in retrievers.names.iter().zip(&retrievers.models) {
        by_model.insert(name.clone(), model.recommend(&stage.users, k, seen)?);
    }
    let scores = ranker.predict(&stage.features);
    by_model.insert(
        "ranker".to_string(),
        top_k_per_user(&stage.frame.group_sizes, &stage.frame.items, &scores, k)?,
    );
    save(
        path,
        &TestRecommendations {
            users: stage.users.clone(),
            width: k,
            by_model,
        },
    )
}
