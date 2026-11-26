use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use rand::seq::index::sample;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use tower::ServiceExt;

use super::api::router;
use super::bundle::{build_bundle, ServingBundle};
use super::cache::Cache;
use super::recommender::TwoStageRecommender;
use crate::config::*;
use crate::ranking::candidates::scorer_names;
use crate::ranking::candidates::CandidateFrame;
use crate::ranking::features::{build_features, item_statistics, user_statistics};
use crate::table::{time_split, Table};

const N_USERS: usize = 200;
const N_ITEMS: usize = 60;

fn small_config() -> Config {
    Config {
        data: DataConfig {
            category: "synthetic".into(),
        },
        split: SplitConfig {
            val_fraction: 0.2,
            test_fraction: 0.2,
        },
        eval: EvalConfig { k_values: vec![10] },
        als: AlsConfig {
            factors: 8,
            regularization: 0.01,
            iterations: 5,
            alpha: 40.0,
        },
        two_tower: TwoTowerConfig {
            embedding_dim: 8,
            output_dim: 8,
            epochs: 3,
            batch_size: 64,
            learning_rate: 0.03,
            temperature: 0.3,
        },
        baselines: BaselineConfig {
            recent_window_days: 90,
            item_cf_neighbors: 10,
        },
        ranking: RankingConfig {
            candidates_per_retriever: 20,
            holdout_fraction: 0.2,
            early_stopping_rounds: 5,
            lightgbm: BoostingConfig {
                n_estimators: 20,
                learning_rate: 0.1,
                num_leaves: 31,
                min_child_samples: 5,
                subsample: 1.0,
                subsample_freq: 0,
                colsample_bytree: 1.0,
            },
        },
        experiment: ExperimentConfig {
            top_k: 10,
            alpha: 0.05,
            power: 0.8,
            treatment_share: 0.5,
            n_repeats: 10,
            seed: 0,
            planning_relative_lifts: vec![0.1],
        },
        serving: ServingConfig {
            cache_ttl_seconds: 60,
            latency_budget_ms: LatencyBudgets {
                cache_hit: 50.0,
                cache_miss: 250.0,
                cold_start: 25.0,
            },
        },
    }
}

/// A log where users in group g like the items with `item % 4 == g`.
fn synthetic_table() -> Table {
    let mut rng = ChaCha12Rng::seed_from_u64(0);
    let mut rows = Vec::new();
    for user in 0..N_USERS {
        let preferred: Vec<u32> = (0..N_ITEMS as u32)
            .filter(|item| item % 4 == (user % 4) as u32)
            .collect();
        for (step, pick) in sample(&mut rng, preferred.len(), 12).iter().enumerate() {
            rows.push((
                user as u32,
                preferred[pick],
                1_000 * (step * N_USERS + user) as i64,
            ));
        }
    }
    rows.sort_by_key(|row| row.2);
    Table {
        user: rows.iter().map(|r| r.0).collect(),
        item: rows.iter().map(|r| r.1).collect(),
        rating: vec![5.0; rows.len()],
        timestamp: rows.iter().map(|r| r.2).collect(),
    }
}

/// Training the bundle takes a moment, so every test shares one.
fn bundle_and_history() -> (ServingBundle, Table) {
    let split = time_split(&synthetic_table(), 0.2, 0.2);
    let bundle = build_bundle(&small_config(), &split, N_USERS, N_ITEMS).unwrap();
    (bundle, Table::concat(&split.train, &split.val))
}

fn memory_cache(bundle: &ServingBundle) -> Cache {
    Cache::Memory(
        Mutex::new(HashMap::new()),
        bundle.retrievers.models.len(),
        bundle.scorer_names.len(),
    )
}

async fn get_json(app: axum::Router, uri: &str) -> (StatusCode, serde_json::Value) {
    let response = app
        .oneshot(Request::get(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

fn item_ids(body: &serde_json::Value) -> Vec<u64> {
    body["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["item_id"].as_u64().unwrap())
        .collect()
}

#[tokio::test]
async fn serving_behaves_like_the_python_service() {
    let (bundle, history) = bundle_and_history();
    let recommender = Arc::new(TwoStageRecommender::new(bundle, Cache::Off));
    let bundle = &recommender.bundle;

    // Online ranking must match the offline feature pipeline: precomputed statistics equal
    // recomputed ones, so there is no train/serve skew.
    let user = 9;
    let candidates = recommender.retrieve(user);
    let n_rows = candidates.items.len();
    let frame = CandidateFrame {
        users: vec![user; n_rows],
        items: candidates.items.clone(),
        n_retrievers: bundle.retrievers.models.len(),
        n_scorers: bundle.scorer_names.len(),
        ranks: candidates.ranks.clone(),
        scores: candidates.scores.clone(),
        group_sizes: vec![n_rows],
        query_users: vec![user],
    };
    let offline = build_features(
        &frame,
        &bundle.retrievers.names,
        &scorer_names(&bundle.retrievers),
        &user_statistics(&history, N_USERS, N_ITEMS),
        &item_statistics(&history, N_ITEMS),
    );
    let offline_scores = bundle.ranker.predict(&offline);
    let mut order: Vec<usize> = (0..n_rows).collect();
    order.sort_by(|&a, &b| {
        offline_scores[b]
            .total_cmp(&offline_scores[a])
            .then(a.cmp(&b))
    });
    let expected: Vec<u32> = order
        .iter()
        .take(10)
        .map(|&row| candidates.items[row])
        .collect();
    let served = recommender.recommend(user, 10).await;
    assert_eq!(
        served
            .items
            .iter()
            .map(|(item, _)| *item)
            .collect::<Vec<_>>(),
        expected
    );

    // Ranked items come back best first and never include something the user has seen.
    let cold_fallback: Vec<u64> = bundle
        .fallback_items
        .iter()
        .take(5)
        .map(|&i| u64::from(i))
        .collect();
    let app = router(Arc::clone(&recommender));
    let (status, body) = get_json(app.clone(), "/recommend?user_id=3&k=5").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        (body["source"].as_str(), body["cache"].as_str()),
        (Some("ranked"), Some("off"))
    );
    let scores: Vec<f64> = body["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["score"].as_f64().unwrap())
        .collect();
    assert_eq!(scores.len(), 5);
    assert!(scores.windows(2).all(|pair| pair[0] >= pair[1]));
    let seen: Vec<u64> = (0..history.len())
        .filter(|&row| history.user[row] == 3)
        .map(|row| u64::from(history.item[row]))
        .collect();
    assert!(item_ids(&body).iter().all(|item| !seen.contains(item)));

    // A user with no history gets the fallback list and no scores.
    let (status, body) = get_json(app.clone(), "/recommend?user_id=999999&k=5").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["source"], "cold_start");
    assert_eq!(item_ids(&body), cold_fallback);
    assert!(body["items"]
        .as_array()
        .unwrap()
        .iter()
        .all(|item| item["score"].is_null()));

    // Invalid parameters are rejected, and health reports the bundle version.
    for uri in [
        "/recommend?user_id=-1",
        "/recommend?user_id=1&k=0",
        "/recommend?user_id=1&k=101",
    ] {
        assert_eq!(
            get_json(app.clone(), uri).await.0,
            StatusCode::UNPROCESSABLE_ENTITY,
            "{uri}"
        );
    }
    let (_, health) = get_json(app, "/health").await;
    assert_eq!(health["status"], "ok");
    assert!(health["bundle_version"].is_string());
}

#[tokio::test]
async fn the_second_request_is_served_from_the_cache_with_the_same_items() {
    let (bundle, _) = bundle_and_history();
    let cache = memory_cache(&bundle);
    let app = router(Arc::new(TwoStageRecommender::new(bundle, cache)));

    let (_, first) = get_json(app.clone(), "/recommend?user_id=7&k=10").await;
    let (_, second) = get_json(app, "/recommend?user_id=7&k=10").await;

    assert_eq!(
        (first["cache"].as_str(), second["cache"].as_str()),
        (Some("miss"), Some("hit"))
    );
    assert_eq!(item_ids(&first), item_ids(&second));
}

#[tokio::test]
async fn a_redis_outage_slows_requests_but_does_not_fail_them() {
    let (bundle, _) = bundle_and_history();
    let cache = Cache::redis("redis://127.0.0.1:1/0", &bundle.version, 5, 2, 60).unwrap();
    let app = router(Arc::new(TwoStageRecommender::new(bundle, cache)));

    for user in [5, 6, 7] {
        let (status, body) = get_json(app.clone(), &format!("/recommend?user_id={user}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["cache"], "miss");
        assert_eq!(body["items"].as_array().unwrap().len(), 10);
    }
}
