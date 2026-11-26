//! HTTP API: `GET /recommend?user_id=...&k=...` and `GET /health`.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Query, Request, State};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::serving::recommender::TwoStageRecommender;

pub const MAX_K: i64 = 100;
const DEFAULT_K: i64 = 10;

#[derive(Deserialize)]
struct RecommendParams {
    user_id: i64,
    k: Option<i64>,
}

#[derive(Serialize)]
struct ScoredItem {
    item_id: u32,
    /// Null for cold-start users, who are not ranked.
    score: Option<f32>,
}

#[derive(Serialize)]
struct RecommendResponse {
    user_id: u32,
    source: &'static str,
    cache: &'static str,
    items: Vec<ScoredItem>,
}

pub fn router(recommender: Arc<TwoStageRecommender>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/recommend", get(recommend))
        .layer(middleware::from_fn(add_process_time))
        .with_state(recommender)
}

async fn add_process_time(request: Request, next: Next) -> Response {
    let started = Instant::now();
    let mut response = next.run(request).await;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if let Ok(value) = HeaderValue::from_str(&format!("{elapsed_ms:.2}")) {
        response.headers_mut().insert("X-Process-Time-Ms", value);
    }
    response
}

async fn health(State(recommender): State<Arc<TwoStageRecommender>>) -> Json<serde_json::Value> {
    Json(json!({"status": "ok", "bundle_version": recommender.bundle.version}))
}

fn invalid(detail: &str) -> Response {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({"detail": detail})),
    )
        .into_response()
}

async fn recommend(
    State(recommender): State<Arc<TwoStageRecommender>>,
    Query(params): Query<RecommendParams>,
) -> Response {
    let k = params.k.unwrap_or(DEFAULT_K);
    if params.user_id < 0 || params.user_id > i64::from(u32::MAX) {
        return invalid("user_id must be a non-negative integer");
    }
    if !(1..=MAX_K).contains(&k) {
        return invalid("k must be between 1 and 100");
    }

    let result = recommender
        .recommend(params.user_id as u32, k as usize)
        .await;
    Json(RecommendResponse {
        user_id: result.user_id,
        source: result.source,
        cache: result.cache,
        items: result
            .items
            .into_iter()
            .map(|(item_id, score)| ScoredItem { item_id, score })
            .collect(),
    })
    .into_response()
}
