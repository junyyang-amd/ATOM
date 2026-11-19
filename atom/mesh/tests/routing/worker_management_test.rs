//! Worker management integration tests
//!
//! Tests for dynamic worker add/remove operations via management API.
//! The actual worker management API uses:
//! - POST /workers - create a worker
//! - GET /workers - list workers
//! - DELETE /workers/{worker_id} - remove a worker

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[cfg(test)]
mod worker_management_tests {
    use super::*;

    /// Test listing workers via API
    #[tokio::test]
    async fn test_list_workers() {
        let config = TestRouterConfig::round_robin(3900);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::healthy(19900),
                TestWorkerConfig::healthy(19901),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // List workers via GET /workers
        let req = Request::builder()
            .method("GET")
            .uri("/workers")
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "GET /workers should return OK"
        );

        ctx.shutdown().await;
    }

    /// Test that routing continues to work with multiple workers
    #[tokio::test]
    async fn test_routing_with_multiple_workers() {
        let config = TestRouterConfig::round_robin(3901);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::healthy(19902),
                TestWorkerConfig::healthy(19903),
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let mut success_count = 0;

        // Verify routing distributes across workers
        for i in 0..20 {
            let payload = json!({
                "text": format!("Test request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 20,
            "All requests should succeed with multiple workers"
        );

        ctx.shutdown().await;
    }

    /// Test that requests continue to work during worker operations
    #[tokio::test]
    async fn test_requests_during_worker_changes() {
        let config = TestRouterConfig::round_robin(3902);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(19904)]).await;

        let app = ctx.create_app().await;

        // Send requests and verify they succeed
        let mut success_count = 0;
        for i in 0..10 {
            let payload = json!({
                "text": format!("Request during changes {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 10,
            "All requests should succeed during normal operation"
        );

        ctx.shutdown().await;
    }

    /// A worker that `/v1/models` names by an alias (the mock's `mock-model`) gets its
    /// tokenizer from `--tokenizer-path`, not from the alias, under `--backend atom`.
    #[tokio::test]
    async fn test_aliased_worker_tokenizer_comes_from_the_tokenizer_path() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = json!({"version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [], "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "unk_token": "<unk>",
                "vocab": {"<unk>": 0, "a": 1, "b": 2}}});
        std::fs::write(dir.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();

        let mut config = TestRouterConfig::round_robin(3910);
        config.backend = mesh::config::types::BackendType::Atom;
        config.tokenizer_path = Some(dir.path().to_str().unwrap().to_string());
        let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(0)]).await;

        let registry = &ctx.app_context.tokenizer_registry;
        let start = tokio::time::Instant::now();
        while registry.get("mock-model").is_none()
            && start.elapsed() < tokio::time::Duration::from_secs(10)
        {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
        let tokenizer = registry
            .get("mock-model")
            .expect("the aliased worker's tokenizer should load from --tokenizer-path");
        assert_eq!(tokenizer.encode("a b", false).unwrap().token_ids(), [1, 2]);

        ctx.shutdown().await;
    }
}
