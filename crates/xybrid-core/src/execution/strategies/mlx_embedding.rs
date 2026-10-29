//! MLX embedding execution strategy.
//!
//! Routes local BERT-family SafeTensors embedding bundles through
//! `MlxEmbeddingAdapter` so `TemplateExecutor`, SDK, and CLI local-directory
//! paths can produce `EnvelopeKind::Embedding` without callers instantiating the
//! adapter directly.

use std::collections::HashMap;
use std::path::Path;

use super::{ExecutionContext, ExecutionStrategy};
use crate::execution::template::{is_mlx_embedding_safetensors_metadata, ModelMetadata};
use crate::execution::types::ExecutorResult;
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::mlx::{MlxEmbeddingAdapter, MlxEmbeddingConfig};
use crate::runtime_adapter::AdapterError;
use crate::tracing as xybrid_trace;

/// Execution strategy for MLX-backed text embeddings.
pub struct MlxEmbeddingStrategy {
    adapter_cache: std::sync::Mutex<Option<(String, MlxEmbeddingConfig, MlxEmbeddingAdapter)>>,
}

impl MlxEmbeddingStrategy {
    /// Create an empty MLX embedding strategy.
    pub fn new() -> Self {
        Self {
            adapter_cache: std::sync::Mutex::new(None),
        }
    }

    fn config_from_metadata(metadata: &ModelMetadata) -> ExecutorResult<MlxEmbeddingConfig> {
        let mut values = HashMap::new();
        for key in ["pooling", "normalize", "max_seq_len"] {
            if let Some(value) = metadata
                .metadata
                .get(key)
                .and_then(metadata_value_as_string)
            {
                values.insert(key.to_string(), value);
            }
        }
        MlxEmbeddingConfig::from_metadata(&values).map_err(AdapterError::from)
    }

    fn model_path(base_path: &str) -> String {
        Path::new(base_path).to_string_lossy().to_string()
    }
}

impl Default for MlxEmbeddingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionStrategy for MlxEmbeddingStrategy {
    fn can_handle(&self, metadata: &ModelMetadata) -> bool {
        cfg!(feature = "llm-mlx") && is_mlx_embedding_safetensors_metadata(metadata)
    }

    fn execute(
        &self,
        ctx: &mut ExecutionContext<'_>,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> ExecutorResult<Envelope> {
        let _span = xybrid_trace::SpanGuard::new("mlx_embedding_execution");
        xybrid_trace::add_metadata("backend", "mlx");
        xybrid_trace::add_metadata("model", &metadata.model_id);

        let text = match &input.kind {
            EnvelopeKind::Text(text) => text,
            _ => {
                return Err(AdapterError::InvalidInput(
                    "MLX embedding strategy requires text input".to_string(),
                ))
            }
        };

        let model_path = Self::model_path(ctx.base_path);
        let config = Self::config_from_metadata(metadata)?;
        let mut cache = self.adapter_cache.lock().map_err(|e| {
            AdapterError::RuntimeError(format!("Failed to acquire MLX embedding lock: {e}"))
        })?;

        let should_load = !matches!(
            cache.as_ref(),
            Some((cached_path, cached_config, _))
                if cached_path == &model_path && cached_config == &config
        );
        if should_load {
            let adapter = MlxEmbeddingAdapter::load(Path::new(&model_path), &config)
                .map_err(AdapterError::from)?;
            *cache = Some((model_path.clone(), config, adapter));
        }

        let adapter = cache
            .as_ref()
            .map(|(_, _, adapter)| adapter)
            .ok_or_else(|| AdapterError::RuntimeError("MLX embedding adapter not loaded".into()))?;

        adapter.embed(text).map_err(AdapterError::from)
    }

    fn name(&self) -> &'static str {
        "mlx_embedding"
    }
}

fn metadata_value_as_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}
