//! MLX embedding integration tests.
//!
//! These tests require:
//! - Apple Silicon macOS with `llm-mlx-runtime` enabled.
//! - Synthetic canonical BERT runtime coverage always runs when the linked
//!   runtime is available.
//! - Optional real `nomic-ai/nomic-embed-text-v1.5` SafeTensors coverage when
//!   `$XYBRID_MLX_NOMIC_DIR` is staged.
//!
//! Run with:
//! ```bash
//! cargo test -p xybrid-core --no-default-features --features llm-mlx-runtime --test mlx_embedding
//!
//! # Optional real fixture coverage:
//! export XYBRID_MLX_NOMIC_DIR=/path/to/nomic-embed-text-v1.5
//! cargo test -p xybrid-core --no-default-features --features llm-mlx-runtime --test mlx_embedding
//! ```
//!
//! When the env var is absent only the real Nomic tests skip; the synthetic
//! runtime smoke still exercises MLX encoder execution and TemplateExecutor
//! routing.

#![cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::runtime_adapter::mlx::{MlxEmbeddingAdapter, MlxEmbeddingConfig, Pooling};

fn nomic_dir() -> Option<PathBuf> {
    std::env::var_os("XYBRID_MLX_NOMIC_DIR").map(PathBuf::from)
}

fn mlx_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("MLX embedding integration test lock poisoned")
}

#[test]
fn synthetic_bert_template_executor_routes_to_mlx_embedding_without_external_fixture() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    let prompt = "Hello";
    write_synthetic_canonical_bert_bundle(tmp.path(), prompt);

    let mut metadata =
        ModelMetadata::safetensors("synthetic-bert-embed", "1.0", "model.safetensors", "bert");
    metadata.backend = Some("auto".to_string());
    metadata
        .metadata
        .insert("task".to_string(), serde_json::json!("text-embedding"));
    metadata
        .metadata
        .insert("pooling".to_string(), serde_json::json!("mean"));
    metadata
        .metadata
        .insert("normalize".to_string(), serde_json::json!(false));
    metadata
        .metadata
        .insert("max_seq_len".to_string(), serde_json::json!(16));

    let mut executor = TemplateExecutor::with_base_path(&tmp.path().to_string_lossy());
    let out = executor
        .execute(
            &metadata,
            &Envelope::new(EnvelopeKind::Text(prompt.to_string())),
            None,
        )
        .expect("TemplateExecutor should route synthetic BERT metadata through MLX embedding");

    match out.kind {
        EnvelopeKind::Embedding(values) => {
            assert_eq!(values.len(), 2);
            assert!((values[0] + 1.0).abs() < 1.0e-5, "got {values:?}");
            assert!((values[1] - 1.0).abs() < 1.0e-5, "got {values:?}");
        }
        other => panic!("expected embedding envelope, got {other:?}"),
    }
}

#[test]
fn synthetic_bert_indexed_shards_route_to_mlx_embedding_without_external_fixture() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    let prompt = "Hello";
    write_synthetic_canonical_bert_bundle(tmp.path(), prompt);
    convert_single_weights_to_indexed_shard(tmp.path());

    let mut metadata =
        ModelMetadata::safetensors("synthetic-bert-embed", "1.0", "model.safetensors", "bert");
    metadata.backend = Some("auto".to_string());
    metadata
        .metadata
        .insert("task".to_string(), serde_json::json!("text-embedding"));
    metadata
        .metadata
        .insert("pooling".to_string(), serde_json::json!("mean"));
    metadata
        .metadata
        .insert("normalize".to_string(), serde_json::json!(false));
    metadata
        .metadata
        .insert("max_seq_len".to_string(), serde_json::json!(16));

    let mut executor = TemplateExecutor::with_base_path(&tmp.path().to_string_lossy());
    let out = executor
        .execute(
            &metadata,
            &Envelope::new(EnvelopeKind::Text(prompt.to_string())),
            None,
        )
        .expect("TemplateExecutor should route indexed BERT shards through MLX embedding");

    match out.kind {
        EnvelopeKind::Embedding(values) => {
            assert_eq!(values.len(), 2);
            assert!((values[0] + 1.0).abs() < 1.0e-5, "got {values:?}");
            assert!((values[1] - 1.0).abs() < 1.0e-5, "got {values:?}");
        }
        other => panic!("expected embedding envelope, got {other:?}"),
    }
}

#[test]
fn real_nomic_bundle_produces_normalized_embedding() {
    let _guard = mlx_test_lock();
    let Some(dir) = nomic_dir() else {
        eprintln!("SKIP: XYBRID_MLX_NOMIC_DIR not set");
        return;
    };

    let adapter = MlxEmbeddingAdapter::load(
        &dir,
        &MlxEmbeddingConfig {
            max_seq_len: 32,
            pooling: Pooling::Mean,
            normalize: true,
        },
    )
    .expect("load nomic embedding bundle");
    let hidden_dim = adapter
        .bert_config()
        .expect("loaded bert config")
        .hidden_size;

    let out = adapter
        .embed("search_document: Xybrid routes inference between device and cloud.")
        .expect("embed real nomic prompt");

    let EnvelopeKind::Embedding(values) = out.kind else {
        panic!("expected embedding envelope");
    };
    assert_eq!(values.len(), hidden_dim);
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| v.abs() > 1.0e-6));

    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1.0e-4,
        "normalized embedding should have unit norm, got {norm}"
    );
}

#[test]
fn template_executor_routes_real_nomic_bundle_to_mlx_embedding_strategy() {
    let _guard = mlx_test_lock();
    let Some(dir) = nomic_dir() else {
        eprintln!("SKIP: XYBRID_MLX_NOMIC_DIR not set");
        return;
    };

    let mut metadata = ModelMetadata::safetensors(
        "nomic-embed-text-v1.5",
        "1.0",
        "model.safetensors",
        "nomic_bert",
    );
    metadata.backend = Some("auto".to_string());
    metadata
        .metadata
        .insert("task".to_string(), serde_json::json!("text-embedding"));
    metadata
        .metadata
        .insert("pooling".to_string(), serde_json::json!("mean"));
    metadata
        .metadata
        .insert("normalize".to_string(), serde_json::json!(true));
    metadata
        .metadata
        .insert("max_seq_len".to_string(), serde_json::json!(32));

    let mut executor = TemplateExecutor::with_base_path(&dir.to_string_lossy());
    let out = executor
        .execute(
            &metadata,
            &Envelope::new(EnvelopeKind::Text(
                "search_document: Xybrid routes inference between device and cloud.".to_string(),
            )),
            None,
        )
        .expect("TemplateExecutor should route nomic metadata through MLX embedding strategy");

    let EnvelopeKind::Embedding(values) = out.kind else {
        panic!("expected embedding envelope");
    };
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| v.abs() > 1.0e-6));

    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1.0e-4,
        "normalized executor embedding should have unit norm, got {norm}"
    );
}

fn write_synthetic_canonical_bert_bundle(dir: &Path, prompt: &str) {
    let tok_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("qwen_tokenizer.json");
    std::fs::copy(&tok_src, dir.join("tokenizer.json")).expect("copy tokenizer");
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_src).expect("load tokenizer");
    let encoding = tokenizer.encode(prompt, true).expect("encode prompt");
    let token_ids = encoding.get_ids();
    let seq_len = token_ids.len().max(1);
    let vocab_size = token_ids
        .iter()
        .copied()
        .max()
        .map(|id| id as usize + 1)
        .unwrap_or(1);

    let cfg = serde_json::json!({
        "model_type": "bert",
        "hidden_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "intermediate_size": 4,
        "vocab_size": vocab_size,
        "max_position_embeddings": seq_len,
        "type_vocab_size": 2,
        "layer_norm_eps": 0.0,
        "use_rotary_embeddings": false,
        "use_swiglu": false
    });
    let mut f = std::fs::File::create(dir.join("config.json")).expect("create config");
    f.write_all(cfg.to_string().as_bytes())
        .expect("write config");

    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    push_f32_tensor(
        &mut tensors,
        "embeddings.word_embeddings.weight",
        vec![vocab_size, 2],
        vec![0.0; vocab_size * 2],
    );
    let mut position_values = Vec::with_capacity(seq_len * 2);
    for pos in 0..seq_len {
        let base = pos as f32 + 1.0;
        position_values.push(base);
        position_values.push(base * 10.0);
    }
    push_f32_tensor(
        &mut tensors,
        "embeddings.position_embeddings.weight",
        vec![seq_len, 2],
        position_values,
    );
    push_f32_tensor(
        &mut tensors,
        "embeddings.token_type_embeddings.weight",
        vec![2, 2],
        vec![0.0; 4],
    );
    push_f32_tensor(
        &mut tensors,
        "embeddings.LayerNorm.weight",
        vec![2],
        vec![1.0; 2],
    );
    push_f32_tensor(
        &mut tensors,
        "embeddings.LayerNorm.bias",
        vec![2],
        vec![0.0; 2],
    );

    let base = "encoder.layer.0";
    for name in [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
    ] {
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.{name}.weight"),
            vec![2, 2],
            vec![0.0; 4],
        );
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.{name}.bias"),
            vec![2],
            vec![0.0; 2],
        );
    }
    for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.{name}.weight"),
            vec![2],
            vec![1.0; 2],
        );
        push_f32_tensor(
            &mut tensors,
            &format!("{base}.{name}.bias"),
            vec![2],
            vec![0.0; 2],
        );
    }
    push_f32_tensor(
        &mut tensors,
        &format!("{base}.intermediate.dense.weight"),
        vec![4, 2],
        vec![0.0; 8],
    );
    push_f32_tensor(
        &mut tensors,
        &format!("{base}.intermediate.dense.bias"),
        vec![4],
        vec![0.0; 4],
    );
    push_f32_tensor(
        &mut tensors,
        &format!("{base}.output.dense.weight"),
        vec![2, 4],
        vec![0.0; 8],
    );
    push_f32_tensor(
        &mut tensors,
        &format!("{base}.output.dense.bias"),
        vec![2],
        vec![0.0; 2],
    );

    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).expect("tensor view"),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, &None, &dir.join("model.safetensors"))
        .expect("write safetensors");
}

fn convert_single_weights_to_indexed_shard(dir: &Path) {
    let single = dir.join("model.safetensors");
    let shard_name = "model-00001-of-00001.safetensors";
    let shard = dir.join(shard_name);
    std::fs::rename(&single, &shard).expect("rename single weights to shard");

    let bytes = std::fs::read(&shard).expect("read shard");
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).expect("read shard metadata");
    let weight_map = meta
        .tensors()
        .into_keys()
        .map(|name| (name, serde_json::Value::String(shard_name.to_string())))
        .collect::<serde_json::Map<_, _>>();
    let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
    std::fs::write(dir.join("model.safetensors.index.json"), index.to_string())
        .expect("write shard index");
}

fn push_f32_tensor(
    tensors: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
    name: &str,
    shape: Vec<usize>,
    values: Vec<f32>,
) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    tensors.push((name.to_string(), shape, bytes));
}
