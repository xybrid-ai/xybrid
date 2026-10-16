//! EmbeddingGemma 2 through xybrid's ONNX text path reproduces the model's own
//! sentence embeddings.
//!
//! The goldens are the fixture graph's in-graph `sentence_embedding` output,
//! written by `tools/scripts/gen_embedding_goldens.py`. Matching them takes the
//! whole text path: empty image/video/audio inputs, `<bos>`/`<eos>` from the
//! tokenizer, mean pooling and L2 normalization.
//!
//! Run with:
//!   ./integration-tests/download.sh embeddinggemma-2
//!   cargo test -p integration-tests --test embedding_parity

use integration_tests::fixtures;
use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};

const MODEL: &str = "embeddinggemma-2";
const EMBEDDING_DIM: usize = 768;
/// Same graph and ONNX Runtime as the goldens: only float rounding remains.
const MIN_COSINE: f32 = 0.9999;

/// A reference text and the model's own embedding of it.
struct Golden {
    text: String,
    embedding: Vec<f32>,
}

/// The goldens, in file order: English query, French query, the document both
/// queries ask for, an unrelated document.
fn goldens() -> Vec<Golden> {
    let path = fixtures::fixtures_dir().join("embedding/embeddinggemma-2.json");
    let contents =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let json: serde_json::Value = serde_json::from_str(&contents).expect("goldens parse");
    json["cases"]
        .as_array()
        .expect("goldens have cases")
        .iter()
        .map(|case| Golden {
            text: case["text"].as_str().expect("case text").to_string(),
            embedding: case["embedding"]
                .as_array()
                .expect("case embedding")
                .iter()
                .map(|value| value.as_f64().expect("embedding value") as f32)
                .collect(),
        })
        .collect()
}

fn load(model_dir: &std::path::Path) -> (TemplateExecutor, ModelMetadata) {
    let metadata = std::fs::read_to_string(model_dir.join("model_metadata.json"))
        .expect("model_metadata.json reads");
    let metadata = serde_json::from_str(&metadata).expect("model_metadata.json parses");
    let executor = TemplateExecutor::with_base_path(model_dir.to_str().expect("UTF-8 model path"));
    (executor, metadata)
}

fn embed(executor: &mut TemplateExecutor, metadata: &ModelMetadata, text: &str) -> Vec<f32> {
    let input = Envelope::new(EnvelopeKind::Text(text.to_string()));
    match executor
        .execute(metadata, &input, None)
        .expect("embedding runs")
        .kind
    {
        EnvelopeKind::Embedding(values) => values,
        other => panic!("expected an embedding, got {}", other.as_str()),
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt())
}

#[test]
fn embeddinggemma2_reproduces_the_reference_embeddings() {
    let Some(model_dir) = fixtures::model_for_test(MODEL) else {
        eprintln!("Skipping: {MODEL} not downloaded. Run: ./integration-tests/download.sh {MODEL}");
        return;
    };
    let (mut executor, metadata) = load(&model_dir);

    for golden in goldens() {
        let embedding = embed(&mut executor, &metadata, &golden.text);

        assert_eq!(embedding.len(), EMBEDDING_DIM, "{}", golden.text);
        let norm = dot(&embedding, &embedding).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "{}: norm {norm}, expected a unit vector",
            golden.text
        );
        let similarity = cosine(&embedding, &golden.embedding);
        assert!(
            similarity >= MIN_COSINE,
            "{}: cosine {similarity} to the reference",
            golden.text
        );
        // The second run reuses the cached session and must give the same vector.
        assert_eq!(
            embed(&mut executor, &metadata, &golden.text),
            embedding,
            "{}",
            golden.text
        );
    }
}

#[test]
fn embeddinggemma2_ranks_the_matching_document_first() {
    let Some(model_dir) = fixtures::model_for_test(MODEL) else {
        eprintln!("Skipping: {MODEL} not downloaded. Run: ./integration-tests/download.sh {MODEL}");
        return;
    };
    let (mut executor, metadata) = load(&model_dir);
    let texts: Vec<String> = goldens().into_iter().map(|golden| golden.text).collect();
    let [english_query, french_query, answer, unrelated] = texts.as_slice() else {
        panic!("expected four goldens, got {}", texts.len());
    };
    let answer = embed(&mut executor, &metadata, answer);
    let unrelated = embed(&mut executor, &metadata, unrelated);

    for query in [english_query, french_query] {
        let query_embedding = embed(&mut executor, &metadata, query);
        let to_answer = dot(&query_embedding, &answer);
        let to_unrelated = dot(&query_embedding, &unrelated);

        assert!(
            to_answer > to_unrelated,
            "{query}: {to_answer} to the answer, {to_unrelated} to the unrelated document"
        );
    }
}
