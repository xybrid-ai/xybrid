//! Rank sentences by semantic similarity with all-MiniLM-L6-v2.
//!
//! This example embeds an intelligence-search query and a small candidate corpus,
//! then sorts the candidates by cosine similarity. It demonstrates the retrieval
//! step behind semantic search and lightweight RAG while running fully on-device.

use std::collections::HashMap;
use std::error::Error;
use std::io;

use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::testing::model_fixtures;

fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, &'static str> {
    if a.is_empty() || a.len() != b.len() {
        return Err("embedding vectors must be non-empty and have equal dimensions");
    }
    if a.iter().chain(b).any(|value| !value.is_finite()) {
        return Err("embedding vectors must contain only finite values");
    }

    let dot_product = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum::<f64>();
    let norm_a = a
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    let norm_b = b
        .iter()
        .map(|&y| f64::from(y) * f64::from(y))
        .sum::<f64>()
        .sqrt();
    let denominator = norm_a * norm_b;

    // Finite f32 components are squared in f64, so even a nonzero f32
    // subnormal has a representable norm. An epsilon cutoff would incorrectly
    // reject valid embeddings just because they have a small scale.
    if denominator == 0.0 {
        return Err("cosine similarity is undefined for a zero vector");
    }

    Ok((dot_product / denominator).clamp(-1.0, 1.0) as f32)
}

fn rank_by_similarity<'a>(
    query_embedding: &[f32],
    candidates: Vec<(&'a str, Vec<f32>)>,
) -> Result<Vec<(&'a str, f32)>, &'static str> {
    let mut ranked = candidates
        .into_iter()
        .map(|(sentence, embedding)| {
            cosine_similarity(query_embedding, &embedding).map(|score| (sentence, score))
        })
        .collect::<Result<Vec<_>, _>>()?;

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(ranked)
}

fn encode_sentence(
    executor: &mut TemplateExecutor,
    metadata: &ModelMetadata,
    sentence: &str,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let input = Envelope {
        kind: EnvelopeKind::Text(sentence.to_string()),
        metadata: HashMap::new(),
    };
    let output = executor.execute(metadata, &input, None)?;

    match output.kind {
        EnvelopeKind::Embedding(embedding) => Ok(embedding),
        _ => Err(io::Error::other("all-minilm returned a non-embedding output").into()),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let query = "Which reports describe disruptions to maritime trade?";
    let candidates = [
        "Container ships are queueing outside the canal after a grounding blocked traffic.",
        "Port workers began a strike that delayed cargo handling across the harbor.",
        "The central bank left interest rates unchanged after its monthly meeting.",
        "A new battery chemistry increased storage capacity for electric vehicles.",
        "Heavy rain caused flash flooding in several inland towns.",
    ];

    let model_dir = model_fixtures::require_model("all-minilm");
    let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(
        model_dir.join("model_metadata.json"),
    )?)?;
    let base_path = model_dir
        .to_str()
        .ok_or_else(|| io::Error::other("model path is not valid UTF-8"))?;
    let mut executor = TemplateExecutor::with_base_path(base_path);

    println!("Sentence ranking with {}", metadata.model_id);
    println!("Query: {query}\n");

    let query_embedding = encode_sentence(&mut executor, &metadata, query)?;
    let candidate_embeddings = candidates
        .iter()
        .map(|sentence| {
            let embedding = encode_sentence(&mut executor, &metadata, sentence)?;
            Ok((*sentence, embedding))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let ranked =
        rank_by_similarity(&query_embedding, candidate_embeddings).map_err(io::Error::other)?;

    for (rank, (sentence, score)) in ranked.iter().enumerate() {
        println!("{:>2}. {score:.4}  {sentence}", rank + 1);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_recognizes_parallel_and_orthogonal_vectors() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]), Ok(1.0));
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), Ok(0.0));
    }

    #[test]
    fn cosine_similarity_rejects_invalid_vectors() {
        assert!(cosine_similarity(&[], &[]).is_err());
        assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_err());
        assert!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]).is_err());
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(cosine_similarity(&[invalid], &[1.0]).is_err());
            assert!(cosine_similarity(&[1.0], &[invalid]).is_err());
        }
    }

    #[test]
    fn cosine_similarity_is_scale_invariant_for_finite_f32_values() {
        for scale in [f32::from_bits(1), 1e-20, 1.0, f32::MAX] {
            assert_eq!(cosine_similarity(&[scale, scale], &[scale, scale]), Ok(1.0));
            assert_eq!(cosine_similarity(&[scale], &[-scale]), Ok(-1.0));
        }
    }

    #[test]
    fn ranking_preserves_input_order_for_ties_and_keeps_negative_scores() {
        let ranked = rank_by_similarity(
            &[1.0, 0.0],
            vec![
                ("opposite", vec![-1.0, 0.0]),
                ("first match", vec![1.0, 0.0]),
                ("second match", vec![2.0, 0.0]),
            ],
        )
        .unwrap();
        assert_eq!(
            ranked,
            vec![
                ("first match", 1.0),
                ("second match", 1.0),
                ("opposite", -1.0)
            ]
        );
    }

    #[test]
    fn rank_by_similarity_sorts_best_match_first() {
        let ranked = rank_by_similarity(
            &[1.0, 0.0],
            vec![
                ("orthogonal", vec![0.0, 1.0]),
                ("same direction", vec![2.0, 0.0]),
                ("partly aligned", vec![1.0, 1.0]),
            ],
        )
        .expect("test vectors should be valid");

        assert_eq!(
            ranked.iter().map(|(label, _)| *label).collect::<Vec<_>>(),
            ["same direction", "partly aligned", "orthogonal"]
        );
    }
}
