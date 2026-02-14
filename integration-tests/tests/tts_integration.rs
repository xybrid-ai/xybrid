//! Integration tests for TTS (Text-to-Speech) pipeline components.
//!
//! This tests:
//! - Phonemizer with CMU dictionary
//! - Token conversion using tokens.txt
//! - Voice embeddings loading from voices.bin and voices.npz
//! - ONNX model inference (if available)
//!
//! Run with: cargo test -p integration-tests --test tts_integration

use integration_tests::fixtures;

#[test]
fn test_load_tokens_file() {
    let Some(model_dir) = fixtures::model_if_available("kitten-tts-nano-0.2") else {
        eprintln!("Skipping test: kitten-tts-nano-0.2 not downloaded");
        eprintln!("Run: ./integration-tests/download.sh kitten-tts-nano-0.2");
        return;
    };

    let tokens_path = model_dir.join("tokens.txt");
    let tokens_content = std::fs::read_to_string(&tokens_path).expect("Failed to read tokens.txt");

    let tokens_map = xybrid_core::phonemizer::load_tokens_map(&tokens_content);

    // Verify some expected mappings
    assert!(tokens_map.contains_key(&'$'), "Should have padding token $");
    assert!(
        tokens_map.contains_key(&'a'),
        "Should have lowercase letters"
    );
    assert!(tokens_map.contains_key(&'ɑ'), "Should have IPA symbols");
    assert!(tokens_map.contains_key(&'ˈ'), "Should have stress markers");

    println!("Loaded {} tokens from tokens.txt", tokens_map.len());

    // Print some IPA symbols we'll use
    let ipa_symbols = ['ɑ', 'ɛ', 'ɪ', 'ʌ', 'ɹ', 'ŋ', 'θ', 'ð', 'ʃ', 'ʒ'];
    for sym in ipa_symbols {
        if let Some(id) = tokens_map.get(&sym) {
            println!("  {} -> {}", sym, id);
        }
    }
}

// NOTE: test_phonemize_text and test_phonemes_to_token_ids were removed.
// Phonemizer backends (CMU, Misaki, espeak) are now internal to xybrid-core
// and tested via crate-internal unit tests in preprocessing/text.rs.

#[test]
fn test_load_voice_embeddings_bin() {
    let Some(model_dir) = fixtures::model_if_available("kitten-tts-nano-0.2") else {
        eprintln!("Skipping test: kitten-tts-nano-0.2 not downloaded");
        return;
    };

    let voices_path = model_dir.join("voices.bin");
    let voices_data = std::fs::read(&voices_path).expect("Failed to read voices.bin");

    // voices.bin contains 8 voices, each with 256 float32 values
    // Total size: 8 * 256 * 4 bytes = 8192 bytes
    assert_eq!(
        voices_data.len(),
        8192,
        "voices.bin should be 8192 bytes (8 voices x 256 dims x 4 bytes)"
    );

    // Parse voice embeddings
    let num_voices = 8;
    let embedding_dim = 256;

    let voices: Vec<Vec<f32>> = (0..num_voices)
        .map(|i| {
            let start = i * embedding_dim * 4;
            let end = start + embedding_dim * 4;
            voices_data[start..end]
                .chunks(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        })
        .collect();

    println!("Loaded {} voice embeddings from voices.bin", voices.len());
    let voice_names = [
        "expr-voice-2-m",
        "expr-voice-2-f",
        "expr-voice-3-m",
        "expr-voice-3-f",
        "expr-voice-4-m",
        "expr-voice-4-f",
        "expr-voice-5-m",
        "expr-voice-5-f",
    ];
    for (i, voice) in voices.iter().enumerate() {
        let mean: f32 = voice.iter().sum::<f32>() / voice.len() as f32;
        let max = voice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = voice.iter().cloned().fold(f32::INFINITY, f32::min);
        println!(
            "  Voice {} ({}): mean={:.4}, min={:.4}, max={:.4}",
            i, voice_names[i], mean, min, max
        );
    }

    // Verify embeddings look reasonable (not all zeros, reasonable range)
    for (i, voice) in voices.iter().enumerate() {
        let sum: f32 = voice.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "Voice {} should not be all zeros", i);
    }
}

#[test]
fn test_model_metadata_loading() {
    let Some(model_dir) = fixtures::model_if_available("kitten-tts-nano-0.2") else {
        eprintln!("Skipping test: kitten-tts-nano-0.2 not downloaded");
        return;
    };

    let metadata_path = model_dir.join("model_metadata.json");
    let metadata_content =
        std::fs::read_to_string(&metadata_path).expect("Failed to read model_metadata.json");

    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_content).expect("Failed to parse model_metadata.json");

    // Verify metadata structure
    assert_eq!(metadata["model_id"], "kitten-tts-nano-0.2");
    assert_eq!(metadata["version"], "1.0");
    assert!(metadata["preprocessing"].is_array());

    // Check preprocessing has Phonemize step
    let preprocessing = metadata["preprocessing"].as_array().unwrap();
    assert!(!preprocessing.is_empty());
    assert_eq!(preprocessing[0]["type"], "Phonemize");
    assert_eq!(preprocessing[0]["tokens_file"], "tokens.txt");

    // Check metadata fields
    let task_metadata = &metadata["metadata"];
    assert_eq!(task_metadata["task"], "text-to-speech");
    assert_eq!(task_metadata["sample_rate"], 24000);
    assert_eq!(task_metadata["num_voices"], 8);

    println!("Model metadata verified:");
    println!("  model_id: {}", metadata["model_id"]);
    println!("  task: {}", task_metadata["task"]);
    println!("  sample_rate: {}", task_metadata["sample_rate"]);
    println!("  voices: {}", task_metadata["num_voices"]);
}

#[test]
fn test_voice_embedding_loader_npz() {
    use xybrid_core::tts::voice_embedding::VoiceEmbeddingLoader;

    // Check if we have the kokoro model with NPZ file
    let Some(model_dir) = fixtures::model_if_available("kokoro-82m") else {
        eprintln!("Skipping test: kokoro-82m not downloaded (has voices.npz)");
        eprintln!("Run: ./integration-tests/download.sh kokoro-82m");
        return;
    };

    let npz_path = model_dir.join("voices.npz");
    if !npz_path.exists() {
        eprintln!("Skipping test: voices.npz not found in kokoro-82m");
        return;
    }

    let loader = VoiceEmbeddingLoader::new(256);

    // Test loading by index
    let embedding = loader
        .load(&npz_path, 0)
        .expect("Failed to load voice embedding by index");
    assert_eq!(embedding.len(), 256, "Embedding should have 256 dimensions");

    // Verify it's not all zeros
    let sum: f32 = embedding.iter().map(|v| v.abs()).sum();
    assert!(sum > 0.0, "Embedding should not be all zeros");

    println!("Successfully loaded NPZ voice embedding:");
    println!("  Length: {}", embedding.len());
    println!("  First 5 values: {:?}", &embedding[..5]);

    // Test listing voice names
    let names = loader
        .list_voice_names(&npz_path)
        .expect("Failed to list voice names");
    if let Some(names) = names {
        println!("  Voice names: {:?}", names);
        assert!(!names.is_empty(), "Should have voice names");
    }

    // Test count_voices
    let count = loader
        .count_voices(&npz_path)
        .expect("Failed to count voices");
    println!("  Voice count: {}", count);
    assert!(count > 0, "Should have at least one voice");
}
