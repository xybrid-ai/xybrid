//! Integration tests for TTS (Text-to-Speech) pipeline with KittenTTS.
//!
//! This tests:
//! - Phonemizer with CMU dictionary
//! - Token conversion using tokens.txt
//! - Voice embeddings loading from voices.bin
//! - ONNX model inference (if available)

use std::path::PathBuf;

/// Get the path to the KittenTTS test model directory
fn get_kitten_tts_dir() -> Option<PathBuf> {
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent()?.to_path_buf();
    let model_dir = project_root.join("test_models/kitten-tts/kitten-nano-en-v0_1-fp16");
    if model_dir.exists() {
        Some(model_dir)
    } else {
        None
    }
}

/// Get the path to the CMU dictionary
fn get_cmudict_path() -> Option<PathBuf> {
    if let Some(home) = dirs::home_dir() {
        let dict_path = home.join(".xybrid/cmudict.dict");
        if dict_path.exists() {
            return Some(dict_path);
        }
    }
    None
}

#[test]
fn test_load_tokens_file() {
    let Some(model_dir) = get_kitten_tts_dir() else {
        eprintln!("Skipping test: KittenTTS model not found");
        return;
    };

    let tokens_path = model_dir.join("tokens.txt");
    let tokens_content = std::fs::read_to_string(&tokens_path)
        .expect("Failed to read tokens.txt");

    let tokens_map = xybrid_core::phonemizer::load_tokens_map(&tokens_content);

    // Verify some expected mappings
    assert!(tokens_map.contains_key(&'$'), "Should have padding token $");
    assert!(tokens_map.contains_key(&'a'), "Should have lowercase letters");
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

#[test]
fn test_phonemize_text() {
    let Some(dict_path) = get_cmudict_path() else {
        eprintln!("Skipping test: CMU dictionary not found at ~/.xybrid/cmudict.dict");
        return;
    };

    let phonemizer = xybrid_core::phonemizer::Phonemizer::new(&dict_path)
        .expect("Failed to create phonemizer");

    // Test some common words
    let test_cases = [
        ("hello", "hʌˈloʊ"),     // Should be close to /həˈloʊ/
        ("world", "wˈɝld"),       // Should be close to /wɜːrld/
        ("the", "ðə"),            // Should be /ðə/
        ("computer", "kʌmpjˈutɝ"), // /kəmˈpjuːtər/
    ];

    for (word, _expected_pattern) in test_cases {
        let phonemes = phonemizer.phonemize(word);
        println!("{} -> {}", word, phonemes);
        assert!(!phonemes.is_empty(), "Phonemes should not be empty for '{}'", word);
    }

    // Test a full sentence
    let sentence = "Hello world, this is a test.";
    let phonemes = phonemizer.phonemize(sentence);
    println!("\nSentence: {}", sentence);
    println!("Phonemes: {}", phonemes);
    assert!(!phonemes.is_empty());
}

#[test]
fn test_phonemes_to_token_ids() {
    let Some(model_dir) = get_kitten_tts_dir() else {
        eprintln!("Skipping test: KittenTTS model not found");
        return;
    };

    let Some(dict_path) = get_cmudict_path() else {
        eprintln!("Skipping test: CMU dictionary not found");
        return;
    };

    // Load tokens map
    let tokens_path = model_dir.join("tokens.txt");
    let tokens_content = std::fs::read_to_string(&tokens_path)
        .expect("Failed to read tokens.txt");
    let tokens_map = xybrid_core::phonemizer::load_tokens_map(&tokens_content);

    // Create phonemizer
    let phonemizer = xybrid_core::phonemizer::Phonemizer::new(&dict_path)
        .expect("Failed to create phonemizer");

    // Convert text to token IDs
    let text = "Hello world";
    let token_ids = phonemizer.text_to_token_ids(text, &tokens_map, true);

    println!("Text: {}", text);
    println!("Token IDs: {:?}", token_ids);
    println!("Token count: {}", token_ids.len());

    // Should have at least: padding + some tokens + padding
    assert!(token_ids.len() >= 3, "Should have at least 3 tokens (with padding)");

    // First and last should be padding (0)
    assert_eq!(token_ids[0], 0, "First token should be padding (0)");
    assert_eq!(*token_ids.last().unwrap(), 0, "Last token should be padding (0)");

    // Test without padding
    let token_ids_no_pad = phonemizer.text_to_token_ids(text, &tokens_map, false);
    assert!(token_ids_no_pad.len() < token_ids.len(), "Without padding should be shorter");
}

#[test]
fn test_load_voice_embeddings() {
    let Some(model_dir) = get_kitten_tts_dir() else {
        eprintln!("Skipping test: KittenTTS model not found");
        return;
    };

    let voices_path = model_dir.join("voices.bin");
    let voices_data = std::fs::read(&voices_path)
        .expect("Failed to read voices.bin");

    // voices.bin contains 8 voices, each with 256 float32 values
    // Total size: 8 * 256 * 4 bytes = 8192 bytes
    assert_eq!(voices_data.len(), 8192, "voices.bin should be 8192 bytes (8 voices × 256 dims × 4 bytes)");

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

    println!("Loaded {} voice embeddings", voices.len());
    for (i, voice) in voices.iter().enumerate() {
        let voice_names = [
            "expr-voice-2-m", "expr-voice-2-f",
            "expr-voice-3-m", "expr-voice-3-f",
            "expr-voice-4-m", "expr-voice-4-f",
            "expr-voice-5-m", "expr-voice-5-f",
        ];
        let mean: f32 = voice.iter().sum::<f32>() / voice.len() as f32;
        let max = voice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = voice.iter().cloned().fold(f32::INFINITY, f32::min);
        println!("  Voice {} ({}): mean={:.4}, min={:.4}, max={:.4}",
            i, voice_names[i], mean, min, max);
    }

    // Verify embeddings look reasonable (not all zeros, reasonable range)
    for (i, voice) in voices.iter().enumerate() {
        let sum: f32 = voice.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "Voice {} should not be all zeros", i);
    }
}

// NOTE: Full ONNX inference test is in a separate example binary
// since the ort API requires special handling. Run with:
// cargo run --example tts_inference -p xybrid-core

#[test]
fn test_model_metadata_loading() {
    let Some(model_dir) = get_kitten_tts_dir() else {
        eprintln!("Skipping test: KittenTTS model not found");
        return;
    };

    let metadata_path = model_dir.join("model_metadata.json");
    let metadata_content = std::fs::read_to_string(&metadata_path)
        .expect("Failed to read model_metadata.json");

    let metadata: serde_json::Value = serde_json::from_str(&metadata_content)
        .expect("Failed to parse model_metadata.json");

    // Verify metadata structure
    assert_eq!(metadata["model_id"], "kitten-tts-nano");
    assert_eq!(metadata["version"], "0.1");
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
