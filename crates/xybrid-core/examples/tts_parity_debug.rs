//! TTS Parity Debug: Dump intermediate values at each pipeline stage.
//!
//! Runs the KittenTTS pipeline step-by-step (reimplemented from public APIs
//! and direct dictionary loading) and dumps intermediate values as JSON
//! and binary files for comparison against the Python reference.
//!
//! Usage:
//!   cargo run --example tts_parity_debug -p xybrid-core -- \
//!     --model-dir integration-tests/fixtures/models/kitten-tts-nano-0.2 \
//!     --voice expr-voice-2-f \
//!     --output-dir tests/tts_parity/outputs/rust

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use xybrid_core::phonemizer::load_tokens_map;
use xybrid_core::tts::voice_embedding::VoiceEmbeddingLoader;

const TEST_SENTENCES: &[&str] = &[
    "Hello.",
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world! How are you today?",
    "Doctor Smith ordered three items.",
    "It's a beautiful day, isn't it?",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let model_dir = get_arg(&args, "--model-dir")
        .unwrap_or_else(|| "integration-tests/fixtures/models/kitten-tts-nano-0.2".to_string());
    let voice_name = get_arg(&args, "--voice").unwrap_or_else(|| "expr-voice-2-f".to_string());
    let output_dir = get_arg(&args, "--output-dir")
        .unwrap_or_else(|| "tests/tts_parity/outputs/rust".to_string());

    let model_path = PathBuf::from(&model_dir);
    let output_path = PathBuf::from(&output_dir);
    std::fs::create_dir_all(&output_path)?;

    println!("Model dir:  {}", model_path.display());
    println!("Voice:      {}", voice_name);
    println!("Output dir: {}", output_path.display());
    println!();

    // =========================================================================
    // Load resources
    // =========================================================================

    // Load tokens map
    let tokens_path = model_path.join("tokens.txt");
    let tokens_content = std::fs::read_to_string(&tokens_path)?;
    let tokens_map = load_tokens_map(&tokens_content);
    println!(
        "Loaded {} tokens from {}",
        tokens_map.len(),
        tokens_path.display()
    );

    // Load voice embedding
    let voices_path = model_path.join("voices.npz");
    let loader = VoiceEmbeddingLoader::new(256);

    // KittenTTS nano voices are indexed alphabetically in NPZ
    let voice_index = match voice_name.as_str() {
        "expr-voice-2-f" => 0, // First alphabetically after sorting
        "expr-voice-2-m" => 1,
        "expr-voice-3-f" => 2,
        "expr-voice-3-m" => 3,
        "expr-voice-4-f" => 4,
        "expr-voice-4-m" => 5,
        "expr-voice-5-f" => 6,
        "expr-voice-5-m" => 7,
        _ => 0,
    };
    // NPZ keys might not be in the order we expect - use by-name loading
    let voice_embedding = loader.load_npz_by_name(&voices_path, &voice_name, None)?;
    println!(
        "Loaded voice '{}' (index {}): {} floats",
        voice_name,
        voice_index,
        voice_embedding.len()
    );

    // Save voice embedding as binary f32
    save_f32_binary(&output_path.join("voice_embedding.f32"), &voice_embedding)?;

    // Load Misaki dictionaries
    let misaki_dir = model_path.join("misaki");
    let gold_dict = load_json_dict(&misaki_dir.join("us_gold.json"))?;
    let silver_dict = load_json_dict(&misaki_dir.join("us_silver.json"))?;
    let gold_grown = grow_dictionary(&gold_dict);
    let silver_grown = grow_dictionary(&silver_dict);
    println!(
        "Loaded Misaki dictionaries: gold={} silver={} (grown: gold={} silver={})",
        gold_dict.len(),
        silver_dict.len(),
        gold_grown.len(),
        silver_grown.len()
    );
    println!();

    // =========================================================================
    // Process each test sentence
    // =========================================================================

    for (idx, &sentence) in TEST_SENTENCES.iter().enumerate() {
        let sentence_num = idx + 1;
        println!("--- Sentence {}: \"{}\" ---", sentence_num, sentence);

        let mut result = serde_json::Map::new();
        result.insert(
            "input_text".to_string(),
            serde_json::Value::String(sentence.to_string()),
        );

        // Stage A: Text normalization
        let normalized = normalize_text_for_tts(sentence);
        result.insert(
            "stage_a_normalized".to_string(),
            serde_json::Value::String(normalized.clone()),
        );
        println!("  A. Normalized: \"{}\"", normalized);

        // Stage B: Phonemization with Misaki
        let phonemes = phonemize_misaki(&normalized, &gold_grown, &silver_grown, &tokens_map);
        result.insert(
            "stage_b_phonemes".to_string(),
            serde_json::Value::String(phonemes.clone()),
        );
        println!("  B. Phonemes:   \"{}\"", phonemes);

        // Stage C: Token IDs
        let mut base_ids: Vec<i64> = vec![0]; // Start pad
        for c in phonemes.chars() {
            if let Some(&id) = tokens_map.get(&c) {
                base_ids.push(id);
            } else if c == ' ' {
                if let Some(&id) = tokens_map.get(&' ') {
                    base_ids.push(id);
                }
            }
        }

        // With end token (KittenTTS Python style): [0, ...ids..., 10, 0]
        let mut ids_with_end = base_ids.clone();
        ids_with_end.push(10); // End-of-sequence token
        ids_with_end.push(0); // End pad

        // Without end token (current Xybrid style): [0, ...ids..., 0]
        let mut ids_no_end = base_ids;
        ids_no_end.push(0); // End pad

        result.insert("stage_c_token_ids".to_string(), ids_to_json(&ids_with_end));
        result.insert(
            "stage_c_token_ids_no_end".to_string(),
            ids_to_json(&ids_no_end),
        );
        println!(
            "  C. Token IDs (with end):  {:?}... (len={})",
            &ids_with_end[..ids_with_end.len().min(10)],
            ids_with_end.len()
        );
        println!(
            "  C'. Token IDs (no end):   {:?}... (len={})",
            &ids_no_end[..ids_no_end.len().min(10)],
            ids_no_end.len()
        );

        // Stage E: Speed
        let speed: f32 = 1.0;
        let speed_with_prior = speed * 0.8; // KittenTTS nano default prior
        result.insert(
            "stage_e_speed".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(speed_with_prior as f64).unwrap(),
            ),
        );
        result.insert(
            "stage_e_speed_no_prior".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(speed as f64).unwrap()),
        );
        println!("  E. Speed: {} (with prior: {})", speed, speed_with_prior);

        // Stage F: ONNX inference
        let onnx_path = model_path.join("model.onnx");
        if onnx_path.exists() {
            match run_onnx_inference(
                &onnx_path,
                &ids_with_end,
                &voice_embedding,
                speed_with_prior,
            ) {
                Ok(raw_waveform) => {
                    let rms = compute_rms(&raw_waveform);
                    result.insert(
                        "stage_f_raw_output_samples".to_string(),
                        serde_json::Value::Number(raw_waveform.len().into()),
                    );
                    result.insert(
                        "stage_f_raw_output_rms".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(rms as f64).unwrap(),
                        ),
                    );
                    result.insert(
                        "stage_f_raw_output_first_20".to_string(),
                        floats_to_json(&raw_waveform[..raw_waveform.len().min(20)]),
                    );
                    result.insert(
                        "stage_f_raw_output_last_20".to_string(),
                        floats_to_json(&raw_waveform[raw_waveform.len().saturating_sub(20)..]),
                    );

                    // Trim last 5000 samples (KittenTTS style)
                    let trimmed = if raw_waveform.len() > 5000 {
                        &raw_waveform[..raw_waveform.len() - 5000]
                    } else {
                        &raw_waveform[..]
                    };
                    result.insert(
                        "stage_g_trimmed_samples".to_string(),
                        serde_json::Value::Number(trimmed.len().into()),
                    );
                    result.insert(
                        "stage_g_trimmed_rms".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(compute_rms(trimmed) as f64).unwrap(),
                        ),
                    );

                    save_f32_binary(
                        &output_path.join(format!("sentence_{}_raw.f32", sentence_num)),
                        trimmed,
                    )?;
                    save_wav(
                        &output_path.join(format!("sentence_{}.wav", sentence_num)),
                        trimmed,
                        24000,
                    )?;

                    println!(
                        "  F. Raw output: {} samples, RMS={:.6}",
                        raw_waveform.len(),
                        rms
                    );
                    println!("  G. Trimmed:    {} samples", trimmed.len());
                }
                Err(e) => println!("  F. ONNX ERROR: {}", e),
            }
        } else {
            println!("  F. SKIP (model.onnx not found)");
        }

        // Save JSON
        let json_path = output_path.join(format!("sentence_{}.json", sentence_num));
        let json_str = serde_json::to_string_pretty(&serde_json::Value::Object(result))?;
        std::fs::write(&json_path, json_str)?;
        println!();
    }

    println!("All outputs saved to {}", output_path.display());
    Ok(())
}

// =============================================================================
// Text normalization (reimplemented to match Xybrid's normalize_text_for_tts)
// =============================================================================

fn normalize_text_for_tts(text: &str) -> String {
    let mut s = text.to_string();

    // Smart quotes
    s = s.replace(['\u{201c}', '\u{201d}'], "\"");
    s = s.replace(['\u{2018}', '\u{2019}'], "'");

    // Abbreviations
    let abbrevs = [
        ("Dr.", "Doctor"),
        ("Mr.", "Mister"),
        ("Mrs.", "Misses"),
        ("Ms.", "Miss"),
        ("Prof.", "Professor"),
        ("Jr.", "Junior"),
        ("Sr.", "Senior"),
        ("St.", "Saint"),
        ("vs.", "versus"),
        ("etc.", "etcetera"),
    ];
    for (abbr, expanded) in abbrevs {
        s = s.replace(abbr, expanded);
    }

    // Currency ($X.YY)
    let re_currency = regex::Regex::new(r"\$(\d+)\.(\d{2})").unwrap();
    s = re_currency
        .replace_all(&s, |caps: &regex::Captures| {
            let dollars: u64 = caps[1].parse().unwrap_or(0);
            let cents: u64 = caps[2].parse().unwrap_or(0);
            let mut parts = Vec::new();
            if dollars > 0 {
                parts.push(format!(
                    "{} dollar{}",
                    num_to_words(dollars),
                    if dollars == 1 { "" } else { "s" }
                ));
            }
            if cents > 0 {
                parts.push(format!(
                    "{} cent{}",
                    num_to_words(cents),
                    if cents == 1 { "" } else { "s" }
                ));
            }
            parts.join(" and ")
        })
        .to_string();

    // Simple $X
    let re_dollar = regex::Regex::new(r"\$(\d+)").unwrap();
    s = re_dollar
        .replace_all(&s, |caps: &regex::Captures| {
            let n: u64 = caps[1].parse().unwrap_or(0);
            format!("{} dollars", num_to_words(n))
        })
        .to_string();

    // Percentages
    let re_pct = regex::Regex::new(r"(\d+)%").unwrap();
    s = re_pct
        .replace_all(&s, |caps: &regex::Captures| {
            let n: u64 = caps[1].parse().unwrap_or(0);
            format!("{} percent", num_to_words(n))
        })
        .to_string();

    // Numbers to words
    let re_num = regex::Regex::new(r"\b(\d+)\b").unwrap();
    s = re_num
        .replace_all(&s, |caps: &regex::Captures| {
            let n: u64 = caps[1].parse().unwrap_or(0);
            num_to_words(n)
        })
        .to_string();

    // Ellipsis normalization
    s = s.replace("...", "\u{2026}");

    // Collapse whitespace
    let re_ws = regex::Regex::new(r"\s+").unwrap();
    s = re_ws.replace_all(&s, " ").trim().to_string();

    s
}

fn num_to_words(n: u64) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    let ones = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    let tens_words = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    if n < 20 {
        return ones[n as usize].to_string();
    }
    if n < 100 {
        let t = tens_words[(n / 10) as usize];
        let o = n % 10;
        return if o == 0 {
            t.to_string()
        } else {
            format!("{} {}", t, ones[o as usize])
        };
    }
    if n < 1000 {
        let h = ones[(n / 100) as usize];
        let rem = n % 100;
        return if rem == 0 {
            format!("{} hundred", h)
        } else {
            format!("{} hundred {}", h, num_to_words(rem))
        };
    }
    if n < 1_000_000 {
        let t = num_to_words(n / 1000);
        let rem = n % 1000;
        return if rem == 0 {
            format!("{} thousand", t)
        } else {
            format!("{} thousand {}", t, num_to_words(rem))
        };
    }
    format!("{}", n) // fallback
}

// =============================================================================
// Misaki phonemization (reimplemented matching Xybrid's MisakiBackend)
// =============================================================================

fn load_json_dict(
    path: &Path,
) -> Result<HashMap<String, serde_json::Value>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let content = std::fs::read_to_string(path)?;
    let dict: HashMap<String, serde_json::Value> = serde_json::from_str(&content)?;
    Ok(dict)
}

fn grow_dictionary(
    dict: &HashMap<String, serde_json::Value>,
) -> HashMap<String, serde_json::Value> {
    let mut grown: HashMap<String, serde_json::Value> = HashMap::new();

    for (k, v) in dict {
        if k.len() < 2 {
            continue;
        }
        let lower = k.to_lowercase();
        if *k == lower {
            let mut chars = k.chars();
            if let Some(first) = chars.next() {
                let capitalized: String = first.to_uppercase().chain(chars).collect();
                if capitalized != *k {
                    grown.entry(capitalized).or_insert_with(|| v.clone());
                }
            }
        } else {
            let mut chars = k.chars();
            let first = chars.next().unwrap();
            let rest: String = chars.collect();
            if first.is_uppercase() && rest == rest.to_lowercase() {
                grown.entry(lower).or_insert_with(|| v.clone());
            }
        }
    }

    for (k, v) in dict {
        grown.insert(k.clone(), v.clone());
    }
    grown
}

fn lookup_word(word: &str, dict: &HashMap<String, serde_json::Value>) -> Option<String> {
    dict.get(word).and_then(|v| match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(obj) => obj
            .get("DEFAULT")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string()),
        _ => None,
    })
}

fn phonemize_word(
    word: &str,
    gold: &HashMap<String, serde_json::Value>,
    silver: &HashMap<String, serde_json::Value>,
) -> String {
    let lower = word.to_lowercase();

    // Dictionary lookup
    if let Some(ps) = lookup_word(&lower, gold)
        .or_else(|| lookup_word(&lower, silver))
        .or_else(|| lookup_word(word, gold))
        .or_else(|| lookup_word(word, silver))
    {
        return ps;
    }

    // Acronym detection
    if word.len() >= 2 && word.chars().all(|c| c.is_ascii_uppercase()) {
        return spell_as_letters(word);
    }

    // Fallback: return the word as-is (will be filtered by vocab later)
    lower
}

fn spell_as_letters(word: &str) -> String {
    let mut parts = Vec::new();
    for c in word.chars() {
        let phoneme = match c.to_ascii_uppercase() {
            'A' => "ˈeɪ",
            'B' => "bˈiː",
            'C' => "sˈiː",
            'D' => "dˈiː",
            'E' => "ˈiː",
            'F' => "ˈɛf",
            'G' => "ʤˈiː",
            'H' => "ˈeɪʧ",
            'I' => "ˈaɪ",
            'J' => "ʤˈeɪ",
            'K' => "kˈeɪ",
            'L' => "ˈɛl",
            'M' => "ˈɛm",
            'N' => "ˈɛn",
            'O' => "ˈoʊ",
            'P' => "pˈiː",
            'Q' => "kjˈuː",
            'R' => "ˈɑːɹ",
            'S' => "ˈɛs",
            'T' => "tˈiː",
            'U' => "jˈuː",
            'V' => "vˈiː",
            'W' => "dˈʌbəljˌuː",
            'X' => "ˈɛks",
            'Y' => "wˈaɪ",
            'Z' => "zˈiː",
            _ => "",
        };
        if !phoneme.is_empty() {
            parts.push(phoneme);
        }
    }
    parts.join(" ")
}

fn phonemize_misaki(
    text: &str,
    gold: &HashMap<String, serde_json::Value>,
    silver: &HashMap<String, serde_json::Value>,
    vocab: &HashMap<char, i64>,
) -> String {
    let mut result = String::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        // Extract leading/trailing punctuation
        let trimmed_start = word.trim_start_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        let leading_punct = &word[..word.len() - trimmed_start.len()];
        let clean_word =
            trimmed_start.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        let trailing_punct = &trimmed_start[clean_word.len()..];

        // Emit leading punctuation if in vocab
        for c in leading_punct.chars() {
            if vocab.contains_key(&c) {
                result.push(c);
            }
        }

        // Phonemize the word
        if !clean_word.is_empty() {
            if clean_word.contains('-') {
                for part in clean_word.split('-') {
                    if !part.is_empty() {
                        result.push_str(&phonemize_word(part, gold, silver));
                    }
                }
            } else {
                result.push_str(&phonemize_word(clean_word, gold, silver));
            }
        }

        // Emit trailing punctuation
        for c in trailing_punct.chars() {
            if vocab.contains_key(&c) {
                result.push(c);
            }
        }

        // Space between words
        if i < words.len() - 1 {
            result.push(' ');
        }
    }

    // Post-phonemization replacements
    let result = result.replace('ɾ', "T").replace('ʔ', "t");

    // Filter to vocabulary
    let filtered: String = result.chars().filter(|c| vocab.contains_key(c)).collect();
    filtered.trim().to_string()
}

// =============================================================================
// ONNX inference
// =============================================================================

fn run_onnx_inference(
    model_path: &Path,
    token_ids: &[i64],
    voice_embedding: &[f32],
    speed: f32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use ort::session::{Session, SessionInputs};

    let mut session = Session::builder()?.commit_from_file(model_path)?;

    // Collect input names
    let input_names: Vec<String> = session
        .inputs()
        .iter()
        .map(|i| i.name().to_string())
        .collect();
    let mut ort_inputs: Vec<(String, ort::value::DynValue)> = Vec::new();

    for name in &input_names {
        if name.contains("input_id") || name.contains("token") {
            let data: Vec<i64> = token_ids.to_vec();
            let tensor = ndarray::Array2::from_shape_vec((1, data.len()), data)?;
            ort_inputs.push((name.clone(), ort::value::Value::from_array(tensor)?.into()));
        } else if name.contains("style") {
            let data: Vec<f32> = voice_embedding.to_vec();
            let tensor = ndarray::Array2::from_shape_vec((1, 256), data)?;
            ort_inputs.push((name.clone(), ort::value::Value::from_array(tensor)?.into()));
        } else if name.contains("speed") {
            let tensor = ndarray::Array1::from_vec(vec![speed]);
            ort_inputs.push((name.clone(), ort::value::Value::from_array(tensor)?.into()));
        }
    }

    // Sort to match session input order
    ort_inputs.sort_by_key(|(name, _)| {
        input_names
            .iter()
            .position(|n| n == name)
            .unwrap_or(usize::MAX)
    });

    let outputs = session.run(SessionInputs::from(ort_inputs))?;
    let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
    Ok(output_tensor.1.to_vec())
}

// =============================================================================
// Utility functions
// =============================================================================

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn ids_to_json(ids: &[i64]) -> serde_json::Value {
    serde_json::Value::Array(
        ids.iter()
            .map(|&id| serde_json::Value::Number(serde_json::Number::from(id)))
            .collect(),
    )
}

fn floats_to_json(data: &[f32]) -> serde_json::Value {
    serde_json::Value::Array(
        data.iter()
            .map(|&v| {
                serde_json::Value::Number(
                    serde_json::Number::from_f64(v as f64)
                        .unwrap_or_else(|| serde_json::Number::from_f64(0.0).unwrap()),
                )
            })
            .collect(),
    )
}

fn save_f32_binary(path: &Path, data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::create(path)?;
    for &val in data {
        file.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}

fn save_wav(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::create(path)?;
    let pcm16: Vec<i16> = samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect();
    let data_size = (pcm16.len() * 2) as u32;
    let file_size = 36 + data_size;

    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&(sample_rate * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for sample in &pcm16 {
        file.write_all(&sample.to_le_bytes())?;
    }
    Ok(())
}
