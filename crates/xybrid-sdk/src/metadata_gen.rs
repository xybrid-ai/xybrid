//! Auto-generation of `model_metadata.json` from HuggingFace model cards and model file inspection.
//!
//! When a HuggingFace repo doesn't contain a `model_metadata.json`, this module attempts to
//! generate one by:
//!
//! 1. Parsing the HF model card (README.md YAML frontmatter) for task type and library info
//! 2. Scanning the directory for model files (.onnx, .gguf, .safetensors)
//! 3. For GGUF models: reading binary metadata (architecture, context_length)
//! 4. For ONNX models: inspecting input/output tensor names and shapes via ort
//! 5. Falling back to a generic template if auto-detection fails

use crate::model::{SdkError, SdkResult};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use xybrid_core::execution::ModelMetadata;

// ============================================================================
// Public API
// ============================================================================

/// Attempt to auto-generate a `model_metadata.json` for a downloaded HuggingFace repo.
///
/// Scans the directory for model files, parses the README.md model card, and inspects
/// model file headers to produce a best-effort `ModelMetadata`.
///
/// Returns `Ok(metadata)` on success and writes `model_metadata.json` to `cache_dir`.
/// Falls back to a generic template if specific detection fails, logging a warning.
pub fn generate_metadata(cache_dir: &Path, repo: &str) -> SdkResult<ModelMetadata> {
    log::info!(
        target: "xybrid_sdk",
        "Auto-generating model_metadata.json for HuggingFace repo '{}'",
        repo
    );

    // 1. Parse HF model card if README.md exists
    let model_card = parse_hf_model_card(&cache_dir.join("README.md"));

    // 2. Scan for model files
    let model_files = detect_model_files(cache_dir);

    if model_files.is_empty() {
        return Err(SdkError::LoadError(format!(
            "No model files (.onnx, .gguf, .safetensors) found in '{}'",
            cache_dir.display()
        )));
    }

    // 3. Inspect model files for metadata
    let gguf_info = model_files
        .iter()
        .find(|f| f.format == ModelFormat::Gguf)
        .and_then(|f| read_gguf_metadata(&cache_dir.join(&f.filename)));

    let onnx_info = model_files
        .iter()
        .find(|f| f.format == ModelFormat::Onnx)
        .and_then(|f| inspect_onnx_model(&cache_dir.join(&f.filename)));

    // 4. Build metadata from collected info
    let metadata = build_metadata(
        repo,
        &model_files,
        model_card.as_ref(),
        gguf_info.as_ref(),
        onnx_info.as_ref(),
        cache_dir,
    );

    // 5. Write to cache directory
    let metadata_path = cache_dir.join("model_metadata.json");
    let json = serde_json::to_string_pretty(&metadata).map_err(|e| {
        SdkError::MetadataInvalid(format!("Failed to serialize generated metadata: {}", e))
    })?;
    std::fs::write(&metadata_path, &json)?;

    log::info!(
        target: "xybrid_sdk",
        "Generated model_metadata.json at {}",
        metadata_path.display()
    );

    Ok(metadata)
}

/// Generate `ModelMetadata` for a standalone GGUF file on disk.
///
/// Reads the GGUF binary header to extract architecture and context length,
/// then returns a ready-to-use `ModelMetadata` without writing anything to disk.
/// This enables `--model-file ./path/to/model.gguf` workflows.
pub fn generate_metadata_for_gguf_file(gguf_path: &Path) -> SdkResult<ModelMetadata> {
    if !gguf_path.exists() {
        return Err(SdkError::LoadError(format!(
            "GGUF file not found: {}",
            gguf_path.display()
        )));
    }

    let filename = gguf_path
        .file_name()
        .and_then(|f| f.to_str())
        .ok_or_else(|| SdkError::LoadError("Invalid GGUF file path".to_string()))?
        .to_string();

    let model_id = filename
        .trim_end_matches(".gguf")
        .to_lowercase()
        .replace(' ', "-");

    let gguf_info = read_gguf_metadata(gguf_path);

    let file_info = ModelFileInfo {
        filename: filename.clone(),
        format: ModelFormat::Gguf,
        size_bytes: std::fs::metadata(gguf_path).map(|m| m.len()).unwrap_or(0),
    };

    Ok(build_gguf_metadata(
        &model_id,
        &filename,
        &file_info,
        "text-generation",
        None,
        gguf_info.as_ref(),
    ))
}

// ============================================================================
// HuggingFace Model Card Parsing
// ============================================================================

/// Parsed information from a HuggingFace model card (README.md YAML frontmatter).
#[derive(Debug, Clone, Default)]
pub(crate) struct HfModelCard {
    /// The `pipeline_tag` field (e.g., "text-generation", "text-to-speech", "automatic-speech-recognition")
    pub pipeline_tag: Option<String>,
    /// The `library_name` field (e.g., "transformers", "onnx", "gguf")
    pub library_name: Option<String>,
    /// The `tags` field (e.g., ["gguf", "llama", "text-generation"])
    pub tags: Vec<String>,
    /// The `model_name` field
    pub model_name: Option<String>,
    /// The `language` or `languages` field
    pub languages: Vec<String>,
    /// The `license` field
    pub license: Option<String>,
}

/// Intermediate deserialization target for YAML frontmatter.
#[derive(Debug, Deserialize, Default)]
struct RawFrontmatter {
    pipeline_tag: Option<String>,
    library_name: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    model_name: Option<String>,
    #[serde(default)]
    language: LanguageField,
    license: Option<String>,
}

/// The `language` field can be a string or list of strings.
#[derive(Debug, Deserialize, Default)]
#[serde(untagged)]
enum LanguageField {
    Single(String),
    Multiple(Vec<String>),
    #[default]
    None,
}

fn parse_hf_model_card(readme_path: &Path) -> Option<HfModelCard> {
    let content = std::fs::read_to_string(readme_path).ok()?;

    // Extract YAML frontmatter between --- delimiters
    let frontmatter = extract_yaml_frontmatter(&content)?;

    let raw: RawFrontmatter = serde_yaml::from_str(&frontmatter)
        .map_err(|e| {
            log::debug!(target: "xybrid_sdk", "Failed to parse model card YAML: {}", e);
            e
        })
        .ok()?;

    let languages = match raw.language {
        LanguageField::Single(s) => vec![s],
        LanguageField::Multiple(v) => v,
        LanguageField::None => Vec::new(),
    };

    Some(HfModelCard {
        pipeline_tag: raw.pipeline_tag,
        library_name: raw.library_name,
        tags: raw.tags,
        model_name: raw.model_name,
        languages,
        license: raw.license,
    })
}

fn extract_yaml_frontmatter(content: &str) -> Option<String> {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return None;
    }

    // Find the closing ---
    let after_first = &trimmed[3..];
    let end_pos = after_first.find("\n---")?;
    Some(after_first[..end_pos].to_string())
}

// ============================================================================
// Model File Detection
// ============================================================================

/// Detected model file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelFormat {
    Onnx,
    Gguf,
    SafeTensors,
}

/// Information about a detected model file.
#[derive(Debug, Clone)]
pub(crate) struct ModelFileInfo {
    pub filename: String,
    pub format: ModelFormat,
    pub size_bytes: u64,
}

fn detect_model_files(dir: &Path) -> Vec<ModelFileInfo> {
    let mut files = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return files,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            // Also follow symlinks
            if path.is_symlink() {
                if let Ok(target) = std::fs::metadata(&path) {
                    if !target.is_file() {
                        continue;
                    }
                } else {
                    continue;
                }
            } else {
                continue;
            }
        }

        let filename = entry.file_name().to_string_lossy().to_string();
        let format = match filename.rsplit('.').next() {
            Some("onnx") => ModelFormat::Onnx,
            Some("gguf") => ModelFormat::Gguf,
            Some("safetensors") => ModelFormat::SafeTensors,
            _ => continue,
        };

        let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

        files.push(ModelFileInfo {
            filename,
            format,
            size_bytes,
        });
    }

    // Sort by size descending (largest model file first — likely the main one)
    files.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    files
}

// ============================================================================
// GGUF Metadata Reading
// ============================================================================

/// Extracted metadata from a GGUF file header.
#[derive(Debug, Clone, Default)]
pub(crate) struct GgufInfo {
    /// The model architecture (e.g., "llama", "qwen2", "gemma")
    pub architecture: Option<String>,
    /// The model name from GGUF metadata
    pub model_name: Option<String>,
    /// Context length from GGUF metadata
    pub context_length: Option<u64>,
    /// Number of parameters (if available)
    pub parameter_count: Option<u64>,
    /// Quantization type inferred from filename
    pub quantization: Option<String>,
}

// GGUF value types
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

fn read_gguf_metadata(path: &Path) -> Option<GgufInfo> {
    let file = std::fs::File::open(path)
        .map_err(|e| {
            log::debug!(target: "xybrid_sdk", "Failed to open GGUF file: {}", e);
            e
        })
        .ok()?;

    let mut reader = std::io::BufReader::new(file);
    let mut info = GgufInfo::default();

    // Read magic number (4 bytes: "GGUF")
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        log::debug!(target: "xybrid_sdk", "Not a GGUF file: invalid magic");
        return None;
    }

    // Read version (u32 LE)
    let version = read_u32_le(&mut reader)?;
    if !(2..=3).contains(&version) {
        log::debug!(target: "xybrid_sdk", "Unsupported GGUF version: {}", version);
        return None;
    }

    // Read tensor count and metadata KV count
    let _tensor_count = if version >= 3 {
        read_u64_le(&mut reader)?
    } else {
        read_u32_le(&mut reader)? as u64
    };

    let metadata_kv_count = if version >= 3 {
        read_u64_le(&mut reader)?
    } else {
        read_u32_le(&mut reader)? as u64
    };

    // Limit to prevent runaway reads on corrupted files
    let kv_limit = metadata_kv_count.min(1000);

    // Read metadata key-value pairs
    for _ in 0..kv_limit {
        let key = match read_gguf_string(&mut reader) {
            Some(k) => k,
            None => break,
        };
        let value_type = match read_u32_le(&mut reader) {
            Some(v) => v,
            None => break,
        };

        // We only care about specific keys
        match key.as_str() {
            "general.architecture" => {
                if value_type == GGUF_TYPE_STRING {
                    info.architecture = read_gguf_string(&mut reader);
                } else {
                    skip_gguf_value(&mut reader, value_type);
                }
            }
            "general.name" => {
                if value_type == GGUF_TYPE_STRING {
                    info.model_name = read_gguf_string(&mut reader);
                } else {
                    skip_gguf_value(&mut reader, value_type);
                }
            }
            k if k.ends_with(".context_length") => {
                info.context_length = read_gguf_uint_value(&mut reader, value_type);
            }
            _ => {
                // Skip values we don't need
                if !skip_gguf_value(&mut reader, value_type) {
                    break;
                }
            }
        }

        // Early exit if we have everything we need
        if info.architecture.is_some() && info.context_length.is_some() {
            break;
        }
    }

    // Infer quantization from filename
    let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
    info.quantization = infer_quantization_from_filename(filename);

    log::debug!(target: "xybrid_sdk", "GGUF metadata: {:?}", info);
    Some(info)
}

fn infer_quantization_from_filename(filename: &str) -> Option<String> {
    let lower = filename.to_lowercase();
    // Common GGUF quantization patterns
    for q in &[
        "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q5_0", "q5_1",
        "q5_k_s", "q5_k_m", "q6_k", "q8_0", "f16", "f32",
    ] {
        if lower.contains(q) {
            return Some(q.to_uppercase());
        }
    }
    None
}

// GGUF binary reading helpers

fn read_u32_le<R: Read>(reader: &mut R) -> Option<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).ok()?;
    Some(u32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(reader: &mut R) -> Option<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).ok()?;
    Some(u64::from_le_bytes(buf))
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Option<String> {
    let len = read_u64_le(reader)? as usize;
    if len > 1_000_000 {
        return None; // Sanity limit
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

fn read_gguf_uint_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> Option<u64> {
    match value_type {
        GGUF_TYPE_UINT8 => {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf).ok()?;
            Some(buf[0] as u64)
        }
        GGUF_TYPE_UINT16 => {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf).ok()?;
            Some(u16::from_le_bytes(buf) as u64)
        }
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 => Some(read_u32_le(reader)? as u64),
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 => read_u64_le(reader),
        _ => {
            skip_gguf_value(reader, value_type);
            None
        }
    }
}

/// Skip a GGUF value of the given type. Returns false if the reader is at EOF or corrupted.
fn skip_gguf_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> bool {
    let skip_bytes: i64 = match value_type {
        GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 | GGUF_TYPE_BOOL => 1,
        GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => 2,
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => 4,
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => 8,
        GGUF_TYPE_STRING => {
            if let Some(len) = read_u64_le(reader) {
                return reader.seek(SeekFrom::Current(len as i64)).is_ok();
            }
            return false;
        }
        GGUF_TYPE_ARRAY => {
            // Array: element_type (u32) + count (u64) + elements
            let elem_type = match read_u32_le(reader) {
                Some(t) => t,
                None => return false,
            };
            let count = match read_u64_le(reader) {
                Some(c) => c,
                None => return false,
            };
            let limit = count.min(100_000);
            for _ in 0..limit {
                if !skip_gguf_value(reader, elem_type) {
                    return false;
                }
            }
            return true;
        }
        _ => return false,
    };
    reader.seek(SeekFrom::Current(skip_bytes)).is_ok()
}

// ============================================================================
// ONNX Model Inspection
// ============================================================================

/// Extracted information from an ONNX model's input/output tensors.
#[derive(Debug, Clone, Default)]
pub(crate) struct OnnxInfo {
    /// Input tensor names and their shapes (negative dims = dynamic)
    pub inputs: Vec<TensorInfo>,
    /// Output tensor names and their shapes
    pub outputs: Vec<TensorInfo>,
}

/// Information about a single tensor.
#[derive(Debug, Clone)]
pub(crate) struct TensorInfo {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
}

/// Inspect an ONNX model to extract input/output tensor information.
///
/// Currently returns None — full ONNX inspection requires the ort crate which is
/// a xybrid-core dependency. For ONNX models, task-based inference from the HF
/// model card provides sufficient preprocessing/postprocessing hints in most cases.
///
/// Future improvement: add optional ort dependency or delegate inspection to xybrid-core.
fn inspect_onnx_model(_path: &Path) -> Option<OnnxInfo> {
    log::debug!(
        target: "xybrid_sdk",
        "ONNX tensor inspection not yet available in SDK. \
         Using task-based inference from model card instead."
    );
    None
}

// ============================================================================
// Metadata Construction
// ============================================================================

fn build_metadata(
    repo: &str,
    model_files: &[ModelFileInfo],
    card: Option<&HfModelCard>,
    gguf_info: Option<&GgufInfo>,
    onnx_info: Option<&OnnxInfo>,
    cache_dir: &Path,
) -> ModelMetadata {
    // Determine the primary model file (largest file of the detected format)
    let primary = &model_files[0];

    // Derive model_id from repo name (last component)
    let model_id = repo
        .rsplit('/')
        .next()
        .unwrap_or(repo)
        .to_lowercase()
        .replace(' ', "-");

    // Determine task from model card or inference
    let task = card
        .and_then(|c| c.pipeline_tag.clone())
        .or_else(|| infer_task_from_tags(card))
        .unwrap_or_else(|| "unknown".to_string());

    match primary.format {
        ModelFormat::Gguf => build_gguf_metadata(&model_id, repo, primary, &task, card, gguf_info),
        ModelFormat::Onnx => build_onnx_metadata(
            &model_id,
            repo,
            primary,
            &task,
            card,
            onnx_info,
            model_files,
            cache_dir,
        ),
        ModelFormat::SafeTensors => {
            build_safetensors_metadata(&model_id, repo, primary, &task, card, model_files)
        }
    }
}

fn build_gguf_metadata(
    model_id: &str,
    repo: &str,
    primary: &ModelFileInfo,
    task: &str,
    card: Option<&HfModelCard>,
    gguf_info: Option<&GgufInfo>,
) -> ModelMetadata {
    use xybrid_core::execution::ExecutionTemplate;

    let context_length = gguf_info.and_then(|g| g.context_length).unwrap_or(4096) as usize;

    let architecture = gguf_info
        .and_then(|g| g.architecture.clone())
        .unwrap_or_else(|| "unknown".to_string());

    let mut metadata_map = HashMap::new();
    metadata_map.insert(
        "task".to_string(),
        serde_json::Value::String(task.to_string()),
    );
    metadata_map.insert(
        "architecture".to_string(),
        serde_json::Value::String(architecture.clone()),
    );
    metadata_map.insert(
        "backend".to_string(),
        serde_json::Value::String("llamacpp".to_string()),
    );
    metadata_map.insert(
        "context_length".to_string(),
        serde_json::json!(gguf_info
            .and_then(|g| g.context_length)
            .unwrap_or(context_length as u64)),
    );
    metadata_map.insert(
        "source_repo".to_string(),
        serde_json::Value::String(repo.to_string()),
    );
    metadata_map.insert("auto_generated".to_string(), serde_json::Value::Bool(true));

    if let Some(q) = gguf_info.and_then(|g| g.quantization.clone()) {
        metadata_map.insert("quantization".to_string(), serde_json::Value::String(q));
    }

    if let Some(card) = card {
        if !card.languages.is_empty() {
            metadata_map.insert("languages".to_string(), serde_json::json!(card.languages));
        }
        if let Some(license) = &card.license {
            metadata_map.insert(
                "license".to_string(),
                serde_json::Value::String(license.clone()),
            );
        }
    }

    let description = gguf_info
        .and_then(|g| g.model_name.clone())
        .or_else(|| card.and_then(|c| c.model_name.clone()))
        .unwrap_or_else(|| format!("{} (auto-generated from {})", model_id, repo));

    ModelMetadata {
        model_id: model_id.to_string(),
        version: "1.0".to_string(),
        execution_template: ExecutionTemplate::Gguf {
            model_file: primary.filename.clone(),
            chat_template: None,
            context_length,
        },
        preprocessing: Vec::new(),
        postprocessing: Vec::new(),
        files: vec![primary.filename.clone()],
        description: Some(description),
        metadata: metadata_map,
        voices: None,
        max_chunk_chars: None,
        trim_trailing_samples: None,
    }
}

fn build_onnx_metadata(
    model_id: &str,
    repo: &str,
    primary: &ModelFileInfo,
    task: &str,
    card: Option<&HfModelCard>,
    onnx_info: Option<&OnnxInfo>,
    all_files: &[ModelFileInfo],
    cache_dir: &Path,
) -> ModelMetadata {
    use xybrid_core::execution::template::TokenizerType;
    use xybrid_core::execution::{PostprocessingStep, PreprocessingStep};

    let mut preprocessing = Vec::new();
    let mut postprocessing = Vec::new();
    let mut files: Vec<String> = vec![primary.filename.clone()];

    // Prefer tokenizer.json (HuggingFace fast tokenizer format) over vocab.txt.
    // The `tokenizers` crate expects the JSON format; vocab.txt is a plain word list
    // that cannot be loaded directly.
    let has_tokenizer_json = cache_dir.join("tokenizer.json").exists();
    let tokenizer_file = if has_tokenizer_json {
        "tokenizer.json"
    } else {
        "vocab.txt"
    };

    // Infer preprocessing/postprocessing from task + ONNX tensor info
    match task {
        "automatic-speech-recognition" | "speech-recognition" => {
            preprocessing.push(PreprocessingStep::AudioDecode {
                sample_rate: 16000,
                channels: 1,
            });
            postprocessing.push(PostprocessingStep::CTCDecode {
                vocab_file: "vocab.json".to_string(),
                blank_index: 0,
            });
            files.push("vocab.json".to_string());
        }
        "text-to-speech" | "tts" => {
            preprocessing.push(PreprocessingStep::Phonemize {
                tokens_file: "tokens.txt".to_string(),
                dict_file: None,
                backend: Default::default(),
                language: None,
                add_padding: true,
                normalize_text: false,
                silence_tokens: None,
            });
            postprocessing.push(PostprocessingStep::TTSAudioEncode {
                sample_rate: 24000,
                apply_postprocessing: true,
                trim_trailing_silence: false,
            });
            files.push("tokens.txt".to_string());
        }
        "text-classification" | "sentiment-analysis" => {
            preprocessing.push(PreprocessingStep::Tokenize {
                vocab_file: tokenizer_file.to_string(),
                tokenizer_type: TokenizerType::WordPiece,
                max_length: Some(512),
            });
            postprocessing.push(PostprocessingStep::Argmax { dim: None });
            files.push(tokenizer_file.to_string());
        }
        "token-classification" | "ner" => {
            preprocessing.push(PreprocessingStep::Tokenize {
                vocab_file: tokenizer_file.to_string(),
                tokenizer_type: TokenizerType::WordPiece,
                max_length: Some(512),
            });
            postprocessing.push(PostprocessingStep::Argmax { dim: None });
            files.push(tokenizer_file.to_string());
        }
        "image-classification" => {
            preprocessing.push(PreprocessingStep::Normalize {
                mean: vec![0.485, 0.456, 0.406],
                std: vec![0.229, 0.224, 0.225],
            });
            postprocessing.push(PostprocessingStep::Argmax { dim: None });
        }
        "feature-extraction" | "sentence-similarity" => {
            preprocessing.push(PreprocessingStep::Tokenize {
                vocab_file: tokenizer_file.to_string(),
                tokenizer_type: TokenizerType::WordPiece,
                max_length: Some(512),
            });
            files.push(tokenizer_file.to_string());
            // Output is typically embeddings — no postprocessing needed
        }
        _ => {
            // Generic: try to infer from ONNX input names
            if let Some(info) = onnx_info {
                infer_steps_from_onnx(
                    info,
                    &mut preprocessing,
                    &mut postprocessing,
                    &mut files,
                    tokenizer_file,
                );
            } else {
                log::warn!(
                    target: "xybrid_sdk",
                    "Could not determine preprocessing/postprocessing for task '{}'. \
                     The generated model_metadata.json may need manual adjustment.",
                    task
                );
            }
        }
    }

    // Add any additional ONNX files found
    for f in all_files {
        if f.format == ModelFormat::Onnx && f.filename != primary.filename {
            files.push(f.filename.clone());
        }
    }

    let mut metadata_map = HashMap::new();
    metadata_map.insert(
        "task".to_string(),
        serde_json::Value::String(task.to_string()),
    );
    metadata_map.insert(
        "source_repo".to_string(),
        serde_json::Value::String(repo.to_string()),
    );
    metadata_map.insert("auto_generated".to_string(), serde_json::Value::Bool(true));

    if let Some(info) = onnx_info {
        let input_names: Vec<String> = info.inputs.iter().map(|i| i.name.clone()).collect();
        let output_names: Vec<String> = info.outputs.iter().map(|o| o.name.clone()).collect();
        metadata_map.insert("onnx_inputs".to_string(), serde_json::json!(input_names));
        metadata_map.insert("onnx_outputs".to_string(), serde_json::json!(output_names));
    }

    if let Some(card) = card {
        if !card.languages.is_empty() {
            metadata_map.insert("languages".to_string(), serde_json::json!(card.languages));
        }
    }

    let description = card
        .and_then(|c| c.model_name.clone())
        .unwrap_or_else(|| format!("{} (auto-generated from {})", model_id, repo));

    ModelMetadata {
        model_id: model_id.to_string(),
        version: "1.0".to_string(),
        execution_template: xybrid_core::execution::ExecutionTemplate::Onnx {
            model_file: primary.filename.clone(),
        },
        preprocessing,
        postprocessing,
        files,
        description: Some(description),
        metadata: metadata_map,
        voices: None,
        max_chunk_chars: None,
        trim_trailing_samples: None,
    }
}

fn build_safetensors_metadata(
    model_id: &str,
    repo: &str,
    primary: &ModelFileInfo,
    task: &str,
    card: Option<&HfModelCard>,
    all_files: &[ModelFileInfo],
) -> ModelMetadata {
    let mut files: Vec<String> = vec![primary.filename.clone()];

    for f in all_files {
        if f.format == ModelFormat::SafeTensors && f.filename != primary.filename {
            files.push(f.filename.clone());
        }
    }

    let architecture = card
        .and_then(|c| {
            c.tags.iter().find(|t| {
                matches!(
                    t.as_str(),
                    "whisper"
                        | "llama"
                        | "gpt2"
                        | "bert"
                        | "t5"
                        | "mistral"
                        | "phi"
                        | "gemma"
                        | "qwen"
                )
            })
        })
        .cloned();

    let mut metadata_map = HashMap::new();
    metadata_map.insert(
        "task".to_string(),
        serde_json::Value::String(task.to_string()),
    );
    metadata_map.insert(
        "source_repo".to_string(),
        serde_json::Value::String(repo.to_string()),
    );
    metadata_map.insert("auto_generated".to_string(), serde_json::Value::Bool(true));

    let description = card
        .and_then(|c| c.model_name.clone())
        .unwrap_or_else(|| format!("{} (auto-generated from {})", model_id, repo));

    ModelMetadata {
        model_id: model_id.to_string(),
        version: "1.0".to_string(),
        execution_template: xybrid_core::execution::ExecutionTemplate::SafeTensors {
            model_file: primary.filename.clone(),
            architecture,
            config_file: None,
            tokenizer_file: None,
        },
        preprocessing: Vec::new(),
        postprocessing: Vec::new(),
        files,
        description: Some(description),
        metadata: metadata_map,
        voices: None,
        max_chunk_chars: None,
        trim_trailing_samples: None,
    }
}

/// Try to infer preprocessing/postprocessing from ONNX input/output tensor names.
fn infer_steps_from_onnx(
    info: &OnnxInfo,
    preprocessing: &mut Vec<xybrid_core::execution::PreprocessingStep>,
    postprocessing: &mut Vec<xybrid_core::execution::PostprocessingStep>,
    files: &mut Vec<String>,
    tokenizer_file: &str,
) {
    use xybrid_core::execution::template::TokenizerType;
    use xybrid_core::execution::{PostprocessingStep, PreprocessingStep};

    let input_names: Vec<&str> = info.inputs.iter().map(|i| i.name.as_str()).collect();

    // Check for tokenized text inputs (input_ids, attention_mask)
    let has_token_inputs = input_names
        .iter()
        .any(|n| *n == "input_ids" || *n == "tokens" || *n == "token_ids");

    if has_token_inputs {
        preprocessing.push(PreprocessingStep::Tokenize {
            vocab_file: tokenizer_file.to_string(),
            tokenizer_type: TokenizerType::WordPiece,
            max_length: Some(512),
        });
        files.push(tokenizer_file.to_string());
    }

    // Check for audio inputs
    let has_audio_inputs = input_names
        .iter()
        .any(|n| n.contains("audio") || n.contains("waveform") || n.contains("mel"));

    if has_audio_inputs {
        preprocessing.push(PreprocessingStep::AudioDecode {
            sample_rate: 16000,
            channels: 1,
        });
    }

    // Check outputs for logits (likely classification)
    let output_names: Vec<&str> = info.outputs.iter().map(|o| o.name.as_str()).collect();
    let has_logits = output_names.iter().any(|n| n.contains("logits"));

    if has_logits {
        postprocessing.push(PostprocessingStep::Argmax { dim: None });
    }
}

fn infer_task_from_tags(card: Option<&HfModelCard>) -> Option<String> {
    let card = card?;
    for tag in &card.tags {
        match tag.as_str() {
            "text-generation" | "text-generation-inference" => {
                return Some("text-generation".to_string())
            }
            "text-to-speech" | "tts" => return Some("text-to-speech".to_string()),
            "automatic-speech-recognition" | "asr" => {
                return Some("automatic-speech-recognition".to_string())
            }
            "text-classification" | "sentiment-analysis" => {
                return Some("text-classification".to_string())
            }
            "token-classification" | "ner" => return Some("token-classification".to_string()),
            "image-classification" => return Some("image-classification".to_string()),
            "feature-extraction" | "sentence-similarity" => {
                return Some("feature-extraction".to_string())
            }
            _ => {}
        }
    }
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_extract_yaml_frontmatter() {
        let content = "---\npipeline_tag: text-generation\ntags:\n  - gguf\n---\n# Model Card\nSome description";
        let fm = extract_yaml_frontmatter(content).unwrap();
        assert!(fm.contains("pipeline_tag"));
        assert!(fm.contains("text-generation"));
    }

    #[test]
    fn test_extract_yaml_frontmatter_missing() {
        let content = "# Just a README\nNo frontmatter here.";
        assert!(extract_yaml_frontmatter(content).is_none());
    }

    #[test]
    fn test_parse_hf_model_card() {
        let dir = TempDir::new().unwrap();
        let readme = dir.path().join("README.md");
        std::fs::write(
            &readme,
            "---\npipeline_tag: text-generation\nlibrary_name: gguf\nlanguage:\n  - en\n  - zh\ntags:\n  - gguf\n  - llama\nlicense: apache-2.0\n---\n# Model\n",
        )
        .unwrap();

        let card = parse_hf_model_card(&readme).unwrap();
        assert_eq!(card.pipeline_tag.as_deref(), Some("text-generation"));
        assert_eq!(card.library_name.as_deref(), Some("gguf"));
        assert_eq!(card.languages, vec!["en", "zh"]);
        assert_eq!(card.license.as_deref(), Some("apache-2.0"));
        assert!(card.tags.contains(&"gguf".to_string()));
    }

    #[test]
    fn test_detect_model_files() {
        let dir = TempDir::new().unwrap();

        // Create dummy model files
        std::fs::write(dir.path().join("model.onnx"), b"dummy onnx").unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"dummy gguf").unwrap();
        std::fs::write(dir.path().join("readme.md"), b"not a model").unwrap();

        let files = detect_model_files(dir.path());
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|f| f.format == ModelFormat::Onnx));
        assert!(files.iter().any(|f| f.format == ModelFormat::Gguf));
    }

    #[test]
    fn test_infer_quantization_from_filename() {
        assert_eq!(
            infer_quantization_from_filename("Qwen3.5-0.8B-Q4_K_M.gguf"),
            Some("Q4_K_M".to_string())
        );
        assert_eq!(
            infer_quantization_from_filename("model-Q8_0.gguf"),
            Some("Q8_0".to_string())
        );
        assert_eq!(
            infer_quantization_from_filename("model-F16.gguf"),
            Some("F16".to_string())
        );
        assert_eq!(infer_quantization_from_filename("model.gguf"), None);
    }

    #[test]
    fn test_read_gguf_metadata_valid() {
        let dir = TempDir::new().unwrap();
        let gguf_path = dir.path().join("test.gguf");

        // Write a minimal valid GGUF v3 file
        let mut f = std::fs::File::create(&gguf_path).unwrap();
        // Magic
        f.write_all(b"GGUF").unwrap();
        // Version (3)
        f.write_all(&3u32.to_le_bytes()).unwrap();
        // Tensor count (0)
        f.write_all(&0u64.to_le_bytes()).unwrap();
        // Metadata KV count (2)
        f.write_all(&2u64.to_le_bytes()).unwrap();

        // KV 1: general.architecture = "llama"
        write_gguf_test_string(&mut f, "general.architecture");
        f.write_all(&GGUF_TYPE_STRING.to_le_bytes()).unwrap();
        write_gguf_test_string(&mut f, "llama");

        // KV 2: llama.context_length = 8192 (uint32)
        write_gguf_test_string(&mut f, "llama.context_length");
        f.write_all(&GGUF_TYPE_UINT32.to_le_bytes()).unwrap();
        f.write_all(&8192u32.to_le_bytes()).unwrap();

        drop(f);

        let info = read_gguf_metadata(&gguf_path).unwrap();
        assert_eq!(info.architecture.as_deref(), Some("llama"));
        assert_eq!(info.context_length, Some(8192));
    }

    #[test]
    fn test_read_gguf_metadata_invalid_magic() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.gguf");
        std::fs::write(&path, b"NOT_GGUF_data").unwrap();
        assert!(read_gguf_metadata(&path).is_none());
    }

    #[test]
    fn test_generate_metadata_gguf() {
        let dir = TempDir::new().unwrap();

        // Write README.md with frontmatter
        std::fs::write(
            dir.path().join("README.md"),
            "---\npipeline_tag: text-generation\nlanguage: en\n---\n# Test Model\n",
        )
        .unwrap();

        // Write minimal GGUF file
        let gguf_path = dir.path().join("model-Q4_K_M.gguf");
        let mut f = std::fs::File::create(&gguf_path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        write_gguf_test_string(&mut f, "general.architecture");
        f.write_all(&GGUF_TYPE_STRING.to_le_bytes()).unwrap();
        write_gguf_test_string(&mut f, "qwen2");
        drop(f);

        let metadata = generate_metadata(dir.path(), "test-org/test-model").unwrap();

        assert_eq!(metadata.model_id, "test-model");
        assert_eq!(metadata.version, "1.0");
        assert!(metadata.preprocessing.is_empty());
        assert!(metadata.postprocessing.is_empty());
        assert!(metadata.files.contains(&"model-Q4_K_M.gguf".to_string()));

        // Check execution template
        match &metadata.execution_template {
            xybrid_core::execution::ExecutionTemplate::Gguf { model_file, .. } => {
                assert_eq!(model_file, "model-Q4_K_M.gguf");
            }
            _ => panic!("Expected Gguf execution template"),
        }

        // Check metadata fields
        assert_eq!(
            metadata.metadata.get("task").and_then(|v| v.as_str()),
            Some("text-generation")
        );
        assert_eq!(
            metadata
                .metadata
                .get("architecture")
                .and_then(|v| v.as_str()),
            Some("qwen2")
        );
        assert_eq!(
            metadata
                .metadata
                .get("auto_generated")
                .and_then(|v| v.as_bool()),
            Some(true)
        );

        // Verify JSON was written
        let metadata_path = dir.path().join("model_metadata.json");
        assert!(metadata_path.exists());

        // Verify it round-trips
        let json = std::fs::read_to_string(&metadata_path).unwrap();
        let parsed: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model_id, "test-model");
    }

    #[test]
    fn test_generate_metadata_no_model_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("README.md"), "# Empty repo").unwrap();

        let result = generate_metadata(dir.path(), "test/empty");
        assert!(result.is_err());
    }

    fn write_gguf_test_string(f: &mut std::fs::File, s: &str) {
        f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
        f.write_all(s.as_bytes()).unwrap();
    }
}
