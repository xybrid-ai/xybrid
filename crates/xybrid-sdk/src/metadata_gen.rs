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
/// Scans the directory for model files, parses the README.md model card, inspects
/// supporting files (tokenizer.json, config.json, etc.), and uses ONNX tensor
/// inspection (when the `onnx-inspect` feature is enabled) to produce a best-effort
/// `ModelMetadata`.
///
/// Returns `Ok((metadata, task_inference))` on success and writes `model_metadata.json`
/// to `cache_dir`. The `TaskInference` is `Some` when ONNX tensor info was available
/// and can be used to check confidence and alternatives.
///
/// Falls back to a generic template if specific detection fails, logging a warning.
/// Inspect a model directory and generate `ModelMetadata` without writing to disk.
///
/// This is the core generation logic used by both `generate_metadata()` (which writes)
/// and `xybrid init` (which may prompt before writing).
///
/// The `model_id_override` parameter allows callers to override the auto-derived model_id.
pub fn inspect_and_generate(
    dir: &Path,
    repo: &str,
    model_id_override: Option<&str>,
) -> SdkResult<(ModelMetadata, Option<TaskInference>)> {
    log::info!(
        target: "xybrid_sdk",
        "Inspecting model directory '{}'",
        dir.display()
    );

    // 1. Parse HF model card if README.md exists
    let model_card = parse_hf_model_card(&dir.join("README.md"));

    // 2. Scan for model files
    let model_files = detect_model_files(dir);

    if model_files.is_empty() {
        return Err(SdkError::LoadError(format!(
            "No model files (.onnx, .gguf, .safetensors) found in '{}'",
            dir.display()
        )));
    }

    // 3. Inspect supporting files (tokenizer.json, config.json, preprocessor_config.json)
    let supporting_files = inspect_supporting_files(dir);

    // 4. Inspect model files for metadata
    let gguf_info = model_files
        .iter()
        .find(|f| f.format == ModelFormat::Gguf)
        .and_then(|f| read_gguf_metadata(&dir.join(&f.filename)));

    let onnx_info = model_files
        .iter()
        .find(|f| f.format == ModelFormat::Onnx)
        .and_then(|f| inspect_onnx_model(&dir.join(&f.filename)));

    // 5. Infer task from tensor patterns when ONNX info is available
    let task_inference = onnx_info
        .as_ref()
        .map(|onnx| infer_task_from_tensors(onnx, &supporting_files, model_card.as_ref()));

    // 6. Derive model_id: use override, prefer repo name (HF case), fall back to directory name
    let model_id = if let Some(id) = model_id_override {
        id.to_string()
    } else {
        let raw_name = if repo.is_empty() {
            dir.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown-model")
        } else {
            repo.rsplit('/').next().unwrap_or(repo)
        };
        sanitize_model_id(raw_name)
    };

    // 7. List all non-hidden files in directory (excluding README.md, model_metadata.json)
    let all_files = list_model_files(dir);

    // 8. Build metadata from collected info
    let metadata = build_metadata(
        &model_id,
        &model_files,
        model_card.as_ref(),
        gguf_info.as_ref(),
        onnx_info.as_ref(),
        &supporting_files,
        task_inference.as_ref(),
        &all_files,
        dir,
    );

    Ok((metadata, task_inference))
}

/// Auto-generate a `model_metadata.json` for a downloaded HuggingFace repo.
///
/// Calls [`inspect_and_generate`] and writes the result to disk.
pub fn generate_metadata(
    cache_dir: &Path,
    repo: &str,
) -> SdkResult<(ModelMetadata, Option<TaskInference>)> {
    let (metadata, task_inference) = inspect_and_generate(cache_dir, repo, None)?;

    // Write to cache directory
    write_metadata(cache_dir, &metadata)?;

    Ok((metadata, task_inference))
}

/// Write a `ModelMetadata` to `model_metadata.json` in the given directory.
pub fn write_metadata(dir: &Path, metadata: &ModelMetadata) -> SdkResult<()> {
    let metadata_path = dir.join("model_metadata.json");
    let json = serde_json::to_string_pretty(metadata).map_err(|e| {
        SdkError::MetadataInvalid(format!("Failed to serialize generated metadata: {}", e))
    })?;
    std::fs::write(&metadata_path, &json)?;

    log::info!(
        target: "xybrid_sdk",
        "Generated model_metadata.json at {}",
        metadata_path.display()
    );
    Ok(())
}

/// Sanitize a name to lowercase kebab-case for use as a model_id.
/// Strips non-alphanumeric characters (except hyphens), replaces spaces and underscores with hyphens.
fn sanitize_model_id(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '.' {
                c.to_ascii_lowercase()
            } else if c == ' ' || c == '_' {
                '-'
            } else {
                // skip non-alphanumeric characters
                '\0'
            }
        })
        .filter(|&c| c != '\0')
        .collect::<String>()
        // Collapse multiple consecutive hyphens
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// List all non-hidden files in a directory, excluding README.md and model_metadata.json.
/// Public wrapper for use by CLI commands.
pub fn list_model_files_pub(dir: &Path) -> Vec<String> {
    list_model_files(dir)
}

/// List all non-hidden files in a directory, excluding README.md and model_metadata.json.
fn list_model_files(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            // Skip hidden files, README.md, and model_metadata.json
            if name.starts_with('.') || name == "README.md" || name == "model_metadata.json" {
                continue;
            }
            if entry.path().is_file() || entry.path().is_symlink() {
                files.push(name);
            }
        }
    }
    files.sort();
    files
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
// Supporting File Inspection
// ============================================================================

/// Information extracted from supporting files in a model directory.
#[derive(Debug, Clone, Default)]
pub(crate) struct SupportingFileInfo {
    /// Tokenizer type detected from tokenizer.json `model.type` field
    pub tokenizer_type: Option<String>,
    /// Vocabulary size from tokenizer.json
    pub vocab_size: Option<usize>,
    /// Hidden size from config.json
    pub hidden_size: Option<u64>,
    /// Number of labels from config.json
    pub num_labels: Option<u64>,
    /// Vocab size from config.json (may differ from tokenizer.json)
    pub config_vocab_size: Option<u64>,
    /// Max position embeddings from config.json
    pub max_position_embeddings: Option<u64>,
    /// Model type from config.json (e.g., "bert", "gpt2", "whisper")
    pub model_type: Option<String>,
    /// Image normalization mean from config.json or preprocessor_config.json
    pub image_mean: Option<Vec<f64>>,
    /// Image normalization std from config.json or preprocessor_config.json
    pub image_std: Option<Vec<f64>>,
    /// Image size from preprocessor_config.json
    pub image_size: Option<u64>,
    /// Whether normalization is enabled from preprocessor_config.json
    pub do_normalize: Option<bool>,
    /// Detected supporting files by name
    pub has_tokens_txt: bool,
    pub has_voices_bin: bool,
    pub has_npz_files: bool,
    pub has_vocab_json: bool,
    pub has_vocab_txt: bool,
    pub has_tokenizer_json: bool,
}

/// Inspect supporting files (tokenizer.json, config.json, preprocessor_config.json, etc.)
/// in a model directory and extract structured information.
///
/// All file reads are best-effort: missing files result in None fields, never errors.
pub(crate) fn inspect_supporting_files(dir: &Path) -> SupportingFileInfo {
    let mut info = SupportingFileInfo {
        has_tokens_txt: dir.join("tokens.txt").exists(),
        has_voices_bin: dir.join("voices.bin").exists(),
        has_vocab_json: dir.join("vocab.json").exists(),
        has_vocab_txt: dir.join("vocab.txt").exists(),
        has_tokenizer_json: dir.join("tokenizer.json").exists(),
        ..Default::default()
    };

    // Check for any .npz files
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "npz" {
                    info.has_npz_files = true;
                    break;
                }
            }
        }
    }

    // Parse tokenizer.json
    if let Some(tokenizer_val) = read_json_file(&dir.join("tokenizer.json")) {
        // Detect tokenizer type from model.type
        if let Some(model_type) = tokenizer_val
            .get("model")
            .and_then(|m| m.get("type"))
            .and_then(|t| t.as_str())
        {
            info.tokenizer_type = Some(model_type.to_string());
        }

        // Extract vocab size from model.vocab length
        let vocab_len = tokenizer_val
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| {
                if let Some(obj) = v.as_object() {
                    Some(obj.len())
                } else {
                    v.as_array().map(|arr| arr.len())
                }
            });

        if let Some(len) = vocab_len.filter(|&l| l > 0) {
            info.vocab_size = Some(len);
        } else if let Some(added) = tokenizer_val.get("added_tokens").and_then(|a| a.as_array()) {
            if !added.is_empty() {
                info.vocab_size = Some(added.len());
            }
        }
    }

    // Parse config.json
    if let Some(config_val) = read_json_file(&dir.join("config.json")) {
        info.hidden_size = config_val.get("hidden_size").and_then(|v| v.as_u64());
        info.num_labels = config_val.get("num_labels").and_then(|v| v.as_u64());
        info.config_vocab_size = config_val.get("vocab_size").and_then(|v| v.as_u64());
        info.max_position_embeddings = config_val
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64());
        info.model_type = config_val
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Image mean/std can be in config.json for some vision models
        if let Some(mean) = extract_f64_array(&config_val, "image_mean") {
            info.image_mean = Some(mean);
        }
        if let Some(std_val) = extract_f64_array(&config_val, "image_std") {
            info.image_std = Some(std_val);
        }
    }

    // Parse preprocessor_config.json
    if let Some(preproc_val) = read_json_file(&dir.join("preprocessor_config.json")) {
        // Image size: can be a direct number, or nested in size.height/size.width
        if info.image_size.is_none() {
            if let Some(size) = preproc_val.get("image_size").and_then(|v| v.as_u64()) {
                info.image_size = Some(size);
            } else if let Some(size_obj) = preproc_val.get("size").and_then(|v| v.as_object()) {
                // Try height or shortest_edge
                if let Some(h) = size_obj.get("height").and_then(|v| v.as_u64()) {
                    info.image_size = Some(h);
                } else if let Some(se) = size_obj.get("shortest_edge").and_then(|v| v.as_u64()) {
                    info.image_size = Some(se);
                }
            }
        }

        info.do_normalize = preproc_val.get("do_normalize").and_then(|v| v.as_bool());

        // Image mean/std from preprocessor_config override config.json values
        if let Some(mean) = extract_f64_array(&preproc_val, "image_mean") {
            info.image_mean = Some(mean);
        }
        if let Some(std_val) = extract_f64_array(&preproc_val, "image_std") {
            info.image_std = Some(std_val);
        }
    }

    info
}

/// Read a JSON file and return it as a serde_json::Value. Returns None on any error.
fn read_json_file(path: &Path) -> Option<serde_json::Value> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Extract an array of f64 values from a JSON object by key.
fn extract_f64_array(val: &serde_json::Value, key: &str) -> Option<Vec<f64>> {
    val.get(key)?
        .as_array()?
        .iter()
        .map(|v| v.as_f64())
        .collect()
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
/// Loads the ONNX file via ORT Session and extracts tensor names, shapes, and element types.
/// Requires the `onnx-inspect` feature flag.
#[cfg(feature = "onnx-inspect")]
fn inspect_onnx_model(path: &Path) -> Option<OnnxInfo> {
    use ort::session::Session;

    let session = match Session::builder().and_then(|b| b.commit_from_file(path)) {
        Ok(s) => s,
        Err(e) => {
            log::warn!(
                target: "xybrid_sdk",
                "Failed to load ONNX model at {}: {}",
                path.display(),
                e
            );
            return None;
        }
    };

    let inputs = session
        .inputs()
        .iter()
        .map(|input| {
            let shape = input
                .dtype()
                .tensor_shape()
                .map(|s| s.to_vec())
                .unwrap_or_default();

            let dtype = input
                .dtype()
                .tensor_type()
                .map(|t| format!("{:?}", t))
                .unwrap_or_else(|| "unknown".to_string());

            TensorInfo {
                name: input.name().to_string(),
                shape,
                dtype,
            }
        })
        .collect();

    let outputs = session
        .outputs()
        .iter()
        .map(|output| {
            let shape = output
                .dtype()
                .tensor_shape()
                .map(|s| s.to_vec())
                .unwrap_or_default();

            let dtype = output
                .dtype()
                .tensor_type()
                .map(|t| format!("{:?}", t))
                .unwrap_or_else(|| "unknown".to_string());

            TensorInfo {
                name: output.name().to_string(),
                shape,
                dtype,
            }
        })
        .collect();

    Some(OnnxInfo { inputs, outputs })
}

/// Inspect an ONNX model to extract input/output tensor information.
///
/// Without the `onnx-inspect` feature, returns None and falls back to
/// task-based inference from the HF model card.
#[cfg(not(feature = "onnx-inspect"))]
fn inspect_onnx_model(_path: &Path) -> Option<OnnxInfo> {
    log::debug!(
        target: "xybrid_sdk",
        "ONNX tensor inspection not available (enable 'onnx-inspect' feature). \
         Using task-based inference from model card instead."
    );
    None
}

// ============================================================================
// Metadata Construction
// ============================================================================

fn build_metadata(
    model_id: &str,
    model_files: &[ModelFileInfo],
    card: Option<&HfModelCard>,
    gguf_info: Option<&GgufInfo>,
    onnx_info: Option<&OnnxInfo>,
    supporting_files: &SupportingFileInfo,
    task_inference: Option<&TaskInference>,
    all_files: &[String],
    cache_dir: &Path,
) -> ModelMetadata {
    // Determine the primary model file (largest file of the detected format)
    let primary = &model_files[0];

    // Determine task: prefer task inference, then model card, then tags
    let task = task_inference
        .map(|ti| ti.task.clone())
        .or_else(|| card.and_then(|c| c.pipeline_tag.clone()))
        .or_else(|| infer_task_from_tags(card))
        .unwrap_or_else(|| "unknown".to_string());

    match primary.format {
        ModelFormat::Gguf => {
            build_gguf_metadata(model_id, model_id, primary, &task, card, gguf_info)
        }
        ModelFormat::Onnx => build_onnx_metadata(
            model_id,
            model_id,
            primary,
            &task,
            card,
            onnx_info,
            supporting_files,
            task_inference,
            all_files,
            cache_dir,
        ),
        ModelFormat::SafeTensors => {
            build_safetensors_metadata(model_id, model_id, primary, &task, card, model_files)
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
    _repo: &str,
    primary: &ModelFileInfo,
    task: &str,
    card: Option<&HfModelCard>,
    onnx_info: Option<&OnnxInfo>,
    supporting_files: &SupportingFileInfo,
    task_inference: Option<&TaskInference>,
    all_files: &[String],
    cache_dir: &Path,
) -> ModelMetadata {
    use xybrid_core::execution::template::TokenizerType;
    use xybrid_core::execution::{PostprocessingStep, PreprocessingStep};

    // Use task inference results when available (they incorporate supporting file info)
    let (preprocessing, postprocessing) = if let Some(ti) = task_inference {
        (ti.preprocessing.clone(), ti.postprocessing.clone())
    } else {
        // Fallback: manual step inference from task string
        let has_tokenizer_json = cache_dir.join("tokenizer.json").exists();
        let tokenizer_file = if has_tokenizer_json {
            "tokenizer.json"
        } else {
            "vocab.txt"
        };

        let mut preprocessing = Vec::new();
        let mut postprocessing = Vec::new();

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
            }
            "text-classification" | "sentiment-analysis" => {
                preprocessing.push(PreprocessingStep::Tokenize {
                    vocab_file: tokenizer_file.to_string(),
                    tokenizer_type: TokenizerType::WordPiece,
                    max_length: supporting_files.max_position_embeddings.map(|v| v as usize),
                });
                postprocessing.push(PostprocessingStep::Argmax { dim: None });
            }
            "token-classification" | "ner" => {
                preprocessing.push(PreprocessingStep::Tokenize {
                    vocab_file: tokenizer_file.to_string(),
                    tokenizer_type: TokenizerType::WordPiece,
                    max_length: supporting_files.max_position_embeddings.map(|v| v as usize),
                });
                postprocessing.push(PostprocessingStep::Argmax { dim: None });
            }
            "image-classification" => {
                let mean = supporting_files
                    .image_mean
                    .as_ref()
                    .map(|v| v.iter().map(|&x| x as f32).collect())
                    .unwrap_or_else(|| vec![0.485, 0.456, 0.406]);
                let std = supporting_files
                    .image_std
                    .as_ref()
                    .map(|v| v.iter().map(|&x| x as f32).collect())
                    .unwrap_or_else(|| vec![0.229, 0.224, 0.225]);
                preprocessing.push(PreprocessingStep::Normalize { mean, std });
                postprocessing.push(PostprocessingStep::Argmax { dim: None });
            }
            "feature-extraction" | "sentence-similarity" => {
                preprocessing.push(PreprocessingStep::Tokenize {
                    vocab_file: tokenizer_file.to_string(),
                    tokenizer_type: TokenizerType::WordPiece,
                    max_length: supporting_files.max_position_embeddings.map(|v| v as usize),
                });
            }
            _ => {
                if let Some(info) = onnx_info {
                    let mut files_vec = Vec::new();
                    infer_steps_from_onnx(
                        info,
                        &mut preprocessing,
                        &mut postprocessing,
                        &mut files_vec,
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
        (preprocessing, postprocessing)
    };

    // Use the full file list from directory scanning
    let files = if all_files.is_empty() {
        // Fallback: at least include the primary model file
        vec![primary.filename.clone()]
    } else {
        all_files.to_vec()
    };

    let mut metadata_map = HashMap::new();
    metadata_map.insert(
        "task".to_string(),
        serde_json::Value::String(task.to_string()),
    );
    metadata_map.insert("auto_generated".to_string(), serde_json::Value::Bool(true));

    // Add architecture from supporting files
    if let Some(model_type) = &supporting_files.model_type {
        metadata_map.insert(
            "architecture".to_string(),
            serde_json::Value::String(model_type.clone()),
        );
    }

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
        if let Some(license) = &card.license {
            metadata_map.insert(
                "license".to_string(),
                serde_json::Value::String(license.clone()),
            );
        }
    }

    let description = card
        .and_then(|c| c.model_name.clone())
        .unwrap_or_else(|| format!("{} (auto-generated)", model_id));

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

// ============================================================================
// Task Inference from Tensor Patterns
// ============================================================================

/// Confidence level for task inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    /// Strong signal: HF pipeline_tag or unambiguous tensor pattern
    High,
    /// Reasonable guess: tensor names match known patterns but could be wrong
    Medium,
    /// Ambiguous: multiple interpretations possible, check `alternatives`
    Low,
}

/// Result of task inference from ONNX tensor patterns and supporting files.
#[derive(Debug, Clone)]
pub struct TaskInference {
    /// Inferred task name (e.g., "text-classification", "text-to-speech")
    pub task: String,
    /// Suggested preprocessing steps
    pub preprocessing: Vec<xybrid_core::execution::PreprocessingStep>,
    /// Suggested postprocessing steps
    pub postprocessing: Vec<xybrid_core::execution::PostprocessingStep>,
    /// Confidence level of the inference
    pub confidence: Confidence,
    /// Alternative interpretations when confidence is Low
    pub alternatives: Vec<TaskInference>,
}

/// Infer the task, preprocessing, and postprocessing from ONNX tensor metadata,
/// supporting file info, and an optional HuggingFace model card.
///
/// Uses a 3-priority heuristic:
/// 1. HF model card `pipeline_tag` (highest confidence)
/// 2. ONNX input name patterns (medium confidence)
/// 3. ONNX output shape analysis (lowest confidence)
pub(crate) fn infer_task_from_tensors(
    onnx: &OnnxInfo,
    files: &SupportingFileInfo,
    hf_card: Option<&HfModelCard>,
) -> TaskInference {
    use xybrid_core::execution::template::TokenizerType;
    use xybrid_core::execution::{PostprocessingStep, PreprocessingStep};

    // Helper: choose tokenizer file
    let tokenizer_file = if files.has_tokenizer_json {
        "tokenizer.json"
    } else {
        "vocab.txt"
    };

    // Helper: determine TokenizerType from supporting files
    let tokenizer_type = match files.tokenizer_type.as_deref() {
        Some("BPE") => TokenizerType::BPE,
        Some("Unigram") => TokenizerType::SentencePiece,
        _ => TokenizerType::WordPiece, // default for BERT-family
    };

    let max_length = files.max_position_embeddings.map(|v| v as usize);

    // ImageNet defaults for vision models
    let image_mean = files
        .image_mean
        .as_ref()
        .map(|v| v.iter().map(|&x| x as f32).collect::<Vec<f32>>())
        .unwrap_or_else(|| vec![0.485, 0.456, 0.406]);
    let image_std = files
        .image_std
        .as_ref()
        .map(|v| v.iter().map(|&x| x as f32).collect::<Vec<f32>>())
        .unwrap_or_else(|| vec![0.229, 0.224, 0.225]);

    // -----------------------------------------------------------------------
    // Priority 1: HF model card pipeline_tag
    // -----------------------------------------------------------------------
    if let Some(card) = hf_card {
        if let Some(tag) = &card.pipeline_tag {
            if let Some(inf) = infer_from_pipeline_tag(
                tag,
                tokenizer_file,
                tokenizer_type.clone(),
                max_length,
                &image_mean,
                &image_std,
                files,
            ) {
                return inf;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Priority 2: ONNX input name patterns
    // -----------------------------------------------------------------------
    let input_names: Vec<&str> = onnx.inputs.iter().map(|i| i.name.as_str()).collect();

    let has_input_ids = input_names.contains(&"input_ids") || input_names.contains(&"token_ids");
    let has_attention_mask = input_names.contains(&"attention_mask");
    let has_tokens = input_names.contains(&"tokens");
    let has_style = input_names.contains(&"style");
    let has_speed = input_names.contains(&"speed");
    let has_pixel_values = input_names.contains(&"pixel_values");
    let has_input_features =
        input_names.contains(&"input_features") || input_names.contains(&"input_values");

    // TTS pattern: tokens + style + speed
    if has_tokens && has_style && has_speed {
        return TaskInference {
            task: "text-to-speech".to_string(),
            preprocessing: vec![PreprocessingStep::Phonemize {
                tokens_file: "tokens.txt".to_string(),
                dict_file: None,
                backend: Default::default(),
                language: None,
                add_padding: true,
                normalize_text: false,
                silence_tokens: None,
            }],
            postprocessing: vec![PostprocessingStep::TTSAudioEncode {
                sample_rate: 24000,
                apply_postprocessing: true,
                trim_trailing_silence: false,
            }],
            confidence: Confidence::High,
            alternatives: Vec::new(),
        };
    }

    // ASR pattern: input_features or input_values
    if has_input_features {
        return TaskInference {
            task: "automatic-speech-recognition".to_string(),
            preprocessing: vec![PreprocessingStep::AudioDecode {
                sample_rate: 16000,
                channels: 1,
            }],
            postprocessing: vec![PostprocessingStep::CTCDecode {
                vocab_file: if files.has_vocab_json {
                    "vocab.json".to_string()
                } else {
                    tokenizer_file.to_string()
                },
                blank_index: 0,
            }],
            confidence: Confidence::Medium,
            alternatives: Vec::new(),
        };
    }

    // Vision pattern: pixel_values
    if has_pixel_values {
        return TaskInference {
            task: "image-classification".to_string(),
            preprocessing: vec![PreprocessingStep::Normalize {
                mean: image_mean,
                std: image_std,
            }],
            postprocessing: vec![PostprocessingStep::Softmax { dim: None }],
            confidence: Confidence::Medium,
            alternatives: Vec::new(),
        };
    }

    // NLP pattern: input_ids + attention_mask
    if has_input_ids && has_attention_mask {
        // Ambiguous: could be classification, feature-extraction, token-classification, etc.
        // Use output shapes to disambiguate (Priority 3)
        return infer_nlp_task_from_outputs(
            onnx,
            tokenizer_file,
            tokenizer_type,
            max_length,
            files,
        );
    }

    // -----------------------------------------------------------------------
    // Priority 3: Output shape analysis (fallback)
    // -----------------------------------------------------------------------
    infer_from_output_shapes(
        onnx,
        tokenizer_file,
        tokenizer_type,
        max_length,
        &image_mean,
        &image_std,
        files,
    )
}

/// Map a HF pipeline_tag to task inference with preprocessing/postprocessing.
fn infer_from_pipeline_tag(
    tag: &str,
    tokenizer_file: &str,
    tokenizer_type: xybrid_core::execution::template::TokenizerType,
    max_length: Option<usize>,
    image_mean: &[f32],
    image_std: &[f32],
    files: &SupportingFileInfo,
) -> Option<TaskInference> {
    use xybrid_core::execution::{PostprocessingStep, PreprocessingStep};

    let inf = match tag {
        "automatic-speech-recognition" | "speech-recognition" => TaskInference {
            task: "automatic-speech-recognition".to_string(),
            preprocessing: vec![PreprocessingStep::AudioDecode {
                sample_rate: 16000,
                channels: 1,
            }],
            postprocessing: vec![PostprocessingStep::CTCDecode {
                vocab_file: if files.has_vocab_json {
                    "vocab.json".to_string()
                } else {
                    tokenizer_file.to_string()
                },
                blank_index: 0,
            }],
            confidence: Confidence::High,
            alternatives: Vec::new(),
        },
        "text-to-speech" | "tts" => TaskInference {
            task: "text-to-speech".to_string(),
            preprocessing: vec![PreprocessingStep::Phonemize {
                tokens_file: "tokens.txt".to_string(),
                dict_file: None,
                backend: Default::default(),
                language: None,
                add_padding: true,
                normalize_text: false,
                silence_tokens: None,
            }],
            postprocessing: vec![PostprocessingStep::TTSAudioEncode {
                sample_rate: 24000,
                apply_postprocessing: true,
                trim_trailing_silence: false,
            }],
            confidence: Confidence::High,
            alternatives: Vec::new(),
        },
        "text-classification" | "sentiment-analysis" => TaskInference {
            task: "text-classification".to_string(),
            preprocessing: vec![PreprocessingStep::Tokenize {
                vocab_file: tokenizer_file.to_string(),
                tokenizer_type,
                max_length,
            }],
            postprocessing: vec![PostprocessingStep::Softmax { dim: None }],
            confidence: Confidence::High,
            alternatives: Vec::new(),
        },
        "token-classification" | "ner" => TaskInference {
            task: "token-classification".to_string(),
            preprocessing: vec![PreprocessingStep::Tokenize {
                vocab_file: tokenizer_file.to_string(),
                tokenizer_type,
                max_length,
            }],
            postprocessing: vec![PostprocessingStep::Argmax { dim: None }],
            confidence: Confidence::High,
            alternatives: Vec::new(),
        },
        "image-classification" => TaskInference {
            task: "image-classification".to_string(),
            preprocessing: vec![PreprocessingStep::Normalize {
                mean: image_mean.to_vec(),
                std: image_std.to_vec(),
            }],
            postprocessing: vec![PostprocessingStep::Softmax { dim: None }],
            confidence: Confidence::High,
            alternatives: Vec::new(),
        },
        "feature-extraction" | "sentence-similarity" => TaskInference {
            task: "feature-extraction".to_string(),
            preprocessing: vec![PreprocessingStep::Tokenize {
                vocab_file: tokenizer_file.to_string(),
                tokenizer_type,
                max_length,
            }],
            postprocessing: vec![PostprocessingStep::MeanPool { dim: 1 }],
            confidence: Confidence::High,
            alternatives: Vec::new(),
        },
        _ => return None,
    };

    Some(inf)
}

/// For NLP models (input_ids + attention_mask), use output shapes to disambiguate.
fn infer_nlp_task_from_outputs(
    onnx: &OnnxInfo,
    tokenizer_file: &str,
    tokenizer_type: xybrid_core::execution::template::TokenizerType,
    max_length: Option<usize>,
    files: &SupportingFileInfo,
) -> TaskInference {
    use xybrid_core::execution::{PostprocessingStep, PreprocessingStep};

    let tokenize_step = PreprocessingStep::Tokenize {
        vocab_file: tokenizer_file.to_string(),
        tokenizer_type: tokenizer_type.clone(),
        max_length,
    };

    // Analyze output shapes
    if let Some(output) = onnx.outputs.first() {
        let shape = &output.shape;

        // [batch, num_classes] where num_classes is small (<1000) → classification
        if shape.len() == 2 {
            let num_classes = shape.get(1).copied().unwrap_or(0);
            if num_classes > 0 && num_classes < 1000 {
                // Could also be num_labels for token classification, but 2D suggests sequence-level
                return TaskInference {
                    task: "text-classification".to_string(),
                    preprocessing: vec![tokenize_step],
                    postprocessing: vec![PostprocessingStep::Softmax { dim: None }],
                    confidence: Confidence::Medium,
                    alternatives: Vec::new(),
                };
            }
        }

        // [batch, seq, hidden] → could be feature-extraction or token-classification
        if shape.len() == 3 {
            let hidden_dim = shape.get(2).copied().unwrap_or(0);
            let num_labels = files.num_labels.map(|v| v as i64);

            // If hidden dim matches a known small label count, it's token classification
            if let Some(labels) = num_labels {
                if hidden_dim == labels {
                    return TaskInference {
                        task: "token-classification".to_string(),
                        preprocessing: vec![tokenize_step],
                        postprocessing: vec![PostprocessingStep::Argmax { dim: None }],
                        confidence: Confidence::Medium,
                        alternatives: Vec::new(),
                    };
                }
            }

            // Large hidden dim → likely embeddings (feature-extraction)
            // But could also be token-classification with many labels
            let feature_extraction = TaskInference {
                task: "feature-extraction".to_string(),
                preprocessing: vec![tokenize_step.clone()],
                postprocessing: vec![PostprocessingStep::MeanPool { dim: 1 }],
                confidence: Confidence::Low,
                alternatives: Vec::new(),
            };

            let token_classification = TaskInference {
                task: "token-classification".to_string(),
                preprocessing: vec![tokenize_step],
                postprocessing: vec![PostprocessingStep::Argmax { dim: None }],
                confidence: Confidence::Low,
                alternatives: Vec::new(),
            };

            // Default to feature-extraction with token-classification as alternative
            return TaskInference {
                task: feature_extraction.task.clone(),
                preprocessing: feature_extraction.preprocessing.clone(),
                postprocessing: feature_extraction.postprocessing.clone(),
                confidence: Confidence::Low,
                alternatives: vec![token_classification],
            };
        }
    }

    // Can't disambiguate from outputs — generic NLP with MeanPool
    TaskInference {
        task: "feature-extraction".to_string(),
        preprocessing: vec![tokenize_step],
        postprocessing: vec![PostprocessingStep::MeanPool { dim: 1 }],
        confidence: Confidence::Low,
        alternatives: Vec::new(),
    }
}

/// Fallback: infer from output shapes alone when input patterns don't match.
fn infer_from_output_shapes(
    onnx: &OnnxInfo,
    tokenizer_file: &str,
    tokenizer_type: xybrid_core::execution::template::TokenizerType,
    max_length: Option<usize>,
    image_mean: &[f32],
    image_std: &[f32],
    _files: &SupportingFileInfo,
) -> TaskInference {
    use xybrid_core::execution::{PostprocessingStep, PreprocessingStep};

    if let Some(output) = onnx.outputs.first() {
        let shape = &output.shape;

        // [batch, small_classes] → classification (could be image or text)
        if shape.len() == 2 {
            let num_classes = shape.get(1).copied().unwrap_or(0);
            if num_classes > 0 && num_classes <= 1000 {
                // Check if any input looks image-like: [batch, channels, H, W]
                let has_image_input = onnx
                    .inputs
                    .iter()
                    .any(|i| i.shape.len() == 4 && i.shape.get(1).copied().unwrap_or(0) <= 4);

                if has_image_input {
                    return TaskInference {
                        task: "image-classification".to_string(),
                        preprocessing: vec![PreprocessingStep::Normalize {
                            mean: image_mean.to_vec(),
                            std: image_std.to_vec(),
                        }],
                        postprocessing: vec![PostprocessingStep::Softmax { dim: None }],
                        confidence: Confidence::Medium,
                        alternatives: Vec::new(),
                    };
                }

                return TaskInference {
                    task: "text-classification".to_string(),
                    preprocessing: vec![PreprocessingStep::Tokenize {
                        vocab_file: tokenizer_file.to_string(),
                        tokenizer_type,
                        max_length,
                    }],
                    postprocessing: vec![PostprocessingStep::Softmax { dim: None }],
                    confidence: Confidence::Low,
                    alternatives: Vec::new(),
                };
            }
        }

        // [batch, large_dim] where large_dim > 1000 → likely TTS audio output
        if shape.len() == 2 {
            let dim = shape.get(1).copied().unwrap_or(0);
            if dim > 10000 {
                return TaskInference {
                    task: "text-to-speech".to_string(),
                    preprocessing: vec![PreprocessingStep::Phonemize {
                        tokens_file: "tokens.txt".to_string(),
                        dict_file: None,
                        backend: Default::default(),
                        language: None,
                        add_padding: true,
                        normalize_text: false,
                        silence_tokens: None,
                    }],
                    postprocessing: vec![PostprocessingStep::TTSAudioEncode {
                        sample_rate: 24000,
                        apply_postprocessing: true,
                        trim_trailing_silence: false,
                    }],
                    confidence: Confidence::Low,
                    alternatives: Vec::new(),
                };
            }
        }
    }

    // Completely ambiguous — return unknown with Low confidence
    TaskInference {
        task: "unknown".to_string(),
        preprocessing: Vec::new(),
        postprocessing: Vec::new(),
        confidence: Confidence::Low,
        alternatives: Vec::new(),
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

        let (metadata, _) = generate_metadata(dir.path(), "test-org/test-model").unwrap();

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

    #[test]
    fn test_inspect_supporting_files_tokenizer_bpe() {
        let dir = TempDir::new().unwrap();

        // Create a minimal tokenizer.json with model.type = "BPE"
        let tokenizer_json = serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "hello": 0,
                    "world": 1,
                    "foo": 2
                }
            },
            "added_tokens": []
        });
        std::fs::write(
            dir.path().join("tokenizer.json"),
            serde_json::to_string(&tokenizer_json).unwrap(),
        )
        .unwrap();

        let info = inspect_supporting_files(dir.path());

        assert_eq!(info.tokenizer_type.as_deref(), Some("BPE"));
        assert_eq!(info.vocab_size, Some(3));
        assert!(info.has_tokenizer_json);
        assert!(!info.has_tokens_txt);
        assert!(!info.has_voices_bin);
    }

    #[test]
    fn test_inspect_supporting_files_config_json() {
        let dir = TempDir::new().unwrap();

        let config_json = serde_json::json!({
            "hidden_size": 768,
            "num_labels": 2,
            "vocab_size": 30522,
            "max_position_embeddings": 512,
            "model_type": "bert"
        });
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&config_json).unwrap(),
        )
        .unwrap();

        let info = inspect_supporting_files(dir.path());

        assert_eq!(info.hidden_size, Some(768));
        assert_eq!(info.num_labels, Some(2));
        assert_eq!(info.config_vocab_size, Some(30522));
        assert_eq!(info.max_position_embeddings, Some(512));
        assert_eq!(info.model_type.as_deref(), Some("bert"));
    }

    #[test]
    fn test_inspect_supporting_files_preprocessor_config() {
        let dir = TempDir::new().unwrap();

        let preproc_json = serde_json::json!({
            "size": { "height": 224, "width": 224 },
            "do_normalize": true,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225]
        });
        std::fs::write(
            dir.path().join("preprocessor_config.json"),
            serde_json::to_string(&preproc_json).unwrap(),
        )
        .unwrap();

        let info = inspect_supporting_files(dir.path());

        assert_eq!(info.image_size, Some(224));
        assert_eq!(info.do_normalize, Some(true));
        assert_eq!(info.image_mean, Some(vec![0.485, 0.456, 0.406]));
        assert_eq!(info.image_std, Some(vec![0.229, 0.224, 0.225]));
    }

    #[test]
    fn test_inspect_supporting_files_file_detection() {
        let dir = TempDir::new().unwrap();

        std::fs::write(dir.path().join("tokens.txt"), "a\nb\nc").unwrap();
        std::fs::write(dir.path().join("voices.bin"), b"dummy").unwrap();
        std::fs::write(dir.path().join("vocab.json"), "{}").unwrap();
        std::fs::write(dir.path().join("embeddings.npz"), b"dummy").unwrap();

        let info = inspect_supporting_files(dir.path());

        assert!(info.has_tokens_txt);
        assert!(info.has_voices_bin);
        assert!(info.has_vocab_json);
        assert!(info.has_npz_files);
        assert!(!info.has_vocab_txt);
        assert!(!info.has_tokenizer_json);
    }

    #[test]
    fn test_inspect_supporting_files_missing_files() {
        let dir = TempDir::new().unwrap();
        // Empty directory — all fields should be None/false
        let info = inspect_supporting_files(dir.path());

        assert!(info.tokenizer_type.is_none());
        assert!(info.vocab_size.is_none());
        assert!(info.hidden_size.is_none());
        assert!(info.model_type.is_none());
        assert!(!info.has_tokens_txt);
        assert!(!info.has_voices_bin);
    }

    #[test]
    fn test_inspect_supporting_files_preprocessor_shortest_edge() {
        let dir = TempDir::new().unwrap();

        let preproc_json = serde_json::json!({
            "size": { "shortest_edge": 256 },
            "do_normalize": false
        });
        std::fs::write(
            dir.path().join("preprocessor_config.json"),
            serde_json::to_string(&preproc_json).unwrap(),
        )
        .unwrap();

        let info = inspect_supporting_files(dir.path());

        assert_eq!(info.image_size, Some(256));
        assert_eq!(info.do_normalize, Some(false));
    }

    #[test]
    fn test_inspect_supporting_files_tokenizer_wordpiece() {
        let dir = TempDir::new().unwrap();

        let tokenizer_json = serde_json::json!({
            "model": {
                "type": "WordPiece",
                "vocab": {}
            },
            "added_tokens": [
                { "id": 0, "content": "[PAD]" },
                { "id": 1, "content": "[UNK]" },
                { "id": 2, "content": "[CLS]" }
            ]
        });
        std::fs::write(
            dir.path().join("tokenizer.json"),
            serde_json::to_string(&tokenizer_json).unwrap(),
        )
        .unwrap();

        let info = inspect_supporting_files(dir.path());

        assert_eq!(info.tokenizer_type.as_deref(), Some("WordPiece"));
        // vocab is empty, so falls back to added_tokens count
        assert_eq!(info.vocab_size, Some(3));
    }

    #[test]
    #[cfg(feature = "onnx-inspect")]
    fn test_inspect_onnx_model_mnist() {
        // The mnist fixture has input: Input3 [1,1,28,28] float32
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../integration-tests/fixtures/models/mnist/model.onnx");
        if !model_path.exists() {
            eprintln!(
                "Skipping test: mnist fixture not found at {}",
                model_path.display()
            );
            return;
        }

        let info = inspect_onnx_model(&model_path).expect("inspect_onnx_model returned None");

        // Verify inputs
        assert!(!info.inputs.is_empty(), "Expected at least one input");
        let first_input = &info.inputs[0];
        assert_eq!(first_input.name, "Input3");
        assert_eq!(first_input.shape, vec![1, 1, 28, 28]);

        // Verify outputs
        assert!(!info.outputs.is_empty(), "Expected at least one output");
    }

    // ========================================================================
    // infer_task_from_tensors tests
    // ========================================================================

    #[test]
    fn test_infer_nlp_inputs_with_2d_output() {
        // input_ids + attention_mask with [batch, num_classes] output → text-classification
        let onnx = OnnxInfo {
            inputs: vec![
                TensorInfo {
                    name: "input_ids".into(),
                    shape: vec![1, 128],
                    dtype: "Int64".into(),
                },
                TensorInfo {
                    name: "attention_mask".into(),
                    shape: vec![1, 128],
                    dtype: "Int64".into(),
                },
            ],
            outputs: vec![TensorInfo {
                name: "logits".into(),
                shape: vec![1, 2],
                dtype: "Float32".into(),
            }],
        };
        let files = SupportingFileInfo {
            has_tokenizer_json: true,
            tokenizer_type: Some("WordPiece".into()),
            max_position_embeddings: Some(512),
            ..Default::default()
        };

        let result = infer_task_from_tensors(&onnx, &files, None);
        assert_eq!(result.task, "text-classification");
        assert_eq!(result.confidence, Confidence::Medium);
        assert!(!result.preprocessing.is_empty());
        assert!(matches!(
            result.preprocessing[0],
            xybrid_core::execution::PreprocessingStep::Tokenize { .. }
        ));
        assert!(matches!(
            result.postprocessing[0],
            xybrid_core::execution::PostprocessingStep::Softmax { .. }
        ));
    }

    #[test]
    fn test_infer_nlp_inputs_with_3d_output_meanpool() {
        // input_ids + attention_mask with [batch, seq, hidden] output → feature-extraction (ambiguous)
        let onnx = OnnxInfo {
            inputs: vec![
                TensorInfo {
                    name: "input_ids".into(),
                    shape: vec![1, 128],
                    dtype: "Int64".into(),
                },
                TensorInfo {
                    name: "attention_mask".into(),
                    shape: vec![1, 128],
                    dtype: "Int64".into(),
                },
            ],
            outputs: vec![TensorInfo {
                name: "last_hidden_state".into(),
                shape: vec![1, 128, 384],
                dtype: "Float32".into(),
            }],
        };
        let files = SupportingFileInfo::default();

        let result = infer_task_from_tensors(&onnx, &files, None);
        assert_eq!(result.task, "feature-extraction");
        assert!(matches!(
            result.postprocessing[0],
            xybrid_core::execution::PostprocessingStep::MeanPool { .. }
        ));
        // Should have token-classification as alternative
        assert_eq!(result.confidence, Confidence::Low);
        assert!(!result.alternatives.is_empty());
        assert_eq!(result.alternatives[0].task, "token-classification");
    }

    #[test]
    fn test_infer_vision_pixel_values() {
        // pixel_values input → image-classification
        let onnx = OnnxInfo {
            inputs: vec![TensorInfo {
                name: "pixel_values".into(),
                shape: vec![1, 3, 224, 224],
                dtype: "Float32".into(),
            }],
            outputs: vec![TensorInfo {
                name: "logits".into(),
                shape: vec![1, 1000],
                dtype: "Float32".into(),
            }],
        };
        let files = SupportingFileInfo {
            image_mean: Some(vec![0.5, 0.5, 0.5]),
            image_std: Some(vec![0.5, 0.5, 0.5]),
            ..Default::default()
        };

        let result = infer_task_from_tensors(&onnx, &files, None);
        assert_eq!(result.task, "image-classification");
        assert_eq!(result.confidence, Confidence::Medium);
        // Should use custom mean/std from files
        if let xybrid_core::execution::PreprocessingStep::Normalize { mean, .. } =
            &result.preprocessing[0]
        {
            assert_eq!(mean, &[0.5, 0.5, 0.5]);
        } else {
            panic!("Expected Normalize preprocessing");
        }
    }

    #[test]
    fn test_infer_vision_from_output_shape() {
        // 4D image-like input + [batch, classes<1000] output → image-classification (fallback)
        let onnx = OnnxInfo {
            inputs: vec![TensorInfo {
                name: "Input3".into(),
                shape: vec![1, 1, 28, 28],
                dtype: "Float32".into(),
            }],
            outputs: vec![TensorInfo {
                name: "Plus214_Output_0".into(),
                shape: vec![1, 10],
                dtype: "Float32".into(),
            }],
        };
        let files = SupportingFileInfo::default();

        let result = infer_task_from_tensors(&onnx, &files, None);
        assert_eq!(result.task, "image-classification");
        assert!(matches!(
            result.postprocessing[0],
            xybrid_core::execution::PostprocessingStep::Softmax { .. }
        ));
    }

    #[test]
    fn test_infer_tts_inputs() {
        // tokens + style + speed → text-to-speech
        let onnx = OnnxInfo {
            inputs: vec![
                TensorInfo {
                    name: "tokens".into(),
                    shape: vec![1, -1],
                    dtype: "Int64".into(),
                },
                TensorInfo {
                    name: "style".into(),
                    shape: vec![1, 256],
                    dtype: "Float32".into(),
                },
                TensorInfo {
                    name: "speed".into(),
                    shape: vec![1],
                    dtype: "Float32".into(),
                },
            ],
            outputs: vec![TensorInfo {
                name: "audio".into(),
                shape: vec![1, -1],
                dtype: "Float32".into(),
            }],
        };
        let files = SupportingFileInfo {
            has_tokens_txt: true,
            has_voices_bin: true,
            ..Default::default()
        };

        let result = infer_task_from_tensors(&onnx, &files, None);
        assert_eq!(result.task, "text-to-speech");
        assert_eq!(result.confidence, Confidence::High);
        assert!(matches!(
            result.preprocessing[0],
            xybrid_core::execution::PreprocessingStep::Phonemize { .. }
        ));
        assert!(matches!(
            result.postprocessing[0],
            xybrid_core::execution::PostprocessingStep::TTSAudioEncode { .. }
        ));
    }

    #[test]
    fn test_infer_hf_card_overrides_tensors() {
        // HF card pipeline_tag should take priority over tensor patterns
        let onnx = OnnxInfo {
            inputs: vec![
                TensorInfo {
                    name: "input_ids".into(),
                    shape: vec![1, 128],
                    dtype: "Int64".into(),
                },
                TensorInfo {
                    name: "attention_mask".into(),
                    shape: vec![1, 128],
                    dtype: "Int64".into(),
                },
            ],
            outputs: vec![TensorInfo {
                name: "output".into(),
                shape: vec![1, 128, 384],
                dtype: "Float32".into(),
            }],
        };
        let files = SupportingFileInfo::default();
        let card = HfModelCard {
            pipeline_tag: Some("feature-extraction".into()),
            ..Default::default()
        };

        let result = infer_task_from_tensors(&onnx, &files, Some(&card));
        assert_eq!(result.task, "feature-extraction");
        assert_eq!(result.confidence, Confidence::High);
        assert!(matches!(
            result.postprocessing[0],
            xybrid_core::execution::PostprocessingStep::MeanPool { .. }
        ));
    }

    #[test]
    fn test_infer_asr_input_features() {
        // input_features → ASR
        let onnx = OnnxInfo {
            inputs: vec![TensorInfo {
                name: "input_features".into(),
                shape: vec![1, 80, 3000],
                dtype: "Float32".into(),
            }],
            outputs: vec![TensorInfo {
                name: "logits".into(),
                shape: vec![1, 1500, 51865],
                dtype: "Float32".into(),
            }],
        };
        let files = SupportingFileInfo {
            has_vocab_json: true,
            ..Default::default()
        };

        let result = infer_task_from_tensors(&onnx, &files, None);
        assert_eq!(result.task, "automatic-speech-recognition");
        assert_eq!(result.confidence, Confidence::Medium);
    }

    #[test]
    fn test_infer_uses_supporting_file_params() {
        // Verify that tokenizer type and max_length from supporting files are used
        let onnx = OnnxInfo {
            inputs: vec![
                TensorInfo {
                    name: "input_ids".into(),
                    shape: vec![1, 128],
                    dtype: "Int64".into(),
                },
                TensorInfo {
                    name: "attention_mask".into(),
                    shape: vec![1, 128],
                    dtype: "Int64".into(),
                },
            ],
            outputs: vec![TensorInfo {
                name: "logits".into(),
                shape: vec![1, 3],
                dtype: "Float32".into(),
            }],
        };
        let files = SupportingFileInfo {
            has_tokenizer_json: true,
            tokenizer_type: Some("BPE".into()),
            max_position_embeddings: Some(1024),
            ..Default::default()
        };

        let result = infer_task_from_tensors(&onnx, &files, None);
        if let xybrid_core::execution::PreprocessingStep::Tokenize {
            vocab_file,
            tokenizer_type,
            max_length,
        } = &result.preprocessing[0]
        {
            assert_eq!(vocab_file, "tokenizer.json");
            assert!(matches!(
                tokenizer_type,
                xybrid_core::execution::template::TokenizerType::BPE
            ));
            assert_eq!(*max_length, Some(1024));
        } else {
            panic!("Expected Tokenize preprocessing");
        }
    }

    // ========================================================================
    // sanitize_model_id tests
    // ========================================================================

    #[test]
    fn test_sanitize_model_id_basic() {
        assert_eq!(sanitize_model_id("My Model"), "my-model");
        assert_eq!(sanitize_model_id("my_model_v2"), "my-model-v2");
        assert_eq!(sanitize_model_id("UPPER-case"), "upper-case");
        assert_eq!(sanitize_model_id("model (copy)"), "model-copy");
        assert_eq!(sanitize_model_id("a--b__c  d"), "a-b-c-d");
        assert_eq!(sanitize_model_id("model.onnx"), "model.onnx");
    }

    #[test]
    fn test_sanitize_model_id_kebab_case() {
        assert_eq!(
            sanitize_model_id("Qwen3.5-0.8B-Q4_K_M"),
            "qwen3.5-0.8b-q4-k-m"
        );
        assert_eq!(sanitize_model_id("all-MiniLM-L6-v2"), "all-minilm-l6-v2");
    }

    // ========================================================================
    // generate_metadata integration tests
    // ========================================================================

    #[test]
    fn test_generate_metadata_returns_task_inference_for_onnx() {
        let dir = TempDir::new().unwrap();

        // Create a dummy ONNX file (won't be loaded without onnx-inspect feature)
        std::fs::write(dir.path().join("model.onnx"), b"dummy onnx data").unwrap();

        let (metadata, _task_inference) =
            generate_metadata(dir.path(), "test-org/test-onnx").unwrap();

        assert_eq!(metadata.model_id, "test-onnx");
        assert_eq!(metadata.version, "1.0");
        assert!(
            metadata
                .metadata
                .get("auto_generated")
                .and_then(|v| v.as_bool())
                == Some(true)
        );
    }

    #[test]
    fn test_generate_metadata_empty_repo_uses_dir_name() {
        let dir = tempfile::Builder::new()
            .prefix("my-custom-model")
            .tempdir()
            .unwrap();

        std::fs::write(dir.path().join("model.onnx"), b"dummy").unwrap();

        let (metadata, _) = generate_metadata(dir.path(), "").unwrap();

        // Model ID should be derived from directory name (with tempdir suffix stripped won't be exact,
        // but should start with the prefix)
        assert!(
            metadata.model_id.starts_with("my-custom-model"),
            "Expected model_id to start with 'my-custom-model', got: {}",
            metadata.model_id
        );
    }

    #[test]
    fn test_generate_metadata_populates_all_files() {
        let dir = TempDir::new().unwrap();

        std::fs::write(dir.path().join("model.onnx"), b"dummy onnx").unwrap();
        std::fs::write(dir.path().join("vocab.json"), "{}").unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        // These should be excluded:
        std::fs::write(dir.path().join("README.md"), "# Model").unwrap();
        std::fs::write(dir.path().join(".gitkeep"), "").unwrap();

        let (metadata, _) = generate_metadata(dir.path(), "test/model").unwrap();

        // Should include model.onnx, vocab.json, config.json but NOT README.md or .gitkeep
        assert!(metadata.files.contains(&"model.onnx".to_string()));
        assert!(metadata.files.contains(&"vocab.json".to_string()));
        assert!(metadata.files.contains(&"config.json".to_string()));
        assert!(!metadata.files.contains(&"README.md".to_string()));
        assert!(!metadata.files.contains(&".gitkeep".to_string()));
        // model_metadata.json itself should also be excluded
        assert!(!metadata.files.contains(&"model_metadata.json".to_string()));
    }

    #[test]
    fn test_generate_metadata_uses_supporting_files() {
        let dir = TempDir::new().unwrap();

        // ONNX model + preprocessor config for an image model
        std::fs::write(dir.path().join("model.onnx"), b"dummy").unwrap();
        std::fs::write(
            dir.path().join("README.md"),
            "---\npipeline_tag: image-classification\n---\n# Vision\n",
        )
        .unwrap();
        let preproc = serde_json::json!({
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.25, 0.25, 0.25],
            "size": { "height": 384, "width": 384 }
        });
        std::fs::write(
            dir.path().join("preprocessor_config.json"),
            serde_json::to_string(&preproc).unwrap(),
        )
        .unwrap();

        let (metadata, _) = generate_metadata(dir.path(), "test/vision").unwrap();

        // Without onnx-inspect, task inference won't have ONNX info, so
        // the fallback path should still pick up image_mean/std from supporting files
        if let Some(xybrid_core::execution::PreprocessingStep::Normalize { mean, std }) =
            metadata.preprocessing.first()
        {
            // Should use custom values from preprocessor_config, not ImageNet defaults
            assert_eq!(mean, &[0.5f32, 0.5, 0.5]);
            assert_eq!(std, &[0.25f32, 0.25, 0.25]);
        } else {
            panic!("Expected Normalize preprocessing for image-classification");
        }
    }

    #[test]
    #[cfg(feature = "onnx-inspect")]
    fn test_generate_metadata_mnist_fixture() {
        // Integration test: run generate_metadata on the real mnist fixture
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../integration-tests/fixtures/models/mnist");
        if !fixture_dir.exists() {
            eprintln!(
                "Skipping test: mnist fixture not found at {}",
                fixture_dir.display()
            );
            return;
        }

        // Create a temp copy so we don't write model_metadata.json into the fixture dir
        let tmp = TempDir::new().unwrap();
        let model_src = fixture_dir.join("model.onnx");
        std::fs::copy(&model_src, tmp.path().join("model.onnx")).unwrap();

        let (metadata, task_inference) = generate_metadata(tmp.path(), "").unwrap();

        // Should detect as ONNX with image-classification task
        assert!(matches!(
            metadata.execution_template,
            xybrid_core::execution::ExecutionTemplate::Onnx { .. }
        ));

        // Task inference should be available since we have onnx-inspect
        let ti = task_inference.expect("Should have TaskInference with onnx-inspect feature");

        // MNIST has input: Input3 [1,1,28,28] and output: Plus214_Output_0 [1,10]
        // This should match the image-classification pattern (4D input + small class count output)
        assert_eq!(ti.task, "image-classification");
        assert!(matches!(ti.confidence, Confidence::Medium));

        // Should have Normalize preprocessing and Softmax postprocessing
        assert!(
            !metadata.preprocessing.is_empty(),
            "Should have preprocessing"
        );
        assert!(matches!(
            metadata.preprocessing[0],
            xybrid_core::execution::PreprocessingStep::Normalize { .. }
        ));
        assert!(
            !metadata.postprocessing.is_empty(),
            "Should have postprocessing"
        );
        assert!(matches!(
            metadata.postprocessing[0],
            xybrid_core::execution::PostprocessingStep::Softmax { .. }
        ));

        // Files should include model.onnx
        assert!(metadata.files.contains(&"model.onnx".to_string()));
    }
}
