//! Voice embedding loader for TTS models.
//!
//! This module extracts voice loading logic from the executor, making it:
//! - Independently testable
//! - Mockable for unit tests
//! - Reusable across different execution paths

use log::debug;
use std::path::{Path, PathBuf};

use super::template::{ModelMetadata, VoiceConfig, VoiceFormat, VoiceInfo, VoiceLoader};
use super::types::ExecutorResult;
use crate::ir::Envelope;
use crate::runtime_adapter::AdapterError;

/// Default embedding dimension for voice vectors.
pub const DEFAULT_EMBEDDING_DIM: usize = 256;

/// Trait for loading voice embeddings.
///
/// This trait enables mocking voice loading in tests without file system access.
pub trait VoiceEmbeddingSource: Send + Sync {
    /// Load voice embedding by index from a binary file.
    fn load_by_index(&self, path: &Path, index: usize) -> Result<Vec<f32>, String>;

    /// Load voice embedding by name from an NPZ file.
    fn load_by_name(&self, path: &Path, name: &str) -> Result<Vec<f32>, String>;
}

/// Default implementation using the crate's VoiceEmbeddingLoader.
pub struct DefaultVoiceSource {
    embedding_dim: usize,
}

impl DefaultVoiceSource {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }
}

impl Default for DefaultVoiceSource {
    fn default() -> Self {
        Self::new(DEFAULT_EMBEDDING_DIM)
    }
}

impl VoiceEmbeddingSource for DefaultVoiceSource {
    fn load_by_index(&self, path: &Path, index: usize) -> Result<Vec<f32>, String> {
        let loader = crate::tts::voice_embedding::VoiceEmbeddingLoader::new(self.embedding_dim);
        loader.load(path, index).map_err(|e| e.to_string())
    }

    fn load_by_name(&self, path: &Path, name: &str) -> Result<Vec<f32>, String> {
        let loader = crate::tts::voice_embedding::VoiceEmbeddingLoader::new(self.embedding_dim);
        loader
            .load_npz_by_name(path, name, None)
            .map_err(|e| e.to_string())
    }
}

/// TTS Voice Loader - handles voice resolution and loading.
///
/// This struct encapsulates the logic for:
/// 1. Resolving voice file path (from config or legacy auto-detect)
/// 2. Resolving voice ID (from envelope or default)
/// 3. Loading the embedding (via VoiceEmbeddingSource trait)
pub struct TtsVoiceLoader<S: VoiceEmbeddingSource = DefaultVoiceSource> {
    base_path: PathBuf,
    source: S,
}

impl TtsVoiceLoader<DefaultVoiceSource> {
    /// Create a new voice loader with the default embedding source.
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
            source: DefaultVoiceSource::default(),
        }
    }
}

impl<S: VoiceEmbeddingSource> TtsVoiceLoader<S> {
    /// Create a voice loader with a custom embedding source (for testing).
    pub fn with_source(base_path: impl Into<PathBuf>, source: S) -> Self {
        Self {
            base_path: base_path.into(),
            source,
        }
    }

    /// Load voice embedding based on metadata and input envelope.
    ///
    /// Resolution order:
    /// 1. `voice_id` from Envelope.metadata (if present)
    /// 2. Default voice from ModelMetadata.voices.default
    /// 3. Index 0 (legacy fallback for models without voice config)
    pub fn load(&self, metadata: &ModelMetadata, input: &Envelope) -> ExecutorResult<Vec<f32>> {
        // Step 1: Resolve voice file path
        let voice_path = match self.resolve_voice_path(metadata)? {
            Some(path) => path,
            None => {
                debug!(target: "xybrid_core", "No voice file found, using zero embedding");
                return Ok(vec![0.0f32; DEFAULT_EMBEDDING_DIM]);
            }
        };

        // Step 2: Check file exists
        if !voice_path.exists() {
            debug!(target: "xybrid_core", "Voice file not found: {:?}, using zero embedding", voice_path);
            return Ok(vec![0.0f32; DEFAULT_EMBEDDING_DIM]);
        }

        // Step 3: Get voice_id from envelope metadata
        let voice_id = input.metadata.get("voice_id");

        // Step 4: Load embedding based on config style
        if let Some(voice_config) = &metadata.voices {
            self.load_with_config(&voice_path, voice_config, metadata, voice_id)
        } else {
            self.load_legacy(&voice_path, voice_id)
        }
    }

    /// Resolve the voice file path from metadata or legacy auto-detection.
    fn resolve_voice_path(&self, metadata: &ModelMetadata) -> ExecutorResult<Option<PathBuf>> {
        if let Some(voice_config) = &metadata.voices {
            match &voice_config.format {
                VoiceFormat::Embedded { file, .. } => {
                    Ok(Some(self.base_path.join(file)))
                }
                VoiceFormat::PerModel { .. } | VoiceFormat::Cloning { .. } => {
                    Err(AdapterError::InvalidInput(
                        "Only embedded voice format is currently supported".to_string(),
                    ))
                }
            }
        } else {
            // Legacy: auto-detect voices.bin or voices.npz
            let voices_bin = self.base_path.join("voices.bin");
            let voices_npz = self.base_path.join("voices.npz");

            if voices_bin.exists() {
                Ok(Some(voices_bin))
            } else if voices_npz.exists() {
                Ok(Some(voices_npz))
            } else {
                Ok(None)
            }
        }
    }

    /// Load voice embedding using structured voice config.
    fn load_with_config(
        &self,
        voice_path: &Path,
        voice_config: &VoiceConfig,
        metadata: &ModelMetadata,
        voice_id: Option<&String>,
    ) -> ExecutorResult<Vec<f32>> {
        // Resolve voice info
        let voice_info = self.resolve_voice_info(voice_config, metadata, voice_id)?;

        debug!(
            target: "xybrid_core",
            "Loading voice: {} (index: {:?})",
            voice_info.id,
            voice_info.index
        );

        // Determine loader type and load
        let is_npz = matches!(
            &voice_config.format,
            VoiceFormat::Embedded {
                loader: VoiceLoader::NumpyNpz,
                ..
            }
        );

        if is_npz {
            self.source
                .load_by_name(voice_path, &voice_info.id)
                .map_err(|e| {
                    AdapterError::RuntimeError(format!(
                        "Failed to load voice '{}' by name: {}",
                        voice_info.id, e
                    ))
                })
        } else if let Some(index) = voice_info.index {
            self.source.load_by_index(voice_path, index).map_err(|e| {
                AdapterError::RuntimeError(format!(
                    "Failed to load voice '{}' (index {}): {}",
                    voice_info.id, index, e
                ))
            })
        } else {
            // Fallback: try by name
            self.source
                .load_by_name(voice_path, &voice_info.id)
                .map_err(|e| {
                    AdapterError::RuntimeError(format!(
                        "Failed to load voice '{}' by name: {}",
                        voice_info.id, e
                    ))
                })
        }
    }

    /// Resolve voice info from config, preferring envelope voice_id over default.
    fn resolve_voice_info<'a>(
        &self,
        voice_config: &'a VoiceConfig,
        metadata: &'a ModelMetadata,
        voice_id: Option<&String>,
    ) -> ExecutorResult<&'a VoiceInfo> {
        if let Some(vid) = voice_id {
            metadata.get_voice(vid).ok_or_else(|| {
                let available: Vec<_> = voice_config
                    .catalog
                    .iter()
                    .map(|v| v.id.as_str())
                    .collect();
                AdapterError::InvalidInput(format!(
                    "Voice '{}' not found. Available voices: {:?}",
                    vid, available
                ))
            })
        } else {
            metadata.default_voice().ok_or_else(|| {
                AdapterError::RuntimeError(format!(
                    "Default voice '{}' not found in catalog",
                    voice_config.default
                ))
            })
        }
    }

    /// Load voice embedding using legacy auto-detection (no structured config).
    fn load_legacy(
        &self,
        voice_path: &Path,
        voice_id: Option<&String>,
    ) -> ExecutorResult<Vec<f32>> {
        if let Some(vid) = voice_id {
            // Try parsing as index first
            if let Ok(index) = vid.parse::<usize>() {
                debug!(target: "xybrid_core", "Loading voice by index: {}", index);
                self.source.load_by_index(voice_path, index)
            } else {
                // Try as name (NPZ only)
                debug!(target: "xybrid_core", "Loading voice by name: {}", vid);
                self.source.load_by_name(voice_path, vid)
            }
        } else {
            // Default to index 0
            debug!(target: "xybrid_core", "Loading default voice (index 0)");
            self.source.load_by_index(voice_path, 0)
        }
        .map_err(|e| AdapterError::RuntimeError(format!("Failed to load voice embedding: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ============================================================================
    // Mock Voice Source for Testing
    // ============================================================================

    struct MockVoiceSource {
        voices_by_index: HashMap<usize, Vec<f32>>,
        voices_by_name: HashMap<String, Vec<f32>>,
    }

    impl MockVoiceSource {
        fn new() -> Self {
            Self {
                voices_by_index: HashMap::new(),
                voices_by_name: HashMap::new(),
            }
        }

        fn with_voice_at_index(mut self, index: usize, embedding: Vec<f32>) -> Self {
            self.voices_by_index.insert(index, embedding);
            self
        }

        fn with_voice_by_name(mut self, name: &str, embedding: Vec<f32>) -> Self {
            self.voices_by_name.insert(name.to_string(), embedding);
            self
        }
    }

    impl VoiceEmbeddingSource for MockVoiceSource {
        fn load_by_index(&self, _path: &Path, index: usize) -> Result<Vec<f32>, String> {
            self.voices_by_index
                .get(&index)
                .cloned()
                .ok_or_else(|| format!("Voice at index {} not found", index))
        }

        fn load_by_name(&self, _path: &Path, name: &str) -> Result<Vec<f32>, String> {
            self.voices_by_name
                .get(name)
                .cloned()
                .ok_or_else(|| format!("Voice '{}' not found", name))
        }
    }

    // ============================================================================
    // Helper Functions
    // ============================================================================

    fn create_test_metadata_with_voices() -> ModelMetadata {
        use super::super::template::{VoiceConfig, VoiceFormat, VoiceInfo, VoiceLoader};

        let mut metadata = ModelMetadata::onnx("test-tts", "1.0", "model.onnx");
        metadata.voices = Some(VoiceConfig {
            format: VoiceFormat::Embedded {
                file: "voices.bin".to_string(),
                loader: VoiceLoader::BinaryF32_256,
            },
            default: "voice_a".to_string(),
            catalog: vec![
                VoiceInfo {
                    id: "voice_a".to_string(),
                    name: "Voice A".to_string(),
                    gender: None,
                    language: None,
                    style: None,
                    index: Some(0),
                    preview_url: None,
                },
                VoiceInfo {
                    id: "voice_b".to_string(),
                    name: "Voice B".to_string(),
                    gender: None,
                    language: None,
                    style: None,
                    index: Some(1),
                    preview_url: None,
                },
            ],
        });
        metadata
    }

    fn create_test_envelope() -> Envelope {
        use crate::ir::EnvelopeKind;
        Envelope::new(EnvelopeKind::Text("Hello world".to_string()))
    }

    fn create_test_envelope_with_voice(voice_id: &str) -> Envelope {
        use crate::ir::EnvelopeKind;
        let mut metadata = HashMap::new();
        metadata.insert("voice_id".to_string(), voice_id.to_string());
        Envelope::with_metadata(EnvelopeKind::Text("Hello world".to_string()), metadata)
    }

    // ============================================================================
    // resolve_voice_path Tests
    // ============================================================================

    #[test]
    fn test_resolve_voice_path_with_config() {
        let source = MockVoiceSource::new();
        let loader = TtsVoiceLoader::with_source("/models/tts", source);
        let metadata = create_test_metadata_with_voices();

        let path = loader.resolve_voice_path(&metadata).unwrap();
        assert!(path.is_some());
        assert!(path.unwrap().ends_with("voices.bin"));
    }

    #[test]
    fn test_resolve_voice_path_no_config_no_files() {
        let source = MockVoiceSource::new();
        let loader = TtsVoiceLoader::with_source("/nonexistent", source);
        let metadata = ModelMetadata::onnx("test", "1.0", "model.onnx");

        let path = loader.resolve_voice_path(&metadata).unwrap();
        assert!(path.is_none());
    }

    // ============================================================================
    // resolve_voice_info Tests
    // ============================================================================

    #[test]
    fn test_resolve_voice_info_default() {
        let source = MockVoiceSource::new();
        let loader = TtsVoiceLoader::with_source("/models", source);
        let metadata = create_test_metadata_with_voices();
        let voice_config = metadata.voices.as_ref().unwrap();

        let info = loader
            .resolve_voice_info(voice_config, &metadata, None)
            .unwrap();
        assert_eq!(info.id, "voice_a");
    }

    #[test]
    fn test_resolve_voice_info_by_id() {
        let source = MockVoiceSource::new();
        let loader = TtsVoiceLoader::with_source("/models", source);
        let metadata = create_test_metadata_with_voices();
        let voice_config = metadata.voices.as_ref().unwrap();
        let voice_id = "voice_b".to_string();

        let info = loader
            .resolve_voice_info(voice_config, &metadata, Some(&voice_id))
            .unwrap();
        assert_eq!(info.id, "voice_b");
        assert_eq!(info.index, Some(1));
    }

    #[test]
    fn test_resolve_voice_info_not_found() {
        let source = MockVoiceSource::new();
        let loader = TtsVoiceLoader::with_source("/models", source);
        let metadata = create_test_metadata_with_voices();
        let voice_config = metadata.voices.as_ref().unwrap();
        let voice_id = "nonexistent".to_string();

        let result = loader.resolve_voice_info(voice_config, &metadata, Some(&voice_id));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found"));
        assert!(err.contains("voice_a")); // Should list available voices
    }

    // ============================================================================
    // load_legacy Tests
    // ============================================================================

    #[test]
    fn test_load_legacy_by_index() {
        let embedding = vec![1.0, 2.0, 3.0];
        let source = MockVoiceSource::new().with_voice_at_index(5, embedding.clone());
        let loader = TtsVoiceLoader::with_source("/models", source);
        let voice_id = "5".to_string();

        let result = loader
            .load_legacy(Path::new("/models/voices.bin"), Some(&voice_id))
            .unwrap();
        assert_eq!(result, embedding);
    }

    #[test]
    fn test_load_legacy_by_name() {
        let embedding = vec![4.0, 5.0, 6.0];
        let source = MockVoiceSource::new().with_voice_by_name("bella", embedding.clone());
        let loader = TtsVoiceLoader::with_source("/models", source);
        let voice_id = "bella".to_string();

        let result = loader
            .load_legacy(Path::new("/models/voices.npz"), Some(&voice_id))
            .unwrap();
        assert_eq!(result, embedding);
    }

    #[test]
    fn test_load_legacy_default_index_0() {
        let embedding = vec![7.0, 8.0, 9.0];
        let source = MockVoiceSource::new().with_voice_at_index(0, embedding.clone());
        let loader = TtsVoiceLoader::with_source("/models", source);

        let result = loader
            .load_legacy(Path::new("/models/voices.bin"), None)
            .unwrap();
        assert_eq!(result, embedding);
    }

    // ============================================================================
    // Integration-style Tests (with Mock)
    // ============================================================================

    #[test]
    fn test_load_returns_zero_embedding_when_no_voice_file() {
        let source = MockVoiceSource::new();
        let loader = TtsVoiceLoader::with_source("/nonexistent/path", source);
        let metadata = ModelMetadata::onnx("test", "1.0", "model.onnx");
        let input = create_test_envelope();

        let result = loader.load(&metadata, &input).unwrap();
        assert_eq!(result.len(), DEFAULT_EMBEDDING_DIM);
        assert!(result.iter().all(|&v| v == 0.0));
    }
}
