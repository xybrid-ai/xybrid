//! Voice embedding loading utilities for TTS models.
//!
//! This module provides utilities for loading voice embeddings from different formats:
//! - **Raw binary format** (KittenTTS): Simple contiguous f32 arrays per voice
//! - **NPZ format** (Kokoro): NumPy ZIP archives with shape (510, 1, 256) per voice
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_core::tts::voice_embedding::{VoiceEmbeddingLoader, VoiceFormat};
//!
//! // Auto-detect format and load
//! let loader = VoiceEmbeddingLoader::new(256);
//! let embedding = loader.load("voices.bin", 0)?;
//!
//! // Or load all embeddings (raw format only)
//! let all_embeddings = loader.load_all_raw("voices.bin")?;
//! ```

use std::path::Path;
use thiserror::Error;

/// Default voice embedding dimension (used by KittenTTS and Kokoro)
pub const DEFAULT_EMBEDDING_DIM: usize = 256;

/// Errors that can occur when loading voice embeddings.
#[derive(Debug, Error)]
pub enum VoiceError {
    /// File not found or cannot be read.
    #[error("Failed to read voice file: {0}")]
    FileError(String),

    /// Invalid voice index (out of range).
    #[error("Voice index {index} out of range (max: {max})")]
    IndexOutOfRange { index: usize, max: usize },

    /// Dimension mismatch between expected and actual embedding.
    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// NPZ parsing or extraction error.
    #[error("NPZ format error: {0}")]
    NpzError(String),

    /// No voices found in file.
    #[error("No voices found in file")]
    NoVoices,

    /// Voice name not found in NPZ.
    #[error("Voice '{0}' not found in NPZ file")]
    VoiceNotFound(String),
}

impl From<std::io::Error> for VoiceError {
    fn from(err: std::io::Error) -> Self {
        VoiceError::FileError(err.to_string())
    }
}

/// Detected format of the voice embedding file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceFormat {
    /// Raw binary format (f32 little-endian, contiguous per voice)
    Raw,
    /// NPZ format (NumPy ZIP archive)
    Npz,
}

/// Voice embedding loader supporting multiple formats.
///
/// This loader auto-detects the file format (NPZ vs raw binary) and provides
/// a unified interface for loading voice embeddings.
pub struct VoiceEmbeddingLoader {
    embedding_dim: usize,
}

impl Default for VoiceEmbeddingLoader {
    fn default() -> Self {
        Self::new(DEFAULT_EMBEDDING_DIM)
    }
}

impl VoiceEmbeddingLoader {
    /// Create a new loader with the specified embedding dimension.
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Detect the format of a voice file by examining its magic bytes.
    ///
    /// NPZ files start with "PK" (ZIP magic bytes).
    pub fn detect_format(bytes: &[u8]) -> VoiceFormat {
        if bytes.len() >= 2 && bytes[0] == b'P' && bytes[1] == b'K' {
            VoiceFormat::Npz
        } else {
            VoiceFormat::Raw
        }
    }

    /// Load a voice embedding from a file, auto-detecting the format.
    ///
    /// # Arguments
    /// * `path` - Path to the voices file (voices.bin, voices.npz)
    /// * `voice_index` - Index of the voice to load
    ///
    /// # Returns
    /// The voice embedding as a Vec<f32> of length `embedding_dim`
    pub fn load(&self, path: impl AsRef<Path>, voice_index: usize) -> Result<Vec<f32>, VoiceError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;

        match Self::detect_format(&bytes) {
            VoiceFormat::Npz => self.load_npz_by_index(path, voice_index),
            VoiceFormat::Raw => self.load_raw(&bytes, voice_index),
        }
    }

    /// Load a voice embedding by name from an NPZ file.
    ///
    /// This is useful for Kokoro-style voices where each voice has a name.
    ///
    /// # Arguments
    /// * `path` - Path to the NPZ file
    /// * `voice_name` - Name of the voice (e.g., "af_bella")
    /// * `token_length` - Token length index (affects embedding selection in Kokoro)
    pub fn load_npz_by_name(
        &self,
        path: impl AsRef<Path>,
        voice_name: &str,
        token_length: Option<usize>,
    ) -> Result<Vec<f32>, VoiceError> {
        use ndarray_npy::NpzReader;
        use std::fs::File;

        let file = File::open(path.as_ref())?;
        let mut npz = NpzReader::new(file).map_err(|e| VoiceError::NpzError(e.to_string()))?;

        // Load the voice array - shape is (510, 1, 256)
        let voice_data: ndarray::Array3<f32> = npz
            .by_name(voice_name)
            .map_err(|_| VoiceError::VoiceNotFound(voice_name.to_string()))?;

        // Select token length index (default to 100 which is mid-range)
        let token_len_idx = token_length.unwrap_or(100).min(voice_data.shape()[0] - 1);

        // Extract embedding at token_len_idx, row 0
        let embedding: Vec<f32> = voice_data
            .slice(ndarray::s![token_len_idx, 0, ..])
            .iter()
            .copied()
            .collect();

        if embedding.len() != self.embedding_dim {
            return Err(VoiceError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: embedding.len(),
            });
        }

        Ok(embedding)
    }

    /// Load a voice embedding by index from an NPZ file.
    fn load_npz_by_index(
        &self,
        path: impl AsRef<Path>,
        voice_index: usize,
    ) -> Result<Vec<f32>, VoiceError> {
        use ndarray_npy::NpzReader;
        use std::fs::File;

        let file = File::open(path.as_ref())?;
        let mut npz = NpzReader::new(file).map_err(|e| VoiceError::NpzError(e.to_string()))?;

        // Get list of voice names
        let voice_names = npz
            .names()
            .map_err(|e| VoiceError::NpzError(e.to_string()))?;

        if voice_names.is_empty() {
            return Err(VoiceError::NoVoices);
        }

        if voice_index >= voice_names.len() {
            return Err(VoiceError::IndexOutOfRange {
                index: voice_index,
                max: voice_names.len() - 1,
            });
        }

        let voice_name = &voice_names[voice_index];
        self.load_npz_by_name_from_reader(&mut npz, voice_name, None)
    }

    /// Internal helper to load from an already-opened NPZ reader.
    fn load_npz_by_name_from_reader<R: std::io::Read + std::io::Seek>(
        &self,
        npz: &mut ndarray_npy::NpzReader<R>,
        voice_name: &str,
        token_length: Option<usize>,
    ) -> Result<Vec<f32>, VoiceError> {
        // Load the voice array - shape is (510, 1, 256)
        let voice_data: ndarray::Array3<f32> = npz
            .by_name(voice_name)
            .map_err(|_| VoiceError::VoiceNotFound(voice_name.to_string()))?;

        // Select token length index (default to 100 which is mid-range)
        let token_len_idx = token_length.unwrap_or(100).min(voice_data.shape()[0] - 1);

        // Extract embedding at token_len_idx, row 0
        let embedding: Vec<f32> = voice_data
            .slice(ndarray::s![token_len_idx, 0, ..])
            .iter()
            .copied()
            .collect();

        if embedding.len() != self.embedding_dim {
            return Err(VoiceError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: embedding.len(),
            });
        }

        Ok(embedding)
    }

    /// Load a voice embedding from raw binary data.
    ///
    /// Raw format stores embeddings contiguously:
    /// - Each voice: `embedding_dim * 4` bytes (f32 little-endian)
    /// - Voices are stored sequentially
    pub fn load_raw(&self, bytes: &[u8], voice_index: usize) -> Result<Vec<f32>, VoiceError> {
        let voice_size = self.embedding_dim * 4;
        let num_voices = bytes.len() / voice_size;

        if num_voices == 0 {
            return Err(VoiceError::NoVoices);
        }

        if voice_index >= num_voices {
            return Err(VoiceError::IndexOutOfRange {
                index: voice_index,
                max: num_voices - 1,
            });
        }

        let start = voice_index * voice_size;
        let end = start + voice_size;
        let voice_bytes = &bytes[start..end];

        let embedding: Vec<f32> = voice_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(bytes)
            })
            .collect();

        Ok(embedding)
    }

    /// Load all voice embeddings from a raw binary file.
    ///
    /// Returns a vector of embeddings, one per voice.
    pub fn load_all_raw(&self, path: impl AsRef<Path>) -> Result<Vec<Vec<f32>>, VoiceError> {
        let bytes = std::fs::read(path)?;
        let voice_size = self.embedding_dim * 4;
        let num_voices = bytes.len() / voice_size;

        if num_voices == 0 {
            return Err(VoiceError::NoVoices);
        }

        let mut embeddings = Vec::with_capacity(num_voices);
        for voice_idx in 0..num_voices {
            embeddings.push(self.load_raw(&bytes, voice_idx)?);
        }

        Ok(embeddings)
    }

    /// Get the number of voices in a file.
    pub fn count_voices(&self, path: impl AsRef<Path>) -> Result<usize, VoiceError> {
        let bytes = std::fs::read(path.as_ref())?;

        match Self::detect_format(&bytes) {
            VoiceFormat::Npz => {
                use ndarray_npy::NpzReader;
                use std::fs::File;

                let file = File::open(path.as_ref())?;
                let mut npz =
                    NpzReader::new(file).map_err(|e| VoiceError::NpzError(e.to_string()))?;
                let names = npz
                    .names()
                    .map_err(|e| VoiceError::NpzError(e.to_string()))?;
                Ok(names.len())
            }
            VoiceFormat::Raw => {
                let voice_size = self.embedding_dim * 4;
                Ok(bytes.len() / voice_size)
            }
        }
    }

    /// List voice names from an NPZ file.
    ///
    /// Returns `None` for raw binary files (voices are unnamed).
    pub fn list_voice_names(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<Option<Vec<String>>, VoiceError> {
        let bytes = std::fs::read(path.as_ref())?;

        match Self::detect_format(&bytes) {
            VoiceFormat::Npz => {
                use ndarray_npy::NpzReader;
                use std::fs::File;

                let file = File::open(path.as_ref())?;
                let mut npz =
                    NpzReader::new(file).map_err(|e| VoiceError::NpzError(e.to_string()))?;
                let names = npz
                    .names()
                    .map_err(|e| VoiceError::NpzError(e.to_string()))?;
                Ok(Some(names))
            }
            VoiceFormat::Raw => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_raw() {
        let bytes = vec![0x00, 0x00, 0x80, 0x3f]; // f32 1.0 in little endian
        assert_eq!(
            VoiceEmbeddingLoader::detect_format(&bytes),
            VoiceFormat::Raw
        );
    }

    #[test]
    fn test_detect_format_npz() {
        let bytes = vec![b'P', b'K', 0x03, 0x04]; // ZIP magic bytes
        assert_eq!(
            VoiceEmbeddingLoader::detect_format(&bytes),
            VoiceFormat::Npz
        );
    }

    #[test]
    fn test_load_raw_single_voice() {
        let loader = VoiceEmbeddingLoader::new(4);
        // Create 2 voices with 4 dimensions each (16 bytes per voice)
        let mut bytes = Vec::new();
        for i in 0..8 {
            bytes.extend_from_slice(&(i as f32).to_le_bytes());
        }

        let voice0 = loader.load_raw(&bytes, 0).unwrap();
        assert_eq!(voice0, vec![0.0, 1.0, 2.0, 3.0]);

        let voice1 = loader.load_raw(&bytes, 1).unwrap();
        assert_eq!(voice1, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_load_raw_index_out_of_range() {
        let loader = VoiceEmbeddingLoader::new(4);
        let bytes = vec![0u8; 16]; // 1 voice

        let result = loader.load_raw(&bytes, 1);
        assert!(matches!(result, Err(VoiceError::IndexOutOfRange { .. })));
    }

    #[test]
    fn test_default_embedding_dim() {
        let loader = VoiceEmbeddingLoader::default();
        assert_eq!(loader.embedding_dim(), DEFAULT_EMBEDDING_DIM);
    }
}
