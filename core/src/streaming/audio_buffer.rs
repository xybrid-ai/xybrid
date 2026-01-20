//! Audio buffer for streaming ASR.
//!
//! Manages audio chunks with ring buffer semantics for efficient
//! streaming inference.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for audio buffering.
#[derive(Debug, Clone)]
pub struct AudioBufferConfig {
    /// Sample rate of audio (default: 16000 Hz for ASR)
    pub sample_rate: u32,
    /// Chunk duration for processing (default: 30s for Whisper)
    pub chunk_duration_secs: f32,
    /// Overlap between chunks for continuity (default: 1s)
    pub overlap_secs: f32,
    /// Maximum buffer duration before oldest samples are discarded
    pub max_buffer_secs: f32,
}

impl Default for AudioBufferConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            chunk_duration_secs: 30.0, // Whisper's 30-second window
            overlap_secs: 1.0,         // 1 second overlap for continuity
            max_buffer_secs: 120.0,    // 2 minutes max buffer
        }
    }
}

impl AudioBufferConfig {
    /// Create config for Whisper ASR (30s chunks, 16kHz)
    pub fn whisper() -> Self {
        Self::default()
    }

    /// Create config for Wav2Vec2 ASR (shorter chunks, 16kHz)
    pub fn wav2vec2() -> Self {
        Self {
            sample_rate: 16000,
            chunk_duration_secs: 10.0, // Wav2Vec2 works better with shorter chunks
            overlap_secs: 0.5,
            max_buffer_secs: 60.0,
        }
    }

    /// Samples per chunk
    pub fn chunk_samples(&self) -> usize {
        (self.sample_rate as f32 * self.chunk_duration_secs) as usize
    }

    /// Overlap samples
    pub fn overlap_samples(&self) -> usize {
        (self.sample_rate as f32 * self.overlap_secs) as usize
    }

    /// Maximum buffer samples
    pub fn max_buffer_samples(&self) -> usize {
        (self.sample_rate as f32 * self.max_buffer_secs) as usize
    }
}

/// A chunk of audio ready for processing.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// PCM samples (f32, mono)
    pub samples: Vec<f32>,
    /// Timestamp when this chunk starts (relative to stream start)
    pub start_time: Duration,
    /// Timestamp when this chunk ends
    pub end_time: Duration,
    /// Sequence number for ordering
    pub sequence: u64,
    /// Whether this is the final chunk (stream ended)
    pub is_final: bool,
}

impl AudioChunk {
    /// Duration of this chunk
    pub fn duration(&self) -> Duration {
        self.end_time - self.start_time
    }

    /// Number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Ring buffer for streaming audio with chunk extraction.
#[derive(Debug)]
pub struct AudioBuffer {
    /// Configuration
    config: AudioBufferConfig,
    /// Sample buffer (ring buffer)
    samples: VecDeque<f32>,
    /// Total samples received (for timing)
    total_samples_received: u64,
    /// Samples already processed (for chunk extraction)
    samples_processed: u64,
    /// Next chunk sequence number
    next_sequence: u64,
    /// Stream start time
    stream_start: Instant,
    /// Whether the stream has ended
    stream_ended: bool,
}

impl AudioBuffer {
    /// Create a new audio buffer with default configuration.
    pub fn new() -> Self {
        Self::with_config(AudioBufferConfig::default())
    }

    /// Create a new audio buffer with custom configuration.
    pub fn with_config(config: AudioBufferConfig) -> Self {
        Self {
            samples: VecDeque::with_capacity(config.max_buffer_samples()),
            config,
            total_samples_received: 0,
            samples_processed: 0,
            next_sequence: 0,
            stream_start: Instant::now(),
            stream_ended: false,
        }
    }

    /// Push audio samples into the buffer.
    ///
    /// # Arguments
    ///
    /// * `samples` - PCM samples (f32, mono, at configured sample rate)
    pub fn push(&mut self, samples: &[f32]) {
        // Add samples to buffer
        for &sample in samples {
            // If buffer is full, remove oldest samples
            if self.samples.len() >= self.config.max_buffer_samples() {
                self.samples.pop_front();
                // Increment processed to maintain timing
                if self.samples_processed < self.total_samples_received {
                    self.samples_processed += 1;
                }
            }
            self.samples.push_back(sample);
        }
        self.total_samples_received += samples.len() as u64;
    }

    /// Mark the stream as ended (no more samples will arrive).
    pub fn end_stream(&mut self) {
        self.stream_ended = true;
    }

    /// Check if stream has ended.
    pub fn is_ended(&self) -> bool {
        self.stream_ended
    }

    /// Get the number of unprocessed samples available.
    pub fn available_samples(&self) -> usize {
        let processed_in_buffer = self
            .samples_processed
            .saturating_sub(self.total_samples_received - self.samples.len() as u64);
        self.samples
            .len()
            .saturating_sub(processed_in_buffer as usize)
    }

    /// Check if a full chunk is ready for processing.
    pub fn has_chunk_ready(&self) -> bool {
        self.available_samples() >= self.config.chunk_samples()
    }

    /// Check if any audio is available (for final flush).
    pub fn has_audio(&self) -> bool {
        self.available_samples() > 0
    }

    /// Extract the next chunk for processing.
    ///
    /// Returns `None` if not enough samples are available.
    /// Use `force = true` to get a partial chunk (e.g., at end of stream).
    pub fn extract_chunk(&mut self, force: bool) -> Option<AudioChunk> {
        let available = self.available_samples();
        let chunk_size = self.config.chunk_samples();

        // Check if we have enough samples (or forcing partial chunk)
        if !force && available < chunk_size {
            return None;
        }

        if available == 0 {
            return None;
        }

        // Calculate how many samples to extract
        let extract_size = if force {
            available.min(chunk_size)
        } else {
            chunk_size
        };

        // Calculate timing
        let sample_rate = self.config.sample_rate as f64;
        let start_sample = self.samples_processed;
        let start_time = Duration::from_secs_f64(start_sample as f64 / sample_rate);
        let end_time =
            Duration::from_secs_f64((start_sample + extract_size as u64) as f64 / sample_rate);

        // Extract samples (keeping overlap)
        let overlap = self.config.overlap_samples();
        let samples: Vec<f32> = self
            .samples
            .iter()
            .skip(self.get_buffer_offset())
            .take(extract_size)
            .copied()
            .collect();

        // Advance processed counter (minus overlap for next chunk)
        let advance = if force {
            extract_size // No overlap on final chunk
        } else {
            extract_size.saturating_sub(overlap)
        };
        self.samples_processed += advance as u64;

        // Create chunk
        let chunk = AudioChunk {
            samples,
            start_time,
            end_time,
            sequence: self.next_sequence,
            is_final: force && self.stream_ended,
        };

        self.next_sequence += 1;

        Some(chunk)
    }

    /// Flush remaining audio as a final chunk.
    pub fn flush(&mut self) -> Option<AudioChunk> {
        if !self.has_audio() {
            return None;
        }
        self.extract_chunk(true)
    }

    /// Reset the buffer state.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.total_samples_received = 0;
        self.samples_processed = 0;
        self.next_sequence = 0;
        self.stream_start = Instant::now();
        self.stream_ended = false;
    }

    /// Get buffer statistics.
    pub fn stats(&self) -> AudioBufferStats {
        AudioBufferStats {
            total_received: self.total_samples_received,
            total_processed: self.samples_processed,
            buffer_size: self.samples.len(),
            available: self.available_samples(),
            chunks_extracted: self.next_sequence,
            elapsed: self.stream_start.elapsed(),
        }
    }

    /// Get current buffer offset for extraction.
    fn get_buffer_offset(&self) -> usize {
        // Calculate offset into buffer for unprocessed samples
        let buffer_start_sample = self
            .total_samples_received
            .saturating_sub(self.samples.len() as u64);
        self.samples_processed.saturating_sub(buffer_start_sample) as usize
    }

    /// Get the configuration.
    pub fn config(&self) -> &AudioBufferConfig {
        &self.config
    }
}

impl Default for AudioBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the audio buffer state.
#[derive(Debug, Clone)]
pub struct AudioBufferStats {
    /// Total samples received
    pub total_received: u64,
    /// Total samples processed (sent to inference)
    pub total_processed: u64,
    /// Current buffer size
    pub buffer_size: usize,
    /// Available unprocessed samples
    pub available: usize,
    /// Number of chunks extracted
    pub chunks_extracted: u64,
    /// Time elapsed since stream start
    pub elapsed: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_config_defaults() {
        let config = AudioBufferConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.chunk_duration_secs, 30.0);
        assert_eq!(config.chunk_samples(), 480000); // 30s * 16000
    }

    #[test]
    fn test_buffer_push() {
        let mut buffer = AudioBuffer::new();
        let samples = vec![0.1, 0.2, 0.3];
        buffer.push(&samples);

        assert_eq!(buffer.available_samples(), 3);
        assert!(!buffer.has_chunk_ready()); // Need 480000 samples
    }

    #[test]
    fn test_buffer_extract_forced() {
        let mut buffer = AudioBuffer::new();
        let samples: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        buffer.push(&samples);
        buffer.end_stream();

        // Force extract partial chunk
        let chunk = buffer.extract_chunk(true).unwrap();
        assert_eq!(chunk.samples.len(), 1000);
        assert!(chunk.is_final);
        assert_eq!(chunk.sequence, 0);
    }

    #[test]
    fn test_buffer_full_chunk() {
        let config = AudioBufferConfig {
            sample_rate: 16000,
            chunk_duration_secs: 0.1, // 100ms = 1600 samples
            overlap_secs: 0.01,       // 10ms = 160 samples overlap
            max_buffer_secs: 1.0,
        };
        let mut buffer = AudioBuffer::with_config(config);

        // Push enough for one chunk
        let samples: Vec<f32> = (0..1600).map(|i| i as f32 / 1600.0).collect();
        buffer.push(&samples);

        assert!(buffer.has_chunk_ready());

        let chunk = buffer.extract_chunk(false).unwrap();
        assert_eq!(chunk.samples.len(), 1600);
        assert_eq!(chunk.sequence, 0);
        assert!(!chunk.is_final);
    }

    #[test]
    fn test_buffer_overlap() {
        let config = AudioBufferConfig {
            sample_rate: 1000, // Simplified for testing
            chunk_duration_secs: 1.0,
            overlap_secs: 0.2, // 200 sample overlap
            max_buffer_secs: 10.0,
        };
        let mut buffer = AudioBuffer::with_config(config);

        // Push 2 chunks worth
        let samples: Vec<f32> = (0..2000).map(|i| i as f32).collect();
        buffer.push(&samples);

        // Extract first chunk
        let chunk1 = buffer.extract_chunk(false).unwrap();
        assert_eq!(chunk1.samples.len(), 1000);

        // Second chunk should start 200 samples before end of first (overlap)
        let chunk2 = buffer.extract_chunk(false).unwrap();
        assert_eq!(chunk2.samples.len(), 1000);
        // First sample of chunk2 should be sample 800 (1000 - 200 overlap)
        assert_eq!(chunk2.samples[0], 800.0);
    }

    #[test]
    fn test_buffer_reset() {
        let mut buffer = AudioBuffer::new();
        buffer.push(&[1.0, 2.0, 3.0]);

        buffer.reset();

        assert_eq!(buffer.available_samples(), 0);
        assert!(!buffer.has_audio());
    }

    #[test]
    fn test_chunk_duration() {
        let chunk = AudioChunk {
            samples: vec![0.0; 16000],
            start_time: Duration::from_secs(0),
            end_time: Duration::from_secs(1),
            sequence: 0,
            is_final: false,
        };

        assert_eq!(chunk.duration(), Duration::from_secs(1));
        assert_eq!(chunk.len(), 16000);
    }
}
