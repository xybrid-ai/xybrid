//! Stream Manager module - Manages streaming data flows between inference stages.
//!
//! The Stream Manager handles buffering, chunking, and flow control for continuous
//! inference pipelines that process streaming inputs (audio, video, sensor data).
//!
//! The stream manager supports the orchestrator's data flow architecture:
//! 1. Receive input envelope (stream chunks)
//! 2. Buffer and chunk management
//! 3. Flow control between stages
//! 4. Stream aggregation and splitting

use crate::ir::Envelope;
use std::collections::VecDeque;

/// Error type for stream manager operations.
#[derive(Debug, Clone)]
pub enum StreamManagerError {
    BufferOverflow(String),
    InvalidChunkSize(String),
    StreamClosed(String),
    InvalidState(String),
}

impl std::fmt::Display for StreamManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamManagerError::BufferOverflow(msg) => {
                write!(f, "Buffer overflow: {}", msg)
            }
            StreamManagerError::InvalidChunkSize(msg) => {
                write!(f, "Invalid chunk size: {}", msg)
            }
            StreamManagerError::StreamClosed(msg) => {
                write!(f, "Stream closed: {}", msg)
            }
            StreamManagerError::InvalidState(msg) => {
                write!(f, "Invalid state: {}", msg)
            }
        }
    }
}

impl std::error::Error for StreamManagerError {}

/// Result type for stream manager operations.
pub type StreamResult<T> = Result<T, StreamManagerError>;

/// Configuration for stream buffering and chunking.
#[derive(Debug, Clone)]
pub struct StreamManagerConfig {
    /// Maximum buffer size in chunks
    pub max_buffer_size: usize,
    /// Preferred chunk size for processing
    pub chunk_size: usize,
    /// Whether to enable backpressure when buffer is full
    pub enable_backpressure: bool,
}

impl Default for StreamManagerConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 100,
            chunk_size: 1024,
            enable_backpressure: true,
        }
    }
}

/// Stream chunk metadata.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Sequence number for ordering
    pub sequence: u64,
    /// Chunk data (in MVP, stored as envelope kind string; future: actual payload)
    pub data: Envelope,
    /// Whether this is the last chunk in the stream
    pub is_last: bool,
    /// Timestamp when chunk was created
    pub timestamp_ms: u64,
}

/// Stream buffer for managing chunks between stages.
#[derive(Debug)]
struct StreamBuffer {
    chunks: VecDeque<StreamChunk>,
    max_size: usize,
    next_sequence: u64,
    closed: bool,
}

impl StreamBuffer {
    fn new(max_size: usize) -> Self {
        Self {
            chunks: VecDeque::new(),
            max_size,
            next_sequence: 0,
            closed: false,
        }
    }

    fn push(&mut self, chunk: StreamChunk) -> StreamResult<()> {
        if self.closed {
            return Err(StreamManagerError::StreamClosed(
                "Cannot push to closed stream".to_string(),
            ));
        }

        if self.chunks.len() >= self.max_size {
            return Err(StreamManagerError::BufferOverflow(format!(
                "Buffer full: {} chunks",
                self.max_size
            )));
        }

        self.chunks.push_back(chunk);
        Ok(())
    }

    fn pop(&mut self) -> Option<StreamChunk> {
        self.chunks.pop_front()
    }

    fn peek(&self) -> Option<&StreamChunk> {
        self.chunks.front()
    }

    fn len(&self) -> usize {
        self.chunks.len()
    }

    fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    fn is_full(&self) -> bool {
        self.chunks.len() >= self.max_size
    }

    fn close(&mut self) {
        self.closed = true;
    }

    fn is_closed(&self) -> bool {
        self.closed
    }
}

/// Stream manager for handling streaming inference data.
pub struct StreamManager {
    config: StreamManagerConfig,
    input_buffer: StreamBuffer,
    output_buffer: StreamBuffer,
}

impl StreamManager {
    /// Creates a new stream manager instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(StreamManagerConfig::default())
    }

    /// Creates a new stream manager with custom configuration.
    pub fn with_config(config: StreamManagerConfig) -> Self {
        Self {
            input_buffer: StreamBuffer::new(config.max_buffer_size),
            output_buffer: StreamBuffer::new(config.max_buffer_size),
            config,
        }
    }

    /// Push a chunk into the input buffer.
    ///
    /// This is used when receiving streaming data from an external source.
    pub fn push_input_chunk(&mut self, envelope: Envelope, is_last: bool) -> StreamResult<()> {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let sequence = self.input_buffer.next_sequence;
        self.input_buffer.next_sequence += 1;

        let chunk = StreamChunk {
            sequence,
            data: envelope,
            is_last,
            timestamp_ms,
        };

        self.input_buffer.push(chunk)
    }

    /// Pop the next chunk from the input buffer.
    ///
    /// This is used when a stage needs to process the next chunk.
    pub fn pop_input_chunk(&mut self) -> Option<StreamChunk> {
        self.input_buffer.pop()
    }

    /// Peek at the next chunk in the input buffer without removing it.
    pub fn peek_input_chunk(&self) -> Option<&StreamChunk> {
        self.input_buffer.peek()
    }

    /// Push a processed chunk into the output buffer.
    ///
    /// This is used after a stage has processed a chunk and produced output.
    pub fn push_output_chunk(&mut self, envelope: Envelope, is_last: bool) -> StreamResult<()> {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let sequence = self.output_buffer.next_sequence;
        self.output_buffer.next_sequence += 1;

        let chunk = StreamChunk {
            sequence,
            data: envelope,
            is_last,
            timestamp_ms,
        };

        self.output_buffer.push(chunk)
    }

    /// Pop the next chunk from the output buffer.
    ///
    /// This is used when sending processed chunks to the next stage.
    pub fn pop_output_chunk(&mut self) -> Option<StreamChunk> {
        self.output_buffer.pop()
    }

    /// Check if the input buffer is full (for backpressure control).
    pub fn is_input_buffer_full(&self) -> bool {
        self.input_buffer.is_full()
    }

    /// Check if the output buffer is full (for backpressure control).
    pub fn is_output_buffer_full(&self) -> bool {
        self.output_buffer.is_full()
    }

    /// Get the current size of the input buffer.
    pub fn input_buffer_size(&self) -> usize {
        self.input_buffer.len()
    }

    /// Get the current size of the output buffer.
    pub fn output_buffer_size(&self) -> usize {
        self.output_buffer.len()
    }

    /// Check if input buffer is empty.
    pub fn is_input_empty(&self) -> bool {
        self.input_buffer.is_empty()
    }

    /// Check if output buffer is empty.
    pub fn is_output_empty(&self) -> bool {
        self.output_buffer.is_empty()
    }

    /// Close the input stream (no more chunks will be accepted).
    pub fn close_input(&mut self) {
        self.input_buffer.close();
    }

    /// Close the output stream (no more chunks will be accepted).
    pub fn close_output(&mut self) {
        self.output_buffer.close();
    }

    /// Check if input stream is closed.
    pub fn is_input_closed(&self) -> bool {
        self.input_buffer.is_closed()
    }

    /// Check if output stream is closed.
    pub fn is_output_closed(&self) -> bool {
        self.output_buffer.is_closed()
    }

    /// Split an envelope into multiple chunks based on the configured chunk size.
    ///
    /// For MVP, this is a simplified implementation. Future versions will
    /// handle actual binary data chunking for audio/video streams.
    pub fn chunk_envelope(&self, envelope: &Envelope) -> Vec<Envelope> {
        // MVP: Simple chunking - in future, this would split actual payload data
        // For now, we just return the envelope as a single chunk
        // This allows the interface to be ready for real chunking later
        vec![envelope.clone()]
    }

    /// Aggregate multiple chunks into a single envelope.
    ///
    /// For MVP, this takes the last chunk. Future versions will properly
    /// aggregate binary data.
    pub fn aggregate_chunks(&self, chunks: &[StreamChunk]) -> Option<Envelope> {
        if chunks.is_empty() {
            return None;
        }
        // MVP: Return the last chunk's envelope
        // Future: Properly aggregate all chunk data
        Some(chunks.last().unwrap().data.clone())
    }

    /// Clear both input and output buffers.
    pub fn clear_buffers(&mut self) {
        self.input_buffer = StreamBuffer::new(self.config.max_buffer_size);
        self.output_buffer = StreamBuffer::new(self.config.max_buffer_size);
    }

    /// Get the stream configuration.
    pub fn config(pub fn config(&self) -> &StreamConfigself) -> &StreamManagerConfig {
        &self.config
    }
}

impl Default for StreamManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::EnvelopeKind;

    fn audio_envelope(bytes: &[u8]) -> Envelope {
        Envelope::new(EnvelopeKind::Audio(bytes.to_vec()))
    }

    fn text_envelope(value: &str) -> Envelope {
        Envelope::new(EnvelopeKind::Text(value.to_string()))
    }

    #[test]
    fn test_stream_manager_creation() {
        let manager = StreamManager::new();
        assert_eq!(manager.input_buffer_size(), 0);
        assert_eq!(manager.output_buffer_size(), 0);
    }

    #[test]
    fn test_push_and_pop_input() {
        let mut manager = StreamManager::new();
        let envelope = audio_envelope(&[0, 1, 2]);

        manager.push_input_chunk(envelope.clone(), false).unwrap();
        assert_eq!(manager.input_buffer_size(), 1);

        let chunk = manager.pop_input_chunk().unwrap();
        assert!(matches!(chunk.data.kind, EnvelopeKind::Audio(_)));
        assert!(!chunk.is_last);
        assert_eq!(manager.input_buffer_size(), 0);
    }

    #[test]
    fn test_push_and_pop_output() {
        let mut manager = StreamManager::new();
        let envelope = text_envelope("Text");

        manager.push_output_chunk(envelope.clone(), true).unwrap();
        assert_eq!(manager.output_buffer_size(), 1);

        let chunk = manager.pop_output_chunk().unwrap();
        assert_eq!(chunk.data.kind, EnvelopeKind::Text("Text".to_string()));
        assert!(chunk.is_last);
    }

    #[test]
    fn test_buffer_overflow() {
        let mut config = StreamConfig::default();
        config.max_buffer_size = 2;
        let mut manager = StreamManager::with_config(config);

        let envelope = text_envelope("test");

        manager.push_input_chunk(envelope.clone(), false).unwrap();
        manager.push_input_chunk(envelope.clone(), false).unwrap();

        // Should fail on third push
        assert!(manager.push_input_chunk(envelope.clone(), false).is_err());
    }

    #[test]
    fn test_peek_input() {
        let mut manager = StreamManager::new();
        let envelope = text_envelope("test");

        manager.push_input_chunk(envelope.clone(), false).unwrap();

        let peeked = manager.peek_input_chunk().unwrap();
        assert_eq!(peeked.data.kind, EnvelopeKind::Text("test".to_string()));

        // Buffer should still have the chunk
        assert_eq!(manager.input_buffer_size(), 1);
    }

    #[test]
    fn test_close_stream() {
        let mut manager = StreamManager::new();
        let envelope = text_envelope("test");

        manager.close_input();
        assert!(manager.is_input_closed());
        assert!(manager.push_input_chunk(envelope, false).is_err());
    }

    #[test]
    fn test_chunk_envelope() {
        let manager = StreamManager::new();
        let envelope = audio_envelope(&[0, 1, 2]);

        let chunks = manager.chunk_envelope(&envelope);
        assert_eq!(chunks.len(), 1);
        assert!(matches!(chunks[0].kind, EnvelopeKind::Audio(_)));
    }

    #[test]
    fn test_aggregate_chunks() {
        let manager = StreamManager::new();
        let mut chunks = Vec::new();

        for i in 0..3 {
            chunks.push(StreamChunk {
                sequence: i,
                data: text_envelope(&format!("chunk_{}", i)),
                is_last: i == 2,
                timestamp_ms: 0,
            });
        }

        let aggregated = manager.aggregate_chunks(&chunks).unwrap();
        assert_eq!(aggregated.kind, EnvelopeKind::Text("chunk_2".to_string())); // MVP: takes last chunk
    }

    #[test]
    fn test_clear_buffers() {
        let mut manager = StreamManager::new();
        let envelope = text_envelope("test");

        manager.push_input_chunk(envelope.clone(), false).unwrap();
        manager.push_output_chunk(envelope.clone(), false).unwrap();

        manager.clear_buffers();
        assert_eq!(manager.input_buffer_size(), 0);
        assert_eq!(manager.output_buffer_size(), 0);
    }

    #[test]
    fn test_sequence_numbers() {
        let mut manager = StreamManager::new();
        let envelope = text_envelope("test");

        manager.push_input_chunk(envelope.clone(), false).unwrap();
        let chunk1 = manager.pop_input_chunk().unwrap();
        assert_eq!(chunk1.sequence, 0);

        manager.push_input_chunk(envelope.clone(), false).unwrap();
        let chunk2 = manager.pop_input_chunk().unwrap();
        assert_eq!(chunk2.sequence, 1);
    }

    #[test]
    fn test_is_last_flag() {
        let mut manager = StreamManager::new();
        let envelope = text_envelope("test");

        manager.push_input_chunk(envelope.clone(), false).unwrap();
        let chunk1 = manager.pop_input_chunk().unwrap();
        assert!(!chunk1.is_last);

        manager.push_input_chunk(envelope.clone(), true).unwrap();
        let chunk2 = manager.pop_input_chunk().unwrap();
        assert!(chunk2.is_last);
    }

    #[test]
    fn test_buffer_size_tracking() {
        let mut manager = StreamManager::new();
        assert!(manager.is_input_empty());
        assert!(manager.is_output_empty());

        let envelope = text_envelope("test");

        manager.push_input_chunk(envelope.clone(), false).unwrap();
        assert!(!manager.is_input_empty());
        assert_eq!(manager.input_buffer_size(), 1);

        manager.push_output_chunk(envelope, false).unwrap();
        assert!(!manager.is_output_empty());
        assert_eq!(manager.output_buffer_size(), 1);
    }
}
