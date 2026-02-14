// Xybrid SDK - OutputType
// Defines the type of output produced by model inference.

namespace Xybrid
{
    /// <summary>
    /// The type of output produced by model inference.
    /// </summary>
    public enum OutputType
    {
        /// <summary>
        /// Text output (ASR transcription, LLM response).
        /// </summary>
        Text,

        /// <summary>
        /// Audio output (TTS waveform). Raw PCM 16-bit signed little-endian.
        /// </summary>
        Audio,

        /// <summary>
        /// Embedding vector output.
        /// </summary>
        Embedding,

        /// <summary>
        /// Unknown or unsupported output type.
        /// </summary>
        Unknown
    }
}
