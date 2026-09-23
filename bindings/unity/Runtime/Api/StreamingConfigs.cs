// Xybrid SDK - Live ASR session presets
// Ready-made XybridStreamingConfig values for Model.Stream.

namespace Xybrid
{
    /// <summary>
    /// Presets for <see cref="Model.Stream"/>.
    /// </summary>
    /// <remarks>
    /// The generated <see cref="XybridBolt.XybridStreamingConfig"/> is a
    /// positional record with no defaults, so every call site would otherwise
    /// have to spell out all five fields — including the 16 kHz sample rate,
    /// which is the only value the ASR backends accept.
    /// </remarks>
    public static class StreamingConfigs
    {
        /// <summary>The only sample rate the ASR backends accept.</summary>
        public const uint RequiredSampleRate = 16000;

        /// <summary>
        /// Fixed time-window chunking at 16 kHz, using the model's own
        /// language. The starting point for dictation.
        /// </summary>
        public static XybridBolt.XybridStreamingConfig Default =>
            new XybridBolt.XybridStreamingConfig(
                RequiredSampleRate,
                new XybridBolt.XybridVadMode.Off(),
                0.5f,
                null,
                null);

        /// <summary>
        /// Chunks on speech boundaries using voice-activity detection, rather
        /// than on a fixed clock.
        /// </summary>
        /// <remarks>
        /// Better transcripts for natural speech — a window cut mid-word is
        /// what makes fixed chunking stutter — at the cost of loading a small
        /// VAD model alongside the ASR one.
        /// </remarks>
        /// <param name="modelDir">
        /// Directory holding a Silero VAD model, containing a <c>model.onnx</c>.
        /// Required: no VAD model ships with the SDK, and the engine falls back
        /// to fixed windows without one.
        /// </param>
        /// <param name="language">Language hint such as "en"; null uses the model default.</param>
        /// <param name="threshold">VAD sensitivity, 0.0-1.0. Lower catches quieter speech, and more background noise with it.</param>
        /// <exception cref="ArgumentNullException">Thrown if modelDir is null.</exception>
        public static XybridBolt.XybridStreamingConfig VoiceActivity(
            string modelDir,
            string language = null,
            float threshold = 0.5f)
        {
            if (modelDir == null)
            {
                throw new System.ArgumentNullException(nameof(modelDir));
            }
            return new XybridBolt.XybridStreamingConfig(
                RequiredSampleRate,
                new XybridBolt.XybridVadMode.Enabled(modelDir),
                threshold,
                language,
                null);
        }
    }
}
