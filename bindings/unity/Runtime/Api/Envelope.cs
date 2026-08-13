// Xybrid SDK - Envelope
// Wrapper for input data passed to model inference.

using System;
using System.Collections.Generic;
using System.Globalization;

namespace Xybrid
{
    /// <summary>
    /// Represents input data for model inference.
    /// Use the static factory methods to create instances.
    /// </summary>
    /// <remarks>
    /// The envelope is an immutable value and can be reused for multiple
    /// inference calls. TTS voice/speed, ASR sample-rate/channels, and message
    /// role are carried as envelope metadata.
    /// </remarks>
    public sealed class Envelope : IDisposable
    {
        /// <summary>The bolt wire value backing this envelope. For internal use.</summary>
        internal XybridBolt.XybridEnvelope Bolt { get; }

        /// <summary>
        /// Gets whether this envelope has been disposed. Retained for source
        /// compatibility; the envelope now holds no native resources.
        /// </summary>
        public bool IsDisposed { get; private set; }

        private Envelope(XybridBolt.XybridEnvelope bolt)
        {
            Bolt = bolt;
        }

        /// <summary>
        /// Creates an envelope containing text data for TTS or LLM inference.
        /// </summary>
        /// <param name="text">The text to process.</param>
        /// <exception cref="ArgumentNullException">Thrown if text is null.</exception>
        public static Envelope Text(string text)
        {
            if (text == null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            return new Envelope(new XybridBolt.XybridEnvelope(
                new XybridBolt.XybridEnvelopeKind.Text(text), System.Array.Empty<XybridBolt.XybridMetadataEntry>()));
        }

        /// <summary>
        /// Creates an envelope containing text data with voice and speed options for TTS.
        /// </summary>
        /// <param name="text">The text to synthesize.</param>
        /// <param name="voiceId">Voice ID (e.g., "af_bella"). Pass null to use the model's default voice.</param>
        /// <param name="speed">Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double).</param>
        /// <exception cref="ArgumentNullException">Thrown if text is null.</exception>
        public static Envelope Text(string text, string voiceId, double speed = 1.0)
        {
            if (text == null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            var metadata = new List<XybridBolt.XybridMetadataEntry>();
            if (voiceId != null)
            {
                metadata.Add(new XybridBolt.XybridMetadataEntry("voice_id", voiceId));
            }
            metadata.Add(new XybridBolt.XybridMetadataEntry(
                "speed", speed.ToString(CultureInfo.InvariantCulture)));

            return new Envelope(new XybridBolt.XybridEnvelope(
                new XybridBolt.XybridEnvelopeKind.Text(text), metadata.ToArray()));
        }

        /// <summary>
        /// Creates an envelope containing text data with a message role.
        /// </summary>
        /// <param name="text">The text to process.</param>
        /// <param name="role">The message role for conversation context.</param>
        /// <exception cref="ArgumentNullException">Thrown if text is null.</exception>
        public static Envelope Text(string text, MessageRole role)
        {
            if (text == null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            var metadata = new[]
            {
                new XybridBolt.XybridMetadataEntry("xybrid.role", RoleString(role)),
            };
            return new Envelope(new XybridBolt.XybridEnvelope(
                new XybridBolt.XybridEnvelopeKind.Text(text), metadata));
        }

        /// <summary>
        /// Creates an envelope containing audio data for ASR inference.
        /// </summary>
        /// <param name="audioBytes">Raw audio bytes (typically PCM or WAV format).</param>
        /// <param name="sampleRate">Sample rate in Hz (e.g., 16000 for 16kHz).</param>
        /// <param name="channels">Number of audio channels (1 = mono, 2 = stereo).</param>
        /// <exception cref="ArgumentNullException">Thrown if audioBytes is null.</exception>
        public static Envelope Audio(byte[] audioBytes, uint sampleRate = 16000, uint channels = 1)
        {
            if (audioBytes == null)
            {
                throw new ArgumentNullException(nameof(audioBytes));
            }

            var metadata = new[]
            {
                new XybridBolt.XybridMetadataEntry(
                    "sample_rate", sampleRate.ToString(CultureInfo.InvariantCulture)),
                new XybridBolt.XybridMetadataEntry(
                    "channels", channels.ToString(CultureInfo.InvariantCulture)),
            };
            return new Envelope(new XybridBolt.XybridEnvelope(
                new XybridBolt.XybridEnvelopeKind.Audio(audioBytes), metadata));
        }

        /// <summary>
        /// Creates an envelope containing encoded image data for vision-language models.
        /// </summary>
        /// <param name="bytes">Encoded PNG, JPEG, or WebP bytes.</param>
        /// <param name="format">Image format: png, jpeg, jpg, or webp.</param>
        /// <exception cref="ArgumentNullException">Thrown if bytes or format is null.</exception>
        /// <exception cref="ArgumentException">Thrown if format is unsupported.</exception>
        public static Envelope Image(byte[] bytes, string format)
        {
            if (bytes == null)
            {
                throw new ArgumentNullException(nameof(bytes));
            }

            string normalizedFormat = NormalizeImageFormat(format);
            return new Envelope(new XybridBolt.XybridEnvelope(
                new XybridBolt.XybridEnvelopeKind.Image(bytes, normalizedFormat),
                System.Array.Empty<XybridBolt.XybridMetadataEntry>()));
        }

        /// <summary>
        /// Creates a multi-part user message with text and image attachments.
        /// </summary>
        /// <param name="text">The user prompt text.</param>
        /// <param name="images">Image envelopes created by <see cref="Image(byte[], string)"/>.</param>
        /// <exception cref="ArgumentNullException">Thrown if text is null.</exception>
        /// <exception cref="ArgumentException">Thrown if any attachment is null or not an image envelope.</exception>
        public static Envelope UserMessage(string text, IList<Envelope> images = null)
        {
            if (text == null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            var parts = new List<XybridBolt.XybridEnvelope>
            {
                new XybridBolt.XybridEnvelope(
                    new XybridBolt.XybridEnvelopeKind.Text(text), System.Array.Empty<XybridBolt.XybridMetadataEntry>()),
            };
            if (images != null)
            {
                foreach (Envelope image in images)
                {
                    if (image == null)
                    {
                        throw new ArgumentException("Image attachment cannot be null.", nameof(images));
                    }
                    if (!(image.Bolt.Kind is XybridBolt.XybridEnvelopeKind.Image))
                    {
                        throw new ArgumentException(
                            "Envelope.UserMessage accepts only image envelopes.", nameof(images));
                    }
                    parts.Add(image.Bolt);
                }
            }

            var metadata = new[]
            {
                new XybridBolt.XybridMetadataEntry("xybrid.role", RoleString(MessageRole.User)),
            };
            return new Envelope(new XybridBolt.XybridEnvelope(
                new XybridBolt.XybridEnvelopeKind.MultiPart(parts.ToArray()), metadata));
        }

        /// <summary>
        /// Creates the continuation envelope for the turn after the model asked
        /// for tools.
        /// </summary>
        /// <remarks>
        /// One Run is one model turn, so the tool loop lives in your code: run a
        /// request carrying <see cref="GenerationConfig.AddTool"/> definitions,
        /// execute every <see cref="InferenceResult.ToolCalls"/> entry, then run
        /// this envelope to feed the outcomes back. Run the continuation with
        /// the same tools as the original turn so the executor rebuilds an
        /// identical chat prefix.
        ///
        /// Only the immediately prior assistant turn is replayed, and
        /// continuation runs on the non-streaming text path only.
        /// </remarks>
        /// <param name="userText">The original user message of the turn being continued.</param>
        /// <param name="priorAssistantText">
        /// That turn's raw output text, tool-call block included — i.e.
        /// <see cref="InferenceResult.Text"/> verbatim.
        /// </param>
        /// <param name="results">Tool outcomes, in call order.</param>
        /// <exception cref="ArgumentNullException">Thrown if any argument is null.</exception>
        /// <exception cref="XybridException">Thrown if a result's content is not valid JSON.</exception>
        public static Envelope ToolResults(
            string userText,
            string priorAssistantText,
            IList<XybridBolt.XybridToolResult> results)
        {
            if (userText == null)
            {
                throw new ArgumentNullException(nameof(userText));
            }
            if (priorAssistantText == null)
            {
                throw new ArgumentNullException(nameof(priorAssistantText));
            }
            if (results == null)
            {
                throw new ArgumentNullException(nameof(results));
            }

            var array = new XybridBolt.XybridToolResult[results.Count];
            results.CopyTo(array, 0);
            return new Envelope(
                XybridBolt.XybridBolt.ToolResultsEnvelope(userText, priorAssistantText, array));
        }

        private static string RoleString(MessageRole role)
        {
            switch (role)
            {
                case MessageRole.System:
                    return "system";
                case MessageRole.User:
                    return "user";
                case MessageRole.Assistant:
                    return "assistant";
                default:
                    throw new ArgumentOutOfRangeException(nameof(role));
            }
        }

        private static string NormalizeImageFormat(string format)
        {
            if (format == null)
            {
                throw new ArgumentNullException(nameof(format));
            }

            string normalized = format.Trim().ToLowerInvariant();
            switch (normalized)
            {
                case "jpg":
                    return "jpeg";
                case "jpeg":
                case "png":
                case "webp":
                    return normalized;
                default:
                    throw new ArgumentException(
                        "Unsupported image format. Supported formats: png, jpeg, jpg, webp.",
                        nameof(format));
            }
        }

        /// <summary>
        /// No-op: the envelope holds no native resources. Retained so existing
        /// <c>using</c> call sites keep compiling.
        /// </summary>
        public void Dispose()
        {
            IsDisposed = true;
        }
    }
}
