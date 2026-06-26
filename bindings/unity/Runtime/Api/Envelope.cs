// Xybrid SDK - Envelope
// Wrapper for input data passed to model inference.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Xybrid.Native;

namespace Xybrid
{
    /// <summary>
    /// Represents input data for model inference.
    /// Use the static factory methods to create instances.
    /// </summary>
    /// <remarks>
    /// This class wraps a native envelope handle and must be disposed when no longer needed.
    /// The envelope can be reused for multiple inference calls.
    /// </remarks>
    public sealed class Envelope : IDisposable
    {
        private enum PayloadKind
        {
            Text,
            Audio,
            Image,
            UserMessage,
        }

        private unsafe XybridEnvelopeHandle* _handle;
        private readonly PayloadKind _kind;
        private bool _disposed;

        /// <summary>
        /// Gets whether this envelope has been disposed.
        /// </summary>
        public bool IsDisposed => _disposed;

        /// <summary>
        /// Gets the internal native handle. For internal use only.
        /// </summary>
        internal unsafe XybridEnvelopeHandle* Handle
        {
            get
            {
                ThrowIfDisposed();
                return _handle;
            }
        }

        private unsafe Envelope(XybridEnvelopeHandle* handle, PayloadKind kind)
        {
            _handle = handle;
            _kind = kind;
        }

        /// <summary>
        /// Creates an envelope containing text data for TTS or LLM inference.
        /// </summary>
        /// <param name="text">The text to process.</param>
        /// <returns>A new Envelope containing the text.</returns>
        /// <exception cref="ArgumentNullException">Thrown if text is null.</exception>
        /// <exception cref="XybridException">Thrown if envelope creation fails.</exception>
        public static unsafe Envelope Text(string text)
        {
            if (text == null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            byte[] textBytes = NativeHelpers.ToUtf8Bytes(text);

            fixed (byte* textPtr = textBytes)
            {
                XybridEnvelopeHandle* handle = NativeMethods.xybrid_envelope_text(textPtr);
                if (handle == null)
                {
                    NativeHelpers.ThrowLastError("Failed to create text envelope");
                }

                return new Envelope(handle, PayloadKind.Text);
            }
        }

        /// <summary>
        /// Creates an envelope containing text data with voice and speed options for TTS.
        /// </summary>
        /// <param name="text">The text to synthesize.</param>
        /// <param name="voiceId">Voice ID (e.g., "af_bella"). Pass null to use the model's default voice.</param>
        /// <param name="speed">Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double).</param>
        /// <returns>A new Envelope containing the text with voice options.</returns>
        /// <exception cref="ArgumentNullException">Thrown if text is null.</exception>
        /// <exception cref="XybridException">Thrown if envelope creation fails.</exception>
        public static unsafe Envelope Text(string text, string voiceId, double speed = 1.0)
        {
            if (text == null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            byte[] textBytes = NativeHelpers.ToUtf8Bytes(text);
            byte[] voiceBytes = voiceId != null ? NativeHelpers.ToUtf8Bytes(voiceId) : null;

            fixed (byte* textPtr = textBytes)
            fixed (byte* voicePtr = voiceBytes)
            {
                XybridEnvelopeHandle* handle = NativeMethods.xybrid_envelope_text_with_voice(
                    textPtr, voicePtr, speed);
                if (handle == null)
                {
                    NativeHelpers.ThrowLastError("Failed to create text envelope with voice");
                }

                return new Envelope(handle, PayloadKind.Text);
            }
        }

        /// <summary>
        /// Creates an envelope containing text data with a message role.
        /// </summary>
        /// <param name="text">The text to process.</param>
        /// <param name="role">The message role for conversation context.</param>
        /// <returns>A new Envelope containing the text with the specified role.</returns>
        /// <exception cref="ArgumentNullException">Thrown if text is null.</exception>
        /// <exception cref="XybridException">Thrown if envelope creation fails.</exception>
        public static unsafe Envelope Text(string text, MessageRole role)
        {
            if (text == null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            byte[] textBytes = NativeHelpers.ToUtf8Bytes(text);

            fixed (byte* textPtr = textBytes)
            {
                XybridEnvelopeHandle* handle = NativeMethods.xybrid_envelope_text_with_role(textPtr, (int)role);
                if (handle == null)
                {
                    NativeHelpers.ThrowLastError("Failed to create text envelope with role");
                }

                return new Envelope(handle, PayloadKind.Text);
            }
        }

        /// <summary>
        /// Creates an envelope containing audio data for ASR inference.
        /// </summary>
        /// <param name="audioBytes">Raw audio bytes (typically PCM or WAV format).</param>
        /// <param name="sampleRate">Sample rate in Hz (e.g., 16000 for 16kHz).</param>
        /// <param name="channels">Number of audio channels (1 = mono, 2 = stereo).</param>
        /// <returns>A new Envelope containing the audio data.</returns>
        /// <exception cref="ArgumentNullException">Thrown if audioBytes is null.</exception>
        /// <exception cref="XybridException">Thrown if envelope creation fails.</exception>
        public static unsafe Envelope Audio(byte[] audioBytes, uint sampleRate = 16000, uint channels = 1)
        {
            if (audioBytes == null)
            {
                throw new ArgumentNullException(nameof(audioBytes));
            }

            fixed (byte* bytesPtr = audioBytes)
            {
                XybridEnvelopeHandle* handle = NativeMethods.xybrid_envelope_audio(
                    bytesPtr,
                    (nuint)audioBytes.Length,
                    sampleRate,
                    channels
                );

                if (handle == null)
                {
                    NativeHelpers.ThrowLastError("Failed to create audio envelope");
                }

                return new Envelope(handle, PayloadKind.Audio);
            }
        }

        /// <summary>
        /// Creates an envelope containing encoded image data for vision-language models.
        /// </summary>
        /// <param name="bytes">Encoded PNG, JPEG, or WebP bytes.</param>
        /// <param name="format">Image format: png, jpeg, jpg, or webp.</param>
        /// <returns>A new Envelope containing the encoded image.</returns>
        /// <exception cref="ArgumentNullException">Thrown if bytes or format is null.</exception>
        /// <exception cref="ArgumentException">Thrown if format is unsupported.</exception>
        /// <exception cref="XybridException">Thrown if native envelope creation fails.</exception>
        public static unsafe Envelope Image(byte[] bytes, string format)
        {
            if (bytes == null)
            {
                throw new ArgumentNullException(nameof(bytes));
            }

            string normalizedFormat = NormalizeImageFormat(format);
            byte[] formatBytes = NativeHelpers.ToUtf8Bytes(normalizedFormat);

            fixed (byte* bytesPtr = bytes)
            fixed (byte* formatPtr = formatBytes)
            {
                XybridEnvelopeHandle* handle = NativeMethods.xybrid_envelope_image(
                    bytesPtr,
                    (nuint)bytes.Length,
                    formatPtr
                );

                if (handle == null)
                {
                    NativeHelpers.ThrowLastError("Failed to create image envelope");
                }

                return new Envelope(handle, PayloadKind.Image);
            }
        }

        /// <summary>
        /// Creates a multi-part user message with text and image attachments.
        /// </summary>
        /// <param name="text">The user prompt text.</param>
        /// <param name="images">Image envelopes created by <see cref="Image(byte[], string)"/>.</param>
        /// <returns>A new Envelope containing the user message.</returns>
        /// <exception cref="ArgumentNullException">Thrown if text is null.</exception>
        /// <exception cref="ArgumentException">Thrown if any attachment is null or not an image envelope.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if an image attachment has been disposed.</exception>
        /// <exception cref="XybridException">Thrown if native envelope creation fails.</exception>
        public static unsafe Envelope UserMessage(string text, IList<Envelope> images = null)
        {
            if (text == null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            int imageCount = images?.Count ?? 0;
            XybridEnvelopeHandle** imageHandles = null;
            if (imageCount > 0)
            {
                // Note: stackalloc must be assigned directly to a pointer-typed local;
                // placing it inside a ternary makes the compiler infer Span<T>, which
                // rejects pointer type arguments (CS0306/CS0029).
                XybridEnvelopeHandle** buffer = stackalloc XybridEnvelopeHandle*[imageCount];
                for (int i = 0; i < imageCount; i++)
                {
                    Envelope image = images[i];
                    if (image == null)
                    {
                        throw new ArgumentException("Image attachment cannot be null.", nameof(images));
                    }
                    if (image._kind != PayloadKind.Image)
                    {
                        throw new ArgumentException("Envelope.UserMessage accepts only image envelopes.", nameof(images));
                    }
                    buffer[i] = image.Handle;
                }
                imageHandles = buffer;
            }

            byte[] textBytes = NativeHelpers.ToUtf8Bytes(text);

            fixed (byte* textPtr = textBytes)
            {
                XybridEnvelopeHandle* handle = NativeMethods.xybrid_envelope_user_message(
                    textPtr,
                    imageHandles,
                    (nuint)imageCount
                );

                if (handle == null)
                {
                    NativeHelpers.ThrowLastError("Failed to create user message envelope");
                }

                return new Envelope(handle, PayloadKind.UserMessage);
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
                        nameof(format)
                    );
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(Envelope));
            }
        }

        /// <summary>
        /// Releases the native resources used by this envelope.
        /// </summary>
        public unsafe void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != null)
                {
                    NativeMethods.xybrid_envelope_free(_handle);
                    _handle = null;
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer to ensure native resources are released.
        /// </summary>
        ~Envelope()
        {
            Dispose();
        }
    }
}
