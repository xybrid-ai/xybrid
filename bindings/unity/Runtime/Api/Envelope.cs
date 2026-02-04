// Xybrid SDK - Envelope
// Wrapper for input data passed to model inference.

using System;
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
        private unsafe XybridEnvelopeHandle* _handle;
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

        private unsafe Envelope(XybridEnvelopeHandle* handle)
        {
            _handle = handle;
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

                return new Envelope(handle);
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

                return new Envelope(handle);
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
