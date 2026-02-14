// Xybrid SDK - Inference Result
// Wrapper for the output of model inference.

using System;
using System.Runtime.InteropServices;
using Xybrid.Native;

namespace Xybrid
{
    /// <summary>
    /// Represents the result of model inference.
    /// </summary>
    /// <remarks>
    /// This class wraps a native result handle and must be disposed when no longer needed.
    /// Access the result properties before disposing.
    /// </remarks>
    public sealed class InferenceResult : IDisposable
    {
        private unsafe XybridResultHandle* _handle;
        private bool _disposed;

        // Cached values (extracted before potential disposal)
        private readonly bool _success;
        private readonly string _error;
        private readonly string _text;
        private readonly uint _latencyMs;
        private readonly OutputType _outputType;
        private readonly byte[] _audioBytes;
        private readonly float[] _embedding;

        /// <summary>
        /// Gets whether this result has been disposed.
        /// </summary>
        public bool IsDisposed => _disposed;

        /// <summary>
        /// Gets whether the inference was successful.
        /// </summary>
        public bool Success => _success;

        /// <summary>
        /// Gets the error message if inference failed, or null if successful.
        /// </summary>
        public string Error => _error;

        /// <summary>
        /// Gets the text output (for ASR or LLM models), or null if not applicable.
        /// </summary>
        public string Text => _text;

        /// <summary>
        /// Gets the inference latency in milliseconds.
        /// </summary>
        public uint LatencyMs => _latencyMs;

        /// <summary>
        /// Gets the type of output produced by inference.
        /// </summary>
        public OutputType OutputType => _outputType;

        /// <summary>
        /// Gets the raw audio bytes (for TTS models), or null if not applicable.
        /// Audio format is raw PCM 16-bit signed little-endian, typically 24kHz mono.
        /// </summary>
        public byte[] AudioBytes => _audioBytes;

        /// <summary>
        /// Gets the embedding vector (for embedding models), or null if not applicable.
        /// </summary>
        public float[] Embedding => _embedding;

        /// <summary>
        /// Gets whether this result contains audio data.
        /// </summary>
        public bool HasAudio => _audioBytes != null && _audioBytes.Length > 0;

        /// <summary>
        /// Gets whether this result contains an embedding.
        /// </summary>
        public bool HasEmbedding => _embedding != null && _embedding.Length > 0;

        internal unsafe InferenceResult(XybridResultHandle* handle)
        {
            _handle = handle;

            // Cache all values immediately so they survive disposal
            _success = NativeMethods.xybrid_result_success(handle) != 0;
            _latencyMs = NativeMethods.xybrid_result_latency_ms(handle);

            if (_success)
            {
                byte* textPtr = NativeMethods.xybrid_result_text(handle);
                _text = NativeHelpers.FromUtf8Ptr(textPtr);
                _error = null;
            }
            else
            {
                byte* errorPtr = NativeMethods.xybrid_result_error(handle);
                _error = NativeHelpers.FromUtf8Ptr(errorPtr);
                _text = null;
            }

            // Cache output type
            byte* outputTypePtr = NativeMethods.xybrid_result_output_type(handle);
            string outputTypeStr = NativeHelpers.FromUtf8Ptr(outputTypePtr);
            _outputType = ParseOutputType(outputTypeStr);

            // Cache audio bytes
            nuint audioLen = NativeMethods.xybrid_result_audio_len(handle);
            if (audioLen > 0)
            {
                byte* audioPtr = NativeMethods.xybrid_result_audio_data(handle);
                if (audioPtr != null)
                {
                    _audioBytes = new byte[(int)audioLen];
                    Marshal.Copy((IntPtr)audioPtr, _audioBytes, 0, (int)audioLen);
                }
            }

            // Cache embedding
            nuint embLen = NativeMethods.xybrid_result_embedding_len(handle);
            if (embLen > 0)
            {
                float* embPtr = NativeMethods.xybrid_result_embedding_data(handle);
                if (embPtr != null)
                {
                    _embedding = new float[(int)embLen];
                    Marshal.Copy((IntPtr)embPtr, _embedding, 0, (int)embLen);
                }
            }
        }

        /// <summary>
        /// Throws an InferenceException if the result indicates failure.
        /// </summary>
        /// <exception cref="InferenceException">Thrown if Success is false.</exception>
        public void ThrowIfFailed()
        {
            if (!_success)
            {
                throw new InferenceException(_error ?? "Unknown inference error");
            }
        }

        /// <summary>
        /// Releases the native resources used by this result.
        /// </summary>
        public unsafe void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != null)
                {
                    NativeMethods.xybrid_result_free(_handle);
                    _handle = null;
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer to ensure native resources are released.
        /// </summary>
        ~InferenceResult()
        {
            Dispose();
        }

        /// <summary>
        /// Returns a string representation of the result.
        /// </summary>
        public override string ToString()
        {
            if (_success)
            {
                return $"InferenceResult(Success, OutputType={_outputType}, LatencyMs={_latencyMs}, " +
                       $"Text=\"{_text ?? "null"}\", AudioBytes={_audioBytes?.Length ?? 0})";
            }
            else
            {
                return $"InferenceResult(Failed, Error=\"{_error}\")";
            }
        }

        private static OutputType ParseOutputType(string type)
        {
            switch (type)
            {
                case "text": return OutputType.Text;
                case "audio": return OutputType.Audio;
                case "embedding": return OutputType.Embedding;
                default: return OutputType.Unknown;
            }
        }
    }
}
