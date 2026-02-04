// Xybrid SDK - Inference Result
// Wrapper for the output of model inference.

using System;
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
        /// Gets the text output (for ASR models), or null if not applicable.
        /// </summary>
        public string Text => _text;

        /// <summary>
        /// Gets the inference latency in milliseconds.
        /// </summary>
        public uint LatencyMs => _latencyMs;

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
                return $"InferenceResult(Success, LatencyMs={_latencyMs}, Text=\"{_text ?? "null"}\")";
            }
            else
            {
                return $"InferenceResult(Failed, Error=\"{_error}\")";
            }
        }
    }
}
