// Xybrid SDK - Model
// Wrapper for a loaded model ready for inference.

using System;
using Xybrid.Native;

namespace Xybrid
{
    /// <summary>
    /// Represents a loaded model ready for inference.
    /// </summary>
    /// <remarks>
    /// Models are created using <see cref="ModelLoader.Load"/>.
    /// This class must be disposed when no longer needed to release native resources.
    /// </remarks>
    public sealed class Model : IDisposable
    {
        private unsafe XybridModelHandle* _handle;
        private bool _disposed;
        private readonly string _modelId;

        /// <summary>
        /// Gets whether this model has been disposed.
        /// </summary>
        public bool IsDisposed => _disposed;

        /// <summary>
        /// Gets the model ID.
        /// </summary>
        public string ModelId => _modelId;

        internal unsafe Model(XybridModelHandle* handle)
        {
            _handle = handle;

            // Cache model ID
            byte* idPtr = NativeMethods.xybrid_model_id(handle);
            if (idPtr != null)
            {
                _modelId = NativeHelpers.FromUtf8Ptr(idPtr);
                NativeMethods.xybrid_free_string(idPtr);
            }
            else
            {
                _modelId = "unknown";
            }
        }

        /// <summary>
        /// Runs inference on this model with the provided input envelope.
        /// </summary>
        /// <param name="envelope">The input data for inference.</param>
        /// <returns>The inference result.</returns>
        /// <exception cref="ArgumentNullException">Thrown if envelope is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this model or the envelope is disposed.</exception>
        /// <exception cref="XybridException">Thrown if inference fails to start.</exception>
        /// <remarks>
        /// The envelope is not consumed - it can be reused for multiple inferences.
        /// Remember to dispose the returned <see cref="InferenceResult"/> when done.
        /// </remarks>
        public unsafe InferenceResult Run(Envelope envelope)
        {
            ThrowIfDisposed();

            if (envelope == null)
            {
                throw new ArgumentNullException(nameof(envelope));
            }

            if (envelope.IsDisposed)
            {
                throw new ObjectDisposedException(nameof(envelope));
            }

            XybridResultHandle* resultHandle = NativeMethods.xybrid_model_run(_handle, envelope.Handle);
            if (resultHandle == null)
            {
                NativeHelpers.ThrowLastError("Failed to run inference");
            }

            return new InferenceResult(resultHandle);
        }

        /// <summary>
        /// Runs inference and returns the text result, throwing on failure.
        /// </summary>
        /// <param name="text">The input text for TTS or LLM inference.</param>
        /// <returns>The text output from the model.</returns>
        /// <exception cref="InferenceException">Thrown if inference fails.</exception>
        /// <remarks>
        /// This is a convenience method that creates an envelope, runs inference,
        /// and extracts the text result. For more control, use <see cref="Run"/>.
        /// </remarks>
        public string RunText(string text)
        {
            using (var envelope = Envelope.Text(text))
            using (var result = Run(envelope))
            {
                result.ThrowIfFailed();
                return result.Text;
            }
        }

        /// <summary>
        /// Runs inference on audio data and returns the transcription.
        /// </summary>
        /// <param name="audioBytes">Raw audio bytes.</param>
        /// <param name="sampleRate">Sample rate in Hz.</param>
        /// <param name="channels">Number of audio channels.</param>
        /// <returns>The transcribed text.</returns>
        /// <exception cref="InferenceException">Thrown if inference fails.</exception>
        public string RunAudio(byte[] audioBytes, uint sampleRate = 16000, uint channels = 1)
        {
            using (var envelope = Envelope.Audio(audioBytes, sampleRate, channels))
            using (var result = Run(envelope))
            {
                result.ThrowIfFailed();
                return result.Text;
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(Model));
            }
        }

        /// <summary>
        /// Releases the native resources used by this model.
        /// </summary>
        public unsafe void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != null)
                {
                    NativeMethods.xybrid_model_free(_handle);
                    _handle = null;
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer to ensure native resources are released.
        /// </summary>
        ~Model()
        {
            Dispose();
        }

        /// <summary>
        /// Returns a string representation of the model.
        /// </summary>
        public override string ToString()
        {
            return $"Model({ModelId})";
        }
    }
}
