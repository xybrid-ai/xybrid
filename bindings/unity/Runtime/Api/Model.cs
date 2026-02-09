// Xybrid SDK - Model
// Wrapper for a loaded model ready for inference.

using System;
using System.Runtime.InteropServices;
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

        /// <summary>
        /// Runs inference with conversation context.
        /// </summary>
        /// <param name="envelope">The input data for inference.</param>
        /// <param name="context">The conversation context with history.</param>
        /// <returns>The inference result.</returns>
        /// <remarks>
        /// The context provides conversation history which is formatted into the prompt
        /// using the model's chat template. The context is NOT automatically updated
        /// with the result - call context.Push() to add the response.
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown if envelope or context is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this model, envelope, or context is disposed.</exception>
        /// <exception cref="XybridException">Thrown if inference fails to start.</exception>
        public unsafe InferenceResult Run(Envelope envelope, ConversationContext context)
        {
            ThrowIfDisposed();

            if (envelope == null)
            {
                throw new ArgumentNullException(nameof(envelope));
            }

            if (context == null)
            {
                throw new ArgumentNullException(nameof(context));
            }

            if (envelope.IsDisposed)
            {
                throw new ObjectDisposedException(nameof(envelope));
            }

            if (context.IsDisposed)
            {
                throw new ObjectDisposedException(nameof(context));
            }

            XybridResultHandle* resultHandle = NativeMethods.xybrid_model_run_with_context(
                _handle, envelope.Handle, context.Handle);
            if (resultHandle == null)
            {
                NativeHelpers.ThrowLastError("Failed to run inference with context");
            }

            return new InferenceResult(resultHandle);
        }

        /// <summary>
        /// Runs inference with conversation context and returns the text result.
        /// </summary>
        /// <param name="text">The input text for LLM inference.</param>
        /// <param name="context">The conversation context with history.</param>
        /// <returns>The text output from the model.</returns>
        /// <remarks>
        /// This is a convenience method. The input message is NOT automatically pushed
        /// to the context - you must call context.Push() for both input and output.
        /// </remarks>
        /// <exception cref="InferenceException">Thrown if inference fails.</exception>
        public string RunText(string text, ConversationContext context)
        {
            using (var envelope = Envelope.Text(text))
            using (var result = Run(envelope, context))
            {
                result.ThrowIfFailed();
                return result.Text;
            }
        }

        /// <summary>
        /// Gets whether this model supports true token-by-token streaming.
        /// </summary>
        /// <remarks>
        /// Returns true for LLM models (GGUF format with LLM features enabled).
        /// Non-LLM models can still use <see cref="RunStreaming"/> but will receive
        /// a single callback with the complete result instead of token-by-token output.
        /// </remarks>
        public unsafe bool SupportsTokenStreaming
        {
            get
            {
                ThrowIfDisposed();
                return NativeMethods.xybrid_model_supports_token_streaming(_handle) != 0;
            }
        }

        /// <summary>
        /// Runs streaming inference, invoking the callback for each generated token.
        /// </summary>
        /// <param name="envelope">The input data for inference.</param>
        /// <param name="onToken">Callback invoked for each token. Called on the calling thread.</param>
        /// <returns>The final inference result after all tokens are emitted.</returns>
        /// <remarks>
        /// This method blocks until inference is complete. For LLM models, the callback
        /// is invoked for each generated token. For non-LLM models, a single callback
        /// is invoked with the complete result.
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown if envelope or onToken is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this model or the envelope is disposed.</exception>
        /// <exception cref="XybridException">Thrown if inference fails to start.</exception>
        public unsafe InferenceResult RunStreaming(Envelope envelope, Action<StreamToken> onToken)
        {
            ThrowIfDisposed();

            if (envelope == null)
                throw new ArgumentNullException(nameof(envelope));
            if (onToken == null)
                throw new ArgumentNullException(nameof(onToken));
            if (envelope.IsDisposed)
                throw new ObjectDisposedException(nameof(envelope));

            // Pin the managed callback so it survives the native call
            var gcHandle = GCHandle.Alloc(onToken);
            try
            {
                XybridResultHandle* resultHandle = NativeMethods.xybrid_model_run_streaming(
                    _handle,
                    envelope.Handle,
                    StreamCallbackTrampoline,
                    (void*)GCHandle.ToIntPtr(gcHandle));

                if (resultHandle == null)
                {
                    NativeHelpers.ThrowLastError("Failed to run streaming inference");
                }

                return new InferenceResult(resultHandle);
            }
            finally
            {
                gcHandle.Free();
            }
        }

        /// <summary>
        /// Runs streaming inference with conversation context.
        /// </summary>
        /// <param name="envelope">The input data for inference.</param>
        /// <param name="context">The conversation context with history.</param>
        /// <param name="onToken">Callback invoked for each token.</param>
        /// <returns>The final inference result after all tokens are emitted.</returns>
        /// <exception cref="ArgumentNullException">Thrown if any argument is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this model, envelope, or context is disposed.</exception>
        /// <exception cref="XybridException">Thrown if inference fails to start.</exception>
        public unsafe InferenceResult RunStreaming(Envelope envelope, ConversationContext context, Action<StreamToken> onToken)
        {
            ThrowIfDisposed();

            if (envelope == null)
                throw new ArgumentNullException(nameof(envelope));
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (onToken == null)
                throw new ArgumentNullException(nameof(onToken));
            if (envelope.IsDisposed)
                throw new ObjectDisposedException(nameof(envelope));
            if (context.IsDisposed)
                throw new ObjectDisposedException(nameof(context));

            var gcHandle = GCHandle.Alloc(onToken);
            try
            {
                XybridResultHandle* resultHandle = NativeMethods.xybrid_model_run_streaming_with_context(
                    _handle,
                    envelope.Handle,
                    context.Handle,
                    StreamCallbackTrampolineWithContext,
                    (void*)GCHandle.ToIntPtr(gcHandle));

                if (resultHandle == null)
                {
                    NativeHelpers.ThrowLastError("Failed to run streaming inference with context");
                }

                return new InferenceResult(resultHandle);
            }
            finally
            {
                gcHandle.Free();
            }
        }

        /// <summary>
        /// Convenience method: stream text inference with a callback.
        /// </summary>
        /// <param name="text">The input text.</param>
        /// <param name="onToken">Callback invoked for each token.</param>
        /// <returns>The full text result after streaming completes.</returns>
        public string RunStreamingText(string text, Action<StreamToken> onToken)
        {
            using (var envelope = Envelope.Text(text))
            using (var result = RunStreaming(envelope, onToken))
            {
                result.ThrowIfFailed();
                return result.Text;
            }
        }

        /// <summary>
        /// Convenience method: stream text inference with conversation context.
        /// </summary>
        /// <param name="text">The input text.</param>
        /// <param name="context">The conversation context.</param>
        /// <param name="onToken">Callback invoked for each token.</param>
        /// <returns>The full text result after streaming completes.</returns>
        public string RunStreamingText(string text, ConversationContext context, Action<StreamToken> onToken)
        {
            using (var envelope = Envelope.Text(text))
            using (var result = RunStreaming(envelope, context, onToken))
            {
                result.ThrowIfFailed();
                return result.Text;
            }
        }

        // ================================================================
        // Static callback trampolines for P/Invoke
        // ================================================================

        /// <summary>
        /// Static trampoline that bridges the unmanaged callback to the managed Action.
        /// Must be static for IL2CPP compatibility.
        /// </summary>
#if ENABLE_IL2CPP
        [AOT.MonoPInvokeCallback(typeof(NativeMethods.xybrid_model_run_streaming_callback_delegate))]
#endif
        private static unsafe void StreamCallbackTrampoline(
            byte* token, long tokenId, uint index,
            byte* cumulativeText, byte* finishReason, void* userData)
        {
            try
            {
                var gcHandle = GCHandle.FromIntPtr((IntPtr)userData);
                var callback = (Action<StreamToken>)gcHandle.Target;

                var streamToken = new StreamToken(
                    token: NativeHelpers.FromUtf8Ptr(token),
                    tokenId: tokenId >= 0 ? tokenId : (long?)null,
                    index: index,
                    cumulativeText: NativeHelpers.FromUtf8Ptr(cumulativeText),
                    finishReason: NativeHelpers.FromUtf8Ptr(finishReason)
                );

                callback(streamToken);
            }
            catch (Exception)
            {
                // Swallow exceptions to prevent unwinding through native frames.
                // Errors are reported via the returned InferenceResult.
            }
        }

#if ENABLE_IL2CPP
        [AOT.MonoPInvokeCallback(typeof(NativeMethods.xybrid_model_run_streaming_with_context_callback_delegate))]
#endif
        private static unsafe void StreamCallbackTrampolineWithContext(
            byte* token, long tokenId, uint index,
            byte* cumulativeText, byte* finishReason, void* userData)
        {
            try
            {
                var gcHandle = GCHandle.FromIntPtr((IntPtr)userData);
                var callback = (Action<StreamToken>)gcHandle.Target;

                var streamToken = new StreamToken(
                    token: NativeHelpers.FromUtf8Ptr(token),
                    tokenId: tokenId >= 0 ? tokenId : (long?)null,
                    index: index,
                    cumulativeText: NativeHelpers.FromUtf8Ptr(cumulativeText),
                    finishReason: NativeHelpers.FromUtf8Ptr(finishReason)
                );

                callback(streamToken);
            }
            catch (Exception)
            {
                // Swallow exceptions to prevent unwinding through native frames.
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
