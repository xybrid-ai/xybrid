// Xybrid SDK - Model
// Wrapper for a loaded model ready for inference.

using System;
using System.Threading;
using System.Threading.Tasks;

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
        private readonly XybridBolt.XybridModel _bolt;
        private readonly string _modelId;
        private bool _disposed;
        private VoiceInfo[] _cachedVoices;

        /// <summary>Gets whether this model has been disposed.</summary>
        public bool IsDisposed => _disposed;

        /// <summary>Gets the model ID.</summary>
        public string ModelId => _modelId;

        internal Model(XybridBolt.XybridModel bolt)
        {
            _bolt = bolt;
            _modelId = bolt.ModelId();
        }

        /// <summary>
        /// Runs inference on this model with the provided input envelope.
        /// </summary>
        /// <param name="envelope">The input data for inference.</param>
        /// <param name="config">Optional generation config for LLM parameters. Pass null for model defaults.</param>
        /// <returns>The inference result (<see cref="InferenceResult.Success"/> is false if inference failed).</returns>
        /// <exception cref="ArgumentNullException">Thrown if envelope is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this model is disposed.</exception>
        /// <exception cref="XybridException">Thrown only on a catastrophic backend failure; ordinary inference failures (including a not-loaded model) set <see cref="InferenceResult.Success"/> to false instead.</exception>
        /// <param name="cancellationToken">
        /// Optional stop button. A batch run is only cancellable <em>before</em>
        /// generation starts — once the backend is producing there is no
        /// token-aware batch path to stop it, so the call finishes normally. Use
        /// <c>RunStreaming</c> when a mid-flight stop button matters. A cancelled
        /// run reports failure rather than throwing
        /// <see cref="OperationCanceledException"/>.
        /// </param>
        public InferenceResult Run(
            Envelope envelope,
            GenerationConfig config = null,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            if (envelope == null)
            {
                throw new ArgumentNullException(nameof(envelope));
            }

            using (var cancel = BoltCancellation.From(cancellationToken))
            {
                return Execute(() => _bolt.Run(envelope.Bolt, ToOptions(config), cancel.Token));
            }
        }

        /// <summary>
        /// Runs inference and returns the text result, throwing on failure.
        /// </summary>
        /// <param name="text">The input text for TTS or LLM inference.</param>
        /// <returns>The text output from the model.</returns>
        /// <exception cref="InferenceException">Thrown if inference fails.</exception>
        public string RunText(string text)
        {
            var envelope = Envelope.Text(text);
            var result = Run(envelope);
            result.ThrowIfFailed();
            return result.Text;
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
            var envelope = Envelope.Audio(audioBytes, sampleRate, channels);
            var result = Run(envelope);
            result.ThrowIfFailed();
            return result.Text;
        }

        /// <summary>
        /// Runs TTS inference and returns the raw audio bytes.
        /// </summary>
        /// <param name="text">The text to synthesize.</param>
        /// <returns>Raw PCM audio bytes (16-bit signed little-endian, typically 24kHz mono).</returns>
        /// <exception cref="InferenceException">Thrown if inference fails.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the result does not contain audio.</exception>
        public byte[] RunTts(string text)
        {
            var envelope = Envelope.Text(text);
            return TtsAudio(Run(envelope));
        }

        /// <summary>
        /// Runs inference with conversation context.
        /// </summary>
        /// <param name="envelope">The input data for inference.</param>
        /// <param name="context">The conversation context with history.</param>
        /// <param name="config">Optional generation config for LLM parameters. Pass null for model defaults.</param>
        /// <returns>The inference result.</returns>
        /// <remarks>
        /// The context provides conversation history which is formatted into the prompt
        /// using the model's chat template. The context is NOT automatically updated
        /// with the result — call context.Push() to add the response.
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown if envelope or context is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this model is disposed.</exception>
        /// <exception cref="XybridException">Thrown only on a catastrophic backend failure; ordinary inference failures set <see cref="InferenceResult.Success"/> to false instead.</exception>
        /// <param name="cancellationToken">
        /// Optional stop button. A batch run is only cancellable <em>before</em>
        /// generation starts — once the backend is producing there is no
        /// token-aware batch path to stop it, so the call finishes normally. Use
        /// <c>RunStreaming</c> when a mid-flight stop button matters. A cancelled
        /// run reports failure rather than throwing
        /// <see cref="OperationCanceledException"/>.
        /// </param>
        public InferenceResult Run(
            Envelope envelope,
            ConversationContext context,
            GenerationConfig config = null,
            CancellationToken cancellationToken = default)
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

            using (var cancel = BoltCancellation.From(cancellationToken))
            {
                return Execute(() =>
                    _bolt.RunWithContext(envelope.Bolt, context.Bolt, ToOptions(config), cancel.Token));
            }
        }

        /// <summary>
        /// Runs inference with conversation context and returns the text result.
        /// </summary>
        /// <param name="text">The input text for LLM inference.</param>
        /// <param name="context">The conversation context with history.</param>
        /// <returns>The text output from the model.</returns>
        /// <exception cref="InferenceException">Thrown if inference fails.</exception>
        public string RunText(string text, ConversationContext context)
        {
            var envelope = Envelope.Text(text);
            var result = Run(envelope, context);
            result.ThrowIfFailed();
            return result.Text;
        }

        // ================================================================
        // Voice Discovery
        // ================================================================

        /// <summary>Gets whether this model has voice support (TTS models with voice catalog).</summary>
        public bool HasVoices
        {
            get
            {
                ThrowIfDisposed();
                return _bolt.HasVoices();
            }
        }

        /// <summary>Gets the number of voices available for this model.</summary>
        public int VoiceCount => Voices.Length;

        /// <summary>Gets the default voice ID for this model, or null if not a TTS model.</summary>
        public string DefaultVoiceId
        {
            get
            {
                ThrowIfDisposed();
                XybridBolt.XybridVoiceInfo? voice = _bolt.DefaultVoice();
                return voice is { } value ? value.Id : null;
            }
        }

        /// <summary>
        /// Gets all available voices for this model. Returns an empty array if the
        /// model has no voice support. The result is cached after the first call.
        /// </summary>
        public VoiceInfo[] Voices
        {
            get
            {
                ThrowIfDisposed();
                if (_cachedVoices != null)
                {
                    return _cachedVoices;
                }

                XybridBolt.XybridVoiceInfo[] boltVoices = _bolt.Voices();
                var voices = new VoiceInfo[boltVoices.Length];
                for (int i = 0; i < boltVoices.Length; i++)
                {
                    voices[i] = MapVoice(boltVoices[i]);
                }

                _cachedVoices = voices;
                return _cachedVoices;
            }
        }

        /// <summary>
        /// Gets a specific voice by ID, or null if not found.
        /// </summary>
        /// <param name="voiceId">The voice identifier (e.g., "af_bella").</param>
        public VoiceInfo GetVoice(string voiceId)
        {
            ThrowIfDisposed();
            XybridBolt.XybridVoiceInfo? voice = _bolt.Voice(voiceId);
            return voice is { } value ? MapVoice(value) : null;
        }

        /// <summary>
        /// Runs TTS inference with a specific voice and returns the raw audio bytes.
        /// </summary>
        /// <param name="text">The text to synthesize.</param>
        /// <param name="voiceId">The voice ID to use (e.g., "af_bella").</param>
        /// <param name="speed">Speed multiplier (1.0 = normal).</param>
        /// <returns>Raw PCM audio bytes.</returns>
        /// <exception cref="InferenceException">Thrown if inference fails.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the result does not contain audio.</exception>
        public byte[] RunTts(string text, string voiceId, double speed = 1.0)
        {
            var envelope = Envelope.Text(text, voiceId, speed);
            return TtsAudio(Run(envelope));
        }

        // ================================================================
        // Streaming & Token Support
        // ================================================================

        /// <summary>
        /// Gets whether this model supports true token-by-token streaming.
        /// </summary>
        /// <remarks>
        /// Returns true for LLM models. Non-LLM models can still use
        /// <see cref="RunStreaming(Envelope, Action{StreamToken}, GenerationConfig)"/>
        /// but will receive a single callback with the complete result.
        /// </remarks>
        public bool SupportsTokenStreaming
        {
            get
            {
                ThrowIfDisposed();
                return _bolt.SupportsTokenStreaming();
            }
        }

        /// <summary>
        /// Gets whether the model bundle declares local tool-calling support.
        /// </summary>
        /// <remarks>
        /// Advisory tri-state: null means the bundle says nothing, so the app
        /// cannot tell. Gate tool UI on it; enforcement stays at run time — a
        /// request carrying <see cref="GenerationConfig.AddTool"/> tools against a
        /// model whose chat template has no tool support fails regardless of what
        /// this reports.
        /// </remarks>
        public bool? SupportsToolCalling
        {
            get
            {
                ThrowIfDisposed();
                return _bolt.SupportsToolCalling();
            }
        }

        /// <summary>
        /// Runs streaming inference, invoking the callback for each generated token.
        /// Blocks until inference is complete.
        /// </summary>
        /// <param name="envelope">The input data for inference.</param>
        /// <param name="onToken">Callback invoked for each token, on the calling thread.</param>
        /// <param name="config">Optional generation config. Pass null for model defaults.</param>
        /// <returns>The final inference result after all tokens are emitted.</returns>
        /// <exception cref="ArgumentNullException">Thrown if envelope or onToken is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this model is disposed.</exception>
        /// <exception cref="XybridException">Thrown only on a catastrophic backend failure; ordinary inference failures set <see cref="InferenceResult.Success"/> to false instead.</exception>
        /// <param name="cancellationToken">
        /// Optional stop button. Cancelling it ends the stream at the next token
        /// boundary; the run then reports failure rather than throwing
        /// <see cref="OperationCanceledException"/>.
        /// </param>
        public InferenceResult RunStreaming(
            Envelope envelope,
            Action<StreamToken> onToken,
            GenerationConfig config = null,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            if (envelope == null)
            {
                throw new ArgumentNullException(nameof(envelope));
            }
            if (onToken == null)
            {
                throw new ArgumentNullException(nameof(onToken));
            }

            using (var cancel = BoltCancellation.From(cancellationToken))
            {
                return Execute(() =>
                    _bolt.RunStreaming(envelope.Bolt, Forward(onToken), ToOptions(config), cancel.Token));
            }
        }

        /// <summary>
        /// Runs streaming inference with conversation context.
        /// </summary>
        /// <param name="envelope">The input data for inference.</param>
        /// <param name="context">The conversation context with history.</param>
        /// <param name="onToken">Callback invoked for each token.</param>
        /// <param name="config">Optional generation config. Pass null for model defaults.</param>
        /// <returns>The final inference result after all tokens are emitted.</returns>
        /// <exception cref="ArgumentNullException">Thrown if any argument is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this model is disposed.</exception>
        /// <exception cref="XybridException">Thrown only on a catastrophic backend failure; ordinary inference failures set <see cref="InferenceResult.Success"/> to false instead.</exception>
        /// <param name="cancellationToken">
        /// Optional stop button. Cancelling it ends the stream at the next token
        /// boundary; the run then reports failure rather than throwing
        /// <see cref="OperationCanceledException"/>.
        /// </param>
        public InferenceResult RunStreaming(
            Envelope envelope,
            ConversationContext context,
            Action<StreamToken> onToken,
            GenerationConfig config = null,
            CancellationToken cancellationToken = default)
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
            if (onToken == null)
            {
                throw new ArgumentNullException(nameof(onToken));
            }

            using (var cancel = BoltCancellation.From(cancellationToken))
            {
                return Execute(() => _bolt.RunStreamingWithContext(
                    envelope.Bolt, Forward(onToken), context.Bolt, ToOptions(config), cancel.Token));
            }
        }

        /// <summary>
        /// Convenience method: stream text inference with a callback.
        /// </summary>
        public string RunStreamingText(string text, Action<StreamToken> onToken)
        {
            var envelope = Envelope.Text(text);
            var result = RunStreaming(envelope, onToken);
            result.ThrowIfFailed();
            return result.Text;
        }

        /// <summary>
        /// Convenience method: stream text inference with conversation context.
        /// </summary>
        public string RunStreamingText(string text, ConversationContext context, Action<StreamToken> onToken)
        {
            var envelope = Envelope.Text(text);
            var result = RunStreaming(envelope, context, onToken);
            result.ThrowIfFailed();
            return result.Text;
        }

        /// <summary>Deliver speech packets on the calling thread; blocks until synthesis ends.</summary>
        /// <remarks>
        /// Call on a worker. Packets are owned PCM16 LE; queue them for playback.
        /// Cancellation discards queued native packets and stops active Kitten
        /// inference; invalidate audio already scheduled by your playback queue.
        /// The native protocol is pull-based, so no managed delegate crosses FFI
        /// and no callback rooting or MonoPInvokeCallback attribute is required.
        /// Dispose the model after this call returns.
        /// </remarks>
        public TtsStreamResult RunTtsStreaming(Envelope envelope, Action<TtsAudioChunk> onAudio = null,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            if (envelope == null) throw new ArgumentNullException(nameof(envelope));
            if (cancellationToken.IsCancellationRequested)
                return new TtsStreamResult(TtsStatus.Cancelled);
            using (var cancel = BoltCancellation.From(cancellationToken))
            {
                ulong? streamId = null;
                bool terminal = false;
                try
                {
                    streamId = _bolt.RunStream(envelope.Bolt, null, cancel.Token);
                    while (true)
                    {
                        var item = _bolt.StreamNext(streamId.Value);
                        if (item.Kind == XybridBolt.XybridStreamEventKind.Complete) { terminal = true; break; }
                        if (item.Kind != XybridBolt.XybridStreamEventKind.Audio)
                            throw new InvalidOperationException("RunTtsStreaming requires an audio model");
                        if (!cancellationToken.IsCancellationRequested && item.Audio.HasValue)
                            onAudio?.Invoke(new TtsAudioChunk(item.Audio.Value));
                    }
                    var result = TtsStreamResult.FromBolt(_bolt.StreamResult(streamId.Value));
                    return cancellationToken.IsCancellationRequested
                        ? new TtsStreamResult(TtsStatus.Cancelled, result.SampleRate, result.Channels,
                            result.Samples, result.Chunks, result.LimitedChunks)
                        : result;
                }
                catch (XybridBolt.XybridErrorException ex)
                {
                    return new TtsStreamResult(TtsStatus.Failed, error: BoltErrors.Describe(ex.Error));
                }
                finally
                {
                    // Also cancels on callback exceptions or a broken consumer.
                    cancel.Token.Cancel();
                    if (streamId.HasValue)
                    {
                        // Even when the host callback throws, retire the native
                        // worker before the caller can dispose/switch providers.
                        if (!terminal)
                        {
                            try
                            {
                                while (_bolt.StreamNext(streamId.Value).Kind !=
                                    XybridBolt.XybridStreamEventKind.Complete) { }
                            }
                            catch (XybridBolt.XybridErrorException) { }
                        }
                        _bolt.StreamClose(streamId.Value);
                    }
                }
            }
        }

        /// <summary>Run speech delivery on a worker, including callbacks.</summary>
        public Task<TtsStreamResult> RunTtsStreamingAsync(Envelope envelope,
            Action<TtsAudioChunk> onAudio = null, CancellationToken cancellationToken = default)
        {
            // Do not pass the token to Task.Run: a queued cancellation must
            // return the explicit Cancelled outcome, not skip the delegate.
            return Task.Run(() => RunTtsStreaming(envelope, onAudio, cancellationToken));
        }

        // ================================================================
        // Internal helpers
        // ================================================================

        // Preserve the pre-bolt contract: xybrid_model_run returned a failed
        // result handle (Success == false) for EVERY SDK run error and only
        // threw for null/invalid handles. Bolt surfaces those SDK errors as
        // XybridErrorException, so map all of them to a failed InferenceResult
        // (callers inspecting Success keep working). A BoltException is the
        // catastrophic analog of the old null-handle path and is rethrown.
        private static InferenceResult Execute(Func<XybridBolt.XybridResult> run)
        {
            try
            {
                return InferenceResult.FromBolt(run());
            }
            catch (XybridBolt.XybridErrorException ex)
            {
                // Match the pre-bolt failure text: "Inference failed: <message>"
                // with the error's inner message (not its record ToString()).
                return InferenceResult.Failed("Inference failed: " + BoltErrors.Describe(ex.Error));
            }
            catch (XybridBolt.BoltException ex)
            {
                throw BoltErrors.Translate(ex);
            }
        }

        // Wrap the user callback so a throw doesn't abort streaming — the
        // pre-bolt native trampoline swallowed callback exceptions; preserved.
        private static Action<XybridBolt.XybridStreamToken> Forward(Action<StreamToken> onToken) =>
            bolt =>
            {
                try
                {
                    onToken(MapToken(bolt));
                }
                catch
                {
                    // Intentionally swallowed (see above).
                }
            };

        private static StreamToken MapToken(XybridBolt.XybridStreamToken token) =>
            new StreamToken(
                token.Token,
                token.TokenId,
                (uint)token.Index,
                token.CumulativeText,
                token.FinishReason,
                token.ToolCalls ?? System.Array.Empty<XybridBolt.XybridToolCall>(),
                token.RawText);

        private static VoiceInfo MapVoice(XybridBolt.XybridVoiceInfo voice) =>
            new VoiceInfo(voice.Id, voice.Name, voice.Gender, voice.Language, voice.Style);

        private static XybridBolt.XybridRunOptions? ToOptions(GenerationConfig config)
        {
            if (config == null)
            {
                return null;
            }
            return new XybridBolt.XybridRunOptions(
                config.ToBolt(),
                Array.Empty<XybridBolt.XybridAbortSignal>(),
                false,
                0u,
                null);
        }

        private static byte[] TtsAudio(InferenceResult result)
        {
            result.ThrowIfFailed();
            if (!result.HasAudio)
            {
                throw new InvalidOperationException(
                    "Model did not produce audio output. " +
                    $"Output type was: {result.OutputType}");
            }
            return result.AudioBytes;
        }

        /// <summary>
        /// Opens a live ASR session: feed microphone PCM in, read partial
        /// transcripts out.
        /// </summary>
        /// <remarks>
        /// This is the live-capture surface. <see cref="Run"/> transcribes a
        /// finished buffer; this transcribes speech as it arrives, which is what
        /// dictation and live captioning need.
        /// <para>
        /// Audio must be PCM <b>float, mono, 16 kHz</b> — converting from Unity's
        /// <c>AudioClip</c> or microphone format is the caller's job.
        /// </para>
        /// <code>
        /// using var session = model.Stream();
        /// // drive the UI from the partial stream
        /// await foreach (var partial in session.Partials(token))
        /// {
        ///     label.text = partial.Text;
        /// }
        /// </code>
        /// </remarks>
        /// <param name="config">
        /// Chunking options. Pass null for fixed-window chunking at 16 kHz with
        /// the model's own language; use
        /// <see cref="StreamingConfigs.VoiceActivity"/> to chunk on speech
        /// boundaries instead.
        /// </param>
        /// <returns>A session; dispose it to stop transcribing.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this model is disposed.</exception>
        /// <exception cref="XybridException">
        /// Thrown if this is not an ASR model, or the sample rate is not 16 kHz.
        /// </exception>
        public XybridBolt.XybridStreamingSession Stream(
            XybridBolt.XybridStreamingConfig? config = null)
        {
            ThrowIfDisposed();

            try
            {
                return XybridBolt.XybridStreamingSession.ForModel(
                    _bolt,
                    config ?? StreamingConfigs.Default);
            }
            catch (Exception ex) when (
                ex is XybridBolt.XybridErrorException || ex is XybridBolt.BoltException)
            {
                throw BoltErrors.Translate(ex);
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(Model));
            }
        }

        /// <summary>Releases the native resources used by this model.</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _bolt.Dispose();
                _disposed = true;
            }
        }

        /// <summary>Returns a string representation of the model.</summary>
        public override string ToString()
        {
            return $"Model({ModelId})";
        }
    }
}
