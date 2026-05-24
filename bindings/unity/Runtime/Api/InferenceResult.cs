// Xybrid SDK - Inference Result
// Wrapper for the output of model inference.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Xybrid.Native;

namespace Xybrid
{
    /// <summary>
    /// Per-stage latency entry for pipeline runs. One entry per executed
    /// stage; <see cref="StageId"/> matches the stage name in the pipeline
    /// definition.
    /// </summary>
    public sealed class StageLatency
    {
        public string StageId { get; }
        public uint LatencyMs { get; }

        internal StageLatency(string stageId, uint latencyMs)
        {
            StageId = stageId;
            LatencyMs = latencyMs;
        }
    }

    /// <summary>
    /// Typed inference metrics surfaced on every <see cref="InferenceResult"/>.
    /// </summary>
    /// <remarks>
    /// LLM-specific fields (<see cref="TtftMs"/>, <see cref="TokensPerSecond"/>,
    /// <see cref="PrefillTps"/>, <see cref="DecodeTps"/>, <see cref="TokensOut"/>)
    /// are <c>null</c> for ASR/TTS/embedding runs. For pipeline runs they are
    /// parsed from the final stage's envelope only, so they are also <c>null</c>
    /// when the final stage isn't the LLM (e.g. an ASR → LLM → TTS pipeline).
    ///
    /// <see cref="StageLatenciesMs"/> is empty for <c>model.Run()</c> and
    /// populated for pipeline runs.
    /// </remarks>
    public sealed class InferenceMetrics
    {
        public uint TotalMs { get; }
        public uint? TtftMs { get; }
        public float? TokensPerSecond { get; }
        public float? PrefillTps { get; }
        public float? DecodeTps { get; }
        public uint? TokensOut { get; }
        public IReadOnlyList<StageLatency> StageLatenciesMs { get; }

        internal InferenceMetrics(
            uint totalMs,
            uint? ttftMs,
            float? tokensPerSecond,
            float? prefillTps,
            float? decodeTps,
            uint? tokensOut,
            IReadOnlyList<StageLatency> stageLatenciesMs)
        {
            TotalMs = totalMs;
            TtftMs = ttftMs;
            TokensPerSecond = tokensPerSecond;
            PrefillTps = prefillTps;
            DecodeTps = decodeTps;
            TokensOut = tokensOut;
            StageLatenciesMs = stageLatenciesMs;
        }
    }

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
        private readonly InferenceMetrics _metrics;

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

        /// <summary>
        /// Gets the typed inference metrics (TTFT, tok/s, per-stage latencies).
        /// </summary>
        /// <remarks>
        /// LLM-specific fields are <c>null</c> for ASR/TTS/embedding runs;
        /// <see cref="InferenceMetrics.StageLatenciesMs"/> is empty for
        /// single-model runs.
        /// </remarks>
        public InferenceMetrics Metrics => _metrics;

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

            // Cache typed metrics. Sentinel conventions match the C ABI:
            // -1 for absent optional integers, NaN for absent optional floats.
            long ttft = NativeMethods.xybrid_result_ttft_ms(handle);
            float tps = NativeMethods.xybrid_result_tokens_per_second(handle);
            float prefill = NativeMethods.xybrid_result_prefill_tps(handle);
            float decode = NativeMethods.xybrid_result_decode_tps(handle);
            long tokensOut = NativeMethods.xybrid_result_tokens_out(handle);

            nuint stageCount = NativeMethods.xybrid_result_stage_count(handle);
            var stages = new List<StageLatency>((int)stageCount);
            for (nuint i = 0; i < stageCount; i++)
            {
                byte* stageIdPtr = NativeMethods.xybrid_result_stage_id(handle, i);
                string stageId = NativeHelpers.FromUtf8Ptr(stageIdPtr) ?? string.Empty;
                uint stageLatencyMs = NativeMethods.xybrid_result_stage_latency_ms(handle, i);
                stages.Add(new StageLatency(stageId, stageLatencyMs));
            }

            _metrics = new InferenceMetrics(
                totalMs: _latencyMs,
                ttftMs: ttft < 0 ? (uint?)null : (uint)ttft,
                tokensPerSecond: float.IsNaN(tps) ? (float?)null : tps,
                prefillTps: float.IsNaN(prefill) ? (float?)null : prefill,
                decodeTps: float.IsNaN(decode) ? (float?)null : decode,
                tokensOut: tokensOut < 0 ? (uint?)null : (uint)tokensOut,
                stageLatenciesMs: stages
            );
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
