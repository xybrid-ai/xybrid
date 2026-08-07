// Xybrid SDK - Inference Result
// Wrapper for the output of model inference.

using System;
using System.Collections.Generic;

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
    /// are <c>null</c> for ASR/TTS/embedding runs. <see cref="StageLatenciesMs"/>
    /// is empty for <c>model.Run()</c> and populated for pipeline runs.
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
    public sealed class InferenceResult : IDisposable
    {
        /// <summary>Gets whether this result has been disposed.</summary>
        public bool IsDisposed { get; private set; }

        /// <summary>Gets whether the inference was successful.</summary>
        public bool Success { get; }

        /// <summary>Gets the error message if inference failed, or null if successful.</summary>
        public string Error { get; }

        /// <summary>Gets the text output (for ASR or LLM models), or null if not applicable.</summary>
        public string Text { get; }

        /// <summary>Gets the inference latency in milliseconds.</summary>
        public uint LatencyMs { get; }

        /// <summary>Gets the type of output produced by inference.</summary>
        public OutputType OutputType { get; }

        /// <summary>
        /// Gets the raw audio bytes (for TTS models), or null if not applicable.
        /// Audio format is raw PCM 16-bit signed little-endian, typically 24kHz mono.
        /// </summary>
        public byte[] AudioBytes { get; }

        /// <summary>Gets the embedding vector (for embedding models), or null if not applicable.</summary>
        public float[] Embedding { get; }

        /// <summary>Gets whether this result contains audio data.</summary>
        public bool HasAudio => AudioBytes != null && AudioBytes.Length > 0;

        /// <summary>Gets whether this result contains an embedding.</summary>
        public bool HasEmbedding => Embedding != null && Embedding.Length > 0;

        /// <summary>Gets the typed inference metrics (TTFT, tok/s, per-stage latencies).</summary>
        public InferenceMetrics Metrics { get; }

        /// <summary>
        /// Gets whether this answer came from the device or the cloud gateway.
        /// Cloud fallback keeps the model id identical on both legs, so this is
        /// the only way to tell them apart. Defaults to
        /// <see cref="XybridBolt.XybridExecutionTarget.Local"/> for synthesized
        /// failures, which never reached the cloud.
        /// </summary>
        public XybridBolt.XybridExecutionTarget ExecutionTarget { get; }

        private InferenceResult(
            bool success,
            string error,
            string text,
            uint latencyMs,
            OutputType outputType,
            byte[] audioBytes,
            float[] embedding,
            InferenceMetrics metrics,
            XybridBolt.XybridExecutionTarget executionTarget =
                XybridBolt.XybridExecutionTarget.Local)
        {
            Success = success;
            Error = error;
            Text = text;
            LatencyMs = latencyMs;
            OutputType = outputType;
            AudioBytes = audioBytes;
            Embedding = embedding;
            Metrics = metrics;
            ExecutionTarget = executionTarget;
        }

        /// <summary>Decode a successful bolt result into the public shape.</summary>
        internal static InferenceResult FromBolt(XybridBolt.XybridResult result)
        {
            string text = null;
            byte[] audio = null;
            float[] embedding = null;
            switch (result.Envelope.Kind)
            {
                case XybridBolt.XybridEnvelopeKind.Text t:
                    text = t.Value;
                    break;
                case XybridBolt.XybridEnvelopeKind.Audio a:
                    audio = a.Bytes;
                    break;
                case XybridBolt.XybridEnvelopeKind.Embedding e:
                    embedding = e.Values;
                    break;
            }

            return new InferenceResult(
                success: true,
                error: null,
                text: text,
                latencyMs: result.LatencyMs,
                outputType: MapOutputType(result.OutputType),
                audioBytes: audio,
                embedding: embedding,
                metrics: MapMetrics(result.Metrics),
                executionTarget: result.ExecutionTarget);
        }

        /// <summary>
        /// A failed result. Bolt inference throws on error; <see cref="Model"/>
        /// catches inference failures and synthesizes this so callers that
        /// inspect <see cref="Success"/> keep their contract.
        /// </summary>
        internal static InferenceResult Failed(string error) =>
            new InferenceResult(
                success: false,
                error: error,
                text: null,
                latencyMs: 0,
                outputType: OutputType.Unknown,
                audioBytes: null,
                embedding: null,
                metrics: new InferenceMetrics(0, null, null, null, null, null, Array.Empty<StageLatency>()));

        private static OutputType MapOutputType(XybridBolt.XybridOutputType outputType)
        {
            switch (outputType)
            {
                case XybridBolt.XybridOutputType.Text:
                    return OutputType.Text;
                case XybridBolt.XybridOutputType.Audio:
                    return OutputType.Audio;
                case XybridBolt.XybridOutputType.Embedding:
                    return OutputType.Embedding;
                default:
                    return OutputType.Unknown;
            }
        }

        private static InferenceMetrics MapMetrics(XybridBolt.XybridInferenceMetrics metrics)
        {
            var stages = new List<StageLatency>(metrics.StageLatenciesMs.Length);
            foreach (XybridBolt.XybridStageLatency stage in metrics.StageLatenciesMs)
            {
                stages.Add(new StageLatency(stage.StageId, stage.LatencyMs));
            }

            return new InferenceMetrics(
                totalMs: metrics.TotalMs,
                ttftMs: metrics.TtftMs,
                tokensPerSecond: metrics.TokensPerSecond,
                prefillTps: metrics.PrefillTps,
                decodeTps: metrics.DecodeTps,
                tokensOut: metrics.TokensOut,
                stageLatenciesMs: stages);
        }

        /// <summary>
        /// Throws an InferenceException if the result indicates failure.
        /// </summary>
        /// <exception cref="InferenceException">Thrown if Success is false.</exception>
        public void ThrowIfFailed()
        {
            if (!Success)
            {
                throw new InferenceException(Error ?? "Unknown inference error");
            }
        }

        /// <summary>
        /// No-op: the result holds no native resources. Retained so existing
        /// <c>using</c> call sites keep compiling.
        /// </summary>
        public void Dispose()
        {
            IsDisposed = true;
        }

        /// <summary>Returns a string representation of the result.</summary>
        public override string ToString()
        {
            return Success
                ? $"InferenceResult(Success, OutputType={OutputType}, LatencyMs={LatencyMs}, " +
                  $"Text=\"{Text ?? "null"}\", AudioBytes={AudioBytes?.Length ?? 0})"
                : $"InferenceResult(Failed, Error=\"{Error}\")";
        }
    }
}
