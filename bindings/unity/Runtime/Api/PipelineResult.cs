// Xybrid SDK - Pipeline Result
// Every stage's output from a pipeline run, not only the final one.

using System.Collections.Generic;

namespace Xybrid
{
    /// <summary>
    /// What one stage of a pipeline run produced.
    /// </summary>
    /// <remarks>
    /// A stage's output is also the next stage's input: the transcript of an
    /// ASR stage, the reply of an LLM stage.
    /// </remarks>
    public sealed class StageResult
    {
        /// <summary>
        /// Gets the stage identifier from the pipeline YAML (<c>id:</c>), or the
        /// model ID when the stage declares none. Matches
        /// <see cref="Pipeline.StageNames"/>.
        /// </summary>
        public string StageId { get; }

        /// <summary>Gets the type of output this stage produced.</summary>
        public OutputType OutputType { get; }

        /// <summary>Gets the text output (an ASR transcript, an LLM reply), or null.</summary>
        public string Text { get; }

        /// <summary>Gets the audio output (from a TTS stage), or null.</summary>
        public byte[] AudioBytes { get; }

        /// <summary>Gets the embedding output, or null.</summary>
        public float[] Embedding { get; }

        /// <summary>Gets this stage's latency in milliseconds.</summary>
        public uint LatencyMs { get; }

        /// <summary>
        /// Gets where this stage ran. Stages of one pipeline can run in
        /// different places.
        /// </summary>
        public XybridBolt.XybridExecutionTarget ExecutionTarget { get; }

        /// <summary>
        /// Gets generation figures (TTFT, tokens per second) when this stage is
        /// a language model. <see cref="InferenceMetrics.TotalMs"/> is the stage
        /// latency.
        /// </summary>
        public InferenceMetrics Metrics { get; }

        private StageResult(XybridBolt.XybridStageResult stage)
        {
            InferenceResult.DecodePayload(
                stage.Envelope, out string text, out byte[] audio, out float[] embedding);
            StageId = stage.StageId;
            OutputType = InferenceResult.MapOutputType(stage.OutputType);
            Text = text;
            AudioBytes = audio;
            Embedding = embedding;
            LatencyMs = stage.LatencyMs;
            ExecutionTarget = stage.ExecutionTarget;
            Metrics = InferenceResult.MapMetrics(stage.Metrics);
        }

        internal static StageResult FromBolt(XybridBolt.XybridStageResult stage) =>
            new StageResult(stage);
    }

    /// <summary>
    /// The result of <see cref="Pipeline.Run"/>: the final output plus every
    /// stage's own output.
    /// </summary>
    /// <example>
    /// <code>
    /// PipelineResult result = pipeline.Run(Envelope.Audio(pcm));
    /// transcript.text = result.Stage("asr")?.Text;
    /// reply.text = result.Stage("llm")?.Text;
    /// Play(result.AudioBytes);
    /// </code>
    /// </example>
    public sealed class PipelineResult
    {
        /// <summary>Gets the type of the final stage's output.</summary>
        public OutputType OutputType { get; }

        /// <summary>Gets the final text output, or null.</summary>
        public string Text { get; }

        /// <summary>Gets the final audio output, or null.</summary>
        public byte[] AudioBytes { get; }

        /// <summary>Gets the final embedding output, or null.</summary>
        public float[] Embedding { get; }

        /// <summary>Gets whether the final output is audio.</summary>
        public bool HasAudio => AudioBytes != null && AudioBytes.Length > 0;

        /// <summary>Gets the whole run's latency in milliseconds.</summary>
        public uint LatencyMs { get; }

        /// <summary>Gets every executed stage, in order.</summary>
        public IReadOnlyList<StageResult> Stages { get; }

        private PipelineResult(XybridBolt.XybridPipelineResult result)
        {
            InferenceResult.DecodePayload(
                result.Envelope, out string text, out byte[] audio, out float[] embedding);
            OutputType = InferenceResult.MapOutputType(result.OutputType);
            Text = text;
            AudioBytes = audio;
            Embedding = embedding;
            LatencyMs = result.LatencyMs;

            var stages = new List<StageResult>(result.Stages.Length);
            foreach (XybridBolt.XybridStageResult stage in result.Stages)
            {
                stages.Add(StageResult.FromBolt(stage));
            }
            Stages = stages;
        }

        internal static PipelineResult FromBolt(XybridBolt.XybridPipelineResult result) =>
            new PipelineResult(result);

        /// <summary>
        /// Returns the stage with this identifier (the YAML <c>id:</c>), or null
        /// if no such stage ran.
        /// </summary>
        public StageResult Stage(string stageId)
        {
            foreach (StageResult stage in Stages)
            {
                if (stage.StageId == stageId)
                {
                    return stage;
                }
            }
            return null;
        }

        /// <summary>Returns a string representation of the result.</summary>
        public override string ToString() =>
            $"PipelineResult(OutputType={OutputType}, LatencyMs={LatencyMs}, " +
            $"Stages={Stages.Count}, Text=\"{Text ?? "null"}\", AudioBytes={AudioBytes?.Length ?? 0})";
    }
}
