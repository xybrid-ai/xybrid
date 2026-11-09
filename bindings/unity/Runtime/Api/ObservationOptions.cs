// Xybrid SDK - Observation Options
// Per-run trace context and trace controls.

namespace Xybrid
{
    /// <summary>
    /// Per-run observability options: parent trace context, conversation id,
    /// and trace suppression.
    /// </summary>
    /// <remarks>
    /// Pass to <see cref="Model.Run(Envelope, GenerationConfig, System.Threading.CancellationToken, ObservationOptions)"/>,
    /// <c>RunStreaming</c>, <c>RunTtsStreaming</c>, <see cref="Pipeline.Run(Envelope, ObservationOptions)"/>,
    /// <see cref="ModelLoader.Load(ObservationOptions)"/>, <c>Model.Stream</c>
    /// or a manual observation such as <see cref="WorkflowObservation.Start"/>.
    /// Has no effect until export is enabled with
    /// <see cref="XybridClient.InitializeObservability"/>.
    /// </remarks>
    /// <example>
    /// <code>
    /// var observation = new ObservationOptions
    /// {
    ///     Traceparent = request.Headers["traceparent"],
    ///     ConversationId = "chat-42",
    /// };
    /// var result = model.Run(envelope, observation: observation);
    /// </code>
    /// </example>
    public sealed class ObservationOptions
    {
        /// <summary>
        /// W3C <c>traceparent</c> of the caller's span; the run becomes its child.
        /// </summary>
        public string Traceparent { get; set; }

        /// <summary>
        /// W3C <c>tracestate</c> accompanying <see cref="Traceparent"/>.
        /// </summary>
        public string Tracestate { get; set; }

        /// <summary>
        /// Caller-owned conversation id recorded on the run.
        /// </summary>
        public string ConversationId { get; set; }

        /// <summary>
        /// Skips the detailed trace for this run; metrics are still recorded.
        /// </summary>
        public bool SuppressTrace { get; set; }

        /// <summary>
        /// The generated record, for APIs that take <see cref="XybridBolt.XybridObservationOptions"/>
        /// directly (for example a hand-built <see cref="XybridBolt.XybridStreamingConfig"/>).
        /// </summary>
        public XybridBolt.XybridObservationOptions ToBolt() =>
            new XybridBolt.XybridObservationOptions(Traceparent, Tracestate, ConversationId, SuppressTrace);

        /// <summary>
        /// Run options carrying <paramref name="config"/> and <paramref name="observation"/>;
        /// null when neither is set, so the native defaults apply.
        /// </summary>
        internal static XybridBolt.XybridRunOptions? ToRunOptions(
            GenerationConfig config,
            ObservationOptions observation)
        {
            if (config == null && observation == null)
            {
                return null;
            }
            return new XybridBolt.XybridRunOptions(
                config?.ToBolt(),
                System.Array.Empty<XybridBolt.XybridAbortSignal>(),
                false,
                0u,
                null,
                observation?.Traceparent,
                observation?.Tracestate,
                observation?.ConversationId,
                observation?.SuppressTrace ?? false);
        }
    }
}
