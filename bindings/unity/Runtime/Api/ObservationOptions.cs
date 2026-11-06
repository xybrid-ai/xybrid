// Xybrid SDK - Observation Options
// Per-run trace context and trace controls.

namespace Xybrid
{
    /// <summary>
    /// Per-run observability options: parent trace context, conversation id,
    /// and trace suppression.
    /// </summary>
    /// <remarks>
    /// Pass to <see cref="Model.Run(Envelope, GenerationConfig, System.Threading.CancellationToken, ObservationOptions)"/>
    /// or <c>RunStreaming</c>. Has no effect until export is enabled with
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
    }
}
