// Xybrid SDK - Manual observations
// IDisposable wrappers over the generated BoltFFI operation handles.

using System;

namespace Xybrid
{
    /// <summary>A manual operation; final disposal cancels unfinished work and never flushes.</summary>
    /// <remarks>
    /// The first terminal call (<c>Complete</c>, <c>Fail</c> or <c>Cancel</c>) wins;
    /// later terminal calls and recordings are ignored. Pass <c>Context</c> to a
    /// child's <see cref="ObservationOptions"/> to nest it.
    /// </remarks>
    public abstract class ManualObservation<T> : IDisposable where T : class, IDisposable
    {
        private readonly object _gate = new object();
        private T _operation;

        /// <summary>Wraps a started generated operation handle.</summary>
        protected ManualObservation(T operation) { _operation = operation; }

        /// <summary>Runs <paramref name="action"/> on the live handle, translating bolt errors.</summary>
        protected R Read<R>(Func<T, R> action)
        {
            lock (_gate)
            {
                if (_operation == null) throw new ObjectDisposedException(GetType().Name);
                return Translated(() => action(_operation));
            }
        }

        /// <summary>Runs <paramref name="action"/> on the live handle, translating bolt errors.</summary>
        protected void Write(Action<T> action) => Read(op => { action(op); return true; });

        /// <summary>Starts a generated handle, mapping bolt failures to <see cref="XybridException"/>.</summary>
        internal static R Translated<R>(Func<R> start)
        {
            try
            {
                return start();
            }
            catch (Exception ex) when (
                ex is XybridBolt.XybridErrorException || ex is XybridBolt.BoltException)
            {
                throw BoltErrors.Translate(ex);
            }
        }

        /// <summary>Releases the native handle; an unfinished operation ends as cancelled.</summary>
        public void Dispose()
        {
            lock (_gate)
            {
                var operation = _operation;
                _operation = null;
                operation?.Dispose();
            }
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>A manual workflow operation (<c>invoke_workflow</c>).</summary>
    public sealed class WorkflowObservation : ManualObservation<XybridBolt.XybridWorkflowObservation>
    {
        private WorkflowObservation(XybridBolt.XybridWorkflowObservation operation) : base(operation) { }

        /// <summary>Starts a workflow under the explicit parent in <paramref name="options"/>, if any.</summary>
        /// <exception cref="XybridException">Thrown for an invalid explicit parent.</exception>
        public static WorkflowObservation Start(string name, ObservationOptions options = null)
        {
            if (name == null) throw new ArgumentNullException(nameof(name));
            return new WorkflowObservation(Translated(() =>
                new XybridBolt.XybridWorkflowObservation(name, options?.ToBolt())));
        }

        /// <summary>Context for child operations, including valid unsampled spans.</summary>
        public XybridBolt.XybridObservationContext? Context => Read(op => op.Context());
        /// <summary>Reference to this operation's own span.</summary>
        public XybridBolt.XybridObservationRef? Reference => Read(op => op.ObservationRef());
        /// <summary>Whether any terminal transition has happened.</summary>
        public bool IsFinished => Read(op => op.IsFinished());
        /// <summary>Ends successfully.</summary>
        public void Complete() => Write(op => op.Complete());
        /// <summary>Ends with an error type and an optional governed message.</summary>
        public void Fail(string errorType, string message = null) => Write(op => op.Fail(errorType, message));
        /// <summary>Ends as cancelled without flushing.</summary>
        public void Cancel() => Write(op => op.Cancel());
    }

    /// <summary>A manual tool execution (<c>execute_tool</c>).</summary>
    public sealed class ToolObservation : ManualObservation<XybridBolt.XybridToolObservation>
    {
        private ToolObservation(XybridBolt.XybridToolObservation operation) : base(operation) { }

        /// <summary>Starts a tool execution under the explicit parent in <paramref name="options"/>, if any.</summary>
        /// <exception cref="XybridException">Thrown for an invalid explicit parent or arguments JSON.</exception>
        public static ToolObservation Start(XybridBolt.XybridToolCallInfo call, ObservationOptions options = null) =>
            new ToolObservation(Translated(() => new XybridBolt.XybridToolObservation(call, options?.ToBolt())));

        /// <summary>Context for child operations, including valid unsampled spans.</summary>
        public XybridBolt.XybridObservationContext? Context => Read(op => op.Context());
        /// <summary>Reference to this operation's own span.</summary>
        public XybridBolt.XybridObservationRef? Reference => Read(op => op.ObservationRef());
        /// <summary>Whether any terminal transition has happened.</summary>
        public bool IsFinished => Read(op => op.IsFinished());
        /// <summary>Ends successfully.</summary>
        public void Complete() => Write(op => op.Complete());
        /// <summary>Ends with an error type and an optional governed message.</summary>
        public void Fail(string errorType, string message = null) => Write(op => op.Fail(errorType, message));
        /// <summary>Ends as cancelled without flushing.</summary>
        public void Cancel() => Write(op => op.Cancel());
        /// <summary>Records the tool result as governed JSON content.</summary>
        public void RecordResult(string resultJson) => Write(op => op.RecordResult(resultJson));
    }

    /// <summary>A manual inference operation for a runtime Xybrid does not execute itself.</summary>
    public sealed class InferenceObservation : ManualObservation<XybridBolt.XybridInferenceObservation>
    {
        private InferenceObservation(XybridBolt.XybridInferenceObservation operation) : base(operation) { }

        /// <summary>Starts an inference under the explicit parent in <paramref name="options"/>, if any.</summary>
        /// <exception cref="XybridException">Thrown for an invalid explicit parent.</exception>
        public static InferenceObservation Start(XybridBolt.XybridInferenceRequest request, ObservationOptions options = null) =>
            new InferenceObservation(Translated(() => new XybridBolt.XybridInferenceObservation(request, options?.ToBolt())));

        /// <summary>Context for child operations, including valid unsampled spans.</summary>
        public XybridBolt.XybridObservationContext? Context => Read(op => op.Context());
        /// <summary>Reference to this operation's own span.</summary>
        public XybridBolt.XybridObservationRef? Reference => Read(op => op.ObservationRef());
        /// <summary>Whether any terminal transition has happened.</summary>
        public bool IsFinished => Read(op => op.IsFinished());
        /// <summary>Ends successfully.</summary>
        public void Complete() => Write(op => op.Complete());
        /// <summary>Ends with an error type and an optional governed message.</summary>
        public void Fail(string errorType, string message = null) => Write(op => op.Fail(errorType, message));
        /// <summary>Ends as cancelled without flushing.</summary>
        public void Cancel() => Write(op => op.Cancel());
        /// <summary>Records system instructions as a JSON array of parts.</summary>
        public void RecordSystemInstructions(string partsJson) => Write(op => op.RecordSystemInstructions(partsJson));
        /// <summary>Records input messages as a JSON array.</summary>
        public void RecordInputMessages(string messagesJson) => Write(op => op.RecordInputMessages(messagesJson));
        /// <summary>Records output messages as a JSON array.</summary>
        public void RecordOutputMessages(string messagesJson) => Write(op => op.RecordOutputMessages(messagesJson));
        /// <summary>Records tool definitions as a JSON array.</summary>
        public void RecordToolDefinitions(string definitionsJson) => Write(op => op.RecordToolDefinitions(definitionsJson));
        /// <summary>Counts one streamed chunk; never implies a first token.</summary>
        public void RecordChunk() => Write(op => op.RecordChunk());
        /// <summary>Records a measured first token, at the given epoch milliseconds or now.</summary>
        public void RecordFirstToken(ulong? atEpochMs = null) => Write(op => op.RecordFirstToken(atEpochMs));
        /// <summary>Records reported token usage; unknown fields stay unavailable.</summary>
        public void RecordUsage(XybridBolt.XybridUsage usage) => Write(op => op.RecordUsage(usage));
        /// <summary>Records the response finish reasons.</summary>
        public void RecordFinishReasons(string[] reasons) => Write(op => op.RecordFinishReasons(reasons));
        /// <summary>Records the provider response id and model.</summary>
        public void RecordResponse(string responseId = null, string model = null) => Write(op => op.RecordResponse(responseId, model));
    }

    /// <summary>A manual model load.</summary>
    public sealed class ModelLoadObservation : ManualObservation<XybridBolt.XybridModelLoadObservation>
    {
        private ModelLoadObservation(XybridBolt.XybridModelLoadObservation operation) : base(operation) { }

        /// <summary>Starts a model load under the explicit parent in <paramref name="options"/>, if any.</summary>
        /// <exception cref="XybridException">Thrown for an invalid explicit parent.</exception>
        public static ModelLoadObservation Start(XybridBolt.XybridModelLoadInfo info, ObservationOptions options = null) =>
            new ModelLoadObservation(Translated(() => new XybridBolt.XybridModelLoadObservation(info, options?.ToBolt())));

        /// <summary>Context for child operations, including valid unsampled spans.</summary>
        public XybridBolt.XybridObservationContext? Context => Read(op => op.Context());
        /// <summary>Reference to this operation's own span.</summary>
        public XybridBolt.XybridObservationRef? Reference => Read(op => op.ObservationRef());
        /// <summary>Whether any terminal transition has happened.</summary>
        public bool IsFinished => Read(op => op.IsFinished());
        /// <summary>Ends successfully.</summary>
        public void Complete() => Write(op => op.Complete());
        /// <summary>Ends with an error type and an optional governed message.</summary>
        public void Fail(string errorType, string message = null) => Write(op => op.Fail(errorType, message));
        /// <summary>Ends as cancelled without flushing.</summary>
        public void Cancel() => Write(op => op.Cancel());
        /// <summary>Records whether the load was served from cache.</summary>
        public void RecordCacheHit(bool hit) => Write(op => op.RecordCacheHit(hit));
    }
}
