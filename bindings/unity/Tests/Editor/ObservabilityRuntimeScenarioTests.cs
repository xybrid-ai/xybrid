// Xybrid SDK - Observability runtime scenario (EditMode).
// Runs tools/scripts/observability_capture/scenario.md through the public C#
// API against capture_server.py. Skips unless pointed at a receiver.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using NUnit.Framework;
using Xybrid;

namespace Xybrid.Tests.Editor
{
    /// <summary>
    /// The shared observability runtime scenario, run in the Unity Editor
    /// through the public C# API.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Implements <c>tools/scripts/observability_capture/scenario.md</c>: every
    /// binding performs the same steps against <c>capture_server.py</c> and
    /// writes the same <c>expectations.json</c>, which <c>check_capture.py</c>
    /// then compares with what the receiver recorded. No model weights are
    /// involved.
    /// </para>
    /// <para>
    /// The test runs only when it is given a receiver endpoint and an
    /// expectations path, and is ignored otherwise, so an ordinary EditMode run
    /// is unaffected. Each setting comes from an environment variable
    /// (<c>XYBRID_SCENARIO_ENDPOINT</c>, <c>XYBRID_SCENARIO_EXPECTATIONS</c>,
    /// <c>XYBRID_SCENARIO_WORKDIR</c>, which <c>run_scenario.py</c> sets) unless
    /// the matching Editor command-line argument (<c>-xybridScenarioEndpoint</c>,
    /// <c>-xybridScenarioExpectations</c>, <c>-xybridScenarioWorkdir</c>) is
    /// given. The arguments exist for launchers that do not forward the
    /// environment, such as game-ci's Editor container in
    /// <c>.github/workflows/unity-editor.yml</c>. Select the test with
    /// <c>-testCategory ObservabilityRuntimeScenario</c>.
    /// </para>
    /// <para>
    /// The scenario constants are written out below; the checker compares every
    /// exported value with <c>scenario.json</c>, so a drifted constant fails the
    /// lane rather than passing silently.
    /// </para>
    /// </remarks>
    [TestFixture]
    [Category(CategoryName)]
    public class ObservabilityRuntimeScenarioTests
    {
        /// <summary>The NUnit category that selects this test.</summary>
        public const string CategoryName = "ObservabilityRuntimeScenario";

        private bool _started;

        [TearDown]
        public void ShutDownObservability()
        {
            // The Editor process outlives the test: never leave it exporting to
            // a receiver that is about to stop, even when the scenario failed.
            if (!_started)
            {
                return;
            }
            try
            {
                XybridClient.ShutdownObservability(TimeSpan.FromSeconds(2));
            }
            catch (Exception)
            {
                // Best effort: the test's own failure, if any, is the one to report.
            }
        }

        [Test]
        public void SharedScenario_ExportsWhatItReports()
        {
            string endpoint = Setting("XYBRID_SCENARIO_ENDPOINT", "-xybridScenarioEndpoint");
            string expectations = Setting("XYBRID_SCENARIO_EXPECTATIONS", "-xybridScenarioExpectations");
            if (endpoint == null || expectations == null)
            {
                Assert.Ignore(
                    "Set XYBRID_SCENARIO_ENDPOINT and XYBRID_SCENARIO_EXPECTATIONS (or pass "
                    + "-xybridScenarioEndpoint and -xybridScenarioExpectations) to run the "
                    + "observability runtime scenario.");
            }
            string workdir = Setting("XYBRID_SCENARIO_WORKDIR", "-xybridScenarioWorkdir")
                ?? Path.Combine(Path.GetDirectoryName(Path.GetFullPath(expectations)), "work");

            _started = true;
            var scenario = new ObservabilityScenario(endpoint, workdir);
            string error = scenario.Run();
            scenario.Write(expectations);
            if (error != null)
            {
                Assert.Fail("The observability scenario stopped: " + error);
            }
        }

        /// <summary>The value after an Editor argument, else an environment variable.</summary>
        private static string Setting(string variable, string argument)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; i++)
            {
                if (string.Equals(args[i], argument, StringComparison.OrdinalIgnoreCase)
                    && !string.IsNullOrEmpty(args[i + 1]))
                {
                    return args[i + 1];
                }
            }
            string value = Environment.GetEnvironmentVariable(variable);
            return string.IsNullOrEmpty(value) ? null : value;
        }
    }

    /// <summary>
    /// Steps (a) to (f) of the scenario, recording what the binding reported.
    /// </summary>
    internal sealed class ObservabilityScenario
    {
        internal const string Binding = "unity";
        internal const string Schema = "xybrid-observability-scenario/1";

        // scenario.json, per binding.
        private const string ServiceName = "xybrid-scenario-" + Binding;
        private const string ServiceVersion = "1.0.0";
        private const string DeviceId = "scenario-device-" + Binding;
        private const string ConversationId = "scenario-conversation-" + Binding;

        // scenario.json, shared.
        private const string Secret = "xybrid-scenario-secret-4c1d";
        private const string Replacement = "[scenario-redacted]";
        private const string OmittedAttribute = "gen_ai.system_instructions";
        private const string WorkflowName = "scenario-workflow";
        private const string AfterShutdownWorkflowName = "scenario-after-shutdown";
        private const string ToolName = "scenario_lookup";
        private const string ToolCallId = "call-scenario-1";
        private const string ToolType = "function";
        private const string ToolArgumentsJson = "{\"query\":\"weather\",\"token\":\"xybrid-scenario-secret-4c1d\"}";
        private const string ToolResultJson = "{\"answer\":\"sunny\",\"echo\":\"xybrid-scenario-secret-4c1d\"}";
        private const string SystemInstructionsJson =
            "[{\"type\":\"text\",\"content\":\"xybrid-scenario-omitted-9b2e\"}]";
        private const string InputMessagesJson =
            "[{\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"content\":\"my token is xybrid-scenario-secret-4c1d\"}]}]";
        private const string OutputMessagesJson =
            "[{\"role\":\"assistant\",\"parts\":[{\"type\":\"text\",\"content\":\"I will not repeat xybrid-scenario-secret-4c1d\"}],\"finish_reason\":\"stop\"}]";
        private const string FailedErrorType = "scenario_error";
        private const string FailedMessage = "upstream rejected xybrid-scenario-secret-4c1d";
        private const string ReleasedInputMessagesJson =
            "[{\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"content\":\"abandoned request\"}]}]";
        private const string ModelDirectoryName = "scenario-weightless";
        private const string ModelMetadataJson =
            "{\"model_id\":\"scenario-weightless\",\"version\":\"1.0\",\"execution_template\":{\"type\":\"Onnx\",\"model_file\":\"model.onnx\"},\"files\":[]}";
        private const string RunInputText = "run input xybrid-scenario-secret-4c1d";
        private static readonly TimeSpan Timeout = TimeSpan.FromMilliseconds(5000);

        private static readonly Encoding Utf8 = new UTF8Encoding(false);

        private readonly string _endpoint;
        private readonly string _workdir;
        private readonly Dictionary<string, object> _expectations;
        private readonly Dictionary<string, object> _references;
        private readonly Dictionary<string, object> _automatic = new Dictionary<string, object>();
        private readonly Dictionary<string, object> _stats = new Dictionary<string, object>();
        private readonly Dictionary<string, object> _lifecycle = new Dictionary<string, object>();

        internal ObservabilityScenario(string endpoint, string workdir)
        {
            _endpoint = endpoint;
            _workdir = workdir;
            _references = new Dictionary<string, object>
            {
                ["workflow"] = null,
                ["tool"] = null,
                ["inference"] = null,
                ["inference_failed"] = null,
                ["inference_released"] = null,
                ["load"] = null,
                ["run"] = null,
            };
            _expectations = new Dictionary<string, object>
            {
                ["schema"] = Schema,
                ["binding"] = Binding,
                ["sdk_version"] = null,
                ["trace_id"] = TraceId,
                ["parent_span_id"] = ParentSpanId,
                ["references"] = _references,
                ["automatic"] = _automatic,
                ["stats"] = _stats,
                ["lifecycle"] = _lifecycle,
                ["scenario_error"] = null,
            };
        }

        /// <summary>T: hex of the UTF-8 <c>"xy:" + binding</c>, right-padded with zeros (scenario.md, Identifiers).</summary>
        internal static string TraceId => Hex("xy:" + Binding, 32);

        /// <summary>P: hex of the UTF-8 binding name, right-padded with zeros and cut to 16 digits.</summary>
        internal static string ParentSpanId => Hex(Binding, 16);

        /// <summary>Runs steps (a) to (e); returns null, or the error that stopped them.</summary>
        internal string Run()
        {
            try
            {
                RunSteps();
                return null;
            }
            catch (Exception error)
            {
                string text = error.ToString();
                _expectations["scenario_error"] = text;
                try
                {
                    XybridClient.ShutdownObservability(Timeout);
                }
                catch (Exception)
                {
                    // The error above is the one to report.
                }
                return text;
            }
        }

        /// <summary>Step (f): writes <c>expectations.json</c>, also after a failed step.</summary>
        internal void Write(string path)
        {
            string full = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(full));
            string temporary = full + ".tmp";
            File.WriteAllText(temporary, Json.Write(_expectations) + "\n", Utf8);
            if (File.Exists(full))
            {
                File.Delete(full);
            }
            File.Move(temporary, full);
        }

        private void RunSteps()
        {
            _expectations["sdk_version"] = XybridClient.Version;
            // The documented startup call, as an app makes it. No API key, so
            // nothing reaches cloud routing.
            XybridClient.Initialize();

            // (a) OTLP/JSON export to the receiver, with content rules and a device id.
            using (var config = ObservabilityConfig.Otlp(_endpoint))
            {
                config.WithJsonProtocol()
                    .WithService(ServiceName, ServiceVersion)
                    .WithDeviceId(DeviceId)
                    .WithLiteralRedaction(Secret, Replacement)
                    .WithoutContentAttribute(OmittedAttribute);
                XybridClient.InitializeObservability(config);
            }

            // (b) Parent-in: the workflow continues the caller's trace T under span P.
            var parent = new ObservationOptions { Traceparent = "00-" + TraceId + "-" + ParentSpanId + "-01" };
            using (var workflow = WorkflowObservation.Start(WorkflowName, parent))
            {
                _references["workflow"] = Reference(workflow.Reference);
                XybridBolt.XybridObservationContext context = workflow.Context
                    ?? throw new InvalidOperationException("the started workflow has no context");
                var child = new ObservationOptions
                {
                    Traceparent = context.Traceparent,
                    Tracestate = context.Tracestate,
                };

                // (c) Manual children: a tool, a completed inference, a failed
                // one and one released unfinished.
                RunManualChildren(child);

                // (d) The automatic path: a weightless model loads lazily and
                // fails to run, both under the workflow with a conversation id.
                RunAutomaticPath(new ObservationOptions
                {
                    Traceparent = context.Traceparent,
                    Tracestate = context.Tracestate,
                    ConversationId = ConversationId,
                });

                workflow.Complete();
            }

            // (e) Lifecycle: flush, stats, shutdown twice, then a no-op workflow.
            _lifecycle["flush"] = Outcome(() => XybridClient.FlushObservability(Timeout));
            _stats["after_flush"] = Snapshot();
            _lifecycle["shutdown"] = new List<object>
            {
                Outcome(() => XybridClient.ShutdownObservability(Timeout)),
                Outcome(() => XybridClient.ShutdownObservability(Timeout)),
            };
            _stats["after_shutdown"] = Snapshot();

            XybridBolt.XybridObservationRef? afterReference;
            XybridBolt.XybridObservationContext? afterContext;
            using (var after = WorkflowObservation.Start(AfterShutdownWorkflowName))
            {
                afterReference = after.Reference;
                afterContext = after.Context;
                after.Complete();
            }
            _lifecycle["after_shutdown"] = new Dictionary<string, object>
            {
                ["workflow_ref"] = Reference(afterReference),
                ["workflow_context"] = afterContext.HasValue ? afterContext.Value.Traceparent : null,
                ["flush"] = Outcome(() => XybridClient.FlushObservability(Timeout)),
            };
        }

        private void RunManualChildren(ObservationOptions child)
        {
            var call = new XybridBolt.XybridToolCallInfo(ToolName, ToolCallId, ToolType, null, ToolArgumentsJson);
            using (var tool = ToolObservation.Start(call, child))
            {
                tool.RecordResult(ToolResultJson);
                _references["tool"] = Reference(tool.Reference);
                tool.Complete();
            }

            var request = new XybridBolt.XybridInferenceRequest(
                "chat", "scenario-provider", "scenario-model", XybridBolt.XybridExecutionLocation.Local,
                "scenario-runtime", null, false, null, null, null, null, null);
            using (var inference = InferenceObservation.Start(request, child))
            {
                inference.RecordSystemInstructions(SystemInstructionsJson);
                inference.RecordInputMessages(InputMessagesJson);
                inference.RecordOutputMessages(OutputMessagesJson);
                _references["inference"] = Reference(inference.Reference);
                inference.Complete();
            }

            using (var failed = InferenceObservation.Start(request, child))
            {
                _references["inference_failed"] = Reference(failed.Reference);
                failed.Fail(FailedErrorType, FailedMessage);
            }

            // Disposing the last owner without a terminal call cancels it
            // synchronously and does not flush: traces.accepted grows by one
            // between the two snapshots.
            var released = InferenceObservation.Start(request, child);
            try
            {
                released.RecordInputMessages(ReleasedInputMessagesJson);
                _references["inference_released"] = Reference(released.Reference);
                _stats["before_release"] = Snapshot();
            }
            finally
            {
                released.Dispose();
            }
            _stats["after_release"] = Snapshot();
        }

        private void RunAutomaticPath(ObservationOptions observation)
        {
            string directory = Path.Combine(_workdir, ModelDirectoryName);
            Directory.CreateDirectory(directory);
            File.WriteAllText(Path.Combine(directory, "model_metadata.json"), ModelMetadataJson, Utf8);

            Model model;
            try
            {
                using (var loader = ModelLoader.FromDirectory(directory))
                {
                    model = loader.Load(observation);
                }
            }
            catch (XybridException error)
            {
                _automatic["load"] = Result("error", Describe(error));
                _automatic["run"] = Result("skipped", null);
                return;
            }

            using (model)
            {
                _automatic["load"] = Result("ok", null);
                _references["load"] = Reference(model.LoadObservation);
                using (var envelope = Envelope.Text(RunInputText))
                {
                    InferenceResult result;
                    try
                    {
                        result = model.Run(envelope, observation: observation);
                    }
                    catch (XybridException error)
                    {
                        _automatic["run"] = Result("error", Describe(error));
                        return;
                    }
                    using (result)
                    {
                        if (result.Success)
                        {
                            _automatic["run"] = Result("ok", null);
                            _references["run"] = Reference(result.Observation);
                        }
                        else
                        {
                            // Model.Run reports an ordinary failure on the
                            // result instead of throwing.
                            _automatic["run"] = Result("error", "InferenceResult.Error: " + result.Error);
                        }
                    }
                }
            }
        }

        private static Dictionary<string, object> Result(string outcome, string error) =>
            new Dictionary<string, object> { ["outcome"] = outcome, ["error"] = error };

        private static string Describe(Exception error) => error.GetType().Name + ": " + error.Message;

        private static string Outcome(Action call)
        {
            try
            {
                call();
            }
            catch (XybridException error)
            {
                return "error: " + Describe(error);
            }
            return "ok";
        }

        private static object Reference(XybridBolt.XybridObservationRef? reference)
        {
            if (!reference.HasValue)
            {
                return null;
            }
            return new Dictionary<string, object>
            {
                ["trace_id"] = reference.Value.TraceId,
                ["span_id"] = reference.Value.SpanId,
            };
        }

        private static Dictionary<string, object> Snapshot()
        {
            var stats = XybridClient.ObservabilityStats();
            return new Dictionary<string, object>
            {
                ["enabled"] = stats.Enabled,
                ["traces"] = Signal(stats.Traces),
                ["logs"] = Signal(stats.Logs),
                ["metrics"] = Signal(stats.Metrics),
            };
        }

        private static Dictionary<string, object> Signal(XybridBolt.XybridSignalStats signal) =>
            new Dictionary<string, object>
            {
                ["accepted"] = signal.Accepted,
                ["exported"] = signal.Exported,
                ["dropped"] = signal.Dropped,
                ["failed"] = signal.Failed,
                ["pending"] = signal.Pending,
            };

        private static string Hex(string text, int digits)
        {
            var hex = new StringBuilder();
            foreach (byte value in Encoding.UTF8.GetBytes(text))
            {
                hex.Append(value.ToString("x2", CultureInfo.InvariantCulture));
            }
            while (hex.Length < digits)
            {
                hex.Append('0');
            }
            return hex.ToString(0, digits);
        }
    }

    /// <summary>
    /// Minimal JSON writer for the expectations document: Unity's
    /// <c>JsonUtility</c> cannot serialize dictionaries or nulls.
    /// </summary>
    internal static class Json
    {
        internal static string Write(object value)
        {
            var output = new StringBuilder();
            Append(output, value, 0);
            return output.ToString();
        }

        private static void Append(StringBuilder output, object value, int depth)
        {
            switch (value)
            {
                case null:
                    output.Append("null");
                    break;
                case string text:
                    AppendString(output, text);
                    break;
                case bool flag:
                    output.Append(flag ? "true" : "false");
                    break;
                case ulong number:
                    output.Append(number.ToString(CultureInfo.InvariantCulture));
                    break;
                case long number:
                    output.Append(number.ToString(CultureInfo.InvariantCulture));
                    break;
                case int number:
                    output.Append(number.ToString(CultureInfo.InvariantCulture));
                    break;
                case IDictionary<string, object> map:
                    AppendObject(output, map, depth);
                    break;
                case IList list:
                    AppendArray(output, list, depth);
                    break;
                default:
                    throw new ArgumentException("cannot write " + value.GetType() + " as JSON");
            }
        }

        private static void AppendObject(StringBuilder output, IDictionary<string, object> map, int depth)
        {
            if (map.Count == 0)
            {
                output.Append("{}");
                return;
            }
            var keys = new List<string>(map.Keys);
            keys.Sort(StringComparer.Ordinal);
            output.Append('{');
            for (int i = 0; i < keys.Count; i++)
            {
                output.Append(i == 0 ? "\n" : ",\n").Append(' ', 2 * (depth + 1));
                AppendString(output, keys[i]);
                output.Append(": ");
                Append(output, map[keys[i]], depth + 1);
            }
            output.Append('\n').Append(' ', 2 * depth).Append('}');
        }

        private static void AppendArray(StringBuilder output, IList list, int depth)
        {
            if (list.Count == 0)
            {
                output.Append("[]");
                return;
            }
            output.Append('[');
            for (int i = 0; i < list.Count; i++)
            {
                output.Append(i == 0 ? "\n" : ",\n").Append(' ', 2 * (depth + 1));
                Append(output, list[i], depth + 1);
            }
            output.Append('\n').Append(' ', 2 * depth).Append(']');
        }

        private static void AppendString(StringBuilder output, string text)
        {
            output.Append('"');
            foreach (char c in text)
            {
                switch (c)
                {
                    case '"': output.Append("\\\""); break;
                    case '\\': output.Append("\\\\"); break;
                    case '\n': output.Append("\\n"); break;
                    case '\r': output.Append("\\r"); break;
                    case '\t': output.Append("\\t"); break;
                    default:
                        if (c < 0x20)
                        {
                            output.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        }
                        else
                        {
                            output.Append(c);
                        }
                        break;
                }
            }
            output.Append('"');
        }
    }
}
