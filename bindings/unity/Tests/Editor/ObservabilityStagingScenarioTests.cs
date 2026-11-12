// Xybrid SDK - Observability staging scenario (EditMode).
// Runs tools/scripts/observability_staging/README.md through the public C#
// API with a real registry model, exporting to Xybrid staging. Skips unless
// given a config.

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using NUnit.Framework;
using Xybrid;

namespace Xybrid.Tests.Editor
{
    /// <summary>
    /// The step 6 staging scenario, run in the Unity Editor through the public
    /// C# API.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Implements <c>tools/scripts/observability_staging/README.md</c> after the
    /// Python reference (<c>bindings/python/tests/observability_staging.py</c>):
    /// two sessions in one Editor process, <c>direct</c> (the Xybrid preset,
    /// metadata only) and <c>collector</c> (OTLP to the pinned Collector,
    /// content captured with one literal redaction), each loading a real
    /// registry model under a remote parent, stopping one stream after three
    /// chunks, streaming a second to the end, and recording feedback and an
    /// evaluation on the run reference. <c>run_staging.py</c> then reads every
    /// row back from staging and checks it against <c>expectations.json</c>.
    /// </para>
    /// <para>
    /// The test runs only when it is given the config and expectations paths,
    /// and is ignored otherwise, so an ordinary EditMode run is unaffected.
    /// Each comes from an environment variable (<c>XYBRID_STAGING_CONFIG</c>,
    /// <c>XYBRID_STAGING_EXPECTATIONS</c>, which <c>run_staging.py</c> sets)
    /// unless the matching Editor argument (<c>-xybridStagingConfig</c>,
    /// <c>-xybridStagingExpectations</c>) is given. The config holds a staging
    /// API key: the test deletes it as soon as it is read and never logs or
    /// writes the key. Select the test with
    /// <c>-testCategory ObservabilityStagingScenario</c>.
    /// </para>
    /// <para>
    /// The streams use the generated pull API (<c>RunStream</c>,
    /// <c>StreamNext</c>, <c>StreamClose</c>): closing a stream before its
    /// terminal event is what cancels the native run, and
    /// <c>Model.RunStreaming</c> drains its stream to the end even after its
    /// cancellation token fires.
    /// </para>
    /// </remarks>
    [TestFixture]
    [Category(CategoryName)]
    public class ObservabilityStagingScenarioTests
    {
        /// <summary>The NUnit category that selects this test.</summary>
        public const string CategoryName = "ObservabilityStagingScenario";

        // The driver gives a lane 30 minutes. The first launch downloads the
        // model, which the test runner's 3-minute default would fail.
        private const int TimeoutMs = 30 * 60 * 1000;

        private bool _started;

        [TearDown]
        public void ShutDownObservability()
        {
            // The Editor process outlives the test: never leave it exporting,
            // even when the scenario failed.
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
        [Timeout(TimeoutMs)]
        public void StagingScenario_ExportsWhatItReports()
        {
            string config = Setting("XYBRID_STAGING_CONFIG", "-xybridStagingConfig");
            string expectations = Setting("XYBRID_STAGING_EXPECTATIONS", "-xybridStagingExpectations");
            if (config == null || expectations == null)
            {
                Assert.Ignore(
                    "Set XYBRID_STAGING_CONFIG and XYBRID_STAGING_EXPECTATIONS (or pass "
                    + "-xybridStagingConfig and -xybridStagingExpectations) to run the "
                    + "observability staging scenario.");
            }

            var scenario = StagingScenario.Read(config);
            _started = true;
            string error = scenario.Run();
            scenario.Write(expectations);
            UnityEngine.Debug.Log("[XybridStaging] " + scenario.Summary());
            if (error != null)
            {
                Assert.Fail("The staging scenario stopped: " + error);
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
    /// Steps 1 to 7 of both sessions, recording what the binding reported.
    /// </summary>
    internal sealed class StagingScenario
    {
        private const string Schema = "xybrid-staging-scenario/1";
        private static readonly string[] Sessions = { "direct", "collector" };
        private static readonly Encoding Utf8 = new UTF8Encoding(false);

        private readonly TimeSpan _timeout;
        private readonly Dictionary<string, object> _expectations;
        private readonly Dictionary<string, object> _sessions = new Dictionary<string, object>();
        private Config _config;

        private StagingScenario(Config config)
        {
            _config = config;
            _timeout = TimeSpan.FromMilliseconds(config.constants.timeout_ms);
            _expectations = new Dictionary<string, object>
            {
                ["schema"] = Schema,
                ["binding"] = config.binding,
                ["lane"] = config.lane,
                ["run"] = config.run,
                ["sdk_version"] = null,
                ["model_id"] = config.model_id,
                ["device"] = new Dictionary<string, object>
                {
                    ["system"] = UnityEngine.SystemInfo.operatingSystem,
                    ["machine"] = RuntimeInformation.ProcessArchitecture.ToString(),
                    ["unity"] = UnityEngine.Application.unityVersion,
                    ["runtime"] = RuntimeInformation.FrameworkDescription,
                },
                ["sessions"] = _sessions,
                ["scenario_error"] = null,
            };
        }

        /// <summary>Reads the config and deletes it at once: it holds the staging key.</summary>
        internal static StagingScenario Read(string path)
        {
            string text = File.ReadAllText(path, Utf8);
            File.Delete(path);
            return new StagingScenario(UnityEngine.JsonUtility.FromJson<Config>(text));
        }

        /// <summary>Runs both sessions; returns null, or the error that stopped them.</summary>
        internal string Run()
        {
            try
            {
                _expectations["sdk_version"] = XybridClient.Version;
                // The documented startup call, as an app makes it. No API key:
                // the staging key only authorizes the direct session's export.
                XybridClient.Initialize();
                foreach (string session in Sessions)
                {
                    var record = new Dictionary<string, object>();
                    _sessions[session] = record;
                    RunSession(session, record);
                }
                return null;
            }
            catch (Exception error)
            {
                string text = error.ToString();
                _expectations["scenario_error"] = text;
                try
                {
                    XybridClient.ShutdownObservability(_timeout);
                }
                catch (Exception)
                {
                    // The error above is the one to report.
                }
                return text;
            }
            finally
            {
                _config = null; // nothing reads the key again
            }
        }

        /// <summary>Writes <c>expectations.json</c>, also after a failed step.</summary>
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

        /// <summary>The references and outcomes each session reported, for the lane log.</summary>
        internal string Summary()
        {
            var summary = new Dictionary<string, object>();
            foreach (KeyValuePair<string, object> session in _sessions)
            {
                var values = (Dictionary<string, object>)session.Value;
                var reported = new Dictionary<string, object>();
                foreach (string key in new[] { "references", "stopped", "completed", "quality" })
                {
                    reported[key] = values.TryGetValue(key, out object value) ? value : null;
                }
                summary[session.Key] = reported;
            }
            return Json.Write(summary);
        }

        private void RunSession(string session, Dictionary<string, object> record)
        {
            Constants constants = _config.constants;
            SessionValues values = session == "direct" ? _config.sessions.direct : _config.sessions.collector;
            string conversation = constants.per_lane.conversation_id
                .Replace("{run}", _config.run)
                .Replace("{session}", session);
            var references = new Dictionary<string, object>
            {
                ["workflow"] = null,
                ["load"] = null,
                ["run"] = null,
            };
            record["trace_id"] = values.trace_id;
            record["parent_span_id"] = values.parent_span_id;
            record["references"] = references;
            record["init"] = Outcome(() => Configure(session, values));
            if ((string)record["init"] != "ok")
            {
                throw new InvalidOperationException(session + " init failed: " + record["init"]);
            }

            // Step 2: the workflow continues the driver's trace T under span P.
            var parent = new ObservationOptions
            {
                Traceparent = "00-" + values.trace_id + "-" + values.parent_span_id + "-01",
            };
            using (var workflow = WorkflowObservation.Start(constants.workflow_name, parent))
            {
                references["workflow"] = Reference(workflow.Reference);
                XybridBolt.XybridObservationContext context = workflow.Context
                    ?? throw new InvalidOperationException("the started workflow has no context");

                // Step 3: a real registry model, loaded under the workflow. This
                // is the call ModelLoader.FromRegistry(id).Load(o) makes; the
                // Model it returns has no stream that can be closed early.
                using (var model = XybridBolt.XybridModel.Load(
                    new XybridBolt.XybridModelSource.Registry(_config.model_id, false),
                    new XybridBolt.XybridObservationOptions(
                        context.Traceparent, context.Tracestate, conversation, false)))
                {
                    references["load"] = Reference(model.LoadObservationRef());

                    // Step 4: stopped after three chunks by closing the stream.
                    StreamConstants stopped = constants.stopped;
                    Streamed result = Stream(model, stopped.prompt,
                        RunOptions(stopped.max_tokens, context, conversation), stopped.stop_after_chunks);
                    record["stopped"] = new Dictionary<string, object>
                    {
                        ["outcome"] = result.Outcome,
                        ["chunks_consumed"] = result.Chunks,
                        ["delivery"] = "pull",
                    };

                    // Step 5: streamed to the end; the result carries the run reference.
                    StreamConstants completed = constants.completed;
                    result = Stream(model, completed.prompt,
                        RunOptions(completed.max_tokens, context, conversation), null);
                    XybridBolt.XybridResult final = result.Final
                        ?? throw new InvalidOperationException("the completed stream returned no result");
                    string text = final.Envelope.Kind is XybridBolt.XybridEnvelopeKind.Text output
                        ? output.Value
                        : null;
                    record["completed"] = new Dictionary<string, object>
                    {
                        ["outcome"] = result.Outcome,
                        ["chunks"] = result.Chunks,
                        ["tokens_out"] = (ulong?)final.Metrics.TokensOut,
                        ["text_chars"] = text?.Length ?? 0,
                        ["finish_reason"] = result.FinishReason,
                    };
                    references["run"] = Reference(final.Observation);

                    // Step 6: quality results on the run reference.
                    XybridBolt.XybridObservationRef run = final.Observation
                        ?? throw new InvalidOperationException("the completed run has no reference");
                    FeedbackConstants feedback = constants.feedback;
                    EvaluationConstants evaluation = constants.evaluation;
                    record["quality"] = new Dictionary<string, object>
                    {
                        ["feedback"] = XybridClient.RecordFeedback(
                            run, feedback.score, feedback.label, feedback.comment),
                        ["evaluation"] = XybridClient.RecordEvaluation(
                            run, evaluation.name, evaluation.score_value, evaluation.score_label,
                            evaluation.explanation),
                    };
                }

                // Step 7: lifecycle (the model is released above).
                workflow.Complete();
            }
            var lifecycle = new Dictionary<string, object>();
            var stats = new Dictionary<string, object>();
            record["lifecycle"] = lifecycle;
            record["stats"] = stats;
            lifecycle["flush"] = Outcome(() => XybridClient.FlushObservability(_timeout));
            stats["after_flush"] = Snapshot();
            lifecycle["shutdown"] = Outcome(() => XybridClient.ShutdownObservability(_timeout));
            stats["after_shutdown"] = Snapshot();
        }

        /// <summary>Step 1: the session's export, identity and content rules, then init.</summary>
        private void Configure(string session, SessionValues values)
        {
            Constants constants = _config.constants;
            using (ObservabilityConfig observability = session == "direct"
                ? ObservabilityConfig.Xybrid(values.api_key)
                : ObservabilityConfig.Otlp(values.endpoint))
            {
                if (session == "direct")
                {
                    observability.WithTracesEndpoint(values.traces_endpoint).WithMetadataOnly();
                }
                else
                {
                    observability.WithLiteralRedaction(constants.secret, constants.replacement);
                }
                observability
                    .WithService(constants.per_lane.service_name.Replace("{binding}", _config.binding),
                        constants.per_lane.service_version)
                    .WithDeviceId(constants.per_lane.device_id.Replace("{lane}", _config.lane))
                    .WithResourceAttribute("xybrid.acceptance.run", _config.run)
                    .WithResourceAttribute("xybrid.acceptance.lane", _config.lane)
                    .WithResourceAttribute("xybrid.acceptance.commit", _config.commit)
                    .WithResourceAttribute("xybrid.acceptance.session", session);
                XybridClient.InitializeObservability(observability);
            }
        }

        /// <summary>
        /// Steps 4 and 5: a pull stream, closed after <paramref name="stopAfter"/>
        /// chunks or read to the end.
        /// </summary>
        private static Streamed Stream(
            XybridBolt.XybridModel model, string prompt, XybridBolt.XybridRunOptions options, int? stopAfter)
        {
            var envelope = new XybridBolt.XybridEnvelope(
                new XybridBolt.XybridEnvelopeKind.Text(prompt), Array.Empty<XybridBolt.XybridMetadataEntry>());
            var streamed = new Streamed();
            // RunStream requires a stop button; closing the stream is the stop.
            using (var cancel = new XybridBolt.XybridCancellationToken())
            {
                ulong? streamId = model.RunStream(envelope, options, cancel);
                try
                {
                    while (true)
                    {
                        XybridBolt.XybridStreamEvent item = model.StreamNext(streamId.Value);
                        if (item.Kind == XybridBolt.XybridStreamEventKind.Complete)
                        {
                            streamed.Final = model.StreamResult(streamId.Value);
                            streamId = null; // taking the result released the stream
                            streamed.Outcome = stopAfter.HasValue ? "completed" : "ok";
                            return streamed;
                        }
                        if (item.Kind == XybridBolt.XybridStreamEventKind.Token && item.Token.HasValue)
                        {
                            streamed.Chunks++;
                            string reason = item.Token.Value.FinishReason;
                            if (!string.IsNullOrEmpty(reason))
                            {
                                streamed.FinishReason = reason;
                            }
                            if (stopAfter.HasValue && streamed.Chunks >= stopAfter.Value)
                            {
                                // Closing before the terminal event cancels the run at
                                // its next token (a user cancellation, error.type cancelled).
                                model.StreamClose(streamId.Value);
                                streamId = null;
                                streamed.Outcome = "cancelled";
                                return streamed;
                            }
                        }
                    }
                }
                finally
                {
                    if (streamId.HasValue)
                    {
                        model.StreamClose(streamId.Value);
                    }
                }
            }
        }

        private static XybridBolt.XybridRunOptions RunOptions(
            int maxTokens, XybridBolt.XybridObservationContext context, string conversationId)
        {
            var generation = new XybridBolt.XybridGenerationConfig(
                (uint)maxTokens, null, null, null, null, null,
                Array.Empty<string>(), null, Array.Empty<XybridBolt.XybridToolDefinition>());
            return new XybridBolt.XybridRunOptions(
                generation, Array.Empty<XybridBolt.XybridAbortSignal>(), false, 0u, null,
                context.Traceparent, context.Tracestate, conversationId, false, null);
        }

        private static string Outcome(Action call)
        {
            try
            {
                call();
            }
            catch (XybridException error)
            {
                return "error: " + error.GetType().Name + ": " + error.Message;
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

        /// <summary>What one stream delivered.</summary>
        private sealed class Streamed
        {
            internal string Outcome;
            internal int Chunks;
            internal string FinishReason;
            internal XybridBolt.XybridResult? Final;
        }

        // config.json, as run_staging.py writes it. JsonUtility fills public
        // fields by name, so these keep the file's snake_case keys.

        [Serializable]
        private sealed class Config
        {
            public string binding;
            public string lane;
            public string run;
            public string commit;
            public string model_id;
            public SessionPair sessions;
            public Constants constants;
        }

        [Serializable]
        private sealed class SessionPair
        {
            public SessionValues direct;
            public SessionValues collector;
        }

        [Serializable]
        private sealed class SessionValues
        {
            public string traces_endpoint;
            public string api_key;
            public string endpoint;
            public string trace_id;
            public string parent_span_id;
        }

        [Serializable]
        private sealed class Constants
        {
            public PerLane per_lane;
            public string workflow_name;
            public StreamConstants stopped;
            public StreamConstants completed;
            public string secret;
            public string replacement;
            public FeedbackConstants feedback;
            public EvaluationConstants evaluation;
            public int timeout_ms;
        }

        [Serializable]
        private sealed class PerLane
        {
            public string service_name;
            public string service_version;
            public string device_id;
            public string conversation_id;
        }

        [Serializable]
        private sealed class StreamConstants
        {
            public string prompt;
            public int max_tokens;
            public int stop_after_chunks;
        }

        [Serializable]
        private sealed class FeedbackConstants
        {
            public double score;
            public string label;
            public string comment;
        }

        [Serializable]
        private sealed class EvaluationConstants
        {
            public string name;
            public double score_value;
            public string score_label;
            public string explanation;
        }
    }
}
