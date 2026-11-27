// Xybrid SDK - Observation options and manual observation EditMode tests.
// The manual-handle tests require the native xybrid library (DllImport "xybrid").

using System;
using NUnit.Framework;
using Xybrid;

namespace Xybrid.Tests.Editor
{
    /// <summary>
    /// EditMode tests for <see cref="ObservationOptions"/> plumbing (pipeline
    /// runs, live ASR configs) and the <see cref="ManualObservation{T}"/> wrappers.
    /// </summary>
    [TestFixture]
    public class ObservationTests
    {
        private const string Parent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

        private static ObservationOptions Options() => new ObservationOptions
        {
            Traceparent = Parent,
            Tracestate = "vendor=1",
            ConversationId = "chat-42",
            SuppressTrace = true,
        };

        [Test]
        public void RunOptions_AreNullWithoutConfigOrObservation()
        {
            Assert.IsNull(ObservationOptions.ToRunOptions(null, null));
        }

        [Test]
        public void PipelineRunOptions_CarryOnlyTheObservationControls()
        {
            // Pipeline.Run passes no generation config: pipelines reject one.
            var options = ObservationOptions.ToRunOptions(null, Options()).Value;

            Assert.IsNull(options.GenerationConfig);
            Assert.AreEqual(0, options.AbortOn.Length);
            Assert.AreEqual(Parent, options.Traceparent);
            Assert.AreEqual("vendor=1", options.Tracestate);
            Assert.AreEqual("chat-42", options.ConversationId);
            Assert.IsTrue(options.SuppressTrace);
            Assert.IsNull(options.FrameSessionId);
        }

        [Test]
        public void RunOptions_CarryTheFrameSession()
        {
            // A live-capture loop names its session on every run.
            var observation = new ObservationOptions { FrameSessionId = "camera-7" };
            var options = ObservationOptions.ToRunOptions(null, observation).Value;

            Assert.AreEqual("camera-7", options.FrameSessionId);
            Assert.IsFalse(options.SuppressTrace);
            Assert.IsNull(options.Traceparent);
        }

        [Test]
        public void StreamingConfig_WithObservation_ReplacesOnlyTheObservation()
        {
            var config = StreamingConfigs.WithObservation(StreamingConfigs.Default, Options());

            Assert.AreEqual(StreamingConfigs.RequiredSampleRate, config.SampleRate);
            Assert.IsTrue(config.Observation.HasValue);
            Assert.AreEqual(Parent, config.Observation.Value.Traceparent);
            Assert.AreEqual("chat-42", config.Observation.Value.ConversationId);
            Assert.IsTrue(config.Observation.Value.SuppressTrace);

            var unchanged = StreamingConfigs.WithObservation(config, null);
            Assert.AreEqual(Parent, unchanged.Observation.Value.Traceparent);
        }

        [Test]
        public void VoiceActivityConfig_CarriesObservation()
        {
            var config = StreamingConfigs.VoiceActivity("/vad", "en", 0.4f, Options());

            Assert.AreEqual("en", config.Language);
            Assert.AreEqual(Parent, config.Observation.Value.Traceparent);
            Assert.IsFalse(StreamingConfigs.VoiceActivity("/vad").Observation.HasValue);
        }

        [Test]
        public void ManualWorkflow_FirstTerminalWins_AndDisposeReleases()
        {
            var workflow = WorkflowObservation.Start("unity-test");
            Assert.IsFalse(workflow.IsFinished);

            workflow.Complete();
            workflow.Cancel(); // ignored: the first terminal call wins
            Assert.IsTrue(workflow.IsFinished);

            workflow.Dispose();
            workflow.Dispose(); // idempotent
            Assert.Throws<ObjectDisposedException>(() => workflow.Complete());
        }

        [Test]
        public void ManualOperations_RejectAnInvalidExplicitParent()
        {
            var invalid = new ObservationOptions { Traceparent = "invalid" };

            Assert.Throws<XybridException>(() => WorkflowObservation.Start("unity-test", invalid));
            Assert.Throws<XybridException>(() => ToolObservation.Start(
                new XybridBolt.XybridToolCallInfo("lookup", null, null, null, null), invalid));
            Assert.Throws<XybridException>(() => ModelLoadObservation.Start(
                new XybridBolt.XybridModelLoadInfo("model", null, null, null, null), invalid));
        }

        [Test]
        public void ManualTool_RejectsInvalidResultJson()
        {
            using (var tool = ToolObservation.Start(
                       new XybridBolt.XybridToolCallInfo("lookup", null, null, null, null)))
            {
                Assert.Throws<XybridException>(() => tool.RecordResult("not json"));
                tool.Complete();
                Assert.IsTrue(tool.IsFinished);
            }
        }
    }
}
