// Xybrid SDK - XybridClient observability lifecycle EditMode tests.
// Require the native xybrid library to be loadable (DllImport "xybrid").

using System;
using NUnit.Framework;
using Xybrid;

namespace Xybrid.Tests.Editor
{
    /// <summary>
    /// EditMode tests for the observability lifecycle methods on <see cref="XybridClient"/>.
    /// </summary>
    /// <remarks>
    /// These tests exercise process-wide state (the installed exporter), so every
    /// test leaves observability shut down. Exports target an unroutable local
    /// collector; nothing leaves the machine. They depend on the native
    /// <c>xybrid</c> library being present on the Unity Editor's load path.
    /// </remarks>
    [TestFixture]
    public class XybridClientObservabilityTests
    {
        private const string TestCollector = "http://127.0.0.1:9";
        private const string TestApiKey = "unit-test-api-key-SECRET-DO-NOT-LOG";
        private static readonly TimeSpan ShortTimeout = TimeSpan.FromMilliseconds(200);

        // A syntactically valid trace/span id pair (W3C trace-context example).
        private static readonly XybridBolt.XybridObservationRef ExampleRef =
            new XybridBolt.XybridObservationRef("0af7651916cd43dd8448eb211c80319c", "b7ad6b7169203331");

        [TearDown]
        public void CleanupObservability()
        {
            // Idempotent; no-op if no test initialized observability. A slow
            // shutdown against the unroutable collector is not a test failure.
            try
            {
                XybridClient.ShutdownObservability(ShortTimeout);
            }
            catch (XybridException)
            {
            }
        }

        [Test]
        public void Initialize_DoesNotEnableExport()
        {
            // No API key: Initialize is first-call-wins, so a fake key would
            // stick to the Editor session and reach cloud routing.
            XybridClient.ShutdownObservability(ShortTimeout);
            XybridClient.Initialize();

            Assert.IsFalse(XybridClient.IsObservabilityEnabled);
            Assert.AreEqual(0UL, XybridClient.ObservabilityStats().SpansExported);
        }

        [Test]
        public void FlushAndShutdown_WithoutInit_AreNoOps()
        {
            XybridClient.ShutdownObservability(ShortTimeout);

            Assert.DoesNotThrow(() => XybridClient.FlushObservability(ShortTimeout));
            Assert.DoesNotThrow(() => XybridClient.ShutdownObservability(ShortTimeout));
            Assert.DoesNotThrow(() => XybridClient.ShutdownObservability());
        }

        [Test]
        public void ObservabilityLifecycle_InitShutdown_TogglesEnabled()
        {
            var config = ObservabilityConfig.Otlp(TestCollector).WithService("unit-test", "1.0");

            XybridClient.InitializeObservability(config);
            Assert.IsTrue(config.IsDisposed, "InitializeObservability takes ownership of the config");
            Assert.IsTrue(XybridClient.IsObservabilityEnabled);

            Assert.DoesNotThrow(() => XybridClient.ShutdownObservability(ShortTimeout));
            Assert.IsFalse(XybridClient.IsObservabilityEnabled);
        }

        [Test]
        public void InitializeObservability_ReplacesThePreviousConfig()
        {
            XybridClient.InitializeObservability(ObservabilityConfig.Otlp(TestCollector));
            Assert.DoesNotThrow(() =>
                XybridClient.InitializeObservability(ObservabilityConfig.Xybrid(TestApiKey)));
            Assert.IsTrue(XybridClient.IsObservabilityEnabled);
        }

        [Test]
        public void InitializeObservability_ConsumedConfig_Throws()
        {
            var config = ObservabilityConfig.Otlp(TestCollector);
            XybridClient.InitializeObservability(config);

            Assert.Throws<ObjectDisposedException>(() => XybridClient.InitializeObservability(config));
        }

        [Test]
        public void InitializeObservability_NullConfig_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => XybridClient.InitializeObservability(null));
        }

        [Test]
        public void Flush_NegativeTimeout_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(
                () => XybridClient.FlushObservability(TimeSpan.FromSeconds(-1)));
        }

        [Test]
        public void Feedback_WithXybridPreset_ReturnsFalse()
        {
            // The Xybrid preset sends traces only, so there is no logs signal
            // to carry feedback or evaluations.
            XybridClient.InitializeObservability(ObservabilityConfig.Xybrid(TestApiKey));

            Assert.IsFalse(XybridClient.RecordFeedback(ExampleRef, score: 1.0, label: "helpful"));
            Assert.IsFalse(XybridClient.RecordEvaluation(ExampleRef, "relevance", scoreValue: 0.9));
        }

        [Test]
        public void Feedback_InvalidTarget_Throws()
        {
            var invalid = new XybridBolt.XybridObservationRef("not-a-trace-id", "nope");

            Assert.Throws<XybridException>(() => XybridClient.RecordFeedback(invalid, score: 1.0));
        }

        [Test]
        public void Evaluation_RequiresAName()
        {
            Assert.Throws<ArgumentException>(() => XybridClient.RecordEvaluation(ExampleRef, " "));
        }
    }
}
