// Xybrid SDK - ObservabilityConfig EditMode Tests
// Requires the native xybrid library to be loadable (DllImport "xybrid").

using System;
using NUnit.Framework;
using Xybrid;

namespace Xybrid.Tests.Editor
{
    /// <summary>
    /// EditMode tests for <see cref="ObservabilityConfig"/>.
    /// </summary>
    /// <remarks>
    /// These tests exercise the managed wrapper. They depend on the native
    /// <c>xybrid</c> library being present on the Unity Editor's load path.
    /// </remarks>
    [TestFixture]
    public class ObservabilityConfigTests
    {
        private const string DefaultTracesEndpoint = "https://ingest.xybrid.dev/v1/traces";
        private const string OverrideEndpoint = "https://traces.example.test/v1/traces";
        private const string CollectorEndpoint = "http://localhost:4318";
        private const string TestApiKey = "unit-test-api-key-SECRET-DO-NOT-LOG";

        [Test]
        public void DefaultTracesEndpoint_IsTheXybridDashboard()
        {
            Assert.AreEqual(DefaultTracesEndpoint, ObservabilityConfig.DefaultTracesEndpoint);
        }

        [Test]
        public void Dispose_IsIdempotent()
        {
            var config = ObservabilityConfig.Xybrid(TestApiKey);
            Assert.IsFalse(config.IsDisposed);

            config.Dispose();
            Assert.IsTrue(config.IsDisposed);

            // Second Dispose must not throw or touch the already-freed handle.
            Assert.DoesNotThrow(() => config.Dispose());
            Assert.IsTrue(config.IsDisposed);
        }

        [Test]
        public void Setters_AfterDispose_Throw()
        {
            var config = ObservabilityConfig.Otlp(CollectorEndpoint);
            config.Dispose();

            Assert.Throws<ObjectDisposedException>(() => config.WithMetadataOnly());
        }

        [Test]
        public void ToString_DoesNotContainApiKeyOrHeaders()
        {
            using (var config = ObservabilityConfig.Xybrid(TestApiKey)
                       .WithTracesEndpoint(OverrideEndpoint)
                       .WithHeader("x-secret", "header-SECRET-DO-NOT-LOG"))
            {
                string description = config.ToString();
                Assert.IsNotNull(description);
                StringAssert.Contains(OverrideEndpoint, description);
                StringAssert.DoesNotContain(TestApiKey, description);
                StringAssert.DoesNotContain("header-SECRET-DO-NOT-LOG", description);
            }
        }

        [Test]
        public void Presets_RejectEmptyArguments()
        {
            Assert.Throws<ArgumentException>(() => ObservabilityConfig.Xybrid(""));
            Assert.Throws<ArgumentException>(() => ObservabilityConfig.Xybrid(null));
            Assert.Throws<ArgumentException>(() => ObservabilityConfig.Xybrid("   "));
            Assert.Throws<ArgumentException>(() => ObservabilityConfig.Otlp(""));
            Assert.Throws<ArgumentException>(() => ObservabilityConfig.Otlp(null));
        }

        [Test]
        public void TracesEndpoint_FollowsThePreset()
        {
            using (var xybrid = ObservabilityConfig.Xybrid(TestApiKey))
            using (var otlp = ObservabilityConfig.Otlp(CollectorEndpoint + "/"))
            {
                Assert.AreEqual(DefaultTracesEndpoint, xybrid.TracesEndpoint);
                Assert.AreEqual(CollectorEndpoint + "/v1/traces", otlp.TracesEndpoint);
            }
        }

        [Test]
        public void WithTracesEndpoint_OverridesThePreset()
        {
            using (var config = ObservabilityConfig.Otlp(CollectorEndpoint))
            {
                var chained = config.WithTracesEndpoint(OverrideEndpoint);
                Assert.AreSame(config, chained);
                Assert.AreEqual(OverrideEndpoint, config.TracesEndpoint);

                Assert.Throws<ArgumentException>(() => config.WithTracesEndpoint(" "));
            }
        }

        [Test]
        public void WithLiveTraceInterval_RejectsNegative()
        {
            using (var config = ObservabilityConfig.Otlp(CollectorEndpoint))
            {
                Assert.Throws<ArgumentOutOfRangeException>(
                    () => config.WithLiveTraceInterval(TimeSpan.FromMilliseconds(-1)));
            }
        }

        [Test]
        public void FluentSetters_ReturnSameInstanceForChaining()
        {
            using (var config = ObservabilityConfig.Otlp(CollectorEndpoint))
            {
                var chained = config
                    .WithTracesEndpoint(OverrideEndpoint)
                    .WithHeader("authorization", "Bearer token")
                    .WithJsonProtocol()
                    .WithService("unit-test", "1.2.3")
                    .WithResourceAttribute("deployment.environment", "test")
                    .WithMetadataOnly()
                    .WithMaxContentBytes(4096)
                    .WithLiveTraceInterval(TimeSpan.FromMilliseconds(500));
                Assert.AreSame(config, chained);
            }
        }
    }
}
