// Xybrid SDK - TelemetryConfig EditMode Tests
// Requires the native xybrid library to be loadable (DllImport "xybrid").

using System;
using NUnit.Framework;
using Xybrid;

namespace Xybrid.Tests.Editor
{
    /// <summary>
    /// EditMode tests for <see cref="TelemetryConfig"/>.
    /// </summary>
    /// <remarks>
    /// These tests exercise the managed wrapper. They depend on the native
    /// <c>xybrid</c> library being present on the Unity Editor's load path.
    /// </remarks>
    [TestFixture]
    public class TelemetryConfigTests
    {
        private const string DefaultEndpoint = "https://ingest.xybrid.dev";
        private const string OverrideEndpoint = "https://telemetry.example.test";
        private const string TestApiKey = "unit-test-api-key-SECRET-DO-NOT-LOG";

        [Test]
        public void Dispose_IsIdempotent()
        {
            var config = new TelemetryConfig(TestApiKey);
            Assert.IsFalse(config.IsDisposed);

            config.Dispose();
            Assert.IsTrue(config.IsDisposed);

            // Second Dispose must not throw or touch the already-freed handle.
            Assert.DoesNotThrow(() => config.Dispose());
            Assert.IsTrue(config.IsDisposed);
        }

        [Test]
        public void ToString_DoesNotContainApiKey()
        {
            using (var config = new TelemetryConfig(TestApiKey).WithEndpoint(OverrideEndpoint))
            {
                string description = config.ToString();
                Assert.IsNotNull(description);
                StringAssert.Contains(OverrideEndpoint, description);
                StringAssert.DoesNotContain(TestApiKey, description);
            }
        }

        [Test]
        public void Constructor_RejectsEmptyApiKey()
        {
            Assert.Throws<ArgumentException>(() => new TelemetryConfig(""));
            Assert.Throws<ArgumentException>(() => new TelemetryConfig(null));
            Assert.Throws<ArgumentException>(() => new TelemetryConfig("   "));
        }

        [Test]
        public void Endpoint_DefaultsToSdkIngestUrl()
        {
            using (var config = new TelemetryConfig(TestApiKey))
            {
                Assert.AreEqual(DefaultEndpoint, config.Endpoint);
            }
        }

        [Test]
        public void WithEndpoint_OverridesDefaultAndIsReportedByProperty()
        {
            using (var config = new TelemetryConfig(TestApiKey))
            {
                Assert.AreEqual(DefaultEndpoint, config.Endpoint);

                var chained = config.WithEndpoint(OverrideEndpoint);
                Assert.AreSame(config, chained);
                Assert.AreEqual(OverrideEndpoint, config.Endpoint);
            }
        }

        [Test]
        public void WithEndpoint_RejectsEmptyOrWhitespace()
        {
            using (var config = new TelemetryConfig(TestApiKey))
            {
                Assert.Throws<ArgumentException>(() => config.WithEndpoint(""));
                Assert.Throws<ArgumentException>(() => config.WithEndpoint(null));
                Assert.Throws<ArgumentException>(() => config.WithEndpoint("   "));
            }
        }

        [Test]
        public void FluentSetters_ReturnSameInstanceForChaining()
        {
            using (var config = new TelemetryConfig(TestApiKey))
            {
                var chained = config
                    .WithEndpoint(OverrideEndpoint)
                    .WithAppVersion("1.2.3")
                    .WithDeviceLabel("iPhone 15")
                    .WithDeviceAttribute("region", "us-west-2")
                    .WithBatchSize(64)
                    .WithFlushInterval(TimeSpan.FromSeconds(30));
                Assert.AreSame(config, chained);
            }
        }
    }
}
