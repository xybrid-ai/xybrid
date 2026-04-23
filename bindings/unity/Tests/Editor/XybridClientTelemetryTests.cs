// Xybrid SDK - XybridClient telemetry lifecycle EditMode tests.
// Require the native xybrid library to be loadable (DllImport "xybrid").

using System;
using NUnit.Framework;
using Xybrid;

namespace Xybrid.Tests.Editor
{
    /// <summary>
    /// EditMode tests for the telemetry lifecycle methods on <see cref="XybridClient"/>.
    /// </summary>
    /// <remarks>
    /// These tests exercise process-wide global state (the SDK's init gate and the
    /// telemetry exporter), so they are ordered and every test leaves telemetry
    /// shut down. They depend on the native <c>xybrid</c> library being present on
    /// the Unity Editor's load path.
    /// </remarks>
    [TestFixture]
    public class XybridClientTelemetryTests
    {
        private const string TestEndpoint = "https://telemetry.example.test";
        private const string TestApiKey = "unit-test-api-key-SECRET-DO-NOT-LOG";

        [TearDown]
        public void CleanupTelemetry()
        {
            // Idempotent; no-op if no test initialized telemetry.
            XybridClient.ShutdownTelemetry();
        }

        [Test, Order(1)]
        public void InitializeTelemetry_BeforeInitialize_Throws()
        {
            if (XybridClient.IsInitialized)
            {
                Assert.Ignore("SDK already initialized in this Editor session; cannot test uninitialized branch.");
            }

            using (var config = new TelemetryConfig(TestApiKey).WithEndpoint(TestEndpoint))
            {
                Assert.Throws<InvalidOperationException>(() => XybridClient.InitializeTelemetry(config));
            }
        }

        [Test, Order(2)]
        public void TelemetryLifecycle_InitFlushShutdown_Succeeds()
        {
            XybridClient.Initialize();

            var config = new TelemetryConfig(TestApiKey).WithEndpoint(TestEndpoint)
                .WithAppVersion("test-1.0")
                .WithBatchSize(16);

            Assert.DoesNotThrow(() => XybridClient.InitializeTelemetry(config));
            Assert.DoesNotThrow(() => XybridClient.FlushTelemetry());
            Assert.DoesNotThrow(() => XybridClient.ShutdownTelemetry());

            // Subsequent shutdown is a safe no-op.
            Assert.DoesNotThrow(() => XybridClient.ShutdownTelemetry());

            // Flush after shutdown is also a safe no-op.
            Assert.DoesNotThrow(() => XybridClient.FlushTelemetry());
        }

        [Test, Order(3)]
        public void DoubleInitializeTelemetry_Throws()
        {
            XybridClient.Initialize();

            var first = new TelemetryConfig(TestApiKey).WithEndpoint(TestEndpoint);
            XybridClient.InitializeTelemetry(first);

            using (var second = new TelemetryConfig(TestApiKey).WithEndpoint(TestEndpoint))
            {
                Assert.Throws<InvalidOperationException>(() => XybridClient.InitializeTelemetry(second));
            }
        }

        [Test, Order(4)]
        public void InitializeTelemetry_NullConfig_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => XybridClient.InitializeTelemetry(null));
        }
    }
}
