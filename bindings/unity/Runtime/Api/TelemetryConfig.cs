// Xybrid SDK - Telemetry Configuration
// Fluent builder wrapping the bolt telemetry config handle.

using System;

namespace Xybrid
{
    /// <summary>
    /// Configures the Xybrid telemetry sender.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Build a configuration with the constructor and the <c>With...</c> fluent methods,
    /// then hand it to <see cref="XybridClient.InitializeTelemetry"/>. Passing a
    /// <see cref="TelemetryConfig"/> to <see cref="XybridClient.InitializeTelemetry"/>
    /// transfers ownership of the underlying native handle, after which calling
    /// <see cref="Dispose"/> on the config is a safe no-op.
    /// </para>
    /// <para>
    /// Each fluent setter mutates the config in place and returns <c>this</c> to support
    /// chaining. If you never hand the config off to <see cref="XybridClient.InitializeTelemetry"/>,
    /// you must dispose it yourself so the native handle is released.
    /// </para>
    /// <para>
    /// The instance is thread-safe for <see cref="Dispose"/>: concurrent calls are serialized
    /// and the second one is a no-op. The fluent builder methods are not intended to be
    /// called concurrently on a single instance.
    /// </para>
    /// </remarks>
    public sealed class TelemetryConfig : IDisposable
    {
        private readonly object _lock = new object();
        private string _endpoint;
        private XybridBolt.XybridTelemetryConfig _bolt;
        private bool _disposed;

        /// <summary>
        /// Gets whether this configuration has been disposed or detached.
        /// </summary>
        public bool IsDisposed
        {
            get
            {
                lock (_lock)
                {
                    return _disposed;
                }
            }
        }

        /// <summary>
        /// Gets the currently resolved ingest endpoint.
        /// </summary>
        /// <remarks>
        /// Reports the SDK's built-in default (<c>https://ingest.xybrid.dev</c>) until
        /// <see cref="WithEndpoint"/> is called, at which point it reports the override.
        /// </remarks>
        public string Endpoint
        {
            get
            {
                lock (_lock)
                {
                    return _endpoint;
                }
            }
        }

        /// <summary>
        /// Creates a new telemetry configuration bound to the SDK's default ingest
        /// endpoint.
        /// </summary>
        /// <param name="apiKey">The API key authorizing this sender.</param>
        /// <remarks>
        /// The default endpoint is <c>https://ingest.xybrid.dev</c>. To target a
        /// self-hosted collector or a non-production environment, chain
        /// <see cref="WithEndpoint"/> after construction.
        /// </remarks>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="apiKey"/> is null, empty, or whitespace.
        /// </exception>
        /// <exception cref="XybridException">Thrown if the native handle cannot be created.</exception>
        public TelemetryConfig(string apiKey)
        {
            if (string.IsNullOrWhiteSpace(apiKey))
            {
                throw new ArgumentException("apiKey must be a non-empty string.", nameof(apiKey));
            }

            try
            {
                _bolt = new XybridBolt.XybridTelemetryConfig(apiKey);
            }
            catch (Exception ex) when (
                ex is XybridBolt.XybridErrorException || ex is XybridBolt.BoltException)
            {
                throw BoltErrors.Translate(ex);
            }

            // Seed the managed Endpoint property from the same default string the
            // native side just bound to, so callers can read it back without a
            // round-trip setter.
            _endpoint = XybridBolt.XybridBolt.TelemetryDefaultEndpoint();
        }

        /// <summary>
        /// Overrides the ingest endpoint. Use for self-hosted collectors or
        /// non-production environments.
        /// </summary>
        /// <param name="endpoint">The telemetry collector endpoint (e.g., <c>https://telemetry.internal</c>).</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="endpoint"/> is null, empty, or whitespace.
        /// </exception>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public TelemetryConfig WithEndpoint(string endpoint)
        {
            if (string.IsNullOrWhiteSpace(endpoint))
            {
                throw new ArgumentException("endpoint must be a non-empty string.", nameof(endpoint));
            }

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetEndpoint(endpoint);
                _endpoint = endpoint;
            }

            return this;
        }

        /// <summary>
        /// Sets the application version reported with every telemetry event.
        /// </summary>
        /// <param name="appVersion">Caller-defined version string (e.g., <c>"1.4.2"</c>).</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="appVersion"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public TelemetryConfig WithAppVersion(string appVersion)
        {
            if (appVersion == null)
            {
                throw new ArgumentNullException(nameof(appVersion));
            }

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetAppVersion(appVersion);
            }

            return this;
        }

        /// <summary>
        /// Sets a human-readable device label reported with every telemetry event.
        /// </summary>
        /// <param name="deviceLabel">Caller-defined device label (e.g., <c>"iPhone 15 Pro"</c>).</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="deviceLabel"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public TelemetryConfig WithDeviceLabel(string deviceLabel)
        {
            if (deviceLabel == null)
            {
                throw new ArgumentNullException(nameof(deviceLabel));
            }

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetDeviceLabel(deviceLabel);
            }

            return this;
        }

        /// <summary>
        /// Adds or replaces a custom device attribute reported with every telemetry event.
        /// </summary>
        /// <param name="key">Attribute key.</param>
        /// <param name="value">Attribute value.</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="key"/> is null or empty.</exception>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="value"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public TelemetryConfig WithDeviceAttribute(string key, string value)
        {
            if (string.IsNullOrEmpty(key))
            {
                throw new ArgumentException("key must be a non-empty string.", nameof(key));
            }
            if (value == null)
            {
                throw new ArgumentNullException(nameof(value));
            }

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetDeviceAttribute(key, value);
            }

            return this;
        }

        /// <summary>
        /// Sets the maximum number of events to buffer before flushing.
        /// </summary>
        /// <param name="batchSize">Batch size in events.</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public TelemetryConfig WithBatchSize(uint batchSize)
        {
            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetBatchSize(batchSize);
            }

            return this;
        }

        /// <summary>
        /// Sets the background flush interval.
        /// </summary>
        /// <param name="interval">Flush interval. Fractional seconds are truncated.</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown if <paramref name="interval"/> is negative or exceeds <see cref="uint.MaxValue"/> seconds.
        /// </exception>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public TelemetryConfig WithFlushInterval(TimeSpan interval)
        {
            double totalSeconds = interval.TotalSeconds;
            if (totalSeconds < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(interval), "interval must not be negative.");
            }
            if (totalSeconds > uint.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(interval), "interval exceeds the maximum supported value.");
            }

            uint seconds = (uint)totalSeconds;

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetFlushIntervalSecs(seconds);
            }

            return this;
        }

        /// <summary>
        /// Transfers ownership of the native handle to the caller and neutralizes this
        /// instance so that subsequent <see cref="Dispose"/> calls are no-ops.
        /// </summary>
        /// <returns>The bolt telemetry config handle.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this config has already been disposed or detached.</exception>
        /// <remarks>
        /// Intended for internal use by <see cref="XybridClient.InitializeTelemetry"/>, which
        /// calls <c>Init()</c> on the handle (a consuming call) and disposes it afterwards.
        /// </remarks>
        internal XybridBolt.XybridTelemetryConfig DetachHandle()
        {
            lock (_lock)
            {
                ThrowIfDisposedLocked();
                XybridBolt.XybridTelemetryConfig bolt = _bolt;
                _bolt = null;
                _disposed = true;
                GC.SuppressFinalize(this);
                return bolt;
            }
        }

        /// <summary>
        /// Returns a diagnostic string representation of this configuration.
        /// </summary>
        /// <returns>A string containing the endpoint but never the API key.</returns>
        public override string ToString()
        {
            string endpoint;
            bool disposed;
            lock (_lock)
            {
                endpoint = _endpoint;
                disposed = _disposed;
            }
            return disposed
                ? $"TelemetryConfig(endpoint={endpoint}, disposed)"
                : $"TelemetryConfig(endpoint={endpoint})";
        }

        private void ThrowIfDisposedLocked()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(TelemetryConfig));
            }
        }

        /// <summary>
        /// Releases the native telemetry config handle. Safe to call multiple times.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            lock (_lock)
            {
                if (_disposed)
                {
                    return;
                }
                if (disposing)
                {
                    // Release the native handle eagerly. On the finalizer path we
                    // leave it to the bolt handle's own finalizer instead of
                    // touching a possibly-finalized managed object.
                    _bolt?.Dispose();
                }
                _bolt = null;
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer to ensure native resources are released if <see cref="Dispose"/> is missed.
        /// </summary>
        ~TelemetryConfig()
        {
            Dispose(false);
        }
    }
}
