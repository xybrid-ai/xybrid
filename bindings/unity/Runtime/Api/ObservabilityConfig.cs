// Xybrid SDK - Observability Configuration
// Fluent builder wrapping the bolt OpenTelemetry observability config handle.

using System;

namespace Xybrid
{
    /// <summary>
    /// Configures OpenTelemetry (OTLP/HTTP) export of model loads, runs, and
    /// pipelines.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Export is off until you build a configuration and hand it to
    /// <see cref="XybridClient.InitializeObservability"/>; an API key passed to
    /// <see cref="XybridClient.Initialize"/> does not enable it. Start from a
    /// preset: <see cref="Xybrid"/> sends traces to the Xybrid dashboard,
    /// <see cref="Otlp"/> sends traces, metrics, and logs to an OTLP/HTTP
    /// endpoint such as an OpenTelemetry Collector. Prompts and outputs are
    /// captured once export is enabled; call <see cref="WithMetadataOnly"/> to
    /// record metadata only.
    /// </para>
    /// <para>
    /// Each <c>With...</c> setter mutates the config in place and returns
    /// <c>this</c> for chaining. Passing the config to
    /// <see cref="XybridClient.InitializeObservability"/> transfers ownership of
    /// the native handle, after which <see cref="Dispose"/> is a safe no-op. If
    /// you never hand it off, dispose it yourself.
    /// </para>
    /// <para>
    /// <see cref="Dispose"/> is thread-safe; the fluent setters are not intended
    /// to be called concurrently on one instance.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// XybridClient.InitializeObservability(
    ///     ObservabilityConfig.Xybrid("sk_live_...")
    ///         .WithService("my-game", Application.version));
    /// </code>
    /// </example>
    public sealed class ObservabilityConfig : IDisposable
    {
        private readonly object _lock = new object();
        private readonly string _preset;
        private readonly string _endpoint;
        private string _tracesEndpointOverride;
        private XybridBolt.XybridObservabilityConfig _bolt;
        private bool _disposed;

        private ObservabilityConfig(
            XybridBolt.XybridObservabilityConfig bolt,
            string preset,
            string endpoint)
        {
            _bolt = bolt;
            _preset = preset;
            _endpoint = endpoint;
        }

        /// <summary>
        /// Gets the Xybrid dashboard's OTLP trace endpoint, the default target of
        /// <see cref="Xybrid"/>.
        /// </summary>
        public static string DefaultTracesEndpoint =>
            XybridBolt.XybridBolt.ObservabilityDefaultTracesEndpoint();

        /// <summary>
        /// Creates a configuration that sends traces to the Xybrid dashboard,
        /// authenticated with <paramref name="apiKey"/>.
        /// </summary>
        /// <remarks>
        /// Traces only: metrics and logs are not sent, so
        /// <see cref="XybridClient.RecordFeedback"/> and
        /// <see cref="XybridClient.RecordEvaluation"/> return false. Route those
        /// through a Collector with <see cref="Otlp"/>.
        /// </remarks>
        /// <param name="apiKey">The Xybrid API key authorizing the export.</param>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="apiKey"/> is null, empty, or whitespace.
        /// </exception>
        /// <exception cref="XybridException">Thrown if the native handle cannot be created.</exception>
        public static ObservabilityConfig Xybrid(string apiKey)
        {
            if (string.IsNullOrWhiteSpace(apiKey))
            {
                throw new ArgumentException("apiKey must be a non-empty string.", nameof(apiKey));
            }

            return Create(
                () => XybridBolt.XybridObservabilityConfig.Xybrid(apiKey),
                "xybrid",
                DefaultTracesEndpoint);
        }

        /// <summary>
        /// Creates a configuration that sends traces, metrics, and logs to an
        /// OTLP/HTTP endpoint such as an OpenTelemetry Collector.
        /// </summary>
        /// <param name="endpoint">
        /// The collector's base URL (for example <c>http://localhost:4318</c>);
        /// signals go to <c>/v1/traces</c>, <c>/v1/metrics</c>, and <c>/v1/logs</c>
        /// under it.
        /// </param>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="endpoint"/> is null, empty, or whitespace.
        /// </exception>
        /// <exception cref="XybridException">Thrown if the native handle cannot be created.</exception>
        public static ObservabilityConfig Otlp(string endpoint)
        {
            if (string.IsNullOrWhiteSpace(endpoint))
            {
                throw new ArgumentException("endpoint must be a non-empty string.", nameof(endpoint));
            }

            return Create(
                () => XybridBolt.XybridObservabilityConfig.Otlp(endpoint),
                "otlp",
                endpoint);
        }

        /// <summary>
        /// Gets whether this configuration has been disposed or handed to
        /// <see cref="XybridClient.InitializeObservability"/>.
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
        /// Gets the URL traces are exported to.
        /// </summary>
        /// <remarks>
        /// <see cref="DefaultTracesEndpoint"/> for <see cref="Xybrid"/>, and
        /// <c>{endpoint}/v1/traces</c> for <see cref="Otlp"/>, until
        /// <see cref="WithTracesEndpoint"/> overrides it.
        /// </remarks>
        public string TracesEndpoint
        {
            get
            {
                lock (_lock)
                {
                    if (_tracesEndpointOverride != null)
                    {
                        return _tracesEndpointOverride;
                    }
                    return _preset == "otlp" ? _endpoint.TrimEnd('/') + "/v1/traces" : _endpoint;
                }
            }
        }

        /// <summary>
        /// Overrides the full trace endpoint URL (staging, self-hosted).
        /// </summary>
        /// <param name="endpoint">The complete trace URL, including its path.</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="endpoint"/> is null, empty, or whitespace.
        /// </exception>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public ObservabilityConfig WithTracesEndpoint(string endpoint)
        {
            if (string.IsNullOrWhiteSpace(endpoint))
            {
                throw new ArgumentException("endpoint must be a non-empty string.", nameof(endpoint));
            }

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetTracesEndpoint(endpoint);
                _tracesEndpointOverride = endpoint;
            }

            return this;
        }

        /// <summary>
        /// Adds an HTTP header sent with every export request (for example a
        /// vendor authorization header).
        /// </summary>
        /// <param name="name">Header name.</param>
        /// <param name="value">Header value.</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="name"/> is null or empty.</exception>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="value"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public ObservabilityConfig WithHeader(string name, string value)
        {
            if (string.IsNullOrEmpty(name))
            {
                throw new ArgumentException("name must be a non-empty string.", nameof(name));
            }
            if (value == null)
            {
                throw new ArgumentNullException(nameof(value));
            }

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetHeader(name, value);
            }

            return this;
        }

        /// <summary>
        /// Sends OTLP JSON instead of protobuf.
        /// </summary>
        /// <param name="json">True for JSON, false for protobuf (the default).</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public ObservabilityConfig WithJsonProtocol(bool json = true)
        {
            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetJsonProtocol(json);
            }

            return this;
        }

        /// <summary>
        /// Sets the reported service (app) name and version.
        /// </summary>
        /// <param name="name">Service name (for example <c>Application.productName</c>).</param>
        /// <param name="version">Optional service version (for example <c>Application.version</c>).</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="name"/> is null, empty, or whitespace.
        /// </exception>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public ObservabilityConfig WithService(string name, string version = null)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("name must be a non-empty string.", nameof(name));
            }

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetService(name, version);
            }

            return this;
        }

        /// <summary>
        /// Attaches a resource attribute to everything exported (for example
        /// <c>deployment.environment</c>).
        /// </summary>
        /// <param name="key">Attribute key.</param>
        /// <param name="value">Attribute value.</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="key"/> is null or empty.</exception>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="value"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public ObservabilityConfig WithResourceAttribute(string key, string value)
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
                _bolt.SetResourceAttribute(key, value);
            }

            return this;
        }

        /// <summary>
        /// Records metadata only: no prompts, outputs, or tool payloads.
        /// </summary>
        /// <param name="metadataOnly">True to drop content; false (the default) captures it.</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public ObservabilityConfig WithMetadataOnly(bool metadataOnly = true)
        {
            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetMetadataOnly(metadataOnly);
            }

            return this;
        }

        /// <summary>Allows context propagation to one exact third-party HTTP(S) origin.</summary>
        public ObservabilityConfig AllowProviderContext(string origin)
        {
            if (origin == null) throw new ArgumentNullException(nameof(origin));
            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.AllowProviderContext(origin);
            }
            return this;
        }

        /// <summary>Sets the caller-owned device identifier for exported resources.</summary>
        public ObservabilityConfig WithDeviceId(string deviceId)
        {
            if (deviceId == null) throw new ArgumentNullException(nameof(deviceId));
            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetDeviceId(deviceId);
            }
            return this;
        }

        /// <summary>Replaces literal text in governed content, in registration order.</summary>
        public ObservabilityConfig WithLiteralRedaction(string find, string replacement)
        {
            if (find == null) throw new ArgumentNullException(nameof(find));
            if (replacement == null) throw new ArgumentNullException(nameof(replacement));
            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.AddLiteralRedaction(find, replacement);
            }
            return this;
        }

        /// <summary>Omits one exact governed content attribute.</summary>
        public ObservabilityConfig WithoutContentAttribute(string key)
        {
            if (key == null) throw new ArgumentNullException(nameof(key));
            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.OmitContentAttribute(key);
            }
            return this;
        }

        /// <summary>
        /// Caps each captured content attribute; longer content is truncated
        /// with a marker.
        /// </summary>
        /// <param name="bytes">Maximum bytes per attribute.</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public ObservabilityConfig WithMaxContentBytes(uint bytes)
        {
            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetMaxContentBytes(bytes);
            }

            return this;
        }

        /// <summary>
        /// Sets the minimum interval between traced frames of one live session
        /// (for example a <see cref="Model.Stream"/> ASR session).
        /// </summary>
        /// <param name="interval">Interval. Fractional milliseconds are truncated.</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown if <paramref name="interval"/> is negative or exceeds <see cref="uint.MaxValue"/> milliseconds.
        /// </exception>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        public ObservabilityConfig WithLiveTraceInterval(TimeSpan interval)
        {
            uint ms = ToMilliseconds(interval, nameof(interval));

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                _bolt.SetLiveTraceIntervalMs(ms);
            }

            return this;
        }

        /// <summary>
        /// Transfers ownership of the native handle to the caller and neutralizes
        /// this instance so that subsequent <see cref="Dispose"/> calls are no-ops.
        /// </summary>
        /// <exception cref="ObjectDisposedException">Thrown if this config has already been disposed or detached.</exception>
        /// <remarks>
        /// Intended for <see cref="XybridClient.InitializeObservability"/>, which
        /// calls <c>Init()</c> on the handle (a consuming call) and disposes it
        /// afterwards.
        /// </remarks>
        internal XybridBolt.XybridObservabilityConfig DetachHandle()
        {
            lock (_lock)
            {
                ThrowIfDisposedLocked();
                XybridBolt.XybridObservabilityConfig bolt = _bolt;
                _bolt = null;
                _disposed = true;
                GC.SuppressFinalize(this);
                return bolt;
            }
        }

        /// <summary>
        /// Converts a timeout or interval to the native millisecond count.
        /// </summary>
        internal static uint ToMilliseconds(TimeSpan value, string paramName)
        {
            double totalMs = value.TotalMilliseconds;
            if (totalMs < 0)
            {
                throw new ArgumentOutOfRangeException(paramName, "value must not be negative.");
            }
            if (totalMs > uint.MaxValue)
            {
                throw new ArgumentOutOfRangeException(paramName, "value exceeds the maximum supported duration.");
            }
            return (uint)totalMs;
        }

        /// <summary>
        /// Returns a diagnostic string representation of this configuration.
        /// </summary>
        /// <returns>The preset and trace endpoint, never the API key or headers.</returns>
        public override string ToString()
        {
            string traces = TracesEndpoint;
            bool disposed = IsDisposed;
            return disposed
                ? $"ObservabilityConfig({_preset}, traces={traces}, disposed)"
                : $"ObservabilityConfig({_preset}, traces={traces})";
        }

        private static ObservabilityConfig Create(
            Func<XybridBolt.XybridObservabilityConfig> create,
            string preset,
            string endpoint)
        {
            XybridBolt.XybridObservabilityConfig bolt;
            try
            {
                bolt = create();
            }
            catch (Exception ex) when (
                ex is XybridBolt.XybridErrorException || ex is XybridBolt.BoltException)
            {
                throw BoltErrors.Translate(ex);
            }
            return new ObservabilityConfig(bolt, preset, endpoint);
        }

        private void ThrowIfDisposedLocked()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(ObservabilityConfig));
            }
        }

        /// <summary>
        /// Releases the native configuration handle. Safe to call multiple times.
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
                    // On the finalizer path the bolt handle's own finalizer
                    // releases it; don't touch a possibly-finalized object.
                    _bolt?.Dispose();
                }
                _bolt = null;
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer to ensure native resources are released if <see cref="Dispose"/> is missed.
        /// </summary>
        ~ObservabilityConfig()
        {
            Dispose(false);
        }
    }
}
