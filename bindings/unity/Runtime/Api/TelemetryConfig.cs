// Xybrid SDK - Telemetry Configuration
// Fluent builder wrapping the native telemetry config handle.

using System;
using Xybrid.Native;

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
        private unsafe XybridTelemetryConfigHandle* _handle;
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
        public unsafe TelemetryConfig(string apiKey)
        {
            if (string.IsNullOrWhiteSpace(apiKey))
            {
                throw new ArgumentException("apiKey must be a non-empty string.", nameof(apiKey));
            }

            byte[] apiKeyBytes = NativeHelpers.ToUtf8Bytes(apiKey);

            fixed (byte* apiKeyPtr = apiKeyBytes)
            {
                XybridTelemetryConfigHandle* handle = NativeMethods.xybrid_telemetry_config_new(apiKeyPtr);
                if (handle == null)
                {
                    NativeHelpers.ThrowLastError("Failed to create telemetry config");
                }

                _handle = handle;
            }

            // Seed the managed Endpoint property from the same static string the
            // native side just bound to, so callers can read it back without a
            // round-trip setter.
            byte* defaultPtr = NativeMethods.xybrid_telemetry_default_endpoint();
            _endpoint = NativeHelpers.FromUtf8Ptr(defaultPtr) ?? string.Empty;
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
        /// <exception cref="XybridException">Thrown if the native setter rejects the value.</exception>
        public unsafe TelemetryConfig WithEndpoint(string endpoint)
        {
            if (string.IsNullOrWhiteSpace(endpoint))
            {
                throw new ArgumentException("endpoint must be a non-empty string.", nameof(endpoint));
            }

            byte[] bytes = NativeHelpers.ToUtf8Bytes(endpoint);

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                fixed (byte* ptr = bytes)
                {
                    int result = NativeMethods.xybrid_telemetry_config_set_endpoint(_handle, ptr);
                    if (result != 0)
                    {
                        NativeHelpers.ThrowLastError("Failed to set telemetry endpoint");
                    }
                }
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
        /// <exception cref="XybridException">Thrown if the native setter rejects the value.</exception>
        public unsafe TelemetryConfig WithAppVersion(string appVersion)
        {
            if (appVersion == null)
            {
                throw new ArgumentNullException(nameof(appVersion));
            }

            byte[] bytes = NativeHelpers.ToUtf8Bytes(appVersion);

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                fixed (byte* ptr = bytes)
                {
                    int result = NativeMethods.xybrid_telemetry_config_set_app_version(_handle, ptr);
                    if (result != 0)
                    {
                        NativeHelpers.ThrowLastError("Failed to set telemetry app version");
                    }
                }
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
        /// <exception cref="XybridException">Thrown if the native setter rejects the value.</exception>
        public unsafe TelemetryConfig WithDeviceLabel(string deviceLabel)
        {
            if (deviceLabel == null)
            {
                throw new ArgumentNullException(nameof(deviceLabel));
            }

            byte[] bytes = NativeHelpers.ToUtf8Bytes(deviceLabel);

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                fixed (byte* ptr = bytes)
                {
                    int result = NativeMethods.xybrid_telemetry_config_set_device_label(_handle, ptr);
                    if (result != 0)
                    {
                        NativeHelpers.ThrowLastError("Failed to set telemetry device label");
                    }
                }
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
        /// <exception cref="XybridException">Thrown if the native setter rejects the value.</exception>
        public unsafe TelemetryConfig WithDeviceAttribute(string key, string value)
        {
            if (string.IsNullOrEmpty(key))
            {
                throw new ArgumentException("key must be a non-empty string.", nameof(key));
            }
            if (value == null)
            {
                throw new ArgumentNullException(nameof(value));
            }

            byte[] keyBytes = NativeHelpers.ToUtf8Bytes(key);
            byte[] valueBytes = NativeHelpers.ToUtf8Bytes(value);

            lock (_lock)
            {
                ThrowIfDisposedLocked();
                fixed (byte* keyPtr = keyBytes)
                fixed (byte* valuePtr = valueBytes)
                {
                    int result = NativeMethods.xybrid_telemetry_config_set_device_attribute(_handle, keyPtr, valuePtr);
                    if (result != 0)
                    {
                        NativeHelpers.ThrowLastError($"Failed to set telemetry device attribute '{key}'");
                    }
                }
            }

            return this;
        }

        /// <summary>
        /// Sets the maximum number of events to buffer before flushing.
        /// </summary>
        /// <param name="batchSize">Batch size in events.</param>
        /// <returns>This configuration, for chaining.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this config has been disposed or detached.</exception>
        /// <exception cref="XybridException">Thrown if the native setter rejects the value.</exception>
        public unsafe TelemetryConfig WithBatchSize(uint batchSize)
        {
            lock (_lock)
            {
                ThrowIfDisposedLocked();
                int result = NativeMethods.xybrid_telemetry_config_set_batch_size(_handle, batchSize);
                if (result != 0)
                {
                    NativeHelpers.ThrowLastError("Failed to set telemetry batch size");
                }
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
        /// <exception cref="XybridException">Thrown if the native setter rejects the value.</exception>
        public unsafe TelemetryConfig WithFlushInterval(TimeSpan interval)
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
                int result = NativeMethods.xybrid_telemetry_config_set_flush_interval_secs(_handle, seconds);
                if (result != 0)
                {
                    NativeHelpers.ThrowLastError("Failed to set telemetry flush interval");
                }
            }

            return this;
        }

        /// <summary>
        /// Transfers ownership of the native handle to the caller and neutralizes this
        /// instance so that subsequent <see cref="Dispose"/> calls are no-ops.
        /// </summary>
        /// <returns>The raw native handle as an <see cref="IntPtr"/>.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if this config has already been disposed or detached.</exception>
        /// <remarks>
        /// Intended for internal use by <see cref="XybridClient.InitializeTelemetry"/>, which
        /// passes the handle to <c>xybrid_telemetry_init</c> (a consuming call).
        /// </remarks>
        internal IntPtr DetachHandle()
        {
            lock (_lock)
            {
                ThrowIfDisposedLocked();
                IntPtr raw;
                unsafe
                {
                    raw = (IntPtr)_handle;
                    _handle = null;
                }
                _disposed = true;
                GC.SuppressFinalize(this);
                return raw;
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
                unsafe
                {
                    if (_handle != null)
                    {
                        NativeMethods.xybrid_telemetry_config_free(_handle);
                        _handle = null;
                    }
                }
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
