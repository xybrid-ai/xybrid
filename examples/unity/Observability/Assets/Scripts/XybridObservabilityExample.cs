// Xybrid Unity Example - Observability Lifecycle Demo
// Demonstrates OpenTelemetry export from Unity: configuring a preset, tracing
// a run, recording feedback on it, and the mobile-safe pause-flush and
// quit-shutdown lifecycle hooks.

using System;
using Xybrid;
using UnityEngine;

/// <summary>
/// Minimal MonoBehaviour that initializes the Xybrid SDK, starts observability
/// export from environment variables (or PlayerPrefs), runs one traced
/// inference, and flushes across Unity's mobile lifecycle events.
/// </summary>
/// <remarks>
/// <para>
/// Attach this script to a single GameObject in your scene.
/// </para>
/// <para>
/// The export target is resolved in order:
/// <list type="number">
///   <item>An OTLP collector from <c>OTEL_EXPORTER_OTLP_ENDPOINT</c> or PlayerPrefs <c>Xybrid.Observability.OtlpEndpoint</c>.</item>
///   <item>The Xybrid dashboard, keyed by <c>XYBRID_API_KEY</c> or PlayerPrefs <c>Xybrid.Observability.ApiKey</c>.</item>
/// </list>
/// If neither is present, export stays off but inference still runs so the
/// scene does not spam errors in the Editor.
/// </para>
/// <para>
/// The API key is NEVER hardcoded in source or serialized into the scene.
/// </para>
/// </remarks>
public class XybridObservabilityExample : MonoBehaviour
{
    // Configuration keys. Credentials and endpoints are resolved at runtime
    // only - they are not serialized fields and never appear in the scene.
    private const string OtlpEndpointEnvVar = "OTEL_EXPORTER_OTLP_ENDPOINT";
    private const string ApiKeyEnvVar = "XYBRID_API_KEY";
    private const string OtlpEndpointPrefKey = "Xybrid.Observability.OtlpEndpoint";
    private const string ApiKeyPrefKey = "Xybrid.Observability.ApiKey";

    // Keep lifecycle hooks short: mobile OSes give a backgrounding app little time.
    private static readonly TimeSpan PauseFlushTimeout = TimeSpan.FromSeconds(2);
    private static readonly TimeSpan QuitShutdownTimeout = TimeSpan.FromSeconds(5);

    [Header("Demo Settings")]
    [Tooltip("Model ID to run a single traced inference against.")]
    [SerializeField] private string modelId = "kokoro-82m";

    [Tooltip("Text prompt for the demo inference.")]
    [SerializeField] private string prompt = "Hello from the Xybrid observability demo.";

    [Tooltip("Service version reported with every span.")]
    [SerializeField] private string appVersion = "1.0.0";

    private bool _observabilityStarted;

    private void Start()
    {
        try
        {
            XybridClient.Initialize();
            Debug.Log($"[XybridObservability] SDK initialized (version={XybridClient.Version}).");
        }
        catch (Exception e)
        {
            Debug.LogError($"[XybridObservability] SDK initialization failed: {e.Message}");
            return;
        }

        _observabilityStarted = TryStartObservability();
        RunOneInference();
    }

    private bool TryStartObservability()
    {
        ObservabilityConfig config = BuildConfig();
        if (config == null)
        {
            Debug.LogWarning(
                $"[XybridObservability] No export target found via ${OtlpEndpointEnvVar}, ${ApiKeyEnvVar}, " +
                "or PlayerPrefs. Observability is off for this run. See README.md for setup.");
            return false;
        }

        // Own the config with `using` so it is disposed even if
        // InitializeObservability throws. InitializeObservability consumes the
        // handle on both success and failure, so a later Dispose is a no-op.
        using (config)
        {
            try
            {
                config
                    .WithService(Application.productName, appVersion)
                    .WithResourceAttribute("device.model.identifier", SystemInfo.deviceModel)
                    .WithResourceAttribute("xybrid.example.platform", Application.platform.ToString());

                string traces = config.TracesEndpoint;
                XybridClient.InitializeObservability(config);
                Debug.Log($"[XybridObservability] Export started -> {traces}");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[XybridObservability] Observability init failed: {e.Message}");
                return false;
            }
        }
    }

    private static ObservabilityConfig BuildConfig()
    {
        string collector = Resolve(OtlpEndpointEnvVar, OtlpEndpointPrefKey);
        if (!string.IsNullOrWhiteSpace(collector))
        {
            // Traces, metrics, and logs to an OpenTelemetry Collector.
            return ObservabilityConfig.Otlp(collector);
        }

        string apiKey = Resolve(ApiKeyEnvVar, ApiKeyPrefKey);
        if (!string.IsNullOrWhiteSpace(apiKey))
        {
            // Traces only, to the Xybrid dashboard.
            return ObservabilityConfig.Xybrid(apiKey);
        }

        return null;
    }

    private static string Resolve(string envVar, string prefKey)
    {
        string value = Environment.GetEnvironmentVariable(envVar);
        if (string.IsNullOrWhiteSpace(value))
        {
            value = PlayerPrefs.GetString(prefKey, null);
        }
        return value;
    }

    private void RunOneInference()
    {
        try
        {
            var observation = new ObservationOptions { ConversationId = "unity-observability-demo" };

            using (var model = XybridClient.LoadModel(modelId))
            using (var input = Envelope.Text(prompt))
            using (var result = model.Run(input, observation: observation))
            {
                if (!result.Success)
                {
                    Debug.LogWarning($"[XybridObservability] Inference returned failure: {result.Error}");
                    return;
                }

                Debug.Log($"[XybridObservability] Inference succeeded ({result.LatencyMs}ms, output={result.OutputType}).");

                if (result.Observation is XybridBolt.XybridObservationRef traced)
                {
                    Debug.Log($"[XybridObservability] Trace {traced.TraceId}, span {traced.SpanId}.");

                    // Feedback travels on the logs signal, so the traces-only
                    // Xybrid preset reports false here; a collector records it.
                    bool recorded = XybridClient.RecordFeedback(traced, score: 1.0, label: "demo");
                    Debug.Log($"[XybridObservability] Feedback recorded: {recorded}.");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[XybridObservability] Inference skipped ({e.Message}). Observability lifecycle still exercised.");
        }
    }

    /// <summary>
    /// Flushes queued spans when the app is backgrounded.
    /// </summary>
    /// <remarks>
    /// On iOS and Android, the OS may suspend or terminate the process after
    /// backgrounding without ever calling <see cref="OnApplicationQuit"/>.
    /// Flushing on pause is the only reliable way to deliver in-flight spans
    /// before the app is frozen.
    /// </remarks>
    private void OnApplicationPause(bool pauseStatus)
    {
        if (!pauseStatus || !_observabilityStarted)
        {
            return;
        }

        try
        {
            XybridClient.FlushObservability(PauseFlushTimeout);
            Debug.Log("[XybridObservability] Flushed on pause.");
        }
        catch (Exception e)
        {
            Debug.LogError($"[XybridObservability] Flush on pause failed: {e.Message}");
        }
    }

    /// <summary>
    /// Flushes and stops export on a clean exit from the Unity editor or a
    /// desktop build.
    /// </summary>
    private void OnApplicationQuit()
    {
        if (!_observabilityStarted)
        {
            return;
        }

        try
        {
            XybridClient.ShutdownObservability(QuitShutdownTimeout);
            Debug.Log("[XybridObservability] Shut down on quit.");
        }
        catch (Exception e)
        {
            Debug.LogError($"[XybridObservability] Shutdown on quit failed: {e.Message}");
        }
    }
}
