// Xybrid Unity Example - Telemetry Lifecycle Demo
// Demonstrates end-to-end wiring of the Xybrid telemetry surface from Unity,
// including mobile-safe pause-flush and quit-shutdown lifecycle hooks.

using System;
using Xybrid;
using UnityEngine;

/// <summary>
/// Minimal MonoBehaviour that initializes the Xybrid SDK, starts the telemetry
/// sender from environment variables (or PlayerPrefs), runs one inference, and
/// cleans up across Unity's mobile lifecycle events.
/// </summary>
/// <remarks>
/// <para>
/// Attach this script to a single GameObject in your scene.
/// </para>
/// <para>
/// Configuration is read in order:
/// <list type="number">
///   <item>Environment variable <c>XYBRID_TELEMETRY_ENDPOINT</c> / <c>XYBRID_TELEMETRY_API_KEY</c> (preferred for CI / dev shells).</item>
///   <item>PlayerPrefs key <c>Xybrid.Telemetry.Endpoint</c> / <c>Xybrid.Telemetry.ApiKey</c> (fallback for on-device runs).</item>
/// </list>
/// If neither is present, telemetry is skipped but inference still runs so the
/// scene does not spam errors in the Editor.
/// </para>
/// <para>
/// The API key is NEVER hardcoded in source or serialized into the scene.
/// </para>
/// </remarks>
public class XybridTelemetryExample : MonoBehaviour
{
    // Configuration keys. API key and endpoint are resolved at runtime only -
    // they are not serialized fields and never appear in the committed scene.
    private const string EndpointEnvVar = "XYBRID_TELEMETRY_ENDPOINT";
    private const string ApiKeyEnvVar = "XYBRID_TELEMETRY_API_KEY";
    private const string EndpointPrefKey = "Xybrid.Telemetry.Endpoint";
    private const string ApiKeyPrefKey = "Xybrid.Telemetry.ApiKey";

    [Header("Demo Settings")]
    [Tooltip("Model ID to run a single inference against after telemetry starts.")]
    [SerializeField] private string modelId = "kokoro-82m";

    [Tooltip("Text prompt for the demo inference.")]
    [SerializeField] private string prompt = "Hello from the Xybrid telemetry demo.";

    [Tooltip("App version reported with every telemetry event.")]
    [SerializeField] private string appVersion = "1.0.0";

    private bool _telemetryStarted;

    private void Start()
    {
        try
        {
            XybridClient.Initialize();
            Debug.Log($"[XybridTelemetry] SDK initialized (version={XybridClient.Version}).");
        }
        catch (Exception e)
        {
            Debug.LogError($"[XybridTelemetry] SDK initialization failed: {e.Message}");
            return;
        }

        _telemetryStarted = TryStartTelemetry();
        RunOneInference();
    }

    private bool TryStartTelemetry()
    {
        string apiKey = ResolveApiKey();
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            Debug.LogWarning(
                $"[XybridTelemetry] No API key found via ${ApiKeyEnvVar} or PlayerPrefs('{ApiKeyPrefKey}'). " +
                "Telemetry disabled for this run. See README.md for setup.");
            return false;
        }

        // Endpoint override is optional. With only an API key set, the SDK
        // routes to its default ingest URL (https://ingest.xybrid.dev).
        string endpointOverride = ResolveEndpointOverride();

        // Own the config with `using` so it is disposed even if InitializeTelemetry
        // throws. InitializeTelemetry consumes the handle on both success and
        // failure, so a subsequent Dispose is a safe no-op.
        using (var config = new TelemetryConfig(apiKey)
            .WithAppVersion(appVersion)
            .WithDeviceLabel(SystemInfo.deviceModel)
            .WithDeviceAttribute("platform", Application.platform.ToString())
            .WithFlushInterval(TimeSpan.FromSeconds(30)))
        {
            if (!string.IsNullOrWhiteSpace(endpointOverride))
            {
                config.WithEndpoint(endpointOverride);
            }

            try
            {
                string resolvedEndpoint = config.Endpoint;
                XybridClient.InitializeTelemetry(config);
                Debug.Log($"[XybridTelemetry] Telemetry started -> {resolvedEndpoint}");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[XybridTelemetry] Telemetry init failed: {e.Message}");
                return false;
            }
        }
    }

    private static string ResolveApiKey()
    {
        string apiKey = Environment.GetEnvironmentVariable(ApiKeyEnvVar);
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            apiKey = PlayerPrefs.GetString(ApiKeyPrefKey, null);
        }
        return apiKey;
    }

    private static string ResolveEndpointOverride()
    {
        string endpoint = Environment.GetEnvironmentVariable(EndpointEnvVar);
        if (string.IsNullOrWhiteSpace(endpoint))
        {
            endpoint = PlayerPrefs.GetString(EndpointPrefKey, null);
        }
        return endpoint;
    }

    private void RunOneInference()
    {
        try
        {
            using (var model = XybridClient.LoadModel(modelId))
            using (var input = Envelope.Text(prompt))
            using (var result = model.Run(input))
            {
                if (result.Success)
                {
                    Debug.Log($"[XybridTelemetry] Inference succeeded ({result.LatencyMs}ms, output={result.OutputType}).");
                }
                else
                {
                    Debug.LogWarning($"[XybridTelemetry] Inference returned failure: {result.Error}");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[XybridTelemetry] Inference skipped ({e.Message}). Telemetry lifecycle still exercised.");
        }
    }

    /// <summary>
    /// Flushes any buffered telemetry events when the app is backgrounded.
    /// </summary>
    /// <remarks>
    /// On iOS and Android, the OS may suspend or terminate the process after
    /// backgrounding without ever calling <see cref="OnApplicationQuit"/>.
    /// Flushing on pause is the only reliable way to ensure in-flight events
    /// reach the collector before the app is frozen.
    /// </remarks>
    private void OnApplicationPause(bool pauseStatus)
    {
        if (!pauseStatus || !_telemetryStarted)
        {
            return;
        }

        try
        {
            XybridClient.FlushTelemetry();
            Debug.Log("[XybridTelemetry] Flushed telemetry on pause.");
        }
        catch (Exception e)
        {
            Debug.LogError($"[XybridTelemetry] Flush on pause failed: {e.Message}");
        }
    }

    /// <summary>
    /// Flushes once more and shuts the telemetry sender down on a clean exit
    /// from the Unity editor or a desktop build.
    /// </summary>
    private void OnApplicationQuit()
    {
        if (!_telemetryStarted)
        {
            return;
        }

        try
        {
            XybridClient.FlushTelemetry();
        }
        catch (Exception e)
        {
            Debug.LogError($"[XybridTelemetry] Flush on quit failed: {e.Message}");
        }

        try
        {
            XybridClient.ShutdownTelemetry();
            Debug.Log("[XybridTelemetry] Telemetry shut down on quit.");
        }
        catch (Exception e)
        {
            Debug.LogError($"[XybridTelemetry] Shutdown on quit failed: {e.Message}");
        }
    }
}
