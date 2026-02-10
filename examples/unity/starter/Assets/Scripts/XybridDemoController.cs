// Xybrid Unity Example - SDK Integration Demo
// This script demonstrates how to use the Xybrid SDK in a Unity project.

using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

// Note: The Xybrid SDK types will be available when native libraries are built.
// For now, this demo simulates the SDK workflow to demonstrate the integration pattern.

/// <summary>
/// Demo controller showing Xybrid SDK initialization and inference workflow.
/// Attach this script to a GameObject with UI references.
/// </summary>
public class XybridDemoController : MonoBehaviour
{
    [Header("UI References")]
    [SerializeField] private InputField inputField;
    [SerializeField] private Button runInferenceButton;
    [SerializeField] private Text resultText;
    [SerializeField] private Text statusText;
    [SerializeField] private Text latencyText;

    [Header("SDK Settings")]
    [SerializeField] private string modelId = "kokoro-82m";

    // SDK state tracking
    private bool isInitialized = false;
    private bool isModelLoaded = false;
    private bool isRunningInference = false;

    // Simulated SDK objects (replace with actual SDK when native libs available)
    // private Xybrid.ModelLoader modelLoader;
    // private Xybrid.Model model;

    private void Start()
    {
        // Initialize UI
        UpdateUI();

        // Set up button listener
        if (runInferenceButton != null)
        {
            runInferenceButton.onClick.AddListener(OnRunInferenceClicked);
        }

        // Set default input text
        if (inputField != null && string.IsNullOrEmpty(inputField.text))
        {
            inputField.text = "Hello, welcome to Xybrid!";
        }

        // Start SDK initialization
        InitializeSDK();
    }

    private void InitializeSDK()
    {
        SetStatus("Initializing SDK...");
        Debug.Log("[Xybrid] Initializing SDK...");

        // Simulated initialization delay
        // In production, this would call Xybrid SDK initialization
        Invoke(nameof(OnSDKInitialized), 0.5f);

        // Production code (when native library is available):
        // try
        // {
        //     Xybrid.Native.xybrid_init();
        //     OnSDKInitialized();
        // }
        // catch (Exception e)
        // {
        //     OnSDKError($"Failed to initialize SDK: {e.Message}");
        // }
    }

    private void OnSDKInitialized()
    {
        isInitialized = true;
        Debug.Log("[Xybrid] SDK initialized successfully");

        // Automatically load the model
        LoadModel();
    }

    private void LoadModel()
    {
        SetStatus($"Loading model '{modelId}'...");
        Debug.Log($"[Xybrid] Loading model: {modelId}");

        // Simulated model loading delay
        Invoke(nameof(OnModelLoaded), 0.8f);

        // Production code (when native library is available):
        // try
        // {
        //     modelLoader = Xybrid.ModelLoader.FromRegistry(modelId);
        //     model = modelLoader.Load();
        //     OnModelLoaded();
        // }
        // catch (Exception e)
        // {
        //     OnSDKError($"Failed to load model: {e.Message}");
        // }
    }

    private void OnModelLoaded()
    {
        isModelLoaded = true;
        SetStatus($"Model '{modelId}' ready");
        Debug.Log($"[Xybrid] Model loaded: {modelId}");
        UpdateUI();
    }

    private void OnRunInferenceClicked()
    {
        if (!isInitialized || !isModelLoaded || isRunningInference)
        {
            return;
        }

        string inputText = inputField != null ? inputField.text : "";
        if (string.IsNullOrEmpty(inputText))
        {
            SetResult("Error: Please enter some text");
            return;
        }

        RunInference(inputText);
    }

    private void RunInference(string text)
    {
        isRunningInference = true;
        UpdateUI();
        SetStatus("Running inference...");
        SetResult("");
        SetLatency("");
        Debug.Log($"[Xybrid] Running inference with input: {text}");

        // Start timing
        var stopwatch = Stopwatch.StartNew();

        // Simulated inference delay (0.2-0.5 seconds to mimic real TTS)
        float simulatedDelay = UnityEngine.Random.Range(0.2f, 0.5f);

        // Store stopwatch for callback
        _pendingStopwatch = stopwatch;
        _pendingInput = text;
        Invoke(nameof(OnInferenceComplete), simulatedDelay);

        // Production code (when native library is available):
        // try
        // {
        //     var envelope = Xybrid.Envelope.Text(text);
        //     var result = model.Run(envelope);
        //
        //     stopwatch.Stop();
        //
        //     if (result.Success)
        //     {
        //         OnInferenceSuccess(result.Text, stopwatch.ElapsedMilliseconds);
        //     }
        //     else
        //     {
        //         OnInferenceError(result.Error);
        //     }
        //
        //     result.Dispose();
        //     envelope.Dispose();
        // }
        // catch (Exception e)
        // {
        //     OnInferenceError(e.Message);
        // }
    }

    // Temporary storage for async simulation
    private Stopwatch _pendingStopwatch;
    private string _pendingInput;

    private void OnInferenceComplete()
    {
        _pendingStopwatch?.Stop();
        long latencyMs = _pendingStopwatch?.ElapsedMilliseconds ?? 0;

        // Simulate successful inference result
        // In production, this would be the actual SDK result
        string simulatedOutput = $"[TTS Audio Generated for: \"{_pendingInput}\"]";

        OnInferenceSuccess(simulatedOutput, latencyMs);
    }

    private void OnInferenceSuccess(string output, long latencyMs)
    {
        isRunningInference = false;
        SetStatus("Inference complete");
        SetResult(output);
        SetLatency($"{latencyMs} ms");
        Debug.Log($"[Xybrid] Inference completed in {latencyMs}ms: {output}");
        UpdateUI();
    }

    private void OnInferenceError(string error)
    {
        isRunningInference = false;
        SetStatus("Inference failed");
        SetResult($"Error: {error}");
        SetLatency("");
        Debug.LogError($"[Xybrid] Inference error: {error}");
        UpdateUI();
    }

    private void OnSDKError(string error)
    {
        SetStatus($"SDK Error: {error}");
        Debug.LogError($"[Xybrid] SDK Error: {error}");
    }

    // UI Update Helpers
    private void SetStatus(string status)
    {
        if (statusText != null)
        {
            statusText.text = status;
        }
    }

    private void SetResult(string result)
    {
        if (resultText != null)
        {
            resultText.text = result;
        }
    }

    private void SetLatency(string latency)
    {
        if (latencyText != null)
        {
            latencyText.text = string.IsNullOrEmpty(latency) ? "" : $"Latency: {latency}";
        }
    }

    private void UpdateUI()
    {
        if (runInferenceButton != null)
        {
            runInferenceButton.interactable = isInitialized && isModelLoaded && !isRunningInference;
        }

        if (inputField != null)
        {
            inputField.interactable = isInitialized && isModelLoaded && !isRunningInference;
        }
    }

    private void OnDestroy()
    {
        // Clean up SDK resources
        // Production code:
        // model?.Dispose();
        // modelLoader = null;

        CancelInvoke();
    }
}
