using System;
using UnityEngine;
using Xybrid;

/// <summary>Calls the public Unity SDK from a built IL2CPP player.</summary>
public sealed class SmokeDriver : MonoBehaviour
{
    private void Start()
    {
        try
        {
            Run();
            Debug.Log("[XybridSmoke] OK");
            Application.Quit(0);
        }
        catch (Exception error)
        {
            Debug.LogException(error);
            Debug.LogError("[XybridSmoke] FAILED");
            Application.Quit(1);
        }
    }

    private static void Run()
    {
        Debug.Log($"[XybridSmoke] version={XybridClient.Version}");

        XybridClient.Initialize();
        Debug.Log($"[XybridSmoke] initialized={XybridClient.IsInitialized}");

        Envelope envelope = Envelope.Text("hello from il2cpp");
        Debug.Log($"[XybridSmoke] envelope={envelope}");

        using (var config =
               new TelemetryConfig("smoke-key").WithEndpoint("https://ingest.example"))
        {
            Debug.Log($"[XybridSmoke] endpoint={config.Endpoint}");
        }

        GenerationConfig generation = GenerationConfig.Greedy();
        Debug.Log($"[XybridSmoke] generation={generation}");

        // Keep the model result-decode path reachable in the AOT compilation.
        Debug.Log($"[XybridSmoke] model-type={typeof(Model).FullName}");
    }
}
