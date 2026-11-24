// Xybrid SDK - Editor Domain Reload Guard
// Ensures the native observability exporter is shut down cleanly across Unity
// editor assembly reloads and editor quits.

using System;
using UnityEditor;
using UnityEngine;

namespace Xybrid.Editor
{
    /// <summary>
    /// Editor-only guard that shuts down Xybrid observability export before a
    /// Unity domain reload or editor quit, so its native export workers do not
    /// outlive the managed AppDomain.
    /// </summary>
    /// <remarks>
    /// The exporter is owned by the native (Rust) library, which the Unity
    /// Editor loads once and keeps in memory across managed-domain reloads
    /// (script recompiles, play-mode entry/exit, "Reload Domain" actions).
    /// Without this guard, a reload leaves the previous session's exporter
    /// running until the next <see cref="XybridClient.InitializeObservability"/>
    /// replaces it.
    ///
    /// <para>
    /// This class subscribes to <see cref="AssemblyReloadEvents.beforeAssemblyReload"/>
    /// and <see cref="EditorApplication.quitting"/> at editor load time and
    /// invokes <see cref="XybridClient.ShutdownObservability"/>, which is a safe
    /// no-op if observability was never initialized.
    /// </para>
    ///
    /// <para>
    /// This file lives in the Editor assembly and therefore does not compile
    /// into player builds.
    /// </para>
    /// </remarks>
    [InitializeOnLoad]
    internal static class ObservabilityDomainReloadGuard
    {
        // Short: a reload blocks the editor while this runs.
        private static readonly TimeSpan ShutdownTimeout = TimeSpan.FromSeconds(2);

        static ObservabilityDomainReloadGuard()
        {
            AssemblyReloadEvents.beforeAssemblyReload += Shutdown;
            EditorApplication.quitting += Shutdown;
        }

        private static void Shutdown()
        {
            try
            {
                XybridClient.ShutdownObservability(ShutdownTimeout);
            }
            catch (DllNotFoundException)
            {
                // Native library not installed yet: nothing can be running.
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Xybrid] Observability shutdown before reload failed: {e.Message}");
            }
        }
    }
}
