// Xybrid SDK - Editor Domain Reload Guard
// Ensures the native telemetry sender is shut down cleanly across Unity
// editor assembly reloads and editor quits.

using UnityEditor;

namespace Xybrid.Editor
{
    /// <summary>
    /// Editor-only guard that shuts down the Xybrid telemetry sender before a
    /// Unity domain reload or editor quit, preventing a stale native handle
    /// from outliving the managed AppDomain.
    /// </summary>
    /// <remarks>
    /// The Xybrid telemetry sender is owned by the native (Rust) library, which
    /// the Unity Editor loads once and keeps in memory across managed-domain
    /// reloads (script recompiles, play-mode entry/exit, "Reload Domain" actions).
    /// Without this guard, the managed <see cref="XybridClient"/> bookkeeping is
    /// reset by the reload while the native sender (and its background worker
    /// thread) keeps running, leaking a handle and producing duplicate senders
    /// the next time <see cref="XybridClient.InitializeTelemetry"/> is called.
    ///
    /// <para>
    /// This class subscribes to <see cref="AssemblyReloadEvents.beforeAssemblyReload"/>
    /// and <see cref="EditorApplication.quitting"/> at editor load time and
    /// invokes <see cref="XybridClient.ShutdownTelemetry"/>, which is a safe
    /// no-op if telemetry was never initialized.
    /// </para>
    ///
    /// <para>
    /// This file lives in the Editor assembly and therefore does not compile
    /// into player builds.
    /// </para>
    /// </remarks>
    [InitializeOnLoad]
    internal static class TelemetryDomainReloadGuard
    {
        static TelemetryDomainReloadGuard()
        {
            AssemblyReloadEvents.beforeAssemblyReload += OnBeforeAssemblyReload;
            EditorApplication.quitting += OnEditorQuitting;
        }

        private static void OnBeforeAssemblyReload()
        {
            XybridClient.ShutdownTelemetry();
        }

        private static void OnEditorQuitting()
        {
            XybridClient.ShutdownTelemetry();
        }
    }
}
