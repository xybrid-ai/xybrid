// Xybrid SDK - Editor Native Library Build Hook
// Ensures the active build target's native libraries are downloaded and
// verified before a player build, so a project that installed the managed-only
// package via OpenUPM/UPM never ships without its natives.

using System.Collections.Generic;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;

namespace Xybrid.Editor
{
    /// <summary>
    /// Build preprocessor that resolves the Xybrid natives for the target
    /// platform before the player build proceeds. Fails the build with an
    /// actionable message if they can't be installed (e.g. offline), rather than
    /// producing a player that crashes on the first P/Invoke.
    /// </summary>
    internal sealed class NativeLibraryBuildHook : IPreprocessBuildWithReport
    {
        // Run early so the natives (and their .meta import settings) are present
        // before Unity collects plugins for the build.
        public int callbackOrder => 0;

        public void OnPreprocessBuild(BuildReport report)
        {
            var platform = NativeLibraryResolver.PlatformForBuildTarget(report.summary.platform);
            if (platform == null)
            {
                // Not a platform we ship natives for — nothing to do.
                return;
            }

            var version = NativeLibraryResolver.ResolveVersion();
            if (version == null)
            {
                throw new BuildFailedException(
                    "[Xybrid] Could not determine the package version, so native " +
                    "libraries can't be resolved for the build.");
            }

            // No modal progress bar during batch/CI builds (-batchmode).
            var ok = NativeLibraryResolver.EnsurePlatform(
                platform, version, interactive: !UnityEngine.Application.isBatchMode,
                force: false, throwOnError: true);
            if (!ok)
            {
                throw new BuildFailedException(
                    $"[Xybrid] Native libraries for '{platform}' (v{version}) are not " +
                    "available; the build would ship without them.");
            }

            if (report.summary.platform == BuildTarget.iOS)
            {
                EnsureIosFrameworkDependencies("Assets/Xybrid/Plugins/iOS/libxybrid_bolt.a");
            }
        }

        /// <summary>Adds the archive's required frameworks before Unity collects plugins.</summary>
        internal static bool EnsureIosFrameworkDependencies(string assetPath)
        {
            var importer = AssetImporter.GetAtPath(assetPath) as PluginImporter;
            if (importer == null)
            {
                throw new BuildFailedException(
                    $"[Xybrid] Could not import the iOS native archive at '{assetPath}'. " +
                    "Download the iOS natives again from Xybrid > Native Libraries.");
            }

            // v0.14.0 bundles have no framework dependencies. Apply this even
            // when the resolver's version marker says they are already current.
            var existing = importer.GetPlatformData(BuildTarget.iOS, "FrameworkDependencies") ?? "";
            var frameworks = new List<string>();
            foreach (var framework in existing.Split(';'))
            {
                var name = framework.Trim();
                if (name.Length != 0) frameworks.Add(name);
            }
            if (frameworks.Contains("Accelerate")) return false;

            // ggml uses Accelerate's vDSP symbols. Unity places these declared
            // system frameworks on the UnityFramework target, not the launcher.
            frameworks.Add("Accelerate");
            importer.SetPlatformData(BuildTarget.iOS, "FrameworkDependencies", string.Join(";", frameworks));
            importer.SaveAndReimport();
            return true;
        }
    }
}
