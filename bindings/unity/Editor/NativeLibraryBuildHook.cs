// Xybrid SDK - Editor Native Library Build Hook
// Ensures the active build target's native libraries are downloaded and
// verified before a player build, so a project that installed the managed-only
// package via OpenUPM/UPM never ships without its natives.

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

            var ok = NativeLibraryResolver.EnsurePlatform(
                platform, version, interactive: true, force: false, throwOnError: true);
            if (!ok)
            {
                throw new BuildFailedException(
                    $"[Xybrid] Native libraries for '{platform}' (v{version}) are not " +
                    "available; the build would ship without them.");
            }
        }
    }
}
