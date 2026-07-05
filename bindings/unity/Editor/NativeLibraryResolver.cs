// Xybrid SDK - Editor Native Library Resolver
// Downloads the per-platform native libraries (libxybrid_ffi.*) from the GitHub
// Release matching the installed package version and installs them into the
// consuming project's Assets/ folder.
//
// Why this exists: the Unity package is distributed managed-only (via OpenUPM /
// UPM), so the heavy native binaries — desktop dylib/dll/so, Android per-ABI
// .so, and the ~326 MB iOS static .a — cannot ship inside the package (size
// limits, and a UPM package installs read-only under Library/PackageCache/).
// Instead they are published as Release assets and fetched on demand here, into
// the project-writable Assets/Xybrid/Plugins/ tree. See tracking issue #319.

using System;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEngine;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Xybrid.SDK.Tests.Editor")]

namespace Xybrid.Editor
{
    /// <summary>
    /// Resolves and installs the Xybrid native libraries for a given platform by
    /// downloading them from the matching GitHub Release, verifying their
    /// SHA-256, and extracting them into <c>Assets/Xybrid/Plugins/</c>.
    /// </summary>
    /// <remarks>
    /// The natives are keyed by the installed package version (see
    /// <see cref="ResolveVersion"/>) and downloaded from
    /// <c>https://github.com/xybrid-ai/xybrid/releases/download/v{version}/…</c>.
    /// A per-platform <c>.xybrid-native-{platform}-version</c> marker records
    /// what was installed so repeated imports/builds don't re-download.
    ///
    /// <para>
    /// On editor load the <b>host</b> platform native is ensured (so Play mode
    /// works immediately); the active <b>build target</b>'s natives are ensured
    /// by <c>NativeLibraryBuildHook</c> before a player build. Both paths route
    /// through <see cref="EnsurePlatform"/>.
    /// </para>
    /// </remarks>
    [InitializeOnLoad]
    internal static class NativeLibraryResolver
    {
        private const string RepoSlug = "xybrid-ai/xybrid";
        private const string ManifestAssetPrefix = "xybrid-unity-natives-v";

        /// <summary>Root under the consuming project where natives are installed.</summary>
        private static string InstallRoot =>
            Path.Combine(Application.dataPath, "Xybrid", "Plugins");

        static NativeLibraryResolver()
        {
            // Defer off the domain-load path so we never block a reload; the
            // ensure itself is a no-op once the marker matches the version.
            EditorApplication.delayCall += EnsureHostPlatformQuiet;
        }

        // ---- Public entry points -------------------------------------------------

        [MenuItem("Xybrid/Native Libraries/Download for Current Editor")]
        private static void MenuDownloadHost()
        {
            var version = ResolveVersion();
            if (version == null)
            {
                EditorUtility.DisplayDialog("Xybrid",
                    "Could not determine the Xybrid package version.", "OK");
                return;
            }
            EnsurePlatform(HostPlatform(), version, interactive: true, force: true);
        }

        [MenuItem("Xybrid/Native Libraries/Download for Active Build Target")]
        private static void MenuDownloadActiveTarget()
        {
            var version = ResolveVersion();
            var platform = PlatformForBuildTarget(EditorUserBuildSettings.activeBuildTarget);
            if (version == null || platform == null)
            {
                EditorUtility.DisplayDialog("Xybrid",
                    "No Xybrid native bundle maps to the active build target.", "OK");
                return;
            }
            EnsurePlatform(platform, version, interactive: true, force: true);
        }

        /// <summary>
        /// Ensures the natives for <paramref name="platform"/> at
        /// <paramref name="version"/> are present under <see cref="InstallRoot"/>,
        /// downloading and verifying them if missing (or if <paramref name="force"/>).
        /// </summary>
        /// <returns><c>true</c> if the natives are present on return.</returns>
        /// <remarks>Throws on a hard failure only when <paramref name="throwOnError"/>.</remarks>
        internal static bool EnsurePlatform(
            string platform, string version, bool interactive, bool force = false,
            bool throwOnError = false)
        {
            var markerPath = MarkerPath(platform);
            if (!force && File.Exists(markerPath) &&
                File.ReadAllText(markerPath).Trim() == version)
            {
                return true;
            }

            try
            {
                var manifest = FetchManifest(version, interactive);
                var entry = manifest.Find(platform);
                if (entry == null)
                {
                    throw new InvalidOperationException(
                        $"Release v{version} has no Unity native bundle for '{platform}'.");
                }

                var url = ReleaseAssetUrl(version, entry.asset);
                var tmpZip = Path.Combine(Path.GetTempPath(), entry.asset);
                var actualSha = DownloadWithProgress(url, tmpZip, entry.asset, interactive);
                if (!string.Equals(actualSha, entry.sha256, StringComparison.OrdinalIgnoreCase))
                {
                    File.Delete(tmpZip);
                    throw new InvalidOperationException(
                        $"SHA-256 mismatch for {entry.asset}: expected {entry.sha256}, got {actualSha}.");
                }

                ExtractInto(tmpZip, InstallRoot);
                File.Delete(tmpZip);

                Directory.CreateDirectory(InstallRoot);
                File.WriteAllText(markerPath, version);
                AssetDatabase.Refresh();
                Debug.Log($"[Xybrid] Installed native libraries for '{platform}' (v{version}).");
                return true;
            }
            catch (Exception ex)
            {
                var msg = $"[Xybrid] Failed to install native libraries for '{platform}' " +
                          $"(v{version}): {ex.Message}\nRun " +
                          "'Xybrid ▸ Native Libraries ▸ Download for …' to retry.";
                if (throwOnError) throw new InvalidOperationException(msg, ex);
                Debug.LogWarning(msg);
                return false;
            }
            finally
            {
                if (interactive) EditorUtility.ClearProgressBar();
            }
        }

        // ---- Version / platform resolution --------------------------------------

        /// <summary>The GitHub Release tag body — the installed package version.</summary>
        internal static string ResolveVersion()
        {
            var pkg = PackageInfo.FindForAssembly(typeof(NativeLibraryResolver).Assembly);
            if (pkg != null && !string.IsNullOrEmpty(pkg.version))
            {
                return pkg.version;
            }
            // Local/embedded fallback: read package.json next to the package root.
            try
            {
                if (pkg != null)
                {
                    var json = File.ReadAllText(Path.Combine(pkg.resolvedPath, "package.json"));
                    var parsed = JsonUtility.FromJson<PackageJson>(json);
                    if (!string.IsNullOrEmpty(parsed?.version)) return parsed.version;
                }
            }
            catch { /* fall through */ }
            return null;
        }

        internal static string HostPlatform()
        {
            switch (Application.platform)
            {
                case RuntimePlatform.OSXEditor: return "macos";
                case RuntimePlatform.WindowsEditor: return "windows";
                case RuntimePlatform.LinuxEditor: return "linux";
                default: return "linux";
            }
        }

        internal static string PlatformForBuildTarget(BuildTarget target)
        {
            switch (target)
            {
                case BuildTarget.StandaloneOSX: return "macos";
                case BuildTarget.StandaloneWindows64: return "windows";
                case BuildTarget.StandaloneLinux64: return "linux";
                case BuildTarget.Android: return "android";
                case BuildTarget.iOS: return "ios";
                default: return null;
            }
        }

        private static void EnsureHostPlatformQuiet()
        {
            var version = ResolveVersion();
            if (version == null) return;
            // Non-interactive, non-fatal: keep first import quiet if offline.
            EnsurePlatform(HostPlatform(), version, interactive: true, throwOnError: false);
        }

        // ---- Networking ----------------------------------------------------------

        private static string ReleaseAssetUrl(string version, string asset) =>
            $"https://github.com/{RepoSlug}/releases/download/v{version}/{asset}";

        private static NativeManifest FetchManifest(string version, bool interactive)
        {
            var asset = $"{ManifestAssetPrefix}{version}.manifest.json";
            var url = ReleaseAssetUrl(version, asset);
            var tmp = Path.Combine(Path.GetTempPath(), asset);
            DownloadWithProgress(url, tmp, asset, interactive);
            var manifest = JsonUtility.FromJson<NativeManifest>(File.ReadAllText(tmp));
            File.Delete(tmp);
            if (manifest?.platforms == null)
            {
                throw new InvalidOperationException($"Malformed native manifest at {url}.");
            }
            return manifest;
        }

        /// <summary>
        /// Streams <paramref name="url"/> to <paramref name="destPath"/> on a
        /// background thread (so the editor stays responsive and large iOS slices
        /// don't buffer in memory), hashing as it goes. Returns the hex SHA-256.
        /// </summary>
        private static string DownloadWithProgress(
            string url, string destPath, string label, bool interactive)
        {
            float progress = 0f;
            string hashHex = null;
            Exception failure = null;

            var task = Task.Run(() =>
            {
                try
                {
                    using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
                    using var resp = client
                        .GetAsync(url, HttpCompletionOption.ResponseHeadersRead)
                        .GetAwaiter().GetResult();
                    resp.EnsureSuccessStatusCode();
                    long total = resp.Content.Headers.ContentLength ?? -1L;

                    using var src = resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult();
                    using var dst = File.Create(destPath);
                    using var sha = SHA256.Create();

                    var buffer = new byte[81920];
                    long read = 0;
                    int n;
                    while ((n = src.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        dst.Write(buffer, 0, n);
                        sha.TransformBlock(buffer, 0, n, null, 0);
                        read += n;
                        progress = total > 0 ? (float)read / total : 0f;
                    }
                    sha.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
                    hashHex = BitConverter.ToString(sha.Hash).Replace("-", "").ToLowerInvariant();
                }
                catch (Exception ex) { failure = ex; }
            });

            while (!task.IsCompleted)
            {
                if (interactive && EditorUtility.DisplayCancelableProgressBar(
                        "Xybrid — downloading native libraries", label, progress))
                {
                    // Best-effort: let the background copy finish/abort, then bail.
                    throw new OperationCanceledException($"Download of {label} cancelled.");
                }
                Thread.Sleep(100);
            }

            if (failure != null)
            {
                throw new InvalidOperationException(
                    $"Download failed for {url}: {failure.Message}", failure);
            }
            return hashHex;
        }

        // ---- Extraction ----------------------------------------------------------

        private static void ExtractInto(string zipPath, string destRoot)
        {
            Directory.CreateDirectory(destRoot);
            using var archive = ZipFile.OpenRead(zipPath);
            foreach (var entry in archive.Entries)
            {
                if (string.IsNullOrEmpty(entry.Name)) continue; // directory entry

                if (!TryResolveSafeExtractPath(destRoot, entry.FullName, out var destPath))
                {
                    throw new InvalidOperationException(
                        $"Refusing to extract '{entry.FullName}' outside {destRoot}.");
                }

                Directory.CreateDirectory(Path.GetDirectoryName(destPath));
                entry.ExtractToFile(destPath, overwrite: true);
            }
        }

        /// <summary>
        /// Resolves <paramref name="entryName"/> under <paramref name="destRoot"/>
        /// and rejects it (returns <c>false</c>) if it would escape the root
        /// (zip-slip). Pure/host-agnostic so it can be unit-tested.
        /// </summary>
        internal static bool TryResolveSafeExtractPath(
            string destRoot, string entryName, out string destPath)
        {
            destPath = Path.GetFullPath(Path.Combine(destRoot, entryName));
            var rootFull = Path.GetFullPath(destRoot) + Path.DirectorySeparatorChar;
            return destPath.StartsWith(rootFull, StringComparison.Ordinal);
        }

        private static string MarkerPath(string platform) =>
            Path.Combine(InstallRoot, $".xybrid-native-{platform}-version");

        // ---- Serializable manifest model ----------------------------------------

        [Serializable]
        internal sealed class NativeManifest
        {
            public string version;
            public NativePlatform[] platforms;

            public NativePlatform Find(string platform)
            {
                if (platforms == null) return null;
                foreach (var p in platforms)
                {
                    if (p != null && p.platform == platform) return p;
                }
                return null;
            }
        }

        [Serializable]
        internal sealed class NativePlatform
        {
            public string platform; // macos | windows | linux | android | ios
            public string asset;    // release asset filename
            public string sha256;   // lowercase hex
            public long size;       // bytes
        }

        [Serializable]
        private sealed class PackageJson
        {
            public string version;
        }
    }
}
