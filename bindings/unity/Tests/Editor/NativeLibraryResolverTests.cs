// Xybrid SDK - Native Library Resolver EditMode Tests
// Covers the host-independent logic of the editor NativeLibraryResolver:
// build-target → platform mapping, manifest JSON parsing/lookup, and the
// zip-slip extraction guard. Network/download and Unity-import paths are not
// exercised here (they require a real release + player build).

using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using Xybrid.Editor;

namespace Xybrid.Tests.Editor
{
    /// <summary>EditMode tests for <see cref="NativeLibraryResolver"/> pure logic.</summary>
    [TestFixture]
    public class NativeLibraryResolverTests
    {
        [TestCase(BuildTarget.StandaloneOSX, "macos")]
        [TestCase(BuildTarget.StandaloneWindows64, "windows")]
        [TestCase(BuildTarget.StandaloneLinux64, "linux")]
        [TestCase(BuildTarget.Android, "android")]
        [TestCase(BuildTarget.iOS, "ios")]
        public void PlatformForBuildTarget_MapsSupportedTargets(BuildTarget target, string expected)
        {
            Assert.AreEqual(expected, NativeLibraryResolver.PlatformForBuildTarget(target));
        }

        [Test]
        public void PlatformForBuildTarget_ReturnsNullForUnsupported()
        {
            Assert.IsNull(NativeLibraryResolver.PlatformForBuildTarget(BuildTarget.WebGL));
        }

        [Test]
        public void Manifest_ParsesAndLooksUpPlatforms()
        {
            const string json =
                "{\"version\":\"0.2.2\",\"platforms\":[" +
                "{\"platform\":\"macos\",\"asset\":\"xybrid-unity-native-macos-v0.2.2.zip\"," +
                "\"sha256\":\"deadbeef\",\"size\":1165}," +
                "{\"platform\":\"ios\",\"asset\":\"xybrid-unity-native-ios-v0.2.2.zip\"," +
                "\"sha256\":\"cafebabe\",\"size\":342177280}]}";

            var manifest = JsonUtility.FromJson<NativeLibraryResolver.NativeManifest>(json);

            Assert.AreEqual("0.2.2", manifest.version);
            Assert.AreEqual(2, manifest.platforms.Length);

            var macos = manifest.Find("macos");
            Assert.IsNotNull(macos);
            Assert.AreEqual("xybrid-unity-native-macos-v0.2.2.zip", macos.asset);
            Assert.AreEqual("deadbeef", macos.sha256);
            Assert.AreEqual(1165, macos.size);

            Assert.AreEqual(342177280L, manifest.Find("ios").size, "iOS size must survive as a 64-bit value");
            Assert.IsNull(manifest.Find("windows"), "absent platform resolves to null");
        }

        [Test]
        public void ExtractPath_AcceptsEntriesInsideRoot()
        {
            var root = Path.Combine(Path.GetTempPath(), "xybrid-extract-root");
            Assert.IsTrue(
                NativeLibraryResolver.TryResolveSafeExtractPath(root, "macOS/libxybrid_ffi.dylib", out var dest));
            StringAssert.Contains("libxybrid_ffi.dylib", dest);
        }

        [Test]
        public void ExtractPath_RejectsZipSlipEscape()
        {
            var root = Path.Combine(Path.GetTempPath(), "xybrid-extract-root");
            Assert.IsFalse(
                NativeLibraryResolver.TryResolveSafeExtractPath(root, "../evil.dll", out _),
                "a parent-traversal entry must be rejected");
            Assert.IsFalse(
                NativeLibraryResolver.TryResolveSafeExtractPath(root, "macOS/../../evil.dll", out _),
                "a nested traversal that escapes the root must be rejected");
        }
    }
}
