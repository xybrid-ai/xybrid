// Xybrid SDK - iOS framework dependency EditMode regression tests.
// Uses a real PluginImporter without loading native code or exporting Xcode.

using System;
using System.IO;
using System.Text;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Build;
using UnityEngine;
using Xybrid.Editor;

namespace Xybrid.Tests.Editor
{
    [TestFixture]
    public class NativeLibraryBuildHookTests
    {
        private string _assetRoot;
        private string _directory;

        [SetUp]
        public void SetUp()
        {
            var folder = "XybridBuildHookTests_" + Guid.NewGuid().ToString("N");
            _assetRoot = "Assets/" + folder;
            _directory = Path.Combine(Application.dataPath, folder);
            Assert.IsNotEmpty(AssetDatabase.CreateFolder("Assets", folder));
        }

        [TearDown]
        public void TearDown()
        {
            if (AssetDatabase.IsValidFolder(_assetRoot))
            {
                Assert.IsTrue(AssetDatabase.DeleteAsset(_assetRoot), "Remove the disposable fixture assets");
            }
        }

        [Test]
        public void EnsureIosFrameworkDependencies_EmptyMetadataPersistsAccelerate()
        {
            var assetPath = CreateArchive();
            Assert.That(ImporterAt(assetPath).GetPlatformData(BuildTarget.iOS, "FrameworkDependencies"),
                Is.Null.Or.Empty, "Reproduce an archive staged before framework dependencies were declared");

            Assert.IsTrue(NativeLibraryBuildHook.EnsureIosFrameworkDependencies(assetPath));

            Reimport(assetPath);
            Assert.AreEqual("Accelerate",
                ImporterAt(assetPath).GetPlatformData(BuildTarget.iOS, "FrameworkDependencies"));
        }

        [Test]
        public void EnsureIosFrameworkDependencies_PreservesCustomDependenciesAndAssetIdentity()
        {
            var assetPath = CreateArchive("Foundation;MyGameFramework");
            var guid = AssetDatabase.AssetPathToGUID(assetPath);
            Assert.IsNotEmpty(guid);
            var importer = ImporterAt(assetPath);
            importer.userData = "game-owned plugin metadata";
            importer.SaveAndReimport();

            Assert.IsTrue(NativeLibraryBuildHook.EnsureIosFrameworkDependencies(assetPath));

            Reimport(assetPath);
            importer = ImporterAt(assetPath);
            Assert.AreEqual("Foundation;MyGameFramework;Accelerate",
                importer.GetPlatformData(BuildTarget.iOS, "FrameworkDependencies"));
            Assert.AreEqual(guid, AssetDatabase.AssetPathToGUID(assetPath), "Existing asset references must survive");
            Assert.AreEqual("game-owned plugin metadata", importer.userData);
            Assert.IsFalse(importer.GetCompatibleWithEditor(), "The archive must remain disabled in the Editor");
        }

        [Test]
        public void EnsureIosFrameworkDependencies_RepeatedInvocationLeavesMetadataUnchanged()
        {
            var assetPath = CreateArchive("Metal");
            Assert.IsTrue(NativeLibraryBuildHook.EnsureIosFrameworkDependencies(assetPath));
            var metadataPath = Path.Combine(_directory, "libxybrid_dependency_test.a.meta");
            var metadata = File.ReadAllBytes(metadataPath);

            Assert.IsFalse(NativeLibraryBuildHook.EnsureIosFrameworkDependencies(assetPath));

            CollectionAssert.AreEqual(metadata, File.ReadAllBytes(metadataPath));
            Assert.AreEqual("Metal;Accelerate",
                ImporterAt(assetPath).GetPlatformData(BuildTarget.iOS, "FrameworkDependencies"));
        }

        [TestCase("Accelerate")]
        [TestCase("Metal;Accelerate;Foundation")]
        [TestCase(" Metal ; Accelerate ; Foundation ;; ")]
        public void EnsureIosFrameworkDependencies_AlreadyDeclaredLeavesMetadataUnchanged(string dependencies)
        {
            var assetPath = CreateArchive(dependencies);
            var metadataPath = Path.Combine(_directory, "libxybrid_dependency_test.a.meta");
            var metadata = File.ReadAllBytes(metadataPath);

            Assert.IsFalse(NativeLibraryBuildHook.EnsureIosFrameworkDependencies(assetPath));

            Assert.AreEqual(dependencies,
                ImporterAt(assetPath).GetPlatformData(BuildTarget.iOS, "FrameworkDependencies"));
            CollectionAssert.AreEqual(metadata, File.ReadAllBytes(metadataPath),
                "An existing dependency must not cause metadata normalization or rewriting");
        }

        [Test]
        public void EnsureIosFrameworkDependencies_MissingAssetFailsWithAssetPath()
        {
            var assetPath = _assetRoot + "/missing.a";
            Assert.IsNull(AssetImporter.GetAtPath(assetPath));

            var error = Assert.Throws<BuildFailedException>(() =>
                NativeLibraryBuildHook.EnsureIosFrameworkDependencies(assetPath));

            StringAssert.Contains(assetPath, error.Message);
        }

        [Test]
        public void EnsureIosFrameworkDependencies_NonPluginAssetFailsWithAssetPath()
        {
            var assetPath = _assetRoot + "/readme.txt";
            File.WriteAllText(Path.Combine(_directory, "readme.txt"), "Not a native plugin");
            Reimport(assetPath);
            var importer = AssetImporter.GetAtPath(assetPath);
            Assert.IsNotNull(importer, "Exercise an existing asset with the wrong importer type");
            Assert.IsFalse(importer is PluginImporter);

            var error = Assert.Throws<BuildFailedException>(() =>
                NativeLibraryBuildHook.EnsureIosFrameworkDependencies(assetPath));

            StringAssert.Contains(assetPath, error.Message);
        }

        private string CreateArchive(string dependencies = null)
        {
            const string filename = "libxybrid_dependency_test.a";
            var assetPath = _assetRoot + "/" + filename;
            var archivePath = Path.Combine(_directory, filename);
            // A valid, empty ar container; there is no machine code to load.
            File.WriteAllBytes(archivePath, Encoding.ASCII.GetBytes("!<arch>\n"));
            // Seed compatibility before the first import, so this fixture never
            // attempts to load an iOS library into a Linux/macOS/Windows Editor.
            File.WriteAllText(archivePath + ".meta", @"fileFormatVersion: 2
guid: FIXTURE_GUID
PluginImporter:
  externalObjects: {}
  serializedVersion: 3
  iconMap: {}
  executionOrder: {}
  defineConstraints: []
  isPreloaded: 0
  isOverridable: 1
  isExplicitlyReferenced: 0
  validateReferences: 1
  platformData:
    Any:
      enabled: 0
      settings:
        Exclude Editor: 1
    Editor:
      enabled: 0
      settings:
        CPU: AnyCPU
        DefaultValueInitialized: true
        OS: AnyOS
    iOS:
      enabled: 1
      settings:
        AddToEmbeddedBinaries: false
        CPU: ARM64
  userData:
  assetBundleName:
  assetBundleVariant:
".Replace("FIXTURE_GUID", Guid.NewGuid().ToString("N")));
            Reimport(assetPath);
            var importer = ImporterAt(assetPath);
            Assert.IsFalse(importer.GetCompatibleWithEditor());
            if (dependencies != null)
            {
                importer.SetPlatformData(BuildTarget.iOS, "FrameworkDependencies", dependencies);
                importer.SaveAndReimport();
                Assert.AreEqual(dependencies,
                    ImporterAt(assetPath).GetPlatformData(BuildTarget.iOS, "FrameworkDependencies"),
                    "The fixture must retain the initial dependency string");
            }
            return assetPath;
        }

        private static PluginImporter ImporterAt(string assetPath)
        {
            var importer = AssetImporter.GetAtPath(assetPath) as PluginImporter;
            Assert.IsNotNull(importer, "Unity must recognize the empty ar fixture as a native plugin");
            return importer;
        }

        private static void Reimport(string assetPath)
        {
            AssetDatabase.ImportAsset(assetPath,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate);
        }
    }
}
