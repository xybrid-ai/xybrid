using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>Builds the platform's IL2CPP runtime smoke player.</summary>
public static class Builder
{
    private const string ScenePath = "Assets/Smoke.unity";

    public static void PerformBuild()
    {
        CreateSmokeScene();

        var standalone = NamedBuildTarget.Standalone;
        PlayerSettings.SetScriptingBackend(standalone, ScriptingImplementation.IL2CPP);
        PlayerSettings.SetManagedStrippingLevel(standalone, ManagedStrippingLevel.Disabled);

        BuildReport report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes = new[] { ScenePath },
            locationPathName = BuildOutputPath(),
            target = BuildTargetForEditor(),
            targetGroup = BuildTargetGroup.Standalone,
            options = BuildOptions.None,
        });

        BuildSummary summary = report.summary;
        Debug.Log(
            $"[XybridBuild] result={summary.result} errors={summary.totalErrors} " +
            $"warnings={summary.totalWarnings} time={summary.totalTime}");
        EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
    }

    private static void CreateSmokeScene()
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        var smoke = new GameObject("Xybrid runtime smoke");
        smoke.AddComponent<SmokeDriver>();
        EditorSceneManager.SaveScene(scene, ScenePath);
    }

    private static BuildTarget BuildTargetForEditor()
    {
#if UNITY_EDITOR_WIN
        return BuildTarget.StandaloneWindows64;
#elif UNITY_EDITOR_OSX
        return BuildTarget.StandaloneOSX;
#elif UNITY_EDITOR_LINUX
        return BuildTarget.StandaloneLinux64;
#else
        throw new PlatformNotSupportedException(
            "The Xybrid IL2CPP smoke supports Windows, macOS, and Linux editors.");
#endif
    }

    private static string BuildOutputPath()
    {
        string relativePath;

#if UNITY_EDITOR_WIN
        relativePath = "Build/windows-il2cpp/XybridSmoke.exe";
#elif UNITY_EDITOR_OSX
        relativePath = "Build/macos-il2cpp/XybridSmoke.app";
#elif UNITY_EDITOR_LINUX
        relativePath = "Build/linux-il2cpp/XybridSmoke.x86_64";
#else
        throw new PlatformNotSupportedException(
            "The Xybrid IL2CPP smoke supports Windows, macOS, and Linux editors.");
#endif

        return Path.GetFullPath(Path.Combine(Application.dataPath, "..", relativePath));
    }
}
