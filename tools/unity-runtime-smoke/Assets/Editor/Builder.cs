using System;
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
#else
        throw new PlatformNotSupportedException(
            "The Xybrid IL2CPP smoke supports Windows and macOS editors.");
#endif
    }

    private static string BuildOutputPath()
    {
#if UNITY_EDITOR_WIN
        return "Build/windows-il2cpp/XybridSmoke.exe";
#elif UNITY_EDITOR_OSX
        return "Build/macos-il2cpp/XybridSmoke.app";
#else
        throw new PlatformNotSupportedException(
            "The Xybrid IL2CPP smoke supports Windows and macOS editors.");
#endif
    }
}
