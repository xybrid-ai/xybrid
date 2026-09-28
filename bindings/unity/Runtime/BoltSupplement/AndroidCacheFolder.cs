// Gives Android apps a model cache folder.
//
// Android has no default one: the native library refuses to load or list models
// until a folder is set. Kotlin and Flutter pass their app's files folder at
// init; here it is <Application.persistentDataPath>/xybrid/models. Unity lets
// only the main thread read persistentDataPath, and may not have it yet in the
// earliest startup phase, so it is read there, again before the first scene
// loads, and on first use when that comes first on the main thread (another
// startup hook loading a model). A first use off the main thread before any of
// these gets the folder as soon as a startup read succeeds. An app that calls
// XybridBolt.InitSdkCacheDir before loading or listing a model keeps its own
// folder: the first one set wins.

#if UNITY_ANDROID && !UNITY_EDITOR
using System;
using System.IO;
using UnityEngine;
#endif

namespace XybridBolt
{
    internal static class AndroidCacheFolder
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        private static readonly object Gate = new object();
        private static string _folder;
        private static bool _needed;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void CaptureEarly() => Capture();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void CaptureBeforeScene() => Capture();

        private static void Capture()
        {
            lock (Gate)
            {
                _folder = _folder ?? Read();
                if (_needed)
                {
                    Apply(_folder);
                }
            }
        }

        // Null off the main thread, where Unity throws, or while Unity has no path.
        private static string Read()
        {
            try
            {
                string root = Application.persistentDataPath;
                return string.IsNullOrEmpty(root) ? null : Path.Combine(root, "xybrid", "models");
            }
            catch (Exception)
            {
                return null;
            }
        }

        private static void Apply(string folder)
        {
            if (folder == null)
            {
                return;
            }
            try
            {
                XybridBolt.InitSdkCacheDir(folder);
            }
            catch (Exception)
            {
                // Runs inside static constructors, which must not throw; the
                // load or listing that follows reports what went wrong.
            }
        }
#endif

        /// <summary>
        /// Sets the default folder unless one is already set. A no-op off Android.
        /// </summary>
        internal static void ApplyDefault()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            lock (Gate)
            {
                _needed = true;
                _folder = _folder ?? Read();
                Apply(_folder);
            }
#endif
        }
    }
}
