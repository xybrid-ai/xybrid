// Gives Android apps a model cache folder.
//
// Android has no default one: the native library refuses to load or list models
// until a folder is set. Kotlin and Flutter pass their app's files folder at
// init; here it is <Application.persistentDataPath>/xybrid/models. Unity only
// lets the main thread read persistentDataPath, and the entry types' static
// constructors can run on any thread, so it is read at startup and applied on
// first use. An app that calls XybridBolt.InitSdkCacheDir before loading or
// listing a model keeps its own folder: the first one set wins.

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
        private static volatile string _folder;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Capture()
        {
            string root = Application.persistentDataPath;
            if (!string.IsNullOrEmpty(root))
            {
                _folder = Path.Combine(root, "xybrid", "models");
            }
        }
#endif

        /// <summary>
        /// Sets the default folder unless one is already set. A no-op off Android.
        /// </summary>
        internal static void ApplyDefault()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            string folder = _folder;
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
#endif
        }
    }
}
