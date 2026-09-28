// Registers the Unity binding before the generated entry types reach native
// code.
//
// Apps reach native code through the hand-written Xybrid API or straight
// through the generated types (the documented pinned-revision constructor,
// XybridModel.FromHuggingfaceWithRevision, exists only there), with or without
// XybridClient.Initialize(). Every generated type whose calls can reach the
// registry or telemetry gets a static constructor, which runs before that
// type's first static call or new instance. The native library's platform
// fallback (`swift` on Apple platforms, `kotlin` on Android) never outranks a
// registration, so a registration that comes late still wins. The types that
// touch the model cache also set the Android default (AndroidCacheFolder.cs);
// XybridBolt does not, so an app's own InitSdkCacheDir call there comes first.

using System;

namespace XybridBolt
{
    internal static class BindingRegistration
    {
        internal static void Register()
        {
            try
            {
                XybridBolt.SetBinding("unity");
            }
            catch (Exception)
            {
                // Throwing from a static constructor would make its type
                // unusable (TypeInitializationException), and a missing native
                // library fails the call that triggered this anyway. The other
                // entry types and XybridClient.Initialize() register again.
            }
        }
    }

    public static partial class XybridBolt
    {
        static XybridBolt() => BindingRegistration.Register();
    }

    public sealed partial class XybridModel
    {
        static XybridModel()
        {
            BindingRegistration.Register();
            AndroidCacheFolder.ApplyDefault();
        }
    }

    public sealed partial class XybridPipeline
    {
        static XybridPipeline()
        {
            BindingRegistration.Register();
            AndroidCacheFolder.ApplyDefault();
        }
    }

    public sealed partial class XybridDownload
    {
        static XybridDownload()
        {
            BindingRegistration.Register();
            AndroidCacheFolder.ApplyDefault();
        }
    }

    public sealed partial class XybridTelemetryConfig
    {
        static XybridTelemetryConfig() => BindingRegistration.Register();
    }
}
