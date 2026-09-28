// Registers the Unity binding before the first native call.
//
// Apps reach native code through the hand-written Xybrid API or straight
// through the generated types (the documented pinned-revision constructor,
// XybridModel.FromHuggingfaceWithRevision, exists only there), with or without
// XybridClient.Initialize(). Every one of those calls goes through
// NativeMethods, so its static constructor runs first. Without it the native
// library reports its platform default (`swift` on Apple platforms, `kotlin`
// on Android) or `rust`. The first registration wins, so this one sticks.

using System;

namespace XybridBolt
{
    internal static partial class NativeMethods
    {
        static NativeMethods()
        {
            try
            {
                XybridBolt.SetBinding("unity");
            }
            catch (Exception)
            {
                // Attribution is best-effort. A missing or mismatched native
                // library still fails the call that triggered this constructor;
                // throwing here would instead fail every native call with
                // TypeInitializationException.
            }
        }
    }
}
