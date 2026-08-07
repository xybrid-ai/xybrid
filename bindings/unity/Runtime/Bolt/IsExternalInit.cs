// Unity C# 9 polyfill (written by tools/scripts/gen_unity_bolt_csharp.py;
// not a BoltFFI output). The generated positional records use `init`
// accessors, which the compiler lowers against
// System.Runtime.CompilerServices.IsExternalInit -- a type .NET 5+ ships
// but Unity's netstandard2.1 scripting profile does not. This empty shim
// satisfies the reference so the records compile under Unity.
namespace System.Runtime.CompilerServices
{
    internal static class IsExternalInit { }
}
