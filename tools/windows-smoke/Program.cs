using System;
using System.Runtime.InteropServices;

// Managed C# smoke for the windows-gnu xybrid_bolt.dll. The first P/Invoke below
// forces the CLR to LoadLibrary the DLL; if the static libc++/libc++abi/libunwind
// trio were NOT self-contained, this throws DllNotFoundException on a clean
// Windows runner (the missing dependent DLL). It then calls a real boltffi entry
// point that returns an owned wire buffer, decodes it exactly as the Unity
// binding does (WireReader.ReadString), and frees it — round-tripping the FfiBuf
// allocator across the C ABI boundary.
internal static class Smoke
{
    private const string Lib = "xybrid_bolt";

    // Mirrors bindings/unity/Runtime/Bolt/XybridBolt.cs (FfiBuf).
    [StructLayout(LayoutKind.Sequential)]
    private struct FfiBuf
    {
        public IntPtr ptr;
        public UIntPtr len;
        public UIntPtr cap;
        public UIntPtr align;
    }

    [DllImport(Lib, EntryPoint = "boltffi_function_xybrid_bolt_version")]
    private static extern FfiBuf Version();

    [DllImport(Lib, EntryPoint = "boltffi_free_buf")]
    private static extern void FreeBuf(FfiBuf buf);

    private static int Main()
    {
        FfiBuf buf;
        try
        {
            buf = Version();
        }
        catch (DllNotFoundException ex)
        {
            Console.Error.WriteLine($"smoke: FAILED to load {Lib}.dll or a dependency: {ex.Message}");
            return 1;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"smoke: FAILED calling the version export: {ex}");
            return 1;
        }

        string version;
        int total = checked((int)(nuint)buf.len);
        try
        {
            if (buf.ptr == IntPtr.Zero || total < 4)
            {
                Console.Error.WriteLine($"smoke: the version export returned an undersized buffer (len={total})");
                return 1;
            }
            // boltffi FfiBuf is wire-encoded (not a raw C string). Mirror
            // WireReader.ReadString: an i32 little-endian length prefix, then that
            // many UTF-8 bytes. (windows-x64 is little-endian, so Marshal.ReadInt32
            // reads the prefix directly.)
            int len = Marshal.ReadInt32(buf.ptr, 0);
            if (len < 0 || 4 + len > total)
            {
                Console.Error.WriteLine($"smoke: corrupt wire buffer (prefix len={len}, total={total})");
                return 1;
            }
            version = len == 0 ? string.Empty : (Marshal.PtrToStringUTF8(buf.ptr + 4, len) ?? string.Empty);
        }
        finally
        {
            FreeBuf(buf);
        }

        if (version.Length == 0)
        {
            Console.Error.WriteLine("smoke: the version export decoded to an empty string");
            return 1;
        }

        Console.WriteLine($"smoke: version -> \"{version}\"");
        Console.WriteLine("windows managed C# bolt smoke: OK");
        return 0;
    }
}
