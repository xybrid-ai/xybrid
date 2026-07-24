using System;
using System.Runtime.InteropServices;

// Managed C# smoke for the windows-gnu xybrid_bolt.dll. The first P/Invoke below
// forces the CLR to LoadLibrary the DLL; if the static libc++/libc++abi/libunwind
// trio were NOT self-contained, this throws DllNotFoundException on a clean
// Windows runner (the missing dependent DLL). Then it exercises a real boltffi
// call that returns an owned buffer and frees it — round-tripping the FfiBuf
// allocator across the C ABI boundary, exactly as the Unity binding does.
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
    }

    [DllImport(Lib, EntryPoint = "boltffi_version")]
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
            Console.Error.WriteLine($"smoke: FAILED calling boltffi_version: {ex}");
            return 1;
        }

        try
        {
            if (buf.ptr == IntPtr.Zero || (nuint)buf.len == 0)
            {
                Console.Error.WriteLine("smoke: boltffi_version returned an empty buffer");
                return 1;
            }

            string version = Marshal.PtrToStringUTF8(buf.ptr, checked((int)(nuint)buf.len)) ?? string.Empty;
            if (version.Length == 0)
            {
                Console.Error.WriteLine("smoke: boltffi_version decoded to an empty string");
                return 1;
            }

            Console.WriteLine($"smoke: boltffi_version -> \"{version}\"");
        }
        finally
        {
            FreeBuf(buf);
        }

        Console.WriteLine("windows managed C# bolt smoke: OK");
        return 0;
    }
}
