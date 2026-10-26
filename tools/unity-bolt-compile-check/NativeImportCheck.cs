using System;
using System.Reflection;
using System.Runtime.InteropServices;

internal static class NativeImportCheck
{
    private static int Main(string[] args)
    {
        if (args.Length != 1)
        {
            Console.Error.WriteLine("Usage: NativeImportCheck <expected native library>");
            return 2;
        }

        int imports = 0;
        int failures = 0;
        foreach (var type in typeof(XybridBolt.XybridBolt).Assembly.GetTypes())
        {
            foreach (var method in type.GetMethods(
                BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                var import = method.GetCustomAttribute<DllImportAttribute>();
                if (import == null)
                    continue;

                imports++;
                if (!string.Equals(import.Value, args[0], StringComparison.Ordinal))
                {
                    Console.Error.WriteLine(
                        $"{type.FullName}.{method.Name}: expected '{args[0]}', got '{import.Value}'");
                    failures++;
                }
            }
        }

        if (imports == 0)
        {
            Console.Error.WriteLine("No native imports were found; the binding sources must be compiled into this check.");
            return 1;
        }

        Console.WriteLine($"Checked {imports} native imports: {failures} failures (expected '{args[0]}').");
        return failures == 0 ? 0 : 1;
    }
}
