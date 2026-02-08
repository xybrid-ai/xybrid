// Xybrid SDK - Native Helpers
// Utility methods for marshaling between C# and native code.

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Xybrid.Native
{
    /// <summary>
    /// Helper methods for native interop operations.
    /// </summary>
    internal static class NativeHelpers
    {
        /// <summary>
        /// Converts a C# string to a null-terminated UTF-8 byte array.
        /// </summary>
        /// <param name="str">The string to convert.</param>
        /// <returns>A null-terminated UTF-8 byte array.</returns>
        public static byte[] ToUtf8Bytes(string str)
        {
            if (str == null)
            {
                return new byte[] { 0 };
            }

            int byteCount = Encoding.UTF8.GetByteCount(str);
            byte[] bytes = new byte[byteCount + 1]; // +1 for null terminator
            Encoding.UTF8.GetBytes(str, 0, str.Length, bytes, 0);
            bytes[byteCount] = 0; // Null terminator
            return bytes;
        }

        /// <summary>
        /// Converts a null-terminated UTF-8 byte pointer to a C# string.
        /// </summary>
        /// <param name="ptr">Pointer to the null-terminated UTF-8 string.</param>
        /// <returns>The C# string, or null if the pointer is null.</returns>
        public static unsafe string FromUtf8Ptr(byte* ptr)
        {
            if (ptr == null)
            {
                return null;
            }

            // Find the null terminator
            int length = 0;
            while (ptr[length] != 0)
            {
                length++;
            }

            if (length == 0)
            {
                return string.Empty;
            }

            return Encoding.UTF8.GetString(ptr, length);
        }

        /// <summary>
        /// Gets the last error message from the native library.
        /// </summary>
        /// <returns>The error message, or "Unknown error" if none available.</returns>
        public static unsafe string GetLastError()
        {
            byte* errorPtr = NativeMethods.xybrid_last_error();
            string error = FromUtf8Ptr(errorPtr);
            return error ?? "Unknown error";
        }

        /// <summary>
        /// Throws an XybridException with the last error message.
        /// </summary>
        /// <param name="context">Additional context to prepend to the error message.</param>
        public static void ThrowLastError(string context = null)
        {
            string error = GetLastError();
            string message = string.IsNullOrEmpty(context) ? error : $"{context}: {error}";
            throw new XybridException(message);
        }

        /// <summary>
        /// Pins a byte array and returns a pointer to it.
        /// The caller must keep the GCHandle alive while using the pointer.
        /// </summary>
        /// <param name="bytes">The byte array to pin.</param>
        /// <param name="handle">The GCHandle that must be freed when done.</param>
        /// <returns>A pointer to the first byte.</returns>
        public static unsafe byte* PinBytes(byte[] bytes, out GCHandle handle)
        {
            handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            return (byte*)handle.AddrOfPinnedObject();
        }
    }
}
