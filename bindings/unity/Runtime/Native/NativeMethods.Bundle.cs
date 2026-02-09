// Xybrid SDK - Bundle Native Methods
// P/Invoke declarations for .xyb bundle operations.
// These supplement the auto-generated NativeMethods.g.cs.

#pragma warning disable CS8500
#pragma warning disable CS8981
using System;
using System.Runtime.InteropServices;

namespace Xybrid.Native
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe partial struct XybridBundleHandle
    {
        private void* _ptr;
    }

    internal static unsafe partial class NativeMethods
    {
        [DllImport(__DllName, EntryPoint = "xybrid_bundle_open", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern XybridBundleHandle* xybrid_bundle_open(byte* path);

        [DllImport(__DllName, EntryPoint = "xybrid_bundle_manifest_json", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern byte* xybrid_bundle_manifest_json(XybridBundleHandle* handle);

        [DllImport(__DllName, EntryPoint = "xybrid_bundle_metadata_json", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern byte* xybrid_bundle_metadata_json(XybridBundleHandle* handle);

        [DllImport(__DllName, EntryPoint = "xybrid_bundle_extract", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern int xybrid_bundle_extract(XybridBundleHandle* handle, byte* output_dir);

        [DllImport(__DllName, EntryPoint = "xybrid_bundle_model_id", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern byte* xybrid_bundle_model_id(XybridBundleHandle* handle);

        [DllImport(__DllName, EntryPoint = "xybrid_bundle_version", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern byte* xybrid_bundle_version(XybridBundleHandle* handle);

        [DllImport(__DllName, EntryPoint = "xybrid_bundle_target", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern byte* xybrid_bundle_target(XybridBundleHandle* handle);

        [DllImport(__DllName, EntryPoint = "xybrid_bundle_hash", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern byte* xybrid_bundle_hash(XybridBundleHandle* handle);

        [DllImport(__DllName, EntryPoint = "xybrid_bundle_has_metadata", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern int xybrid_bundle_has_metadata(XybridBundleHandle* handle);

        [DllImport(__DllName, EntryPoint = "xybrid_bundle_file_count", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern uint xybrid_bundle_file_count(XybridBundleHandle* handle);

        [DllImport(__DllName, EntryPoint = "xybrid_bundle_file_name", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern byte* xybrid_bundle_file_name(XybridBundleHandle* handle, uint index);

        [DllImport(__DllName, EntryPoint = "xybrid_bundle_free", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern void xybrid_bundle_free(XybridBundleHandle* handle);
    }
}
