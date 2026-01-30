// Xybrid Unity SDK - P/Invoke bindings for crates/xybrid-ffi/include/xybrid.h
using System;
using System.Runtime.InteropServices;

namespace Xybrid
{
    /// <summary>Native P/Invoke declarations for xybrid-ffi.</summary>
    internal static class Native
    {
        private const string Lib = "xybrid_ffi";

        // TODO: Uncomment when native library is available
        // [DllImport(Lib)] public static extern int xybrid_init();
        // [DllImport(Lib)] public static extern IntPtr xybrid_version();
        // [DllImport(Lib)] public static extern IntPtr xybrid_last_error();
        // [DllImport(Lib)] public static extern void xybrid_free_string(IntPtr s);
        // [DllImport(Lib)] public static extern IntPtr xybrid_model_loader_from_registry(string modelId);
        // [DllImport(Lib)] public static extern IntPtr xybrid_model_loader_from_bundle(string path);
        // [DllImport(Lib)] public static extern IntPtr xybrid_model_loader_load(IntPtr loader);
        // [DllImport(Lib)] public static extern void xybrid_model_loader_free(IntPtr loader);
        // [DllImport(Lib)] public static extern IntPtr xybrid_envelope_audio(byte[] bytes, UIntPtr len, uint sampleRate, uint channels);
        // [DllImport(Lib)] public static extern IntPtr xybrid_envelope_text(string text);
        // [DllImport(Lib)] public static extern void xybrid_envelope_free(IntPtr envelope);
        // [DllImport(Lib)] public static extern IntPtr xybrid_model_run(IntPtr model, IntPtr envelope);
        // [DllImport(Lib)] public static extern IntPtr xybrid_model_id(IntPtr model);
        // [DllImport(Lib)] public static extern void xybrid_model_free(IntPtr model);
        // [DllImport(Lib)] public static extern int xybrid_result_success(IntPtr result);
        // [DllImport(Lib)] public static extern IntPtr xybrid_result_error(IntPtr result);
        // [DllImport(Lib)] public static extern IntPtr xybrid_result_text(IntPtr result);
        // [DllImport(Lib)] public static extern uint xybrid_result_latency_ms(IntPtr result);
        // [DllImport(Lib)] public static extern void xybrid_result_free(IntPtr result);
    }

    /// <summary>Creates and loads ML models from the registry or local bundles.</summary>
    public class ModelLoader
    {
        public static ModelLoader FromRegistry(string modelId) =>
            throw new NotImplementedException("Native library not yet available");

        public static ModelLoader FromBundle(string path) =>
            throw new NotImplementedException("Native library not yet available");

        public Model Load() =>
            throw new NotImplementedException("Native library not yet available");
    }

    /// <summary>A loaded ML model ready for inference.</summary>
    public class Model : IDisposable
    {
        public Result Run(Envelope envelope) =>
            throw new NotImplementedException("Native library not yet available");

        public void Dispose() { /* TODO: Native.xybrid_model_free */ }
    }

    /// <summary>Input data envelope for inference (audio or text).</summary>
    public class Envelope : IDisposable
    {
        public static Envelope Audio(byte[] bytes, uint sampleRate, uint channels) =>
            throw new NotImplementedException("Native library not yet available");

        public static Envelope Text(string text) =>
            throw new NotImplementedException("Native library not yet available");

        public void Dispose() { /* TODO: Native.xybrid_envelope_free */ }
    }

    /// <summary>Inference result containing output data.</summary>
    public class Result : IDisposable
    {
        public bool Success => throw new NotImplementedException();
        public string Error => throw new NotImplementedException();
        public string Text => throw new NotImplementedException();
        public uint LatencyMs => throw new NotImplementedException();

        public void Dispose() { /* TODO: Native.xybrid_result_free */ }
    }
}
