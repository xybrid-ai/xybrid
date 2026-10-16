using System;
using System.Globalization;

namespace Xybrid
{
    /// <summary>Terminal outcome of a speech request.</summary>
    public enum TtsStatus { Completed, Cancelled, Limited, Failed }

    /// <summary>Owned PCM16 little-endian audio and its utterance offset.</summary>
    public sealed class TtsAudioChunk
    {
        public byte[] Pcm { get; }
        public uint SampleRate { get; }
        public uint Channels { get; }
        /// <summary>Offset in samples per channel, starting at zero.</summary>
        public ulong FirstSample { get; }
        internal TtsAudioChunk(XybridBolt.XybridTtsAudioChunk chunk)
        {
            Pcm = chunk.Pcm;
            SampleRate = chunk.SampleRate;
            Channels = chunk.Channels;
            FirstSample = chunk.FirstSample;
        }
    }

    /// <summary>Speech outcome. Limited audio is partial; failed requests carry an error.</summary>
    public sealed class TtsStreamResult
    {
        public TtsStatus Status { get; }
        public uint SampleRate { get; }
        public uint Channels { get; }
        public ulong Samples { get; }
        public uint Chunks { get; }
        public uint LimitedChunks { get; }
        public string Error { get; }
        internal TtsStreamResult(TtsStatus status, uint rate = 0, uint channels = 0,
            ulong samples = 0, uint chunks = 0, uint limited = 0, string error = null)
        {
            Status = status; SampleRate = rate; Channels = channels;
            Samples = samples; Chunks = chunks; LimitedChunks = limited; Error = error;
        }
        internal static TtsStreamResult FromBolt(XybridBolt.XybridResult result)
        {
            string Find(string key)
            {
                foreach (var entry in result.Envelope.Metadata)
                    if (entry.Key == key) return entry.Value;
                return null;
            }
            ulong Number(string key) => ulong.TryParse(Find(key), NumberStyles.None,
                CultureInfo.InvariantCulture, out var value) ? value : 0;
            var status = Find("tts_status");
            if (status != "completed" && status != "cancelled" && status != "limited")
                return new TtsStreamResult(TtsStatus.Failed, error: "Missing TTS terminal status");
            return new TtsStreamResult(status == "cancelled" ? TtsStatus.Cancelled :
                status == "limited" ? TtsStatus.Limited : TtsStatus.Completed,
                (uint)Number("sample_rate"), (uint)Number("channels"), Number("samples"),
                (uint)Number("chunks"), (uint)Number("zzz_limited_chunks"));
        }
    }
}
