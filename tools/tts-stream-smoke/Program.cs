using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using Xybrid;

// Runs the actual Unity managed API against a locally staged Bolt plugin.
// This is CLR/native-host evidence; Unity Editor and IL2CPP require separate runs.
internal static class Program
{
    private const string Short = "Your name sounds like trouble.";
    private const string Long = "Tell me your name. Then explain your business. The village gate stays closed until I know who you are and what you want here today.";
    private static void Require(bool value, string message)
    {
        if (!value) throw new Exception(message);
    }
    private static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes));
    private static void CheckDeliveryCancellation(Model model)
    {
        using var cancel = new CancellationTokenSource();
        int callbacks = 0;
        var stopped = Speak(model, "first-packet-cancel", Short, cancel.Token, packet =>
        {
            callbacks++;
            Require(packet.FirstSample == 0, "stale packet reached callback after cancellation");
            cancel.Cancel();
        }, expectCompleted: false);
        Require(stopped.Status == TtsStatus.Cancelled && callbacks == 1,
            "delivery cancellation must retain first packet and suppress queued packets");
        Speak(model, "reuse-after-delivery-cancel", Short);
    }
    private static TtsStreamResult Speak(Model model, string name, string text,
        CancellationToken token = default, Action<TtsAudioChunk> extra = null,
        bool expectCompleted = true, bool expectLong = false)
    {
        var watch = Stopwatch.StartNew();
        var times = new List<double>();
        var saved = new List<(TtsAudioChunk packet, string hash)>();
        ulong end = 0;
        var caller = Environment.CurrentManagedThreadId;
        using var envelope = Envelope.Text(text);
        var result = model.RunTtsStreaming(envelope, packet =>
        {
            Require(caller == Environment.CurrentManagedThreadId, "callback changed caller thread");
            Require(packet.SampleRate == 24000 && packet.Channels == 1, "format mismatch");
            Require(packet.FirstSample == end, "noncontiguous packet offset");
            Require(packet.Pcm.Length > 0 && packet.Pcm.Length % 2 == 0, "invalid PCM16");
            end += (ulong)packet.Pcm.Length / 2;
            times.Add(watch.Elapsed.TotalMilliseconds);
            saved.Add((packet, Hash(packet.Pcm)));
            extra?.Invoke(packet);
        }, token);
        var elapsed = watch.Elapsed.TotalMilliseconds;
        foreach (var entry in saved) Require(Hash(entry.packet.Pcm) == entry.hash, "retained PCM changed");
        if (expectCompleted)
        {
            Require(result.Status == TtsStatus.Completed, name + ": " + result.Status + " " + result.Error);
            Require(result.Samples == end, "sample count mismatch");
            Require(saved.Any(e => e.packet.Pcm.Any(b => b != 0)), "silent result");
        }
        if (expectLong)
        {
            Require(end / 24000.0 > 5.1, "speech must exceed 5.1s");
            Require(times.Count > 1 && times[0] < elapsed, "no early packet delivery");
            Require(result.Chunks >= 3, "sentence mode must produce multiple chunks");
        }
        Console.WriteLine(JsonSerializer.Serialize(new {
            name, status = result.Status.ToString(), first_audio_ms = times.Count == 0 ? (double?)null : times[0],
            total_ms = elapsed, samples = result.Samples, chunks = result.Chunks, limited_chunks = result.LimitedChunks,
            packets = times.Count, speech_seconds = end / 24000.0,
            max_packet_gap_ms = times.Count < 2 ? 0 : times.Zip(times.Skip(1), (a,b) => b-a).Max(),
            packet_gaps_over_10ms = times.Zip(times.Skip(1), (a,b) => b-a).Where(gap => gap > 10).ToArray(),
            rss_bytes = Process.GetCurrentProcess().WorkingSet64,
            peak_rss_bytes = Process.GetCurrentProcess().PeakWorkingSet64, error = result.Error
        }));
        return result;
    }
    private static int Main(string[] args)
    {
        try
        {
            if (args.Length < 2) throw new Exception("Usage: kitten <bundle> <capped-bundle> [missing-bundle] | kokoro <bundle> | delivery <bundle>");
            var cold = Stopwatch.StartNew();
            using var loader = ModelLoader.FromDirectory(args[1]);
            using var model = loader.Load();
            Console.WriteLine(JsonSerializer.Serialize(new { name = "loader", total_ms = cold.Elapsed.TotalMilliseconds }));
            var baseline = Speak(model, "cold-short", Short);
            Speak(model, "warm-short", Short);
            if (args[0] == "kokoro")
            {
                using var input = Envelope.Text(Short);
                var batch = model.RunTts(Short);
                Require(batch.Length > 10000 && batch.Length % 2 == 0, "Kokoro batch PCM regression");
                var fallback = model.RunStreaming(input, _ => { });
                Require(fallback.Success && fallback.AudioBytes.Length > 10000, "token API lost audio fallback");
                Console.WriteLine("KOKORO PASS");
                return 0;
            }
            CheckDeliveryCancellation(model);
            if (args[0] == "delivery")
            {
                Console.WriteLine("DELIVERY CANCELLATION PASS (shutdown follows)");
                return 0;
            }
            Speak(model, "long-sentence", Long, expectLong: true);
            using (var cancel = new CancellationTokenSource())
            {
                cancel.Cancel();
                Require(Speak(model, "pre-cancel", Short, cancel.Token, expectCompleted: false).Status == TtsStatus.Cancelled, "pre-cancel failed");
            }
            using (var cancel = new CancellationTokenSource())
            {
                var watch = Stopwatch.StartNew();
                long requestTick = 0;
                using var observed = cancel.Token.Register(() =>
                    Interlocked.Exchange(ref requestTick, watch.ElapsedTicks));
                cancel.CancelAfter(200);
                var stopped = Speak(model, "during-computation-cancel", Long, cancel.Token, expectCompleted: false);
                Require(stopped.Status == TtsStatus.Cancelled, "live cancel failed");
                var returnedTick = watch.ElapsedTicks;
                var cancelledTick = Interlocked.Read(ref requestTick);
                Require(cancelledTick > 0, "cancellation callback not observed");
                Console.WriteLine(JsonSerializer.Serialize(new {
                    name = "cancel-latency",
                    request_ms = cancelledTick * 1000.0 / Stopwatch.Frequency,
                    after_request_ms = (returnedTick - cancelledTick) * 1000.0 / Stopwatch.Frequency
                }));
            }
            Speak(model, "reuse-after-cancel", Short);
            using (var cancel = new CancellationTokenSource())
            using (var input = Envelope.Text(Long))
            {
                cancel.CancelAfter(200);
                var batch = model.Run(input, cancellationToken: cancel.Token);
                Require(!batch.Success && batch.SpeechStatus == TtsStatus.Cancelled, "batch live cancel lost status");
            }
            using (var cancel = new CancellationTokenSource())
            {
                Require(Speak(model, "final-callback-cancel", Short, cancel.Token, packet => {
                    if (packet.FirstSample + (ulong)packet.Pcm.Length / 2 == baseline.Samples) cancel.Cancel();
                }, expectCompleted: false).Status == TtsStatus.Cancelled, "final callback cancel failed");
            }
            Speak(model, "reuse-after-final-cancel", Short);
            using (var input = Envelope.Text(Short))
            {
                var batch = model.Run(input);
                Require(batch.Success && batch.SpeechStatus == TtsStatus.Completed && batch.AudioBytes.Length > 10000, "Kitten batch failed");
            }
            try
            {
                Speak(model, "callback-exception", Short, extra: _ => throw new ApplicationException("host callback"));
                throw new Exception("host callback exception was swallowed");
            }
            catch (ApplicationException) { }
            Speak(model, "reuse-after-callback-exception", Short);
            using (var cappedLoader = ModelLoader.FromDirectory(args[2]))
            using (var capped = cappedLoader.Load())
            {
                var partial = Speak(capped, "explicit-cap", Long, expectCompleted: false);
                Require(partial.Status == TtsStatus.Limited && partial.LimitedChunks > 0 && partial.Samples > 0, "lost partial outcome");
                using var input = Envelope.Text(Long);
                var batch = capped.Run(input);
                Require(!batch.Success && batch.SpeechStatus == TtsStatus.Limited && batch.AudioBytes.Length > 0, "batch limit hidden");
            }
            if (args.Length > 3)
            {
                using var missingLoader = ModelLoader.FromDirectory(args[3]);
                using var missing = missingLoader.Load();
                Require(Speak(missing, "missing-assets", Short, expectCompleted: false).Status == TtsStatus.Failed, "missing assets not failed");
            }
            Console.WriteLine("KITTEN PASS (shutdown follows)");
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
