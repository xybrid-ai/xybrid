// Xybrid SDK - Live ASR streaming
// Public wrapper for rolling-window microphone transcription.

using System;
using System.Threading;
using System.Threading.Tasks;

namespace Xybrid
{
    /// <summary>Configuration for mono 16 kHz live ASR.</summary>
    public sealed class AsrStreamConfig
    {
        public uint SampleRate { get; set; } = 16000;
        public bool EnableVad { get; set; }
        public float VadThreshold { get; set; } = 0.5f;
        public string VadModelDirectory { get; set; }
        public string Language { get; set; }
        public uint? AudioContext { get; set; }

        internal XybridBolt.XybridAsrStreamConfig ToBolt()
        {
            return new XybridBolt.XybridAsrStreamConfig(
                SampleRate,
                EnableVad,
                VadThreshold,
                VadModelDirectory,
                Language,
                AudioContext);
        }
    }

    /// <summary>A rolling transcript emitted as microphone audio is processed.</summary>
    public readonly struct AsrPartialResult
    {
        public string Text { get; }
        public bool IsStable { get; }
        public ulong ChunkIndex { get; }
        public ulong AudioDurationMs { get; }

        internal AsrPartialResult(XybridBolt.XybridAsrPartialResult result)
        {
            Text = result.Text;
            IsStable = result.IsStable;
            ChunkIndex = result.ChunkIndex;
            AudioDurationMs = result.AudioDurationMs;
        }
    }

    /// <summary>The final transcript returned after queued audio is drained.</summary>
    public readonly struct AsrTranscriptionResult
    {
        public string Text { get; }
        public ulong DurationMs { get; }
        public ulong ChunksProcessed { get; }

        internal AsrTranscriptionResult(XybridBolt.XybridAsrTranscriptionResult result)
        {
            Text = result.Text;
            DurationMs = result.DurationMs;
            ChunksProcessed = result.ChunksProcessed;
        }
    }

    /// <summary>
    /// A live ASR session that accepts PCM frames and yields rolling transcripts.
    /// </summary>
    /// <remarks>
    /// <see cref="Next"/> blocks while waiting for inference. Use
    /// <see cref="NextAsync"/> from Unity's main thread. Cancelling that task
    /// stops the native worker, so an outstanding pull does not leak a thread.
    /// </remarks>
    public sealed class AsrStreamSession : IDisposable
    {
        private readonly XybridBolt.XybridAsrStreamSession _bolt;
        private int _stopped;
        private bool _disposed;

        internal AsrStreamSession(XybridBolt.XybridAsrStreamSession bolt)
        {
            _bolt = bolt;
        }

        /// <summary>Queues mono 16 kHz PCM f32 samples for transcription.</summary>
        public void Feed(float[] samples)
        {
            ThrowIfDisposed();
            if (samples == null)
            {
                throw new ArgumentNullException(nameof(samples));
            }
            Execute(() => _bolt.Feed(samples));
        }

        /// <summary>Blocks until the next distinct rolling transcript is ready.</summary>
        public AsrPartialResult? Next()
        {
            ThrowIfDisposed();
            XybridBolt.XybridAsrPartialResult? result = Execute(() => _bolt.Next());
            if (!result.HasValue)
            {
                Interlocked.Exchange(ref _stopped, 1);
                return null;
            }
            return new AsrPartialResult(result.Value);
        }

        /// <summary>Waits for the next partial without blocking Unity's main thread.</summary>
        public async Task<AsrPartialResult?> NextAsync(
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            using (cancellationToken.Register(StopSilently))
            {
                AsrPartialResult? result = await Task.Run(() => Next()).ConfigureAwait(false);
                cancellationToken.ThrowIfCancellationRequested();
                return result;
            }
        }

        /// <summary>Drains queued audio and returns the complete transcript.</summary>
        public AsrTranscriptionResult Flush()
        {
            ThrowIfDisposed();
            XybridBolt.XybridAsrTranscriptionResult result = Execute(() => _bolt.Flush());
            return new AsrTranscriptionResult(result);
        }

        /// <summary>Flushes without blocking Unity's main thread.</summary>
        public Task<AsrTranscriptionResult> FlushAsync()
        {
            ThrowIfDisposed();
            return Task.Run(() => Flush());
        }

        /// <summary>Clears transcript state without reloading the model.</summary>
        public void Reset()
        {
            ThrowIfDisposed();
            Execute(() => _bolt.Reset());
            Interlocked.Exchange(ref _stopped, 0);
        }

        /// <summary>Stops the native worker and ends any outstanding pull.</summary>
        public void Stop()
        {
            ThrowIfDisposed();
            if (Interlocked.Exchange(ref _stopped, 1) == 0)
            {
                Execute(() => _bolt.Stop());
            }
        }

        private void StopSilently()
        {
            if (_disposed || Interlocked.Exchange(ref _stopped, 1) != 0)
            {
                return;
            }
            try
            {
                _bolt.Stop();
            }
            catch
            {
                // Cancellation and disposal are best-effort cleanup paths.
            }
        }

        private static T Execute<T>(Func<T> operation)
        {
            try
            {
                return operation();
            }
            catch (Exception ex) when (
                ex is XybridBolt.XybridErrorException || ex is XybridBolt.BoltException)
            {
                throw BoltErrors.Translate(ex);
            }
        }

        private static void Execute(Action operation)
        {
            Execute(() =>
            {
                operation();
                return true;
            });
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(AsrStreamSession));
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            StopSilently();
            _bolt.Dispose();
            _disposed = true;
        }
    }
}
