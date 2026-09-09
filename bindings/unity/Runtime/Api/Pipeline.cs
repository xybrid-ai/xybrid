// Xybrid SDK - Pipeline
// Public wrapper for multi-stage inference pipelines.

using System;
using System.Collections.Generic;

namespace Xybrid
{
    /// <summary>
    /// A loaded multi-stage inference pipeline.
    /// </summary>
    /// <remarks>
    /// Construction parses the YAML and resolves its model references. The
    /// first <see cref="Run"/> downloads any required local models.
    /// Dispose the pipeline when it is no longer needed to release its native
    /// handle.
    /// </remarks>
    public sealed class Pipeline : IDisposable
    {
        private readonly XybridBolt.XybridPipeline _bolt;
        private bool _disposed;

        private Pipeline(XybridBolt.XybridPipeline bolt)
        {
            _bolt = bolt;
        }

        /// <summary>Gets whether this pipeline has been disposed.</summary>
        public bool IsDisposed => _disposed;

        /// <summary>Gets the optional name from the pipeline definition.</summary>
        public string Name
        {
            get
            {
                ThrowIfDisposed();
                return _bolt.Name();
            }
        }

        /// <summary>Gets stage identifiers in execution order.</summary>
        public IReadOnlyList<string> StageNames
        {
            get
            {
                ThrowIfDisposed();
                return _bolt.StageNames();
            }
        }

        /// <summary>Gets the number of stages.</summary>
        public uint StageCount
        {
            get
            {
                ThrowIfDisposed();
                return _bolt.StageCount();
            }
        }

        /// <summary>Parses and loads a pipeline from YAML content.</summary>
        public static Pipeline FromYaml(string yaml)
        {
            if (yaml == null)
            {
                throw new ArgumentNullException(nameof(yaml));
            }

            return Create(() => XybridBolt.XybridPipeline.FromYaml(yaml));
        }

        /// <summary>Reads, parses, and loads a pipeline from a YAML file.</summary>
        public static Pipeline FromFile(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException(nameof(path));
            }

            return Create(() => XybridBolt.XybridPipeline.FromFile(path));
        }

        /// <summary>Loads a pipeline bundle.</summary>
        public static Pipeline FromBundle(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException(nameof(path));
            }

            return Create(() => XybridBolt.XybridPipeline.FromBundle(path));
        }

        /// <summary>
        /// Runs every stage and returns the final stage's output.
        /// </summary>
        /// <remarks>
        /// <see cref="InferenceResult.Metrics"/> includes one stage-latency
        /// entry per executed stage. <see cref="InferenceResult.ExecutionTarget"/>
        /// describes the final stage that produced the returned output.
        /// </remarks>
        public InferenceResult Run(Envelope envelope)
        {
            ThrowIfDisposed();
            if (envelope == null)
            {
                throw new ArgumentNullException(nameof(envelope));
            }

            try
            {
                return InferenceResult.FromBolt(_bolt.Run(envelope.Bolt));
            }
            catch (Exception ex) when (
                ex is XybridBolt.XybridErrorException || ex is XybridBolt.BoltException)
            {
                throw BoltErrors.Translate(ex);
            }
        }

        private static Pipeline Create(Func<XybridBolt.XybridPipeline> create)
        {
            try
            {
                return new Pipeline(create());
            }
            catch (Exception ex) when (
                ex is XybridBolt.XybridErrorException || ex is XybridBolt.BoltException)
            {
                throw BoltErrors.Translate(ex);
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(Pipeline));
            }
        }

        /// <summary>Releases the native pipeline handle.</summary>
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _bolt.Dispose();
            _disposed = true;
        }
    }
}
