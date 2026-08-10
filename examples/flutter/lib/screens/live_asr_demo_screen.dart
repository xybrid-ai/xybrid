import 'dart:async';

import 'package:flutter/material.dart';
import 'package:xybrid_example/ui.dart';
import 'package:xybrid_example/utils/recorder.dart';
import 'package:xybrid_flutter/xybrid.dart';

/// Live (rolling-window) ASR demo.
///
/// Streams microphone audio into a [XybridStreamSession] and renders partial
/// transcripts as they arrive, then the final transcript on stop. Unlike the
/// batch Speech-to-Text demo, nothing is written to a file — PCM frames are fed
/// straight to the rolling-window engine.
class LiveAsrScreen extends StatefulWidget {
  const LiveAsrScreen({super.key});

  @override
  State<LiveAsrScreen> createState() => _LiveAsrScreenState();
}

/// High-level screen phases.
enum _Phase { idle, loadingModel, loadError, ready, listening, finalizing }

/// Which ASR engine runs the rolling window.
///
/// Only whisper.cpp remains. The Candle arm was removed when Candle left the
/// platform presets — a button that always fails is worse than no button. The
/// A/B harness that produced the comparison lives in git history at the
/// measurement commit, and the numbers are recorded in
/// `.context/whispercpp-design.md`: 9871 ms to first partial on Candle vs
/// 2724 ms on whisper.cpp, same Pixel 8, same session.
enum _Backend {
  /// The multilingual Q5_1 GGML bundle from the registry, run by whisper.cpp.
  ///
  /// Served as its own mask id rather than replacing `whisper-tiny` in place:
  /// that id still resolves to the SafeTensors/Candle bundle, so both
  /// generations of client keep working and neither ever gets an artifact it
  /// cannot execute.
  whisperCpp('whisper.cpp (Q5_1 GGML)', 'whisper-tiny-ggml');

  const _Backend(this.label, this.id);

  final String label;
  final String id;
}

class _LiveAsrScreenState extends State<LiveAsrScreen> {
  final _recorder = XybridRecorder();

  /// Which engine the loaded model runs on.
  _Backend _backend = _Backend.whisperCpp;

  _Phase _phase = _Phase.idle;
  double? _loadProgress;
  String _errorMessage = '';

  XybridModel? _model;
  XybridStreamSession? _session;
  StreamSubscription<LoadEvent>? _loadSub;
  StreamSubscription<FfiPartialResult>? _partialSub;

  /// Running transcript text while listening (the engine emits cumulative
  /// text per rolling-window chunk).
  String _liveText = '';

  /// The committed transcript after [flush].
  String _finalText = '';

  /// Chunks seen this session, for a bit of live feedback.
  int _chunks = 0;

  /// When the microphone started feeding the engine — the origin for the
  /// first-partial latency below.
  DateTime? _streamStartedAt;

  /// When the previous partial arrived, so each new one yields a cadence.
  DateTime? _lastPartialAt;

  /// Milliseconds from stream start to the first partial transcript.
  int? _firstPartialMs;

  /// Gaps between consecutive partials, in milliseconds.
  final List<int> _cadencesMs = [];

  /// Median cadence, which is more honest than a mean here: the first steady
  /// window after the warm-up ones is systematically slower and would drag an
  /// average that a user never experiences.
  int? get _medianCadenceMs {
    if (_cadencesMs.isEmpty) return null;
    final sorted = [..._cadencesMs]..sort();
    return sorted[sorted.length ~/ 2];
  }

  @override
  void dispose() {
    _loadSub?.cancel();
    _partialSub?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  // ── Model loading ──────────────────────────────────────────────────────

  Future<void> _loadModel(_Backend backend) async {
    setState(() {
      _backend = backend;
      _phase = _Phase.loadingModel;
      _loadProgress = null;
      _errorMessage = '';
    });

    // The bundle's `GgmlWhisper` template routes the engine to whisper.cpp.
    final loader = XybridModelLoader.fromRegistry(backend.id);
    _loadSub = loader.loadWithProgress().listen(
      (event) async {
        if (!mounted) return;
        switch (event) {
          case LoadProgress(:final progress):
            setState(() => _loadProgress = progress);
          case LoadComplete():
            try {
              final model = await loader.load();
              if (!mounted) return;
              setState(() {
                _model = model;
                _phase = _Phase.ready;
              });
            } catch (e) {
              _showLoadError(e);
            }
          case LoadError(:final message):
            _showLoadError(message);
        }
      },
      onError: _showLoadError,
    );
  }

  void _showLoadError(Object error) {
    if (!mounted) return;
    setState(() {
      _phase = _Phase.loadError;
      _errorMessage = error.toString();
    });
  }

  // ── Streaming ──────────────────────────────────────────────────────────

  Future<void> _start() async {
    final model = _model;
    if (model == null) return;

    // Microphone permission is required before streaming.
    if (!await _recorder.hasPermission()) {
      await _recorder.requestPermission();
      if (!await _recorder.hasPermission()) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Microphone permission denied')),
          );
        }
        return;
      }
    }

    try {
      final session = await XybridStreamSession.fromModel(model);

      // Listen for partials *before* feeding any audio.
      _partialSub = session.partials.listen((partial) {
        final now = DateTime.now();
        _recordPartialTiming(now);
        if (!mounted) return;
        setState(() {
          _liveText = partial.text;
          _chunks = partial.chunkSequence.toInt() + 1;
        });
      });

      // Pipe microphone PCM (already f32 mono 16 kHz) straight into the engine.
      await _recorder.startStreaming(onSamples: session.feed);
      _streamStartedAt = DateTime.now();
      _lastPartialAt = null;
      _firstPartialMs = null;
      _cadencesMs.clear();

      setState(() {
        _session = session;
        _phase = _Phase.listening;
        _liveText = '';
        _finalText = '';
        _chunks = 0;
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          _phase = _Phase.ready;
          _errorMessage = e.toString();
        });
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Failed to start: $e')));
      }
    }
  }

  /// Record first-partial latency and inter-partial cadence.
  ///
  /// Also logged to the console with a grep-able prefix so a run can be
  /// measured over `adb logcat` without reading the screen.
  void _recordPartialTiming(DateTime now) {
    final startedAt = _streamStartedAt;
    if (startedAt == null) return;

    if (_firstPartialMs == null) {
      _firstPartialMs = now.difference(startedAt).inMilliseconds;
      debugPrint('[ASRPERF] backend=${_backend.id} first_partial_ms=$_firstPartialMs');
    } else if (_lastPartialAt != null) {
      final cadence = now.difference(_lastPartialAt!).inMilliseconds;
      _cadencesMs.add(cadence);
      debugPrint('[ASRPERF] backend=${_backend.id} cadence_ms=$cadence');
    }
    _lastPartialAt = now;
  }

  Future<void> _stop() async {
    final session = _session;
    if (session == null) return;

    setState(() => _phase = _Phase.finalizing);
    try {
      await _recorder.stopStreaming();
      final transcript = await session.flush();
      await _partialSub?.cancel();
      _partialSub = null;
      if (!mounted) return;
      setState(() {
        _finalText = transcript;
        _liveText = '';
        _session = null;
        _phase = _Phase.ready;
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          _phase = _Phase.ready;
          _errorMessage = e.toString();
        });
      }
    }
  }

  // ── UI ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Live ASR'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: switch (_phase) {
          _Phase.idle => _IdleView(onLoad: _loadModel),
          _Phase.loadingModel => _LoadingView(progress: _loadProgress),
          _Phase.loadError => ErrorCard(
            errorMessage: _errorMessage,
            onRetry: () => _loadModel(_backend),
          ),
          _ => _ActiveView(
            modelId: _backend.label,
            firstPartialMs: _firstPartialMs,
            medianCadenceMs: _medianCadenceMs,
            listening: _phase == _Phase.listening,
            finalizing: _phase == _Phase.finalizing,
            liveText: _liveText,
            finalText: _finalText,
            chunks: _chunks,
            onStart: _start,
            onStop: _stop,
          ),
        },
      ),
    );
  }
}

class _IdleView extends StatelessWidget {
  const _IdleView({required this.onLoad});

  final ValueChanged<_Backend> onLoad;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.graphic_eq, size: 72),
          const SizedBox(height: 24),
          Text(
            'Live, rolling-window speech-to-text.',
            style: Theme.of(context).textTheme.bodyLarge,
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 24),
          // One button per engine: the point of this screen is the comparison,
          // so make it a choice rather than a build-time constant.
          for (final backend in _Backend.values) ...[
            FilledButton.icon(
              onPressed: () => onLoad(backend),
              icon: const Icon(Icons.download),
              label: Text(backend.label),
            ),
            const SizedBox(height: 12),
          ],
        ],
      ),
    );
  }
}

class _LoadingView extends StatelessWidget {
  const _LoadingView({required this.progress});

  final double? progress;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          CircularProgressIndicator(value: progress),
          const SizedBox(height: 24),
          Text(
            progress == null
                ? 'Preparing model…'
                : 'Downloading model… ${(progress! * 100).toStringAsFixed(0)}%',
          ),
        ],
      ),
    );
  }
}

class _ActiveView extends StatelessWidget {
  const _ActiveView({
    required this.modelId,
    required this.firstPartialMs,
    required this.medianCadenceMs,
    required this.listening,
    required this.finalizing,
    required this.liveText,
    required this.finalText,
    required this.chunks,
    required this.onStart,
    required this.onStop,
  });

  final String modelId;

  /// Milliseconds from stream start to the first partial — the number a user
  /// actually feels when they start speaking.
  final int? firstPartialMs;

  /// Median gap between partials once the stream is running.
  final int? medianCadenceMs;

  final bool listening;
  final bool finalizing;
  final String liveText;
  final String finalText;
  final int chunks;
  final VoidCallback onStart;
  final VoidCallback onStop;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final showText = listening ? liveText : finalText;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                InfoRow(label: 'Model', value: modelId),
                InfoRow(
                  label: 'Status',
                  value: finalizing
                      ? 'Finalizing…'
                      : listening
                      ? 'Listening ($chunks chunks)'
                      : 'Ready',
                ),
                if (firstPartialMs != null)
                  InfoRow(
                    label: 'First partial',
                    value: '${(firstPartialMs! / 1000).toStringAsFixed(1)} s',
                  ),
                if (medianCadenceMs != null)
                  InfoRow(
                    label: 'Cadence (median)',
                    value: '${(medianCadenceMs! / 1000).toStringAsFixed(1)} s',
                  ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        Expanded(
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: theme.colorScheme.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(12),
            ),
            child: SingleChildScrollView(
              child: Text(
                showText.isEmpty
                    ? (listening ? 'Speak now…' : 'No transcript yet.')
                    : showText,
                style: theme.textTheme.titleMedium?.copyWith(
                  color: showText.isEmpty
                      ? theme.colorScheme.onSurfaceVariant
                      : theme.colorScheme.onSurface,
                  fontStyle: listening ? FontStyle.italic : FontStyle.normal,
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 16),
        if (listening)
          FilledButton.icon(
            onPressed: onStop,
            icon: const Icon(Icons.stop),
            label: const Text('Stop'),
            style: FilledButton.styleFrom(
              backgroundColor: theme.colorScheme.error,
            ),
          )
        else
          FilledButton.icon(
            onPressed: finalizing ? null : onStart,
            icon: const Icon(Icons.mic),
            label: const Text('Start listening'),
          ),
      ],
    );
  }
}
