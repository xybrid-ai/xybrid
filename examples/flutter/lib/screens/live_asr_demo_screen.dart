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

class _LiveAsrScreenState extends State<LiveAsrScreen> {
  final _recorder = XybridRecorder();

  /// ASR model loaded from the registry (backend auto-detected by the engine).
  static const _modelId = 'whisper-tiny';

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

  @override
  void dispose() {
    _loadSub?.cancel();
    _partialSub?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  // ── Model loading ──────────────────────────────────────────────────────

  void _loadModel() {
    setState(() {
      _phase = _Phase.loadingModel;
      _loadProgress = null;
      _errorMessage = '';
    });

    final loader = XybridModelLoader.fromRegistry(_modelId);
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
        if (!mounted) return;
        setState(() {
          _liveText = partial.text;
          _chunks = partial.chunkSequence.toInt() + 1;
        });
      });

      // Pipe microphone PCM (already f32 mono 16 kHz) straight into the engine.
      await _recorder.startStreaming(onSamples: session.feed);

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
            onRetry: _loadModel,
          ),
          _ => _ActiveView(
            modelId: _modelId,
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

  final VoidCallback onLoad;

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
          FilledButton.icon(
            onPressed: onLoad,
            icon: const Icon(Icons.download),
            label: const Text('Load ASR model'),
          ),
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
    required this.listening,
    required this.finalizing,
    required this.liveText,
    required this.finalText,
    required this.chunks,
    required this.onStart,
    required this.onStop,
  });

  final String modelId;
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
