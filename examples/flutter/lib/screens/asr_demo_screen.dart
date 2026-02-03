import 'dart:async';

import 'package:flutter/material.dart';
import 'package:xybrid_example/utils/recorder.dart';
import 'package:xybrid_example/widgets/model_status_card.dart';
import 'package:xybrid_flutter/xybrid.dart';

/// Speech-to-Text demo screen.
///
/// Demonstrates how to run audio-based inference (ASR) using the Xybrid SDK.
class SpeechToTextScreen extends StatefulWidget {
  const SpeechToTextScreen({super.key});

  @override
  State<SpeechToTextScreen> createState() => _SpeechToTextScreenState();
}

/// Sealed class representing ASR screen states.
sealed class AsrScreenState {
  const AsrScreenState();
}

/// Initial state before model is loaded.
class AsrIdle extends AsrScreenState {
  const AsrIdle();
}

/// Model is being downloaded/loaded with progress.
class AsrLoadingModel extends AsrScreenState {
  final double? progress;

  const AsrLoadingModel({this.progress});
}

/// Model failed to load.
class AsrLoadError extends AsrScreenState {
  final String message;

  const AsrLoadError(this.message);
}

/// Model is loaded and ready for inference.
class AsrReady extends AsrScreenState {
  const AsrReady();
}

/// Recording audio from microphone.
class AsrRecording extends AsrScreenState {
  const AsrRecording();
}

/// Transcribing recorded audio.
class AsrTranscribing extends AsrScreenState {
  const AsrTranscribing();
}

/// Inference failed with an error.
class AsrError extends AsrScreenState {
  final String message;

  const AsrError(this.message);
}

class _SpeechToTextScreenState extends State<SpeechToTextScreen> {
  /// Audio recorder instance.
  final _recorder = XybridRecorder();

  /// Current screen state.
  AsrScreenState _state = const AsrIdle();

  /// Loaded ASR model (if any).
  XybridModel? _model;

  /// The model ID used for ASR.
  static const _asrModelId = 'whisper-tiny';

  /// Latency of the last inference in milliseconds.
  int? _latencyMs;

  /// Transcribed text from last inference.
  String? _transcribedText;

  /// Sample rate for audio recording (Whisper expects 16kHz).
  static const _sampleRate = 16000;

  /// Load progress stream subscription.
  StreamSubscription<LoadEvent>? _loadSubscription;

  @override
  void dispose() {
    _recorder.dispose();
    _loadSubscription?.cancel();
    super.dispose();
  }

  /// Load the ASR model from the registry with progress tracking.
  Future<void> _loadModel() async {
    setState(() {
      _state = const AsrLoadingModel();
    });

    try {
      final loader = XybridModelLoader.fromRegistry(_asrModelId);

      _loadSubscription = loader.loadWithProgress().listen(
        (event) {
          if (!mounted) return;

          switch (event) {
            case LoadProgress(:final progress):
              setState(() {
                _state = AsrLoadingModel(progress: progress);
              });
            case LoadComplete():
              _finalizeModelLoad(loader);
            case LoadError(:final message):
              setState(() {
                _state = AsrLoadError(message);
              });
          }
        },
        onError: (e) {
          if (mounted) {
            setState(() {
              _state = AsrLoadError(_formatErrorMessage(e.toString()));
            });
          }
        },
      );
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = AsrLoadError(_formatErrorMessage(e.toString()));
        });
      }
    }
  }

  /// Finalize model load after download completes.
  Future<void> _finalizeModelLoad(XybridModelLoader loader) async {
    try {
      final model = await loader.load();
      if (mounted) {
        setState(() {
          _state = const AsrReady();
          _model = model;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = AsrLoadError(_formatErrorMessage(e.toString()));
        });
      }
    }
  }

  /// Start recording audio from microphone.
  Future<void> _startRecording() async {
    if (_state is AsrRecording) {
      throw StateError('Already recording');
    }

    try {
      // Configure recording for Whisper (16kHz mono WAV)
      await _recorder.start(config: RecordingConfig.asr);

      setState(() {
        _state = const AsrRecording();
        _transcribedText = null;
        _latencyMs = null;
      });

      final recording = await _recorder.waitForCompletion();
      await _transcribeFromFilePath(recording.path);
    } catch (e) {
      setState(() {
        _state = AsrError('Failed to start recording: $e');
      });
    }
  }

  /// Stop recording and run inference.
  Future<void> _stopRecordingAndTranscribe() async {
    // Stop recording
    final recording = await _recorder.stop();
    _transcribeFromFilePath(recording.path);
  }

  Future<void> _transcribeFromFilePath(String path) async {
    try {
      setState(() {
        _state = const AsrTranscribing();
      });

      final audioBytes = await _recorder.readAudioBytes(path);

      if (audioBytes.isEmpty) {
        setState(() {
          _state = const AsrError('Recorded audio is empty.');
        });
        return;
      }

      // Create audio envelope and run inference
      final envelope = XybridEnvelope.audio(
        bytes: audioBytes.toList(),
        sampleRate: _sampleRate,
        channels: 1,
      );

      final result = await _model!.run(envelope);

      if (!result.success) {
        throw XybridException('Inference returned unsuccessful result');
      }

      final transcribedText = result.text;
      if (transcribedText == null || transcribedText.isEmpty) {
        setState(() {
          _state = const AsrReady();
          _transcribedText = '(No speech detected)';
          _latencyMs = result.latencyMs;
        });
        return;
      }

      if (mounted) {
        setState(() {
          _state = const AsrReady();
          _transcribedText = transcribedText;
          _latencyMs = result.latencyMs;
        });
      }
    } on XybridException catch (e) {
      if (mounted) {
        setState(() {
          _state = AsrError(e.message);
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = AsrError(_formatErrorMessage(e.toString()));
        });
      }
    }
  }

  /// Format error message for user display.
  String _formatErrorMessage(String error) {
    if (error.contains('network') || error.contains('Network')) {
      return 'Network error: Please check your internet connection and try again.';
    }
    if (error.contains('not found') || error.contains('NotFound')) {
      return 'Model not found: The model may not be available in the registry.';
    }
    if (error.contains('timeout') || error.contains('Timeout')) {
      return 'Request timed out: The server took too long to respond.';
    }
    if (error.contains('permission') || error.contains('Permission')) {
      return 'Permission denied: Please grant microphone access in your device settings.';
    }
    return error;
  }

  /// Convert screen state to ModelLoadState for the status card.
  ModelLoadState get _modelLoadState {
    return switch (_state) {
      AsrIdle() => const ModelIdle(),
      AsrLoadingModel(:final progress) => ModelLoading(progress: progress),
      AsrLoadError(:final message) => ModelLoadError(message),
      AsrReady() ||
      AsrRecording() ||
      AsrTranscribing() ||
      AsrError() =>
        const ModelReady(),
    };
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        backgroundColor: theme.colorScheme.inversePrimary,
        title: const Text('Speech-to-Text'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Description card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          Icons.info_outline,
                          color: theme.colorScheme.primary,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'About ASR Demo',
                          style: theme.textTheme.titleMedium,
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'This demo shows how to use XybridEnvelope.audio() to run '
                      'speech recognition. Tap the microphone button to record audio, '
                      'then see the transcribed text.',
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Model status section
            ModelStatusCard(
              state: _modelLoadState,
              modelId: _asrModelId,
              idleIcon: Icons.mic,
              idleTitle: 'ASR Model Required',
              idleDescription:
                  'Load the $_asrModelId model to start transcribing speech.',
              onLoad: _loadModel,
            ),
            const SizedBox(height: 24),

            // Recording section (only shown when model is loaded)
            if (_model != null) _buildRecordingSection(theme),

            // Result section
            if (_transcribedText != null || _latencyMs != null)
              _buildResultSection(theme),

            // Error display
            if (_state case AsrError(:final message))
              _buildErrorSection(theme, message),
          ],
        ),
      ),
    );
  }

  /// Build the recording section.
  Widget _buildRecordingSection(ThemeData theme) {
    final isRecording = _state is AsrRecording;
    final isTranscribing = _state is AsrTranscribing;
    final canRecord = _state is AsrReady || _state is AsrError;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            // Recording indicator
            if (isRecording) ...[
              Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  color: theme.colorScheme.error.withAlpha(30),
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  Icons.mic,
                  size: 48,
                  color: theme.colorScheme.error,
                ),
              ),
              const SizedBox(height: 16),
              Text(
                'Recording...',
                style: theme.textTheme.titleMedium?.copyWith(
                  color: theme.colorScheme.error,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Speak clearly into the microphone',
                style: theme.textTheme.bodyMedium,
              ),
            ] else if (isTranscribing) ...[
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text('Transcribing...', style: theme.textTheme.titleMedium),
              const SizedBox(height: 8),
              Text(
                'Running speech recognition model',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ] else ...[
              Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  color: theme.colorScheme.primary.withAlpha(30),
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  Icons.mic_none,
                  size: 48,
                  color: theme.colorScheme.primary,
                ),
              ),
              const SizedBox(height: 16),
              Text('Ready to Record', style: theme.textTheme.titleMedium),
              const SizedBox(height: 8),
              Text(
                'Tap the button below to start recording',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
            const SizedBox(height: 24),

            // Record/Stop button
            if (isRecording)
              FilledButton.icon(
                onPressed: _stopRecordingAndTranscribe,
                icon: const Icon(Icons.stop),
                label: const Text('Stop & Transcribe'),
                style: FilledButton.styleFrom(
                  backgroundColor: theme.colorScheme.error,
                ),
              )
            else
              FilledButton.icon(
                onPressed: canRecord ? _startRecording : null,
                icon: const Icon(Icons.mic),
                label: const Text('Record'),
              ),
          ],
        ),
      ),
    );
  }

  /// Build the result section.
  Widget _buildResultSection(ThemeData theme) {
    return Card(
      color: theme.colorScheme.surfaceContainerHighest.withAlpha(100),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.text_snippet_outlined,
                  color: theme.colorScheme.primary,
                ),
                const SizedBox(width: 8),
                Text(
                  'Transcription Result',
                  style: theme.textTheme.titleMedium?.copyWith(
                    color: theme.colorScheme.primary,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            if (_transcribedText != null)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surface,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: theme.colorScheme.outline.withAlpha(50),
                  ),
                ),
                child: Text(
                  _transcribedText!,
                  style: theme.textTheme.bodyLarge,
                ),
              ),
            if (_latencyMs != null) ...[
              const SizedBox(height: 12),
              Row(
                children: [
                  Icon(
                    Icons.timer_outlined,
                    size: 18,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                  const SizedBox(width: 6),
                  Text(
                    'Inference latency: ${_latencyMs}ms',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  /// Build the error section.
  Widget _buildErrorSection(ThemeData theme, String message) {
    return Card(
      color: theme.colorScheme.errorContainer.withAlpha(100),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.error_outline, color: theme.colorScheme.error),
                const SizedBox(width: 8),
                Text(
                  'Error',
                  style: theme.textTheme.titleMedium?.copyWith(
                    color: theme.colorScheme.error,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              message,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onErrorContainer,
              ),
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _model == null ? _loadModel : _startRecording,
              icon: const Icon(Icons.refresh),
              label: Text(_model == null ? 'Load Model' : 'Try Again'),
            ),
          ],
        ),
      ),
    );
  }
}
