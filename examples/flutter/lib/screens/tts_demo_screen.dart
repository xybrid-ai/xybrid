import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:xybrid_flutter/xybrid.dart';
import 'package:just_audio/just_audio.dart';

/// Text-to-Speech demo screen.
///
/// Demonstrates how to run text-based inference (TTS) using the Xybrid SDK.

class TextToSpeechScreen extends StatefulWidget {
  const TextToSpeechScreen({super.key});

  @override
  State<TextToSpeechScreen> createState() => _TextToSpeechScreenState();
}

class _TextToSpeechScreenState extends State<TextToSpeechScreen> {
  /// Controller for the text input field.
  final _textController = TextEditingController(
    text: 'Hello, welcome to Xybrid!',
  );

  /// Audio player for TTS output.
  final _audioPlayer = AudioPlayer();

  /// Current screen state.
  _TtsState _state = _TtsState.idle;

  /// Loaded TTS model (if any).
  XybridModel? _model;

  /// The model ID used for TTS.
  static const _ttsModelId = 'kokoro-82m';

  /// Error message if operation failed.
  String? _errorMessage;

  /// Latency of the last inference in milliseconds.
  int? _latencyMs;

  /// Whether audio is currently playing.
  bool _isPlaying = false;

  @override
  void initState() {
    super.initState();
    _audioPlayer.playerStateStream.listen((state) {
      if (mounted) {
        setState(() {
          _isPlaying = state.playing;
        });
      }
    });
  }

  @override
  void dispose() {
    _textController.dispose();
    _audioPlayer.dispose();
    super.dispose();
  }

  /// Load the TTS model from the registry.
  Future<void> _loadModel() async {
    setState(() {
      _state = _TtsState.loadingModel;
      _errorMessage = null;
    });

    try {
      final loader = XybridModelLoader.fromRegistry(_ttsModelId);
      final model = await loader.load();
      if (mounted) {
        setState(() {
          _state = _TtsState.ready;
          _model = model;
        });
      }
    } on XybridException catch (e) {
      if (mounted) {
        setState(() {
          _state = _TtsState.error;
          _errorMessage = e.message;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = _TtsState.error;
          _errorMessage = _formatErrorMessage(e.toString());
        });
      }
    }
  }

  /// Run TTS inference on the input text.
  Future<void> _speak() async {
    final text = _textController.text.trim();
    if (text.isEmpty) {
      setState(() {
        _state = _TtsState.error;
        _errorMessage = 'Please enter some text to speak.';
      });
      return;
    }

    if (_model == null) {
      setState(() {
        _state = _TtsState.error;
        _errorMessage = 'Model not loaded. Please load the model first.';
      });
      return;
    }

    setState(() {
      _state = _TtsState.synthesizing;
      _errorMessage = null;
      _latencyMs = null;
    });

    try {
      final envelope = XybridEnvelope.text(text);
      final result = await _model!.run(envelope);

      if (!result.success) {
        throw XybridException('Inference returned unsuccessful result');
      }

      // Get audio as WAV format (ready for playback)
      final wavBytes = result.audioAsWav();
      if (wavBytes == null || wavBytes.isEmpty) {
        throw XybridException('No audio output received from model');
      }

      if (mounted) {
        setState(() {
          _state = _TtsState.playing;
          _latencyMs = result.latencyMs;
        });

        await _playAudio(wavBytes);

        if (mounted) {
          setState(() {
            _state = _TtsState.ready;
          });
        }
      }
    } on XybridException catch (e) {
      if (mounted) {
        setState(() {
          _state = _TtsState.error;
          _errorMessage = e.message;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = _TtsState.error;
          _errorMessage = _formatErrorMessage(e.toString());
        });
      }
    }
  }

  /// Play WAV audio bytes using just_audio.
  Future<void> _playAudio(Uint8List wavBytes) async {
    try {
      final source = _AudioBytesSource(wavBytes);
      await _audioPlayer.setAudioSource(source);
      await _audioPlayer.play();

      // Wait for playback to complete
      await _audioPlayer.processingStateStream.firstWhere(
        (state) => state == ProcessingState.completed,
      );
    } catch (e) {
      throw XybridException('Audio playback failed: $e');
    }
  }

  /// Stop audio playback.
  Future<void> _stopAudio() async {
    await _audioPlayer.stop();
    if (mounted && _state == _TtsState.playing) {
      setState(() {
        _state = _TtsState.ready;
      });
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
    return error;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        backgroundColor: theme.colorScheme.inversePrimary,
        title: const Text('Text-to-Speech'),
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
                          'About TTS Demo',
                          style: theme.textTheme.titleMedium,
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'This demo shows how to use XybridEnvelope.text() to run '
                      'text-to-speech inference. Enter text below and tap '
                      '"Speak" to hear the synthesized audio.',
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Model status section
            _buildModelStatusSection(theme),
            const SizedBox(height: 24),

            // Text input section (only shown when model is loaded)
            if (_model != null) ...[
              TextField(
                controller: _textController,
                decoration: const InputDecoration(
                  labelText: 'Text to speak',
                  hintText: 'Enter text to synthesize...',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.text_fields),
                ),
                maxLines: 3,
                enabled: _state == _TtsState.ready,
              ),
              const SizedBox(height: 16),

              // Speak button
              FilledButton.icon(
                onPressed: _state == _TtsState.ready ? _speak : null,
                icon: _state == _TtsState.synthesizing
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.record_voice_over),
                label: Text(
                  _state == _TtsState.synthesizing
                      ? 'Synthesizing...'
                      : 'Speak',
                ),
              ),
              const SizedBox(height: 24),
            ],

            // Status display
            _buildStatusSection(theme),
          ],
        ),
      ),
    );
  }

  /// Build the model status section.
  Widget _buildModelStatusSection(ThemeData theme) {
    if (_state == _TtsState.loadingModel) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(
                'Loading $_ttsModelId...',
                style: theme.textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              Text(
                'Downloading model from registry...',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      );
    }

    if (_model == null) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              Icon(
                Icons.smart_toy_outlined,
                size: 48,
                color: theme.colorScheme.primary,
              ),
              const SizedBox(height: 16),
              Text('TTS Model Required', style: theme.textTheme.titleMedium),
              const SizedBox(height: 8),
              Text(
                'Load the $_ttsModelId model to start synthesizing speech.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium,
              ),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: _loadModel,
                icon: const Icon(Icons.download),
                label: const Text('Load Model'),
              ),
            ],
          ),
        ),
      );
    }

    return Card(
      color: theme.colorScheme.primaryContainer.withAlpha(100),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Icon(Icons.check_circle, color: theme.colorScheme.primary),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Model Ready',
                    style: theme.textTheme.titleSmall?.copyWith(
                      color: theme.colorScheme.primary,
                    ),
                  ),
                  Text(_ttsModelId, style: theme.textTheme.bodySmall),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Build the status display section.
  Widget _buildStatusSection(ThemeData theme) {
    // Playing state
    if (_state == _TtsState.playing) {
      return Card(
        color: theme.colorScheme.secondaryContainer.withAlpha(100),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    _isPlaying ? Icons.volume_up : Icons.volume_off,
                    color: theme.colorScheme.secondary,
                    size: 32,
                  ),
                  const SizedBox(width: 12),
                  Text(
                    _isPlaying ? 'Playing audio...' : 'Audio playback complete',
                    style: theme.textTheme.titleMedium?.copyWith(
                      color: theme.colorScheme.secondary,
                    ),
                  ),
                ],
              ),
              if (_latencyMs != null) ...[
                const SizedBox(height: 12),
                Text(
                  'Inference latency: ${_latencyMs}ms',
                  style: theme.textTheme.bodyMedium,
                ),
              ],
              if (_isPlaying) ...[
                const SizedBox(height: 16),
                OutlinedButton.icon(
                  onPressed: _stopAudio,
                  icon: const Icon(Icons.stop),
                  label: const Text('Stop'),
                ),
              ],
            ],
          ),
        ),
      );
    }

    // Error state
    if (_state == _TtsState.error && _errorMessage != null) {
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
                _errorMessage!,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onErrorContainer,
                ),
              ),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: _model == null ? _loadModel : _speak,
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    // Show latency from previous inference if available
    if (_latencyMs != null && _state == _TtsState.ready) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Icon(Icons.timer_outlined, color: theme.colorScheme.primary),
              const SizedBox(width: 12),
              Text(
                'Last inference: ${_latencyMs}ms',
                style: theme.textTheme.bodyMedium,
              ),
            ],
          ),
        ),
      );
    }

    return const SizedBox.shrink();
  }
}

/// TTS screen states.
enum _TtsState { idle, loadingModel, ready, synthesizing, playing, error }

/// Custom audio source for playing bytes directly with just_audio.
class _AudioBytesSource extends StreamAudioSource {
  final Uint8List _bytes;

  _AudioBytesSource(this._bytes);

  @override
  Future<StreamAudioResponse> request([int? start, int? end]) async {
    start ??= 0;
    end ??= _bytes.length;
    return StreamAudioResponse(
      sourceLength: _bytes.length,
      contentLength: end - start,
      offset: start,
      stream: Stream.value(_bytes.sublist(start, end)),
      contentType: 'audio/wav',
    );
  }
}
