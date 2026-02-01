import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:xybrid/xybrid.dart';

void main() {
  runApp(const XybridExampleApp());
}

/// The main application widget for the Xybrid SDK example.
class XybridExampleApp extends StatelessWidget {
  const XybridExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Xybrid SDK Example',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const SdkInitializationScreen(),
      routes: {
        '/demos': (context) => const DemoHubScreen(),
        '/demos/model-loading': (context) => const ModelLoadingScreen(),
        '/demos/text-to-speech': (context) => const TextToSpeechScreen(),
        '/demos/speech-to-text': (context) => const SpeechToTextScreen(),
        '/demos/pipelines': (context) => const PipelineScreen(),
        '/demos/error-handling': (context) => const ErrorHandlingScreen(),
      },
    );
  }
}

/// Screen that handles SDK initialization and displays status.
class SdkInitializationScreen extends StatefulWidget {
  const SdkInitializationScreen({super.key});

  @override
  State<SdkInitializationScreen> createState() =>
      _SdkInitializationScreenState();
}

class _SdkInitializationScreenState extends State<SdkInitializationScreen> {
  /// Current initialization state.
  _InitState _initState = _InitState.loading;

  /// Error message if initialization failed.
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _initializeSdk();
  }

  /// Initialize the Xybrid SDK.
  Future<void> _initializeSdk() async {
    try {
      await Xybrid.init();
      if (mounted) {
        setState(() {
          _initState = _InitState.ready;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _initState = _InitState.error;
          _errorMessage = e.toString();
        });
      }
    }
  }

  /// Retry SDK initialization after failure.
  void _retryInitialization() {
    setState(() {
      _initState = _InitState.loading;
      _errorMessage = null;
    });
    _initializeSdk();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Xybrid SDK Example'),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: switch (_initState) {
            _InitState.loading => const _LoadingView(),
            _InitState.ready => const _ReadyView(),
            _InitState.error => _ErrorView(
                errorMessage: _errorMessage ?? 'Unknown error',
                onRetry: _retryInitialization,
              ),
          },
        ),
      ),
    );
  }
}

/// Enum representing SDK initialization states.
enum _InitState { loading, ready, error }

/// Loading view shown during SDK initialization.
class _LoadingView extends StatelessWidget {
  const _LoadingView();

  @override
  Widget build(BuildContext context) {
    return const Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        CircularProgressIndicator(),
        SizedBox(height: 24),
        Text(
          'Initializing Xybrid SDK...',
          style: TextStyle(fontSize: 18),
        ),
      ],
    );
  }
}

/// Ready view shown when SDK is initialized successfully.
class _ReadyView extends StatelessWidget {
  const _ReadyView();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          Icons.check_circle,
          size: 72,
          color: theme.colorScheme.primary,
        ),
        const SizedBox(height: 24),
        Text(
          'SDK Ready',
          style: theme.textTheme.headlineMedium?.copyWith(
            color: theme.colorScheme.primary,
          ),
        ),
        const SizedBox(height: 16),
        Text(
          'Xybrid SDK has been initialized successfully.\n'
          'You can now load models and run inference.',
          textAlign: TextAlign.center,
          style: theme.textTheme.bodyLarge,
        ),
        const SizedBox(height: 32),
        FilledButton.icon(
          onPressed: () => Navigator.pushReplacementNamed(context, '/demos'),
          icon: const Icon(Icons.arrow_forward),
          label: const Text('Continue to Demos'),
        ),
      ],
    );
  }
}

/// Error view shown when SDK initialization fails.
class _ErrorView extends StatelessWidget {
  const _ErrorView({
    required this.errorMessage,
    required this.onRetry,
  });

  final String errorMessage;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          Icons.error_outline,
          size: 72,
          color: theme.colorScheme.error,
        ),
        const SizedBox(height: 24),
        Text(
          'Initialization Failed',
          style: theme.textTheme.headlineMedium?.copyWith(
            color: theme.colorScheme.error,
          ),
        ),
        const SizedBox(height: 16),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: theme.colorScheme.errorContainer.withAlpha(50),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Text(
            errorMessage,
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.error,
            ),
          ),
        ),
        const SizedBox(height: 24),
        FilledButton.icon(
          onPressed: onRetry,
          icon: const Icon(Icons.refresh),
          label: const Text('Retry'),
        ),
      ],
    );
  }
}

/// Demo hub screen showing available SDK demos.
class DemoHubScreen extends StatelessWidget {
  const DemoHubScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        backgroundColor: theme.colorScheme.inversePrimary,
        title: const Text('Xybrid SDK Demos'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _DemoTile(
            icon: Icons.download,
            title: 'Model Loading',
            subtitle: 'Load models from the registry',
            onTap: () => Navigator.pushNamed(context, '/demos/model-loading'),
          ),
          _DemoTile(
            icon: Icons.record_voice_over,
            title: 'Text-to-Speech',
            subtitle: 'Convert text to audio',
            onTap: () => Navigator.pushNamed(context, '/demos/text-to-speech'),
          ),
          _DemoTile(
            icon: Icons.mic,
            title: 'Speech-to-Text',
            subtitle: 'Transcribe audio to text',
            onTap: () => Navigator.pushNamed(context, '/demos/speech-to-text'),
          ),
          _DemoTile(
            icon: Icons.account_tree,
            title: 'Pipelines',
            subtitle: 'Multi-stage inference workflows',
            onTap: () => Navigator.pushNamed(context, '/demos/pipelines'),
          ),
          _DemoTile(
            icon: Icons.warning,
            title: 'Error Handling',
            subtitle: 'Error patterns showcase',
            onTap: () => Navigator.pushNamed(context, '/demos/error-handling'),
          ),
        ],
      ),
    );
  }
}

/// A tile representing a demo in the hub.
class _DemoTile extends StatelessWidget {
  const _DemoTile({
    required this.icon,
    required this.title,
    required this.subtitle,
    this.onTap,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: ListTile(
        leading: Icon(
          icon,
          color: theme.colorScheme.primary,
        ),
        title: Text(title),
        subtitle: Text(subtitle),
        trailing: const Icon(Icons.chevron_right),
        onTap: onTap,
      ),
    );
  }
}

/// Model loading demo screen.
///
/// Demonstrates how to load models from the Xybrid registry.
class ModelLoadingScreen extends StatefulWidget {
  const ModelLoadingScreen({super.key});

  @override
  State<ModelLoadingScreen> createState() => _ModelLoadingScreenState();
}

class _ModelLoadingScreenState extends State<ModelLoadingScreen> {
  /// Controller for the model ID text field.
  final _modelIdController = TextEditingController(text: 'whisper-tiny');

  /// Current loading state.
  _ModelLoadState _state = _ModelLoadState.idle;

  /// The loaded model (if any).
  /// Stored for use in inference demos - the model instance is needed
  /// to call model.run() for text-to-speech or other inference tasks.
  XybridModel? _loadedModel;

  /// Whether a model is currently loaded and ready for inference.
  bool get hasModel => _loadedModel != null;

  /// The model ID that was loaded.
  String? _loadedModelId;

  /// Error message if loading failed.
  String? _errorMessage;

  @override
  void dispose() {
    _modelIdController.dispose();
    super.dispose();
  }

  /// Load a model from the registry.
  Future<void> _loadModel() async {
    final modelId = _modelIdController.text.trim();
    if (modelId.isEmpty) {
      setState(() {
        _state = _ModelLoadState.error;
        _errorMessage = 'Please enter a model ID';
      });
      return;
    }

    setState(() {
      _state = _ModelLoadState.loading;
      _errorMessage = null;
      _loadedModel = null;
      _loadedModelId = null;
    });

    try {
      final loader = XybridModelLoader.fromRegistry(modelId);
      final model = await loader.load();
      if (mounted) {
        setState(() {
          _state = _ModelLoadState.loaded;
          _loadedModel = model;
          _loadedModelId = modelId;
        });
      }
    } on XybridException catch (e) {
      if (mounted) {
        setState(() {
          _state = _ModelLoadState.error;
          _errorMessage = e.message;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = _ModelLoadState.error;
          _errorMessage = _formatErrorMessage(e.toString());
        });
      }
    }
  }

  /// Format error message for user display.
  String _formatErrorMessage(String error) {
    // Check for common error patterns and provide friendly messages
    if (error.contains('network') || error.contains('Network')) {
      return 'Network error: Please check your internet connection and try again.';
    }
    if (error.contains('not found') || error.contains('NotFound')) {
      return 'Model not found: The model ID may be incorrect or the model is not available in the registry.';
    }
    if (error.contains('timeout') || error.contains('Timeout')) {
      return 'Request timed out: The server took too long to respond. Please try again.';
    }
    // Return the original error if no pattern matches
    return error;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        backgroundColor: theme.colorScheme.inversePrimary,
        title: const Text('Model Loading'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Description
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.info_outline,
                            color: theme.colorScheme.primary),
                        const SizedBox(width: 8),
                        Text(
                          'About Model Loading',
                          style: theme.textTheme.titleMedium,
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Load models from the Xybrid registry by entering a model ID. '
                      'Models are downloaded and cached locally for offline use.',
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Model ID input
            TextField(
              controller: _modelIdController,
              decoration: const InputDecoration(
                labelText: 'Model ID',
                hintText: 'e.g., whisper-tiny, kokoro-82m',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.smart_toy),
              ),
              enabled: _state != _ModelLoadState.loading,
              onSubmitted: (_) => _loadModel(),
            ),
            const SizedBox(height: 16),

            // Load button
            FilledButton.icon(
              onPressed:
                  _state == _ModelLoadState.loading ? null : _loadModel,
              icon: _state == _ModelLoadState.loading
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Icon(Icons.download),
              label: Text(
                _state == _ModelLoadState.loading
                    ? 'Loading...'
                    : 'Load Model',
              ),
            ),
            const SizedBox(height: 24),

            // Result display
            if (_state == _ModelLoadState.loading)
              const _LoadingIndicator()
            else if (_state == _ModelLoadState.loaded && hasModel)
              _ModelInfoCard(
                modelId: _loadedModelId!,
                onUnload: () {
                  setState(() {
                    _state = _ModelLoadState.idle;
                    _loadedModel = null;
                    _loadedModelId = null;
                  });
                },
              )
            else if (_state == _ModelLoadState.error && _errorMessage != null)
              _ErrorCard(
                errorMessage: _errorMessage!,
                onRetry: _loadModel,
              ),
          ],
        ),
      ),
    );
  }
}

/// Model loading states.
enum _ModelLoadState { idle, loading, loaded, error }

/// Loading indicator widget.
class _LoadingIndicator extends StatelessWidget {
  const _LoadingIndicator();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 16),
            Text(
              'Downloading model...',
              style: theme.textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Text(
              'This may take a moment depending on model size\nand your network connection.',
              textAlign: TextAlign.center,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Model info display card.
class _ModelInfoCard extends StatelessWidget {
  const _ModelInfoCard({
    required this.modelId,
    required this.onUnload,
  });

  final String modelId;
  final VoidCallback onUnload;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      color: theme.colorScheme.primaryContainer.withAlpha(100),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.check_circle,
                  color: theme.colorScheme.primary,
                ),
                const SizedBox(width: 8),
                Text(
                  'Model Loaded',
                  style: theme.textTheme.titleMedium?.copyWith(
                    color: theme.colorScheme.primary,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _InfoRow(label: 'Model ID', value: modelId),
            const _InfoRow(label: 'Status', value: 'Ready for inference'),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: onUnload,
                    icon: const Icon(Icons.close),
                    label: const Text('Unload'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

/// A row displaying label-value info.
class _InfoRow extends StatelessWidget {
  const _InfoRow({
    required this.label,
    required this.value,
  });

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              '$label:',
              style: theme.textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: theme.textTheme.bodyMedium,
            ),
          ),
        ],
      ),
    );
  }
}

/// Error display card.
class _ErrorCard extends StatelessWidget {
  const _ErrorCard({
    required this.errorMessage,
    required this.onRetry,
  });

  final String errorMessage;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      color: theme.colorScheme.errorContainer.withAlpha(100),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.error_outline,
                  color: theme.colorScheme.error,
                ),
                const SizedBox(width: 8),
                Text(
                  'Failed to Load Model',
                  style: theme.textTheme.titleMedium?.copyWith(
                    color: theme.colorScheme.error,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text(
              errorMessage,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onErrorContainer,
              ),
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }
}

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

      final audioBytes = result.audioBytes;
      if (audioBytes == null || audioBytes.isEmpty) {
        throw XybridException('No audio output received from model');
      }

      if (mounted) {
        setState(() {
          _state = _TtsState.playing;
          _latencyMs = result.latencyMs;
        });

        await _playAudio(audioBytes);

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

  /// Play audio bytes using just_audio.
  Future<void> _playAudio(Uint8List audioBytes) async {
    try {
      // Create a data source from bytes
      // The audio from TTS is typically WAV or raw PCM
      final source = _AudioBytesSource(audioBytes);
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
                        Icon(Icons.info_outline,
                            color: theme.colorScheme.primary),
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
              Text(
                'TTS Model Required',
                style: theme.textTheme.titleMedium,
              ),
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
            Icon(
              Icons.check_circle,
              color: theme.colorScheme.primary,
            ),
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
                  Text(
                    _ttsModelId,
                    style: theme.textTheme.bodySmall,
                  ),
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
                  Icon(
                    Icons.error_outline,
                    color: theme.colorScheme.error,
                  ),
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
              Icon(
                Icons.timer_outlined,
                color: theme.colorScheme.primary,
              ),
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
enum _TtsState {
  idle,
  loadingModel,
  ready,
  synthesizing,
  playing,
  error,
}

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

/// Speech-to-Text demo screen.
///
/// Demonstrates how to run audio-based inference (ASR) using the Xybrid SDK.
class SpeechToTextScreen extends StatefulWidget {
  const SpeechToTextScreen({super.key});

  @override
  State<SpeechToTextScreen> createState() => _SpeechToTextScreenState();
}

class _SpeechToTextScreenState extends State<SpeechToTextScreen> {
  /// Audio recorder instance.
  final _recorder = AudioRecorder();

  /// Current screen state.
  _AsrState _state = _AsrState.idle;

  /// Loaded ASR model (if any).
  XybridModel? _model;

  /// The model ID used for ASR.
  static const _asrModelId = 'whisper-tiny';

  /// Error message if operation failed.
  String? _errorMessage;

  /// Latency of the last inference in milliseconds.
  int? _latencyMs;

  /// Transcribed text from last inference.
  String? _transcribedText;

  /// Sample rate for audio recording (Whisper expects 16kHz).
  static const _sampleRate = 16000;

  @override
  void dispose() {
    _recorder.dispose();
    super.dispose();
  }

  /// Check and request microphone permission.
  Future<bool> _checkMicrophonePermission() async {
    var status = await Permission.microphone.status;
    if (status.isDenied) {
      status = await Permission.microphone.request();
    }
    return status.isGranted;
  }

  /// Load the ASR model from the registry.
  Future<void> _loadModel() async {
    setState(() {
      _state = _AsrState.loadingModel;
      _errorMessage = null;
    });

    try {
      final loader = XybridModelLoader.fromRegistry(_asrModelId);
      final model = await loader.load();
      if (mounted) {
        setState(() {
          _state = _AsrState.ready;
          _model = model;
        });
      }
    } on XybridException catch (e) {
      if (mounted) {
        setState(() {
          _state = _AsrState.error;
          _errorMessage = e.message;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = _AsrState.error;
          _errorMessage = _formatErrorMessage(e.toString());
        });
      }
    }
  }

  /// Start recording audio from microphone.
  Future<void> _startRecording() async {
    // Check permission first
    final hasPermission = await _checkMicrophonePermission();
    if (!hasPermission) {
      setState(() {
        _state = _AsrState.error;
        _errorMessage = 'Microphone permission denied. Please grant permission in settings.';
      });
      return;
    }

    // Check if recorder is available
    if (!await _recorder.hasPermission()) {
      setState(() {
        _state = _AsrState.error;
        _errorMessage = 'Unable to access microphone. Please check your settings.';
      });
      return;
    }

    try {
      // Get temp directory for recording
      final tempDir = await getTemporaryDirectory();
      final recordingPath = '${tempDir.path}/xybrid_recording.wav';

      // Configure recording for Whisper (16kHz mono WAV)
      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: _sampleRate,
          numChannels: 1,
        ),
        path: recordingPath,
      );

      setState(() {
        _state = _AsrState.recording;
        _transcribedText = null;
        _latencyMs = null;
      });
    } catch (e) {
      setState(() {
        _state = _AsrState.error;
        _errorMessage = 'Failed to start recording: $e';
      });
    }
  }

  /// Stop recording and run inference.
  Future<void> _stopRecordingAndTranscribe() async {
    try {
      // Stop recording
      final path = await _recorder.stop();
      if (path == null) {
        setState(() {
          _state = _AsrState.error;
          _errorMessage = 'Recording failed - no audio data captured.';
        });
        return;
      }

      setState(() {
        _state = _AsrState.transcribing;
      });

      // Read the audio file
      final audioFile = File(path);
      if (!await audioFile.exists()) {
        setState(() {
          _state = _AsrState.error;
          _errorMessage = 'Audio file not found.';
        });
        return;
      }

      final audioBytes = await audioFile.readAsBytes();
      if (audioBytes.isEmpty) {
        setState(() {
          _state = _AsrState.error;
          _errorMessage = 'Recorded audio is empty.';
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
          _state = _AsrState.ready;
          _transcribedText = '(No speech detected)';
          _latencyMs = result.latencyMs;
        });
        return;
      }

      if (mounted) {
        setState(() {
          _state = _AsrState.ready;
          _transcribedText = transcribedText;
          _latencyMs = result.latencyMs;
        });
      }
    } on XybridException catch (e) {
      if (mounted) {
        setState(() {
          _state = _AsrState.error;
          _errorMessage = e.message;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = _AsrState.error;
          _errorMessage = _formatErrorMessage(e.toString());
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
                        Icon(Icons.info_outline,
                            color: theme.colorScheme.primary),
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
            _buildModelStatusSection(theme),
            const SizedBox(height: 24),

            // Recording section (only shown when model is loaded)
            if (_model != null) _buildRecordingSection(theme),

            // Result section
            if (_transcribedText != null || _latencyMs != null)
              _buildResultSection(theme),

            // Error display
            if (_state == _AsrState.error && _errorMessage != null)
              _buildErrorSection(theme),
          ],
        ),
      ),
    );
  }

  /// Build the model status section.
  Widget _buildModelStatusSection(ThemeData theme) {
    if (_state == _AsrState.loadingModel) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(
                'Loading $_asrModelId...',
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
              Text(
                'ASR Model Required',
                style: theme.textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              Text(
                'Load the $_asrModelId model to start transcribing speech.',
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
            Icon(
              Icons.check_circle,
              color: theme.colorScheme.primary,
            ),
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
                  Text(
                    _asrModelId,
                    style: theme.textTheme.bodySmall,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Build the recording section.
  Widget _buildRecordingSection(ThemeData theme) {
    final isRecording = _state == _AsrState.recording;
    final isTranscribing = _state == _AsrState.transcribing;
    final canRecord = _state == _AsrState.ready || _state == _AsrState.error;

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
              Text(
                'Transcribing...',
                style: theme.textTheme.titleMedium,
              ),
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
              Text(
                'Ready to Record',
                style: theme.textTheme.titleMedium,
              ),
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
  Widget _buildErrorSection(ThemeData theme) {
    return Card(
      color: theme.colorScheme.errorContainer.withAlpha(100),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.error_outline,
                  color: theme.colorScheme.error,
                ),
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

/// ASR screen states.
enum _AsrState {
  idle,
  loadingModel,
  ready,
  recording,
  transcribing,
  error,
}

/// Pipeline demo screen.
///
/// Demonstrates how to load and run multi-stage inference pipelines.
class PipelineScreen extends StatefulWidget {
  const PipelineScreen({super.key});

  @override
  State<PipelineScreen> createState() => _PipelineScreenState();
}

class _PipelineScreenState extends State<PipelineScreen> {
  /// Example pipeline YAML for the text area.
  static const _examplePipelineYaml = '''name: speech-to-text
description: Simple ASR pipeline
stages:
  - model: whisper-tiny
    input: audio
''';

  /// Controller for the pipeline YAML input.
  final _yamlController = TextEditingController(text: _examplePipelineYaml);

  /// Current screen state.
  _PipelineState _state = _PipelineState.idle;

  /// Loaded pipeline (if any).
  XybridPipeline? _pipeline;

  /// Error message if operation failed.
  String? _errorMessage;

  /// Latency of the last pipeline execution in milliseconds.
  int? _latencyMs;

  /// Result text from the last execution (if any).
  String? _resultText;

  @override
  void dispose() {
    _yamlController.dispose();
    super.dispose();
  }

  /// Load a pipeline from the YAML input.
  void _loadPipeline() {
    final yaml = _yamlController.text.trim();
    if (yaml.isEmpty) {
      setState(() {
        _state = _PipelineState.error;
        _errorMessage = 'Please enter pipeline YAML.';
      });
      return;
    }

    setState(() {
      _state = _PipelineState.loading;
      _errorMessage = null;
      _pipeline = null;
      _resultText = null;
      _latencyMs = null;
    });

    try {
      final pipeline = XybridPipeline.fromYaml(yaml);
      setState(() {
        _state = _PipelineState.loaded;
        _pipeline = pipeline;
      });
    } on XybridException catch (e) {
      setState(() {
        _state = _PipelineState.error;
        _errorMessage = e.message;
      });
    } catch (e) {
      setState(() {
        _state = _PipelineState.error;
        _errorMessage = _formatErrorMessage(e.toString());
      });
    }
  }

  /// Run the loaded pipeline with sample input.
  Future<void> _runPipeline() async {
    if (_pipeline == null) {
      setState(() {
        _state = _PipelineState.error;
        _errorMessage = 'No pipeline loaded. Please load a pipeline first.';
      });
      return;
    }

    setState(() {
      _state = _PipelineState.running;
      _errorMessage = null;
      _resultText = null;
      _latencyMs = null;
    });

    try {
      // Create sample input based on pipeline type
      // For now, we'll use a simple text envelope as sample input
      final envelope = XybridEnvelope.text('Hello, this is a sample input for the pipeline.');

      final stopwatch = Stopwatch()..start();
      final result = await _pipeline!.run(envelope);
      stopwatch.stop();

      if (mounted) {
        setState(() {
          _state = _PipelineState.loaded;
          _latencyMs = result.latencyMs;
          _resultText = result.text ?? '(No text output)';
        });
      }
    } on XybridException catch (e) {
      if (mounted) {
        setState(() {
          _state = _PipelineState.error;
          _errorMessage = e.message;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = _PipelineState.error;
          _errorMessage = _formatErrorMessage(e.toString());
        });
      }
    }
  }

  /// Reset and unload the pipeline.
  void _resetPipeline() {
    setState(() {
      _state = _PipelineState.idle;
      _pipeline = null;
      _errorMessage = null;
      _resultText = null;
      _latencyMs = null;
    });
  }

  /// Format error message for user display.
  String _formatErrorMessage(String error) {
    if (error.contains('yaml') || error.contains('YAML')) {
      return 'Invalid YAML: Please check the pipeline format.';
    }
    if (error.contains('model') || error.contains('Model')) {
      return 'Model error: A model in the pipeline may not be available.';
    }
    if (error.contains('network') || error.contains('Network')) {
      return 'Network error: Please check your internet connection.';
    }
    return error;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        backgroundColor: theme.colorScheme.inversePrimary,
        title: const Text('Pipelines'),
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
                        Icon(Icons.info_outline,
                            color: theme.colorScheme.primary),
                        const SizedBox(width: 8),
                        Text(
                          'About Pipelines',
                          style: theme.textTheme.titleMedium,
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Pipelines chain multiple models together for multi-stage '
                      'inference workflows. Enter a YAML definition below to load '
                      'and run a pipeline.',
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // YAML input section
            _buildYamlInputSection(theme),
            const SizedBox(height: 16),

            // Action buttons
            _buildActionButtons(theme),
            const SizedBox(height: 24),

            // Pipeline info section (when loaded)
            if (_pipeline != null) _buildPipelineInfoSection(theme),

            // Result section (when executed)
            if (_resultText != null || _latencyMs != null)
              _buildResultSection(theme),

            // Error section
            if (_state == _PipelineState.error && _errorMessage != null)
              _buildErrorSection(theme),
          ],
        ),
      ),
    );
  }

  /// Build the YAML input section.
  Widget _buildYamlInputSection(ThemeData theme) {
    final isEditable = _state == _PipelineState.idle ||
        _state == _PipelineState.error ||
        _state == _PipelineState.loaded;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Pipeline YAML',
              style: theme.textTheme.titleSmall,
            ),
            if (_yamlController.text != _examplePipelineYaml)
              TextButton.icon(
                onPressed: isEditable
                    ? () {
                        setState(() {
                          _yamlController.text = _examplePipelineYaml;
                        });
                      }
                    : null,
                icon: const Icon(Icons.restore, size: 18),
                label: const Text('Reset to Example'),
              ),
          ],
        ),
        const SizedBox(height: 8),
        TextField(
          controller: _yamlController,
          decoration: InputDecoration(
            hintText: 'Enter pipeline YAML...',
            border: const OutlineInputBorder(),
            filled: true,
            fillColor: theme.colorScheme.surfaceContainerLowest,
          ),
          maxLines: 8,
          enabled: isEditable,
          style: TextStyle(
            fontFamily: 'monospace',
            fontSize: 13,
            color: theme.colorScheme.onSurface,
          ),
        ),
      ],
    );
  }

  /// Build the action buttons section.
  Widget _buildActionButtons(ThemeData theme) {
    final isLoading = _state == _PipelineState.loading;
    final isRunning = _state == _PipelineState.running;
    final hasLoaded = _pipeline != null;
    final canLoad = !isLoading && !isRunning;
    final canRun = hasLoaded && !isLoading && !isRunning;

    return Row(
      children: [
        Expanded(
          child: FilledButton.icon(
            onPressed: canLoad ? _loadPipeline : null,
            icon: isLoading
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.white,
                    ),
                  )
                : const Icon(Icons.upload_file),
            label: Text(isLoading ? 'Loading...' : 'Load Pipeline'),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: FilledButton.icon(
            onPressed: canRun ? _runPipeline : null,
            icon: isRunning
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.white,
                    ),
                  )
                : const Icon(Icons.play_arrow),
            label: Text(isRunning ? 'Running...' : 'Run Pipeline'),
          ),
        ),
        if (hasLoaded) ...[
          const SizedBox(width: 12),
          IconButton.outlined(
            onPressed: canLoad ? _resetPipeline : null,
            icon: const Icon(Icons.close),
            tooltip: 'Unload Pipeline',
          ),
        ],
      ],
    );
  }

  /// Build the pipeline info section.
  Widget _buildPipelineInfoSection(ThemeData theme) {
    final pipeline = _pipeline!;
    return Card(
      color: theme.colorScheme.primaryContainer.withAlpha(100),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.check_circle,
                  color: theme.colorScheme.primary,
                ),
                const SizedBox(width: 8),
                Text(
                  'Pipeline Loaded',
                  style: theme.textTheme.titleMedium?.copyWith(
                    color: theme.colorScheme.primary,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _InfoRow(
              label: 'Name',
              value: pipeline.name ?? '(unnamed)',
            ),
            _InfoRow(
              label: 'Stages',
              value: '${pipeline.stageCount}',
            ),
            if (pipeline.stageNames.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                'Stage Names:',
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 4),
              Wrap(
                spacing: 8,
                runSpacing: 4,
                children: pipeline.stageNames
                    .map((name) => Chip(
                          label: Text(name),
                          visualDensity: VisualDensity.compact,
                        ))
                    .toList(),
              ),
            ],
          ],
        ),
      ),
    );
  }

  /// Build the result section.
  Widget _buildResultSection(ThemeData theme) {
    return Card(
      color: theme.colorScheme.secondaryContainer.withAlpha(100),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.output,
                  color: theme.colorScheme.secondary,
                ),
                const SizedBox(width: 8),
                Text(
                  'Execution Result',
                  style: theme.textTheme.titleMedium?.copyWith(
                    color: theme.colorScheme.secondary,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            if (_resultText != null) ...[
              Text(
                'Output:',
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 4),
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
                  _resultText!,
                  style: theme.textTheme.bodyMedium,
                ),
              ),
              const SizedBox(height: 12),
            ],
            if (_latencyMs != null)
              Row(
                children: [
                  Icon(
                    Icons.timer_outlined,
                    size: 18,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                  const SizedBox(width: 6),
                  Text(
                    'Total latency: ${_latencyMs}ms',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }

  /// Build the error section.
  Widget _buildErrorSection(ThemeData theme) {
    return Card(
      color: theme.colorScheme.errorContainer.withAlpha(100),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.error_outline,
                  color: theme.colorScheme.error,
                ),
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
              onPressed: _loadPipeline,
              icon: const Icon(Icons.refresh),
              label: const Text('Try Again'),
            ),
          ],
        ),
      ),
    );
  }
}

/// Pipeline screen states.
enum _PipelineState {
  idle,
  loading,
  loaded,
  running,
  error,
}

/// Error handling demo screen.
///
/// Demonstrates proper error handling patterns with the Xybrid SDK,
/// including triggering common errors and showing retry logic.
class ErrorHandlingScreen extends StatefulWidget {
  const ErrorHandlingScreen({super.key});

  @override
  State<ErrorHandlingScreen> createState() => _ErrorHandlingScreenState();
}

class _ErrorHandlingScreenState extends State<ErrorHandlingScreen> {
  /// The last captured exception details.
  _CapturedError? _capturedError;

  /// Whether an operation is currently in progress.
  bool _isLoading = false;

  /// Retry count for demonstrating retry logic.
  int _retryCount = 0;

  /// Maximum number of retries to attempt.
  static const _maxRetries = 3;

  /// Trigger an "invalid model" error.
  Future<void> _triggerInvalidModelError() async {
    await _executeWithErrorCapture(
      operationName: 'Load Invalid Model',
      operation: () async {
        // Attempt to load a model that doesn't exist
        final loader = XybridModelLoader.fromRegistry('non-existent-model-xyz');
        await loader.load();
      },
    );
  }

  /// Trigger a "network timeout" simulation by loading a very large model.
  /// Note: This simulates a slow network by attempting an operation that would
  /// typically timeout or fail in a disconnected scenario.
  Future<void> _triggerNetworkError() async {
    await _executeWithErrorCapture(
      operationName: 'Network Request',
      operation: () async {
        // This will fail if network is unavailable or model doesn't exist
        final loader = XybridModelLoader.fromRegistry('network-test-unavailable');
        await loader.load();
      },
    );
  }

  /// Trigger an "empty input" validation error.
  Future<void> _triggerEmptyInputError() async {
    await _executeWithErrorCapture(
      operationName: 'Empty Input Validation',
      operation: () async {
        // First load a valid model (or fail trying)
        final loader = XybridModelLoader.fromRegistry('whisper-tiny');
        final model = await loader.load();
        // Then run with problematic input - empty audio bytes
        final envelope = XybridEnvelope.audio(
          bytes: [], // Empty bytes - should fail
          sampleRate: 16000,
          channels: 1,
        );
        await model.run(envelope);
      },
    );
  }

  /// Trigger an "invalid YAML" pipeline error.
  Future<void> _triggerInvalidYamlError() async {
    await _executeWithErrorCapture(
      operationName: 'Invalid Pipeline YAML',
      operation: () async {
        // Attempt to parse invalid YAML
        XybridPipeline.fromYaml('''
invalid yaml [[[
  this: is: broken::
''');
      },
    );
  }

  /// Execute an operation with error capture and display.
  ///
  /// This demonstrates the proper try/catch pattern for async operations.
  Future<void> _executeWithErrorCapture({
    required String operationName,
    required Future<void> Function() operation,
  }) async {
    setState(() {
      _isLoading = true;
      _capturedError = null;
      _retryCount = 0;
    });

    final stopwatch = Stopwatch()..start();

    try {
      await operation();

      // If we get here, operation succeeded (unexpected for error demos)
      stopwatch.stop();
      if (mounted) {
        setState(() {
          _isLoading = false;
          _capturedError = _CapturedError(
            operationName: operationName,
            exceptionType: 'Success',
            message: 'Operation completed successfully (no error)',
            rawError: 'No exception thrown',
            durationMs: stopwatch.elapsedMilliseconds,
          );
        });
      }
    } on XybridException catch (e) {
      // XybridException - SDK-specific error
      stopwatch.stop();
      if (mounted) {
        setState(() {
          _isLoading = false;
          _capturedError = _CapturedError(
            operationName: operationName,
            exceptionType: 'XybridException',
            message: e.message,
            rawError: e.toString(),
            durationMs: stopwatch.elapsedMilliseconds,
          );
        });
      }
    } catch (e, stackTrace) {
      // Generic exception - unexpected errors
      stopwatch.stop();
      if (mounted) {
        setState(() {
          _isLoading = false;
          _capturedError = _CapturedError(
            operationName: operationName,
            exceptionType: e.runtimeType.toString(),
            message: _extractUserFriendlyMessage(e.toString()),
            rawError: e.toString(),
            stackTrace: stackTrace.toString(),
            durationMs: stopwatch.elapsedMilliseconds,
          );
        });
      }
    }
  }

  /// Demonstrate retry logic with exponential backoff.
  Future<void> _demonstrateRetryLogic() async {
    setState(() {
      _isLoading = true;
      _capturedError = null;
      _retryCount = 0;
    });

    const operationName = 'Retry Demo (Invalid Model)';
    XybridException? lastError;

    // Attempt the operation with retries
    while (_retryCount < _maxRetries) {
      setState(() {
        _retryCount++;
      });

      try {
        // This will always fail - demonstrating retry behavior
        final loader = XybridModelLoader.fromRegistry('retry-test-model');
        await loader.load();

        // If somehow succeeded, exit
        if (mounted) {
          setState(() {
            _isLoading = false;
            _capturedError = _CapturedError(
              operationName: operationName,
              exceptionType: 'Success',
              message: 'Operation succeeded after $_retryCount attempts',
              rawError: 'No exception',
              durationMs: 0,
            );
          });
        }
        return;
      } on XybridException catch (e) {
        lastError = e;
        // Add delay before retry (exponential backoff simulation)
        if (_retryCount < _maxRetries) {
          final delay = Duration(milliseconds: 200 * _retryCount);
          await Future<void>.delayed(delay);
        }
      } catch (e) {
        lastError = XybridException('Unexpected error: $e');
        if (_retryCount < _maxRetries) {
          final delay = Duration(milliseconds: 200 * _retryCount);
          await Future<void>.delayed(delay);
        }
      }
    }

    // All retries exhausted
    if (mounted && lastError != null) {
      setState(() {
        _isLoading = false;
        _capturedError = _CapturedError(
          operationName: operationName,
          exceptionType: 'XybridException',
          message: lastError!.message,
          rawError: lastError.toString(),
          durationMs: 0,
          retryInfo: 'Failed after $_retryCount attempts (max: $_maxRetries)',
        );
      });
    }
  }

  /// Extract a user-friendly message from an error string.
  String _extractUserFriendlyMessage(String error) {
    if (error.contains('network') || error.contains('Network')) {
      return 'Network error: Unable to connect. Check your internet connection.';
    }
    if (error.contains('not found') || error.contains('NotFound') || error.contains('404')) {
      return 'Not found: The requested resource does not exist.';
    }
    if (error.contains('timeout') || error.contains('Timeout')) {
      return 'Timeout: The operation took too long and was cancelled.';
    }
    if (error.contains('permission') || error.contains('Permission')) {
      return 'Permission denied: Required access was not granted.';
    }
    if (error.contains('invalid') || error.contains('Invalid')) {
      return 'Invalid input: The provided data format is incorrect.';
    }
    // Return truncated original error if no pattern matches
    if (error.length > 100) {
      return '${error.substring(0, 100)}...';
    }
    return error;
  }

  /// Clear the captured error.
  void _clearError() {
    setState(() {
      _capturedError = null;
      _retryCount = 0;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        backgroundColor: theme.colorScheme.inversePrimary,
        title: const Text('Error Handling'),
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
                        Icon(Icons.info_outline,
                            color: theme.colorScheme.primary),
                        const SizedBox(width: 8),
                        Text(
                          'About Error Handling',
                          style: theme.textTheme.titleMedium,
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'This demo shows how to properly handle errors in the Xybrid SDK. '
                      'Tap the buttons below to trigger common error scenarios and see '
                      'how they can be caught and displayed to users.',
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Error trigger section
            _buildErrorTriggerSection(theme),
            const SizedBox(height: 24),

            // Retry demo section
            _buildRetryDemoSection(theme),
            const SizedBox(height: 24),

            // Captured error display
            if (_capturedError != null) _buildErrorDetailsSection(theme),

            // Code example section
            _buildCodeExampleSection(theme),
          ],
        ),
      ),
    );
  }

  /// Build the error trigger section with buttons.
  Widget _buildErrorTriggerSection(ThemeData theme) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.bug_report, color: theme.colorScheme.error),
                const SizedBox(width: 8),
                Text(
                  'Trigger Common Errors',
                  style: theme.textTheme.titleMedium,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              'Tap a button to simulate an error scenario:',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 16),

            // Error trigger buttons in a grid
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _ErrorTriggerButton(
                  icon: Icons.error,
                  label: 'Invalid Model',
                  description: 'Model not found in registry',
                  onPressed: _isLoading ? null : _triggerInvalidModelError,
                ),
                _ErrorTriggerButton(
                  icon: Icons.wifi_off,
                  label: 'Network Error',
                  description: 'Simulated network failure',
                  onPressed: _isLoading ? null : _triggerNetworkError,
                ),
                _ErrorTriggerButton(
                  icon: Icons.input,
                  label: 'Empty Input',
                  description: 'Invalid or empty input data',
                  onPressed: _isLoading ? null : _triggerEmptyInputError,
                ),
                _ErrorTriggerButton(
                  icon: Icons.code_off,
                  label: 'Invalid YAML',
                  description: 'Malformed pipeline definition',
                  onPressed: _isLoading ? null : _triggerInvalidYamlError,
                ),
              ],
            ),

            // Loading indicator
            if (_isLoading) ...[
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    _retryCount > 0
                        ? 'Attempt $_retryCount of $_maxRetries...'
                        : 'Triggering error...',
                    style: theme.textTheme.bodyMedium,
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  /// Build the retry demo section.
  Widget _buildRetryDemoSection(ThemeData theme) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.refresh, color: theme.colorScheme.secondary),
                const SizedBox(width: 8),
                Text(
                  'Retry Logic Demo',
                  style: theme.textTheme.titleMedium,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              'Demonstrates automatic retry with exponential backoff. '
              'This will attempt an operation up to $_maxRetries times before giving up.',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                FilledButton.icon(
                  onPressed: _isLoading ? null : _demonstrateRetryLogic,
                  icon: const Icon(Icons.replay),
                  label: const Text('Run Retry Demo'),
                ),
                if (_retryCount > 0 && !_isLoading) ...[
                  const SizedBox(width: 12),
                  Chip(
                    label: Text('$_retryCount attempts'),
                    avatar: Icon(
                      _capturedError?.exceptionType == 'Success'
                          ? Icons.check
                          : Icons.close,
                      size: 18,
                    ),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }

  /// Build the error details section showing XybridException info.
  Widget _buildErrorDetailsSection(ThemeData theme) {
    final error = _capturedError!;
    final isSuccess = error.exceptionType == 'Success';

    return Card(
      color: isSuccess
          ? theme.colorScheme.primaryContainer.withAlpha(100)
          : theme.colorScheme.errorContainer.withAlpha(100),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  isSuccess ? Icons.check_circle : Icons.error_outline,
                  color: isSuccess
                      ? theme.colorScheme.primary
                      : theme.colorScheme.error,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    isSuccess ? 'Operation Succeeded' : 'Exception Caught',
                    style: theme.textTheme.titleMedium?.copyWith(
                      color: isSuccess
                          ? theme.colorScheme.primary
                          : theme.colorScheme.error,
                    ),
                  ),
                ),
                IconButton(
                  onPressed: _clearError,
                  icon: const Icon(Icons.close),
                  tooltip: 'Clear',
                  iconSize: 20,
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Exception details
            _DetailRow(
              label: 'Operation',
              value: error.operationName,
              theme: theme,
            ),
            _DetailRow(
              label: 'Exception Type',
              value: error.exceptionType,
              theme: theme,
              isCode: true,
            ),
            _DetailRow(
              label: 'User Message',
              value: error.message,
              theme: theme,
            ),
            if (error.durationMs != null && error.durationMs! > 0)
              _DetailRow(
                label: 'Duration',
                value: '${error.durationMs}ms',
                theme: theme,
              ),
            if (error.retryInfo != null)
              _DetailRow(
                label: 'Retry Info',
                value: error.retryInfo!,
                theme: theme,
              ),

            const SizedBox(height: 12),
            ExpansionTile(
              title: Text(
                'Raw Error Details',
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w500,
                ),
              ),
              tilePadding: EdgeInsets.zero,
              childrenPadding: const EdgeInsets.only(bottom: 8),
              children: [
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
                  child: SelectableText(
                    error.rawError,
                    style: TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 12,
                      color: theme.colorScheme.onSurface,
                    ),
                  ),
                ),
                if (error.stackTrace != null) ...[
                  const SizedBox(height: 8),
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
                    child: SelectableText(
                      error.stackTrace!,
                      style: TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 10,
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                      maxLines: 10,
                    ),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }

  /// Build the code example section.
  Widget _buildCodeExampleSection(ThemeData theme) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.code, color: theme.colorScheme.tertiary),
                const SizedBox(width: 8),
                Text(
                  'Code Pattern: try/catch with async',
                  style: theme.textTheme.titleMedium,
                ),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: theme.colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(8),
              ),
              child: SelectableText(
                '''Future<void> loadModel() async {
  try {
    final loader = XybridModelLoader.fromRegistry(modelId);
    final model = await loader.load();
    // Success: use the model
  } on XybridException catch (e) {
    // SDK-specific error
    showError(e.message);
  } catch (e) {
    // Unexpected error
    showError('Unexpected: \$e');
  }
}''',
                style: TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 12,
                  color: theme.colorScheme.onSurface,
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Icon(Icons.code, color: theme.colorScheme.tertiary),
                const SizedBox(width: 8),
                Text(
                  'Code Pattern: Retry with Backoff',
                  style: theme.textTheme.titleMedium,
                ),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: theme.colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(8),
              ),
              child: SelectableText(
                '''Future<XybridModel> loadWithRetry(String id, int maxRetries) async {
  for (int attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await XybridModelLoader.fromRegistry(id).load();
    } on XybridException catch (e) {
      if (attempt == maxRetries) rethrow;
      // Exponential backoff: 200ms, 400ms, 800ms...
      await Future.delayed(Duration(milliseconds: 200 * attempt));
    }
  }
  throw XybridException('Max retries exceeded');
}''',
                style: TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 12,
                  color: theme.colorScheme.onSurface,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// A button to trigger a specific error scenario.
class _ErrorTriggerButton extends StatelessWidget {
  const _ErrorTriggerButton({
    required this.icon,
    required this.label,
    required this.description,
    this.onPressed,
  });

  final IconData icon;
  final String label;
  final String description;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return SizedBox(
      width: 160,
      child: OutlinedButton(
        onPressed: onPressed,
        style: OutlinedButton.styleFrom(
          padding: const EdgeInsets.all(12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: theme.colorScheme.error),
            const SizedBox(height: 4),
            Text(
              label,
              style: theme.textTheme.labelMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 2),
            Text(
              description,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
              textAlign: TextAlign.center,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ),
      ),
    );
  }
}

/// A row displaying a labeled detail.
class _DetailRow extends StatelessWidget {
  const _DetailRow({
    required this.label,
    required this.value,
    required this.theme,
    this.isCode = false,
  });

  final String label;
  final String value;
  final ThemeData theme;
  final bool isCode;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 120,
            child: Text(
              '$label:',
              style: theme.textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          Expanded(
            child: isCode
                ? Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: theme.colorScheme.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      value,
                      style: TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 13,
                        color: theme.colorScheme.primary,
                      ),
                    ),
                  )
                : Text(
                    value,
                    style: theme.textTheme.bodyMedium,
                  ),
          ),
        ],
      ),
    );
  }
}

/// Data class for captured error details.
class _CapturedError {
  final String operationName;
  final String exceptionType;
  final String message;
  final String rawError;
  final String? stackTrace;
  final int? durationMs;
  final String? retryInfo;

  const _CapturedError({
    required this.operationName,
    required this.exceptionType,
    required this.message,
    required this.rawError,
    this.stackTrace,
    this.durationMs,
    this.retryInfo,
  });
}
