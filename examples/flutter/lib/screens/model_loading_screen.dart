import 'package:flutter/material.dart';
import 'package:xybrid_example/ui.dart';
import 'package:xybrid_flutter/xybrid.dart';

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
                        Icon(
                          Icons.info_outline,
                          color: theme.colorScheme.primary,
                        ),
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
              onPressed: _state == _ModelLoadState.loading ? null : _loadModel,
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
                _state == _ModelLoadState.loading ? 'Loading...' : 'Load Model',
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
              ErrorCard(errorMessage: _errorMessage!, onRetry: _loadModel),
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
            Text('Downloading model...', style: theme.textTheme.titleMedium),
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
  const _ModelInfoCard({required this.modelId, required this.onUnload});

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
                Icon(Icons.check_circle, color: theme.colorScheme.primary),
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
            InfoRow(label: 'Model ID', value: modelId),
            const InfoRow(label: 'Status', value: 'Ready for inference'),
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
