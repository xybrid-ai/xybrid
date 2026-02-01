import 'package:flutter/material.dart';
import 'package:xybrid_example/ui.dart';
import 'package:xybrid_flutter/xybrid.dart';

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
      final envelope = XybridEnvelope.text(
        'Hello, this is a sample input for the pipeline.',
      );

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
                        Icon(
                          Icons.info_outline,
                          color: theme.colorScheme.primary,
                        ),
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
    final isEditable =
        _state == _PipelineState.idle ||
        _state == _PipelineState.error ||
        _state == _PipelineState.loaded;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text('Pipeline YAML', style: theme.textTheme.titleSmall),
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
                Icon(Icons.check_circle, color: theme.colorScheme.primary),
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
            InfoRow(label: 'Name', value: pipeline.name ?? '(unnamed)'),
            InfoRow(label: 'Stages', value: '${pipeline.stageCount}'),
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
                    .map(
                      (name) => Chip(
                        label: Text(name),
                        visualDensity: VisualDensity.compact,
                      ),
                    )
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
                Icon(Icons.output, color: theme.colorScheme.secondary),
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
                child: Text(_resultText!, style: theme.textTheme.bodyMedium),
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
enum _PipelineState { idle, loading, loaded, running, error }
