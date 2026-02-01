import 'package:flutter/material.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

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
        final loader = XybridModelLoader.fromRegistry(
          'network-test-unavailable',
        );
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
    if (error.contains('not found') ||
        error.contains('NotFound') ||
        error.contains('404')) {
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
                        Icon(
                          Icons.info_outline,
                          color: theme.colorScheme.primary,
                        ),
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
                Text('Retry Logic Demo', style: theme.textTheme.titleMedium),
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
                : Text(value, style: theme.textTheme.bodyMedium),
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
