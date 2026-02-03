import 'package:flutter/material.dart';

/// State representing model loading progress.
sealed class ModelLoadState {
  const ModelLoadState();
}

/// Model has not been loaded yet.
class ModelIdle extends ModelLoadState {
  const ModelIdle();
}

/// Model is currently loading with progress.
class ModelLoading extends ModelLoadState {
  /// Download progress from 0.0 to 1.0 (null if indeterminate).
  final double? progress;

  const ModelLoading({this.progress});

  /// Progress as percentage (0-100).
  int? get percentage => progress != null ? (progress! * 100).round() : null;
}

/// Model is ready for inference.
class ModelReady extends ModelLoadState {
  const ModelReady();
}

/// Model loading failed with an error.
class ModelLoadError extends ModelLoadState {
  final String message;

  const ModelLoadError(this.message);
}

/// A reusable card that displays model loading status.
///
/// Handles idle, loading (with progress), ready, and error states.
class ModelStatusCard extends StatelessWidget {
  /// The current loading state.
  final ModelLoadState state;

  /// The model ID being loaded.
  final String modelId;

  /// Icon to show when model is idle (before loading).
  final IconData idleIcon;

  /// Title shown when model is idle.
  final String idleTitle;

  /// Description shown when model is idle.
  final String idleDescription;

  /// Callback when the load button is pressed.
  final VoidCallback? onLoad;

  /// Optional trailing widget shown when model is ready (e.g., token count chip).
  final Widget? readyTrailing;

  const ModelStatusCard({
    super.key,
    required this.state,
    required this.modelId,
    this.idleIcon = Icons.smart_toy_outlined,
    this.idleTitle = 'Model Required',
    this.idleDescription = 'Load the model to get started.',
    this.onLoad,
    this.readyTrailing,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return switch (state) {
      ModelIdle() => _buildIdleCard(theme),
      ModelLoading(:final progress) => _buildLoadingCard(theme, progress),
      ModelReady() => _buildReadyCard(theme),
      ModelLoadError(:final message) => _buildErrorCard(theme, message),
    };
  }

  Widget _buildIdleCard(ThemeData theme) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Icon(idleIcon, size: 48, color: theme.colorScheme.primary),
            const SizedBox(height: 16),
            Text(idleTitle, style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            Text(
              idleDescription,
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium,
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: onLoad,
              icon: const Icon(Icons.download),
              label: const Text('Load Model'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingCard(ThemeData theme, double? progress) {
    final percentage = progress != null ? (progress * 100).round() : null;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            Text(
              'Loading $modelId...',
              style: theme.textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            LinearProgressIndicator(value: progress),
            const SizedBox(height: 12),
            Text(
              percentage != null
                  ? 'Downloading... $percentage%'
                  : 'Downloading model from registry...',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildReadyCard(ThemeData theme) {
    return Card(
      color: theme.colorScheme.primaryContainer.withAlpha(100),
      child: Padding(
        padding: const EdgeInsets.all(12),
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
                  Text(modelId, style: theme.textTheme.bodySmall),
                ],
              ),
            ),
            if (readyTrailing != null) readyTrailing!,
          ],
        ),
      ),
    );
  }

  Widget _buildErrorCard(ThemeData theme, String message) {
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
                  'Failed to load model',
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
              onPressed: onLoad,
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }
}
