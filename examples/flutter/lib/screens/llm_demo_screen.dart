import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:xybrid_example/widgets/model_status_card.dart';
import 'package:xybrid_flutter/xybrid.dart';

/// LLM Chat demo screen with token streaming.
///
/// Demonstrates how to run LLM inference with streaming token output,
/// similar to the CLI repl mode with --stream flag.
class LlmDemoScreen extends StatefulWidget {
  const LlmDemoScreen({super.key});

  @override
  State<LlmDemoScreen> createState() => _LlmDemoScreenState();
}

/// Sealed class representing LLM screen states.
sealed class LlmScreenState {
  const LlmScreenState();
}

/// Initial state before model is loaded.
class LlmIdle extends LlmScreenState {
  const LlmIdle();
}

/// Model is being downloaded/loaded with progress.
class LlmLoadingModel extends LlmScreenState {
  /// Download progress from 0.0 to 1.0 (null if indeterminate).
  final double? progress;

  const LlmLoadingModel({this.progress});
}

/// Model failed to load.
class LlmLoadError extends LlmScreenState {
  final String message;

  const LlmLoadError(this.message);
}

/// Model is loaded and ready for inference.
class LlmReady extends LlmScreenState {
  const LlmReady();
}

/// Model is generating a response.
class LlmGenerating extends LlmScreenState {
  const LlmGenerating();
}

/// Inference failed with an error.
class LlmInferenceError extends LlmScreenState {
  final String message;

  const LlmInferenceError(this.message);
}

/// A message in the conversation history.
class _ChatMessage {
  final String role; // 'system', 'user', 'assistant'
  final String content;

  const _ChatMessage({required this.role, required this.content});
}

class _LlmDemoScreenState extends State<LlmDemoScreen> {
  /// Controller for the prompt input field.
  final _promptController = TextEditingController(
    text: 'Tell me a short joke.',
  );

  /// Scroll controller for auto-scrolling response.
  final _scrollController = ScrollController();

  /// Current screen state.
  LlmScreenState _state = const LlmIdle();

  /// Loaded model (if any).
  XybridModel? _model;

  /// The model ID used for LLM.
  static const _llmModelId = 'gemma-3-1b';

  /// Streaming response text.
  String _responseText = '';

  /// Number of tokens generated.
  int _tokenCount = 0;

  /// Generation start time for timing calculations.
  DateTime? _generationStartTime;

  /// Total inference time in milliseconds.
  int? _inferenceTimeMs;

  /// Tokens per second (calculated after generation completes).
  double? _tokensPerSecond;

  /// Generation stream subscription.
  StreamSubscription<StreamToken>? _streamSubscription;

  /// Load progress stream subscription.
  StreamSubscription<LoadEvent>? _loadSubscription;

  /// Whether context mode is enabled (preserves conversation history).
  bool _contextEnabled = false;

  /// Conversation history for context mode.
  List<_ChatMessage> _conversationHistory = [];

  /// Get conversation length (excluding system prompt).
  int get _conversationLength =>
      _conversationHistory.where((m) => m.role != 'system').length;

  @override
  void dispose() {
    _promptController.dispose();
    _scrollController.dispose();
    _streamSubscription?.cancel();
    _loadSubscription?.cancel();
    super.dispose();
  }

  /// Toggle context mode on/off.
  void _toggleContext() {
    setState(() {
      _contextEnabled = !_contextEnabled;
      if (_contextEnabled && _conversationHistory.isEmpty) {
        _conversationHistory = [
          const _ChatMessage(
            role: 'system',
            content: 'You are a helpful AI assistant. Keep responses concise.',
          ),
        ];
      }
    });
  }

  /// Clear the conversation history.
  void _clearContext() {
    setState(() {
      _conversationHistory = [
        const _ChatMessage(
          role: 'system',
          content: 'You are a helpful AI assistant. Keep responses concise.',
        ),
      ];
      _responseText = '';
      _tokenCount = 0;
      _inferenceTimeMs = null;
      _tokensPerSecond = null;
    });
  }

  /// Build a prompt that includes conversation history.
  String _buildContextualPrompt(String userMessage) {
    final buffer = StringBuffer();

    for (final message in _conversationHistory) {
      switch (message.role) {
        case 'system':
          buffer.writeln('System: ${message.content}');
        case 'user':
          buffer.writeln('User: ${message.content}');
        case 'assistant':
          buffer.writeln('Assistant: ${message.content}');
      }
    }

    // Add the new user message
    buffer.writeln('User: $userMessage');
    buffer.writeln('Assistant:');
    return buffer.toString();
  }

  /// Load the LLM model from the registry with progress tracking.
  Future<void> _loadModel() async {
    setState(() {
      _state = const LlmLoadingModel();
    });

    try {
      final loader = XybridModelLoader.fromRegistry(_llmModelId);

      _loadSubscription = loader.loadWithProgress().listen(
        (event) {
          if (!mounted) return;

          switch (event) {
            case LoadProgress(:final progress):
              setState(() {
                _state = LlmLoadingModel(progress: progress);
              });
            case LoadComplete():
              // Model is ready, now load it from cache
              _finalizeModelLoad(loader);
            case LoadError(:final message):
              setState(() {
                _state = LlmLoadError(message);
              });
          }
        },
        onError: (e) {
          if (mounted) {
            setState(() {
              _state = LlmLoadError(_formatErrorMessage(e.toString()));
            });
          }
        },
      );
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = LlmLoadError(_formatErrorMessage(e.toString()));
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
          _state = const LlmReady();
          _model = model;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = LlmLoadError(_formatErrorMessage(e.toString()));
        });
      }
    }
  }

  /// Generate response with streaming.
  Future<void> _generate() async {
    final userMessage = _promptController.text.trim();
    if (userMessage.isEmpty) {
      setState(() {
        _state = const LlmInferenceError('Please enter a prompt.');
      });
      return;
    }

    if (_model == null) {
      setState(() {
        _state =
            const LlmInferenceError('Model not loaded. Please load the model first.');
      });
      return;
    }

    // Build the prompt (with or without context)
    final prompt = _contextEnabled
        ? _buildContextualPrompt(userMessage)
        : userMessage;

    // Add user message to history if context is enabled
    if (_contextEnabled) {
      _conversationHistory.add(_ChatMessage(role: 'user', content: userMessage));
    }

    setState(() {
      _state = const LlmGenerating();
      _responseText = '';
      _tokenCount = 0;
      _inferenceTimeMs = null;
      _tokensPerSecond = null;
      _generationStartTime = DateTime.now();
    });

    try {
      final envelope = XybridEnvelope.text(prompt);
      final stream = _model!.runStreaming(envelope);

      _streamSubscription = stream.listen(
        (token) {
          if (mounted) {
            setState(() {
              _responseText = token.cumulativeText;
              _tokenCount = token.index + 1;
            });

            // Auto-scroll to bottom
            WidgetsBinding.instance.addPostFrameCallback((_) {
              if (_scrollController.hasClients) {
                _scrollController.animateTo(
                  _scrollController.position.maxScrollExtent,
                  duration: const Duration(milliseconds: 100),
                  curve: Curves.easeOut,
                );
              }
            });

            // Check for completion
            if (token.isFinal) {
              _calculateInferenceStats();
              // Add assistant response to history if context is enabled
              if (_contextEnabled && _responseText.isNotEmpty) {
                _conversationHistory.add(
                  _ChatMessage(role: 'assistant', content: _responseText),
                );
              }
              setState(() {
                _state = const LlmReady();
              });
            }
          }
        },
        onError: (e) {
          if (mounted) {
            setState(() {
              _state = LlmInferenceError(_formatErrorMessage(e.toString()));
            });
          }
        },
        onDone: () {
          if (mounted && _state is LlmGenerating) {
            _calculateInferenceStats();
            // Add assistant response to history if context is enabled
            if (_contextEnabled && _responseText.isNotEmpty) {
              _conversationHistory.add(
                _ChatMessage(role: 'assistant', content: _responseText),
              );
            }
            setState(() {
              _state = const LlmReady();
            });
          }
        },
      );
    } on XybridException catch (e) {
      if (mounted) {
        setState(() {
          _state = LlmInferenceError(e.message);
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = LlmInferenceError(_formatErrorMessage(e.toString()));
        });
      }
    }
  }

  /// Stop the current generation.
  void _stopGeneration() {
    _streamSubscription?.cancel();
    _streamSubscription = null;
    _calculateInferenceStats();
    if (mounted) {
      setState(() {
        _state = const LlmReady();
      });
    }
  }

  /// Calculate inference statistics after generation completes.
  void _calculateInferenceStats() {
    if (_generationStartTime != null && _tokenCount > 0) {
      final elapsed = DateTime.now().difference(_generationStartTime!);
      _inferenceTimeMs = elapsed.inMilliseconds;
      if (_inferenceTimeMs! > 0) {
        _tokensPerSecond = (_tokenCount / _inferenceTimeMs!) * 1000;
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
    if (error.contains('LLM features not enabled')) {
      return 'LLM features not enabled in the current build.';
    }
    return error;
  }

  /// Convert screen state to ModelLoadState for the status card.
  ModelLoadState get _modelLoadState {
    return switch (_state) {
      LlmIdle() => const ModelIdle(),
      LlmLoadingModel(:final progress) => ModelLoading(progress: progress),
      LlmLoadError(:final message) => ModelLoadError(message),
      LlmReady() || LlmGenerating() || LlmInferenceError() => const ModelReady(),
    };
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      resizeToAvoidBottomInset: true,
      appBar: AppBar(
        backgroundColor: theme.colorScheme.inversePrimary,
        title: const Text('LLM Chat'),
      ),
      body: SafeArea(
        child: Column(
          children: [
            // Scrollable top section (description + model status)
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    // Description card with context toggle
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: Card(
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
                                    'About LLM Demo',
                                    style: theme.textTheme.titleMedium,
                                  ),
                                  const Spacer(),
                                  // Context toggle
                                  Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Icon(
                                        Icons.chat_bubble_outline,
                                        size: 16,
                                        color: _contextEnabled ? Colors.blue : Colors.grey,
                                      ),
                                      const SizedBox(width: 4),
                                      Text(
                                        'Context',
                                        style: TextStyle(
                                          fontSize: 12,
                                          color: _contextEnabled ? Colors.blue : Colors.grey,
                                        ),
                                      ),
                                      Switch(
                                        value: _contextEnabled,
                                        onChanged: (_) => _toggleContext(),
                                        activeColor: Colors.blue,
                                      ),
                                    ],
                                  ),
                                ],
                              ),
                              const SizedBox(height: 8),
                              const Text(
                                'This demo shows LLM inference with token streaming, '
                                'similar to running "xybrid repl --stream". Tokens are '
                                'displayed as they are generated.',
                              ),
                              // Context info panel (when enabled)
                              if (_contextEnabled) ...[
                                const SizedBox(height: 12),
                                Container(
                                  padding: const EdgeInsets.all(12),
                                  decoration: BoxDecoration(
                                    color: Colors.blue.withOpacity(0.1),
                                    borderRadius: BorderRadius.circular(8),
                                    border: Border.all(color: Colors.blue.withOpacity(0.3)),
                                  ),
                                  child: Row(
                                    children: [
                                      const Icon(Icons.history, size: 16, color: Colors.blue),
                                      const SizedBox(width: 8),
                                      Text(
                                        'Conversation: $_conversationLength messages',
                                        style: const TextStyle(
                                          fontSize: 12,
                                          color: Colors.blue,
                                        ),
                                      ),
                                      const Spacer(),
                                      TextButton.icon(
                                        onPressed: _clearContext,
                                        icon: const Icon(Icons.delete_outline, size: 16),
                                        label: const Text('Clear'),
                                        style: TextButton.styleFrom(
                                          foregroundColor: Colors.blue,
                                          padding: const EdgeInsets.symmetric(horizontal: 8),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ],
                          ),
                        ),
                      ),
                    ),

                    // Model status section
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      child: ModelStatusCard(
                        state: _modelLoadState,
                        modelId: _llmModelId,
                        idleIcon: Icons.psychology_outlined,
                        idleTitle: 'LLM Model Required',
                        idleDescription: 'Load the $_llmModelId model to start chatting.',
                        onLoad: _loadModel,
                        readyTrailing: _tokenCount > 0
                            ? Chip(
                                label: Text('$_tokenCount tokens'),
                                padding: EdgeInsets.zero,
                                visualDensity: VisualDensity.compact,
                              )
                            : null,
                      ),
                    ),

                    // Response area (only shown when model is loaded)
                    if (_model != null) ...[
                      const SizedBox(height: 16),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: _buildResponseSection(theme),
                      ),
                    ],
                  ],
                ),
              ),
            ),

            // Input area - fixed at bottom (only shown when model is loaded)
            if (_model != null)
              Padding(
                padding: const EdgeInsets.all(16),
                child: _buildInputSection(theme),
              ),
          ],
        ),
      ),
    );
  }

  /// Build the response display section.
  Widget _buildResponseSection(ThemeData theme) {
    // Inference error state
    if (_state case LlmInferenceError(:final message)) {
      return Card(
        color: theme.colorScheme.errorContainer.withAlpha(100),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
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
                onPressed: _generate,
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    // Response display
    return Card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                Icon(
                  Icons.smart_toy_outlined,
                  color: theme.colorScheme.primary,
                ),
                const SizedBox(width: 8),
                Text(
                  'Response',
                  style: theme.textTheme.titleSmall?.copyWith(
                    color: theme.colorScheme.primary,
                  ),
                ),
                const Spacer(),
                if (_state is LlmGenerating)
                  SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: theme.colorScheme.primary,
                    ),
                  ),
              ],
            ),
          ),
          const Divider(height: 1),

          // Response text - constrained height to allow scrolling within
          ConstrainedBox(
            constraints: const BoxConstraints(
              minHeight: 150,
              maxHeight: 300,
            ),
            child: SingleChildScrollView(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              child: _responseText.isEmpty
                  ? Text(
                      'Enter a prompt below and tap "Send" to generate a response.',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                        fontStyle: FontStyle.italic,
                      ),
                    )
                  : MarkdownBody(
                      data: _responseText,
                      selectable: true,
                      styleSheet: MarkdownStyleSheet(
                        p: theme.textTheme.bodyMedium,
                        h1: theme.textTheme.headlineMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                        h2: theme.textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                        h3: theme.textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                        strong: theme.textTheme.bodyMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                        em: theme.textTheme.bodyMedium?.copyWith(
                          fontStyle: FontStyle.italic,
                        ),
                        listBullet: theme.textTheme.bodyMedium,
                        code: theme.textTheme.bodyMedium?.copyWith(
                          fontFamily: 'monospace',
                          backgroundColor:
                              theme.colorScheme.surfaceContainerHighest,
                        ),
                        codeblockDecoration: BoxDecoration(
                          color: theme.colorScheme.surfaceContainerHighest,
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                    ),
            ),
          ),

          // Stats row (shown when we have results)
          if (_inferenceTimeMs != null || _state is LlmGenerating)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: theme.colorScheme.surfaceContainerHighest.withAlpha(100),
                border: Border(
                  top: BorderSide(
                    color: theme.colorScheme.outline.withAlpha(50),
                  ),
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.timer_outlined,
                    size: 16,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                  const SizedBox(width: 6),
                  if (_state is LlmGenerating)
                    Text(
                      'Generating...',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    )
                  else if (_inferenceTimeMs != null) ...[
                    Text(
                      'Inference: ${(_inferenceTimeMs! / 1000).toStringAsFixed(2)}s',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                    const SizedBox(width: 16),
                    Icon(
                      Icons.speed,
                      size: 16,
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      '$_tokenCount tokens',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                    if (_tokensPerSecond != null) ...[
                      Text(
                        ' (${_tokensPerSecond!.toStringAsFixed(1)} tok/s)',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ],
                ],
              ),
            ),
        ],
      ),
    );
  }

  /// Build the input section.
  Widget _buildInputSection(ThemeData theme) {
    final isReady = _state is LlmReady;
    final isGenerating = _state is LlmGenerating;

    return Row(
      children: [
        Expanded(
          child: TextField(
            controller: _promptController,
            decoration: const InputDecoration(
              labelText: 'Prompt',
              hintText: 'Enter your message...',
              border: OutlineInputBorder(),
              prefixIcon: Icon(Icons.edit_note),
            ),
            maxLines: 2,
            minLines: 1,
            enabled: isReady,
            textInputAction: TextInputAction.send,
            onSubmitted: (_) {
              if (isReady) {
                _generate();
              }
            },
          ),
        ),
        const SizedBox(width: 12),
        if (isGenerating)
          FilledButton.tonalIcon(
            onPressed: _stopGeneration,
            icon: const Icon(Icons.stop),
            label: const Text('Stop'),
          )
        else
          FilledButton.icon(
            onPressed: isReady ? _generate : null,
            icon: const Icon(Icons.send),
            label: const Text('Send'),
          ),
      ],
    );
  }
}
