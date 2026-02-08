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

  /// Last user message (for non-context mode display).
  String _lastUserMessage = '';

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
            content: 'You are a helpful AI assistant. Always ask the user their name first before helping them. Keep responses concise.',
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
          content: 'You are a helpful AI assistant. Always ask the user their name first before helping them. Keep responses concise.',
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

    // Store user message for display (in non-context mode)
    _lastUserMessage = userMessage;

    // Clear input field after sending
    _promptController.clear();

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
              _completeGeneration();
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
          // Fallback: complete if stream ends without isFinal token
          if (mounted && _state is LlmGenerating) {
            _completeGeneration();
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
    _completeGeneration();
  }

  /// Complete the generation and add response to history.
  /// Guarded to only run once per generation.
  void _completeGeneration() {
    // Guard: only complete if still generating
    if (_state is! LlmGenerating) return;

    _calculateInferenceStats();

    // Add assistant response to history if context is enabled
    if (_contextEnabled && _responseText.isNotEmpty) {
      _conversationHistory.add(
        _ChatMessage(role: 'assistant', content: _responseText),
      );
    }

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
        actions: [
          // Context toggle in app bar
          if (_model != null) ...[
            Icon(
              Icons.chat_bubble_outline,
              size: 18,
              color: _contextEnabled
                  ? theme.colorScheme.primary
                  : theme.colorScheme.onSurfaceVariant,
            ),
            Switch(
              value: _contextEnabled,
              onChanged: (_) => _toggleContext(),
              activeColor: theme.colorScheme.primary,
            ),
            if (_contextEnabled && _conversationLength > 0)
              IconButton(
                icon: const Icon(Icons.delete_outline),
                onPressed: _clearContext,
                tooltip: 'Clear conversation',
              ),
          ],
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            // Model status (only when not loaded)
            if (_model == null)
              Expanded(
                child: Center(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: ModelStatusCard(
                      state: _modelLoadState,
                      modelId: _llmModelId,
                      idleIcon: Icons.psychology_outlined,
                      idleTitle: 'LLM Model Required',
                      idleDescription: 'Load the $_llmModelId model to start chatting.',
                      onLoad: _loadModel,
                    ),
                  ),
                ),
              )
            else ...[
              // Chat messages area
              Expanded(
                child: _buildChatArea(theme),
              ),

              // Stats bar (when generating or after completion)
              if (_inferenceTimeMs != null || _state is LlmGenerating)
                _buildStatsBar(theme),

              // Input area
              _buildInputSection(theme),
            ],
          ],
        ),
      ),
    );
  }

  /// Build the chat messages area.
  Widget _buildChatArea(ThemeData theme) {
    final isGenerating = _state is LlmGenerating;

    // Build list of messages to display
    final messages = <_ChatMessage>[];

    if (_contextEnabled) {
      // Add conversation history (excluding system message)
      messages.addAll(
        _conversationHistory.where((m) => m.role != 'system'),
      );
    } else {
      // In non-context mode, show last exchange
      if (_lastUserMessage.isNotEmpty) {
        messages.add(_ChatMessage(role: 'user', content: _lastUserMessage));
      }
    }

    // Determine if we should show a streaming assistant bubble
    final showStreamingBubble =
        isGenerating || (!_contextEnabled && _responseText.isNotEmpty);

    if (messages.isEmpty && !showStreamingBubble) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.chat_bubble_outline,
              size: 64,
              color: theme.colorScheme.outlineVariant,
            ),
            const SizedBox(height: 16),
            Text(
              'Start a conversation',
              style: theme.textTheme.titleMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Type a message below to begin',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.outline,
              ),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      itemCount: messages.length + (showStreamingBubble ? 1 : 0),
      itemBuilder: (context, index) {
        // Assistant response bubble (streaming or completed)
        if (index == messages.length) {
          return _buildMessageBubble(
            theme,
            _ChatMessage(role: 'assistant', content: _responseText),
            isStreaming: isGenerating,
          );
        }

        final message = messages[index];

        // In context mode, the last assistant message might be streaming
        final isLiveAssistant = _contextEnabled &&
            isGenerating &&
            message.role == 'assistant' &&
            index == messages.length - 1;

        return _buildMessageBubble(
          theme,
          isLiveAssistant
              ? _ChatMessage(role: 'assistant', content: _responseText)
              : message,
          isStreaming: isLiveAssistant,
        );
      },
    );
  }

  /// Build a single chat message bubble.
  Widget _buildMessageBubble(
    ThemeData theme,
    _ChatMessage message, {
    bool isStreaming = false,
  }) {
    final isUser = message.role == 'user';

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment:
            isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!isUser) ...[
            CircleAvatar(
              radius: 16,
              backgroundColor: theme.colorScheme.primaryContainer,
              child: Icon(
                Icons.smart_toy,
                size: 18,
                color: theme.colorScheme.onPrimaryContainer,
              ),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Container(
              constraints: BoxConstraints(
                maxWidth: MediaQuery.of(context).size.width * 0.75,
              ),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: isUser
                    ? theme.colorScheme.primary
                    : theme.colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(16),
                  topRight: const Radius.circular(16),
                  bottomLeft: Radius.circular(isUser ? 16 : 4),
                  bottomRight: Radius.circular(isUser ? 4 : 16),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (isUser)
                    Text(
                      message.content,
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: theme.colorScheme.onPrimary,
                      ),
                    )
                  else
                    MarkdownBody(
                      data: message.content.isEmpty && isStreaming
                          ? '...'
                          : message.content,
                      selectable: true,
                      styleSheet: MarkdownStyleSheet(
                        p: theme.textTheme.bodyMedium?.copyWith(
                          color: theme.colorScheme.onSurface,
                        ),
                        strong: theme.textTheme.bodyMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: theme.colorScheme.onSurface,
                        ),
                        em: theme.textTheme.bodyMedium?.copyWith(
                          fontStyle: FontStyle.italic,
                          color: theme.colorScheme.onSurface,
                        ),
                        code: theme.textTheme.bodySmall?.copyWith(
                          fontFamily: 'monospace',
                          backgroundColor: theme.colorScheme.surface,
                          color: theme.colorScheme.onSurface,
                        ),
                        codeblockDecoration: BoxDecoration(
                          color: theme.colorScheme.surface,
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                    ),
                  if (isStreaming) ...[
                    const SizedBox(height: 4),
                    SizedBox(
                      width: 12,
                      height: 12,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: theme.colorScheme.primary,
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
          if (isUser) ...[
            const SizedBox(width: 8),
            CircleAvatar(
              radius: 16,
              backgroundColor: theme.colorScheme.secondaryContainer,
              child: Icon(
                Icons.person,
                size: 18,
                color: theme.colorScheme.onSecondaryContainer,
              ),
            ),
          ],
        ],
      ),
    );
  }

  /// Build the stats bar.
  Widget _buildStatsBar(ThemeData theme) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest.withAlpha(100),
        border: Border(
          top: BorderSide(
            color: theme.colorScheme.outline.withAlpha(50),
          ),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if (_state is LlmGenerating) ...[
            SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: theme.colorScheme.primary,
              ),
            ),
            const SizedBox(width: 8),
            Text(
              '$_tokenCount tokens',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ] else if (_inferenceTimeMs != null) ...[
            Icon(
              Icons.timer_outlined,
              size: 14,
              color: theme.colorScheme.onSurfaceVariant,
            ),
            const SizedBox(width: 4),
            Text(
              '${(_inferenceTimeMs! / 1000).toStringAsFixed(2)}s',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(width: 12),
            Text(
              '$_tokenCount tokens',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            if (_tokensPerSecond != null) ...[
              const SizedBox(width: 4),
              Text(
                '(${_tokensPerSecond!.toStringAsFixed(1)} tok/s)',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.outline,
                ),
              ),
            ],
          ],
        ],
      ),
    );
  }

  /// Build the input section.
  Widget _buildInputSection(ThemeData theme) {
    final isReady = _state is LlmReady;
    final isGenerating = _state is LlmGenerating;

    // Show error inline if there is one
    if (_state case LlmInferenceError(:final message)) {
      return Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            margin: const EdgeInsets.fromLTRB(16, 0, 16, 8),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: theme.colorScheme.errorContainer.withAlpha(100),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                Icon(Icons.error_outline,
                    size: 16, color: theme.colorScheme.error),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    message,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.error,
                    ),
                  ),
                ),
                TextButton(
                  onPressed: _generate,
                  child: const Text('Retry'),
                ),
              ],
            ),
          ),
          _buildInputRow(theme, isReady: true, isGenerating: false),
        ],
      );
    }

    return _buildInputRow(theme, isReady: isReady, isGenerating: isGenerating);
  }

  Widget _buildInputRow(
    ThemeData theme, {
    required bool isReady,
    required bool isGenerating,
  }) {
    return Container(
      padding: const EdgeInsets.fromLTRB(8, 8, 8, 8),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        border: Border(
          top: BorderSide(
            color: theme.colorScheme.outline.withAlpha(30),
          ),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Expanded(
            child: TextField(
              controller: _promptController,
              decoration: InputDecoration(
                hintText: 'Message...',
                filled: true,
                fillColor: theme.colorScheme.surfaceContainerHighest,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(24),
                  borderSide: BorderSide.none,
                ),
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 10,
                ),
              ),
              maxLines: 4,
              minLines: 1,
              enabled: isReady || _state is LlmInferenceError,
              textInputAction: TextInputAction.send,
              onSubmitted: (_) {
                if (isReady || _state is LlmInferenceError) {
                  _generate();
                }
              },
            ),
          ),
          const SizedBox(width: 8),
          if (isGenerating)
            IconButton.filled(
              onPressed: _stopGeneration,
              icon: const Icon(Icons.stop),
              style: IconButton.styleFrom(
                backgroundColor: theme.colorScheme.errorContainer,
                foregroundColor: theme.colorScheme.onErrorContainer,
              ),
            )
          else
            IconButton.filled(
              onPressed:
                  (isReady || _state is LlmInferenceError) ? _generate : null,
              icon: const Icon(Icons.send),
            ),
        ],
      ),
    );
  }
}
