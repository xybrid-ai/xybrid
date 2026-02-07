/// Conversation context for multi-turn LLM interactions.
///
/// This class wraps the FRB-generated [FfiConversationContext] with a clean,
/// idiomatic Dart API.
library;

import 'envelope.dart';
import 'rust/api/context.dart';

/// Message role for conversation turns.
///
/// Used to tag messages in conversation history with their source.
enum MessageRole {
  /// System prompt that defines assistant behavior.
  /// Persists across [ConversationContext.clear] calls.
  system,

  /// User message.
  user,

  /// Assistant response.
  assistant;

  /// Convert to the FFI enum.
  FfiMessageRole toFfi() => switch (this) {
        MessageRole.system => FfiMessageRole.system,
        MessageRole.user => FfiMessageRole.user,
        MessageRole.assistant => FfiMessageRole.assistant,
      };

  /// Convert from the FFI enum.
  static MessageRole fromFfi(FfiMessageRole ffi) => switch (ffi) {
        FfiMessageRole.system => MessageRole.system,
        FfiMessageRole.user => MessageRole.user,
        FfiMessageRole.assistant => MessageRole.assistant,
      };
}

/// Manages multi-turn conversation history for LLM models.
///
/// ConversationContext stores conversation turns with proper message roles
/// (System, User, Assistant). It supports:
/// - System prompts that persist across [clear] calls
/// - Automatic FIFO pruning when history exceeds max length
/// - Chat template formatting for different LLM formats
///
/// ## Example
///
/// ```dart
/// final context = ConversationContext();
/// context.setSystem('You are a helpful assistant.');
///
/// context.pushText('Hello!', MessageRole.user);
/// final result = await model.runWithContext(
///   XybridEnvelope.text('Hello!'),
///   context,
/// );
/// context.pushText(result.text ?? '', MessageRole.assistant);
/// ```
class ConversationContext {
  /// The underlying FRB conversation context.
  final FfiConversationContext inner;

  /// Create a new conversation context with a generated UUID.
  ConversationContext() : inner = FfiConversationContext.create();

  /// Create a new conversation context with a specific ID.
  ///
  /// Useful for resuming conversations or tracking sessions.
  ConversationContext.withId(String id)
      : inner = FfiConversationContext.withId(id: id);

  /// Get the conversation ID.
  String get id => inner.id();

  /// Get the current history length (excluding system prompt).
  int get historyLength => inner.historyLen();

  /// Get the maximum history length before FIFO pruning.
  int get maxHistoryLength => inner.maxHistoryLen();

  /// Check if a system prompt is set.
  bool get hasSystem => inner.hasSystem();

  /// Set the system prompt for this conversation.
  ///
  /// The system prompt defines the assistant's behavior and persists
  /// across [clear] calls.
  void setSystem(String text) {
    inner.setSystem(text: text);
  }

  /// Set the maximum history length before FIFO pruning.
  ///
  /// When the history exceeds this limit, the oldest messages are dropped.
  /// Default is 50 messages.
  void setMaxHistoryLength(int length) {
    inner.setMaxHistoryLen(len: length);
  }

  /// Add a message to the conversation history.
  ///
  /// The envelope should have a role set via [XybridEnvelope.withRole].
  void push(XybridEnvelope envelope) {
    inner.push(envelope: envelope.inner);
  }

  /// Add a text message with a specific role.
  ///
  /// This is a convenience method that creates an envelope internally.
  void pushText(String text, MessageRole role) {
    inner.pushText(text: text, role: role.toFfi());
  }

  /// Clear the conversation history but preserve the system prompt and ID.
  void clear() {
    inner.clear();
  }

  @override
  String toString() => 'ConversationContext(id: $id, history: $historyLength)';
}
