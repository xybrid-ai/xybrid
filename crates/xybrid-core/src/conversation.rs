//! Conversation Context - Multi-turn conversation management for LLM interactions.
//!
//! This module provides `ConversationContext`, a container that accumulates
//! conversation turns as `Envelope` instances with message roles. It supports:
//!
//! - Automatic FIFO pruning when history exceeds the maximum length
//! - Persistent system prompts that survive `clear()` operations
//! - Ordered retrieval for LLM prompt construction
//!
//! # Example
//!
//! ```rust
//! use xybrid_core::conversation::ConversationContext;
//! use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
//!
//! // Create a new conversation context
//! let mut ctx = ConversationContext::new();
//!
//! // Set a system prompt
//! ctx = ctx.with_system(
//!     Envelope::new(EnvelopeKind::Text("You are a helpful assistant.".to_string()))
//!         .with_role(MessageRole::System)
//! );
//!
//! // Add user and assistant messages
//! ctx.push(
//!     Envelope::new(EnvelopeKind::Text("Hello!".to_string()))
//!         .with_role(MessageRole::User)
//! );
//! ctx.push(
//!     Envelope::new(EnvelopeKind::Text("Hi there!".to_string()))
//!         .with_role(MessageRole::Assistant)
//! );
//!
//! // Get context for LLM (system prompt + history)
//! let messages = ctx.context_for_llm();
//! assert_eq!(messages.len(), 3); // system + user + assistant
//! ```

use crate::ir::Envelope;
use uuid::Uuid;

/// A conversation context that manages multi-turn conversation history.
///
/// `ConversationContext` stores conversation turns as `Envelope` instances,
/// each tagged with a message role (System, User, or Assistant). It provides:
///
/// - **UUID identification**: Each context has a unique ID for tracking
/// - **System prompt persistence**: The system envelope survives `clear()` calls
/// - **FIFO pruning**: When history exceeds `max_history_len`, oldest messages are dropped
/// - **LLM-ready output**: `context_for_llm()` returns messages in the correct order
///
/// # Thread Safety
///
/// `ConversationContext` is not thread-safe. For concurrent access, wrap it
/// in a `Mutex` or `RwLock`.
#[derive(Debug, Clone)]
pub struct ConversationContext {
    /// Unique identifier for this conversation
    id: String,
    /// Conversation history (user and assistant messages)
    history: Vec<Envelope>,
    /// Optional system prompt envelope (persists across clear())
    system_envelope: Option<Envelope>,
    /// Maximum number of history entries before FIFO pruning
    max_history_len: usize,
}

impl Default for ConversationContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationContext {
    /// Default maximum history length.
    const DEFAULT_MAX_HISTORY_LEN: usize = 50;

    /// Creates a new conversation context with a generated UUID and default settings.
    ///
    /// The default `max_history_len` is 50.
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::conversation::ConversationContext;
    ///
    /// let ctx = ConversationContext::new();
    /// assert_eq!(ctx.history().len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            history: Vec::new(),
            system_envelope: None,
            max_history_len: Self::DEFAULT_MAX_HISTORY_LEN,
        }
    }

    /// Creates a new conversation context with a specific ID.
    ///
    /// Useful for resuming a conversation or testing.
    ///
    /// # Arguments
    ///
    /// * `id` - The conversation identifier
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::conversation::ConversationContext;
    ///
    /// let ctx = ConversationContext::with_id("my-conversation-123".to_string());
    /// assert_eq!(ctx.id(), "my-conversation-123");
    /// ```
    pub fn with_id(id: String) -> Self {
        Self {
            id,
            history: Vec::new(),
            system_envelope: None,
            max_history_len: Self::DEFAULT_MAX_HISTORY_LEN,
        }
    }

    /// Sets the maximum history length and returns self (builder pattern).
    ///
    /// When the history length exceeds this limit, the oldest messages
    /// are dropped (FIFO pruning).
    ///
    /// # Arguments
    ///
    /// * `len` - Maximum number of history entries
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::conversation::ConversationContext;
    ///
    /// let ctx = ConversationContext::new().with_max_history_len(100);
    /// ```
    pub fn with_max_history_len(mut self, len: usize) -> Self {
        self.max_history_len = len;
        self
    }

    /// Sets the system prompt envelope and returns self (builder pattern).
    ///
    /// The system envelope is included at the beginning of `context_for_llm()`
    /// output and persists across `clear()` calls.
    ///
    /// # Arguments
    ///
    /// * `envelope` - The system prompt envelope (should have MessageRole::System)
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::conversation::ConversationContext;
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let ctx = ConversationContext::new()
    ///     .with_system(
    ///         Envelope::new(EnvelopeKind::Text("You are helpful.".to_string()))
    ///             .with_role(MessageRole::System)
    ///     );
    /// ```
    pub fn with_system(mut self, envelope: Envelope) -> Self {
        self.system_envelope = Some(envelope);
        self
    }

    /// Returns the conversation ID.
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::conversation::ConversationContext;
    ///
    /// let ctx = ConversationContext::new();
    /// println!("Conversation ID: {}", ctx.id());
    /// ```
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the maximum history length.
    pub fn max_history_len(&self) -> usize {
        self.max_history_len
    }

    /// Returns a reference to the system envelope, if set.
    pub fn system_envelope(&self) -> Option<&Envelope> {
        self.system_envelope.as_ref()
    }

    /// Appends an envelope to the conversation history.
    ///
    /// If the history length exceeds `max_history_len` after adding,
    /// the oldest message is dropped (FIFO pruning).
    ///
    /// # Arguments
    ///
    /// * `envelope` - The envelope to add (should have a message role set)
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::conversation::ConversationContext;
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let mut ctx = ConversationContext::new();
    /// ctx.push(
    ///     Envelope::new(EnvelopeKind::Text("Hello!".to_string()))
    ///         .with_role(MessageRole::User)
    /// );
    /// assert_eq!(ctx.history().len(), 1);
    /// ```
    pub fn push(&mut self, envelope: Envelope) {
        self.history.push(envelope);

        // FIFO pruning: drop oldest message if we exceed the limit
        while self.history.len() > self.max_history_len {
            self.history.remove(0);
        }
    }

    /// Returns a reference to the conversation history.
    ///
    /// This does not include the system envelope; use `context_for_llm()`
    /// for the complete message sequence.
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::conversation::ConversationContext;
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let mut ctx = ConversationContext::new();
    /// ctx.push(Envelope::new(EnvelopeKind::Text("Hi".to_string())).with_role(MessageRole::User));
    /// assert_eq!(ctx.history().len(), 1);
    /// ```
    pub fn history(&self) -> &[Envelope] {
        &self.history
    }

    /// Returns the complete message sequence for LLM prompt construction.
    ///
    /// Returns the system envelope (if set) followed by the history entries,
    /// up to `max_history_len` total entries.
    ///
    /// # Returns
    ///
    /// A vector of envelope references in order: [system, ...history]
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::conversation::ConversationContext;
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let mut ctx = ConversationContext::new()
    ///     .with_system(
    ///         Envelope::new(EnvelopeKind::Text("System prompt".to_string()))
    ///             .with_role(MessageRole::System)
    ///     );
    /// ctx.push(Envelope::new(EnvelopeKind::Text("User msg".to_string())).with_role(MessageRole::User));
    /// ctx.push(Envelope::new(EnvelopeKind::Text("Assistant msg".to_string())).with_role(MessageRole::Assistant));
    ///
    /// let messages = ctx.context_for_llm();
    /// assert_eq!(messages.len(), 3);
    /// assert!(messages[0].is_system_message());
    /// assert!(messages[1].is_user_message());
    /// assert!(messages[2].is_assistant_message());
    /// ```
    pub fn context_for_llm(&self) -> Vec<&Envelope> {
        let mut result = Vec::with_capacity(self.history.len() + 1);

        // Add system envelope first if present
        if let Some(ref system) = self.system_envelope {
            result.push(system);
        }

        // Add history entries (already pruned to max_history_len)
        for envelope in &self.history {
            result.push(envelope);
        }

        result
    }

    /// Clears the conversation history but preserves the system envelope and ID.
    ///
    /// Use this to start a new conversation within the same context.
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::conversation::ConversationContext;
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let mut ctx = ConversationContext::new()
    ///     .with_system(
    ///         Envelope::new(EnvelopeKind::Text("System".to_string()))
    ///             .with_role(MessageRole::System)
    ///     );
    /// ctx.push(Envelope::new(EnvelopeKind::Text("User".to_string())).with_role(MessageRole::User));
    ///
    /// ctx.clear();
    ///
    /// // History is cleared
    /// assert_eq!(ctx.history().len(), 0);
    /// // System envelope is preserved
    /// assert!(ctx.system_envelope().is_some());
    /// ```
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{EnvelopeKind, MessageRole};

    #[test]
    fn test_new_creates_uuid() {
        let ctx1 = ConversationContext::new();
        let ctx2 = ConversationContext::new();

        // Each context should have a unique UUID
        assert_ne!(ctx1.id(), ctx2.id());
        // UUID should be valid format (36 chars with hyphens)
        assert_eq!(ctx1.id().len(), 36);
    }

    #[test]
    fn test_new_has_empty_history() {
        let ctx = ConversationContext::new();
        assert!(ctx.history().is_empty());
        assert!(ctx.system_envelope().is_none());
    }

    #[test]
    fn test_default_max_history_len() {
        let ctx = ConversationContext::new();
        assert_eq!(ctx.max_history_len(), 50);
    }

    #[test]
    fn test_with_id() {
        let ctx = ConversationContext::with_id("custom-id".to_string());
        assert_eq!(ctx.id(), "custom-id");
    }

    #[test]
    fn test_with_max_history_len() {
        let ctx = ConversationContext::new().with_max_history_len(100);
        assert_eq!(ctx.max_history_len(), 100);
    }

    #[test]
    fn test_with_system() {
        let system_envelope = Envelope::new(EnvelopeKind::Text("System prompt".to_string()))
            .with_role(MessageRole::System);

        let ctx = ConversationContext::new().with_system(system_envelope.clone());

        assert!(ctx.system_envelope().is_some());
        assert!(ctx.system_envelope().unwrap().is_system_message());
    }

    #[test]
    fn test_push_and_history() {
        let mut ctx = ConversationContext::new();

        ctx.push(
            Envelope::new(EnvelopeKind::Text("Hello".to_string())).with_role(MessageRole::User),
        );
        ctx.push(
            Envelope::new(EnvelopeKind::Text("Hi there".to_string()))
                .with_role(MessageRole::Assistant),
        );

        let history = ctx.history();
        assert_eq!(history.len(), 2);
        assert!(history[0].is_user_message());
        assert!(history[1].is_assistant_message());
    }

    #[test]
    fn test_fifo_pruning() {
        let mut ctx = ConversationContext::new().with_max_history_len(3);

        // Add 5 messages
        for i in 0..5 {
            ctx.push(
                Envelope::new(EnvelopeKind::Text(format!("Message {}", i)))
                    .with_role(MessageRole::User),
            );
        }

        // Should only have the last 3 messages
        assert_eq!(ctx.history().len(), 3);

        // Verify the oldest messages were dropped
        if let EnvelopeKind::Text(text) = &ctx.history()[0].kind {
            assert_eq!(text, "Message 2");
        }
        if let EnvelopeKind::Text(text) = &ctx.history()[1].kind {
            assert_eq!(text, "Message 3");
        }
        if let EnvelopeKind::Text(text) = &ctx.history()[2].kind {
            assert_eq!(text, "Message 4");
        }
    }

    #[test]
    fn test_context_for_llm_with_system() {
        let mut ctx = ConversationContext::new().with_system(
            Envelope::new(EnvelopeKind::Text("You are helpful.".to_string()))
                .with_role(MessageRole::System),
        );

        ctx.push(
            Envelope::new(EnvelopeKind::Text("Hello".to_string())).with_role(MessageRole::User),
        );
        ctx.push(
            Envelope::new(EnvelopeKind::Text("Hi!".to_string())).with_role(MessageRole::Assistant),
        );

        let messages = ctx.context_for_llm();
        assert_eq!(messages.len(), 3);
        assert!(messages[0].is_system_message());
        assert!(messages[1].is_user_message());
        assert!(messages[2].is_assistant_message());
    }

    #[test]
    fn test_context_for_llm_without_system() {
        let mut ctx = ConversationContext::new();

        ctx.push(
            Envelope::new(EnvelopeKind::Text("Hello".to_string())).with_role(MessageRole::User),
        );

        let messages = ctx.context_for_llm();
        assert_eq!(messages.len(), 1);
        assert!(messages[0].is_user_message());
    }

    #[test]
    fn test_clear_preserves_system_and_id() {
        let original_id = "test-id".to_string();
        let mut ctx = ConversationContext::with_id(original_id.clone()).with_system(
            Envelope::new(EnvelopeKind::Text("System".to_string())).with_role(MessageRole::System),
        );

        ctx.push(
            Envelope::new(EnvelopeKind::Text("User msg".to_string())).with_role(MessageRole::User),
        );
        ctx.push(
            Envelope::new(EnvelopeKind::Text("Assistant msg".to_string()))
                .with_role(MessageRole::Assistant),
        );

        assert_eq!(ctx.history().len(), 2);

        ctx.clear();

        // History should be empty
        assert!(ctx.history().is_empty());
        // System envelope should be preserved
        assert!(ctx.system_envelope().is_some());
        // ID should be preserved
        assert_eq!(ctx.id(), original_id);
    }

    #[test]
    fn test_default_trait() {
        let ctx: ConversationContext = Default::default();
        assert!(ctx.history().is_empty());
        assert_eq!(ctx.max_history_len(), 50);
    }

    #[test]
    fn test_context_for_llm_ordering() {
        let mut ctx = ConversationContext::new().with_system(
            Envelope::new(EnvelopeKind::Text("System".to_string())).with_role(MessageRole::System),
        );

        // Add alternating user/assistant messages
        ctx.push(
            Envelope::new(EnvelopeKind::Text("User 1".to_string())).with_role(MessageRole::User),
        );
        ctx.push(
            Envelope::new(EnvelopeKind::Text("Assistant 1".to_string()))
                .with_role(MessageRole::Assistant),
        );
        ctx.push(
            Envelope::new(EnvelopeKind::Text("User 2".to_string())).with_role(MessageRole::User),
        );
        ctx.push(
            Envelope::new(EnvelopeKind::Text("Assistant 2".to_string()))
                .with_role(MessageRole::Assistant),
        );

        let messages = ctx.context_for_llm();
        assert_eq!(messages.len(), 5);

        // Verify ordering
        if let EnvelopeKind::Text(text) = &messages[0].kind {
            assert_eq!(text, "System");
        }
        if let EnvelopeKind::Text(text) = &messages[1].kind {
            assert_eq!(text, "User 1");
        }
        if let EnvelopeKind::Text(text) = &messages[2].kind {
            assert_eq!(text, "Assistant 1");
        }
        if let EnvelopeKind::Text(text) = &messages[3].kind {
            assert_eq!(text, "User 2");
        }
        if let EnvelopeKind::Text(text) = &messages[4].kind {
            assert_eq!(text, "Assistant 2");
        }
    }
}
