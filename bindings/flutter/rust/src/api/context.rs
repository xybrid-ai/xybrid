//! Conversation context FFI wrappers for Flutter.
//!
//! Provides multi-turn conversation memory for LLM interactions.

use flutter_rust_bridge::frb;
use std::sync::{Arc, RwLock};
use xybrid_core::conversation::ConversationContext;
use xybrid_core::ir::MessageRole;

use super::envelope::FfiEnvelope;

/// Message role for conversation turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiMessageRole {
    /// System prompt that defines assistant behavior.
    System,
    /// User message.
    User,
    /// Assistant response.
    Assistant,
}

impl From<FfiMessageRole> for MessageRole {
    fn from(role: FfiMessageRole) -> Self {
        match role {
            FfiMessageRole::System => MessageRole::System,
            FfiMessageRole::User => MessageRole::User,
            FfiMessageRole::Assistant => MessageRole::Assistant,
        }
    }
}

impl From<MessageRole> for FfiMessageRole {
    fn from(role: MessageRole) -> Self {
        match role {
            MessageRole::System => FfiMessageRole::System,
            MessageRole::User => FfiMessageRole::User,
            MessageRole::Assistant => FfiMessageRole::Assistant,
        }
    }
}

/// FFI wrapper for ConversationContext.
///
/// Manages multi-turn conversation history for LLM models.
/// Supports system prompts, automatic FIFO pruning, and chat template formatting.
#[frb(opaque)]
pub struct FfiConversationContext(pub(crate) Arc<RwLock<ConversationContext>>);

impl FfiConversationContext {
    /// Create a new conversation context with a generated UUID.
    #[frb(sync)]
    pub fn create() -> FfiConversationContext {
        FfiConversationContext(Arc::new(RwLock::new(ConversationContext::new())))
    }

    /// Create a new conversation context with a specific ID.
    ///
    /// Useful for resuming conversations or tracking sessions.
    #[frb(sync)]
    pub fn with_id(id: String) -> FfiConversationContext {
        FfiConversationContext(Arc::new(RwLock::new(ConversationContext::with_id(id))))
    }

    /// Set the system prompt for this conversation.
    ///
    /// The system prompt defines the assistant's behavior and persists
    /// across `clear()` calls.
    #[frb(sync)]
    pub fn set_system(&self, text: String) {
        use xybrid_core::ir::{Envelope, EnvelopeKind};

        let system_envelope =
            Envelope::new(EnvelopeKind::Text(text)).with_role(MessageRole::System);

        if let Ok(mut ctx) = self.0.write() {
            // We need to recreate with system since with_system consumes self
            let id = ctx.id().to_string();
            let max_len = ctx.max_history_len();
            let new_ctx = ConversationContext::with_id(id)
                .with_max_history_len(max_len)
                .with_system(system_envelope);
            *ctx = new_ctx;
        }
    }

    /// Set the maximum history length before FIFO pruning.
    ///
    /// Default is 50 messages.
    #[frb(sync)]
    pub fn set_max_history_len(&self, len: u32) {
        if let Ok(mut ctx) = self.0.write() {
            let id = ctx.id().to_string();
            let system = ctx.system_envelope().cloned();
            let history: Vec<_> = ctx.history().to_vec();

            let mut new_ctx = ConversationContext::with_id(id).with_max_history_len(len as usize);

            if let Some(sys) = system {
                new_ctx = new_ctx.with_system(sys);
            }

            for envelope in history {
                new_ctx.push(envelope);
            }

            *ctx = new_ctx;
        }
    }

    /// Add a message to the conversation history.
    ///
    /// The envelope should have a role set via `with_role()`.
    #[frb(sync)]
    pub fn push(&self, envelope: FfiEnvelope) {
        if let Ok(mut ctx) = self.0.write() {
            ctx.push(envelope.0);
        }
    }

    /// Add a text message with a specific role.
    ///
    /// Convenience method that creates an envelope with the given role.
    #[frb(sync)]
    pub fn push_text(&self, text: String, role: FfiMessageRole) {
        use xybrid_core::ir::{Envelope, EnvelopeKind};

        let envelope = Envelope::new(EnvelopeKind::Text(text)).with_role(role.into());

        if let Ok(mut ctx) = self.0.write() {
            ctx.push(envelope);
        }
    }

    /// Clear the conversation history but preserve the system prompt and ID.
    #[frb(sync)]
    pub fn clear(&self) {
        if let Ok(mut ctx) = self.0.write() {
            ctx.clear();
        }
    }

    /// Get the conversation ID.
    #[frb(sync)]
    pub fn id(&self) -> String {
        self.0
            .read()
            .map(|ctx| ctx.id().to_string())
            .unwrap_or_default()
    }

    /// Get the current history length (excluding system prompt).
    #[frb(sync)]
    pub fn history_len(&self) -> u32 {
        self.0
            .read()
            .map(|ctx| ctx.history().len() as u32)
            .unwrap_or(0)
    }

    /// Get the maximum history length.
    #[frb(sync)]
    pub fn max_history_len(&self) -> u32 {
        self.0
            .read()
            .map(|ctx| ctx.max_history_len() as u32)
            .unwrap_or(50)
    }

    /// Check if a system prompt is set.
    #[frb(sync)]
    pub fn has_system(&self) -> bool {
        self.0
            .read()
            .map(|ctx| ctx.system_envelope().is_some())
            .unwrap_or(false)
    }
}

// Allow cloning the context handle (shares the same underlying context)
impl Clone for FfiConversationContext {
    fn clone(&self) -> Self {
        FfiConversationContext(Arc::clone(&self.0))
    }
}
