//! Manual operations retain their original session until the last owner drops.
use super::*;
use std::ops::Deref;

/// A portable operation pinned to the SDK session that created it.
///
/// All portable recording and lifecycle methods are available through `Deref`.
/// Drop cancels unfinished operations before releasing the session; it never flushes.
#[derive(Debug)]
pub struct ManualObservation<T> {
    // Drop the operation before the session, including during retirement.
    operation: T,
    session: ObservabilitySession,
    conversation_id: Option<String>,
}

impl<T> Deref for ManualObservation<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.operation
    }
}

fn options(options: &ObservationOptions) -> StartOptions {
    StartOptions {
        parent: options.parent.clone().or_else(ambient_parent),
        trace: if options.parent.is_some() {
            options.trace
        } else {
            inherited_trace(options.trace)
        },
        attributes: options.attributes.clone(),
        ..Default::default()
    }
}

/// [`options`] plus the caller's conversation id, for operations whose
/// request type has no conversation field (workflow, tool, model load).
fn options_with_conversation(controls: &ObservationOptions) -> StartOptions {
    let mut start = options(controls);
    if let Some(id) = conversation_id(Some(controls)) {
        start
            .attributes
            .push(KeyValue::new(semconv::attr::CONVERSATION_ID, id));
    }
    start
}

macro_rules! manual {
    ($start:ident, $operation:ty, $argument:ident: $ty:ty) => {
        /// Starts a manual operation, resolving the session and parent before any worker hop.
        /// `controls.conversation_id` is recorded as `gen_ai.conversation.id`.
        pub fn $start(
            $argument: $ty,
            controls: &ObservationOptions,
        ) -> ManualObservation<$operation> {
            let session = resolve_session(Some(controls));
            let operation = session
                .observer()
                .$start($argument, options_with_conversation(controls));
            ManualObservation {
                operation,
                session,
                conversation_id: conversation_id(Some(controls)),
            }
        }

        impl ManualObservation<$operation> {
            /// Installs this operation's parent, suppression and original session on this thread.
            /// Do not hold this guard across an async await; capture context for a worker instead.
            #[must_use]
            pub fn enter(&self) -> CapturedScope {
                CapturedScope::for_operation(
                    self.session.clone(),
                    self.operation.context(),
                    self.operation.context().is_none(),
                    self.conversation_id.clone(),
                )
            }
        }
    };
}

manual!(start_workflow, WorkflowOperation, name: &str);
manual!(start_tool, ToolOperation, call: ToolCallInfo);
manual!(start_model_load, ModelLoadOperation, info: ModelLoadInfo);

/// Starts a manual inference using the SDK's session, parent and content controls.
pub fn start_inference(
    mut request: InferenceRequest,
    controls: &ObservationOptions,
) -> ManualObservation<InferenceOperation> {
    let session = resolve_session(Some(controls));
    request.conversation_id = controls
        .conversation_id
        .clone()
        .or(request.conversation_id)
        .or_else(|| conversation_id(None));
    request.prompt = controls.prompt.clone().or(request.prompt);
    let conversation_id = request.conversation_id.clone();
    let operation = session
        .observer()
        .start_inference(request, options(controls));
    ManualObservation {
        operation,
        session,
        conversation_id,
    }
}

impl ManualObservation<InferenceOperation> {
    /// Installs this inference's context and original session on this thread.
    #[must_use]
    pub fn enter(&self) -> CapturedScope {
        CapturedScope::for_operation(
            self.session.clone(),
            self.operation.context(),
            self.operation.context().is_none(),
            self.conversation_id.clone(),
        )
    }
}
