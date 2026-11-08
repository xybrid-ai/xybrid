//! ASR session lifetime and scopes for real decoder executions.
//!
//! One `invoke_workflow asr_session` per utterance, started by the first
//! `feed` (never by opening or warming a stream), ended by `flush` and
//! cancelled by `reset` or by dropping the stream. Each real decoder execution
//! is a child inference; cached partials record nothing.
use super::*;
use xybrid_core::streaming::{DecoderObservation, DecoderObserver, StreamResult};

/// `gen_ai.workflow.name` of an ASR utterance.
pub(crate) const ASR_WORKFLOW: &str = "asr_session";

pub(crate) struct AsrObservation {
    session: ObservabilitySession,
    options: ObservationOptions,
    model: Option<Arc<LoadSubject>>,
    workflow: Mutex<Option<WorkflowOperation>>,
    reference: Mutex<Option<ObservationRef>>,
    decoder: Mutex<Option<ObservationRef>>,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl AsrObservation {
    /// Captures the parent and session on the caller's thread. No span yet.
    pub(crate) fn new(options: Option<&ObservationOptions>) -> Arc<Self> {
        Self::for_model(options, None)
    }

    /// [`Self::new`] for a known model, so warmup loads name it.
    pub(crate) fn for_model(
        options: Option<&ObservationOptions>,
        model: Option<(&str, &ModelMetadata)>,
    ) -> Arc<Self> {
        let session = resolve_session(options);
        let mut options = options.cloned().unwrap_or_default();
        options.trace = option_trace(Some(&options));
        options.parent = options.parent.or_else(ambient_parent);
        options.conversation_id = conversation_id(Some(&options));
        Arc::new(Self {
            session,
            options,
            model: model.map(|(id, metadata)| {
                LoadSubject::new(
                    id,
                    RunSubject::from_metadata(id, metadata).format,
                    "warmup",
                    declared_model_size(metadata),
                )
            }),
            workflow: Mutex::new(None),
            reference: Mutex::new(None),
            decoder: Mutex::new(None),
        })
    }

    /// Starts the utterance workflow unless one is open.
    fn ensure_started(&self) {
        let mut workflow = lock(&self.workflow);
        if workflow.is_some() {
            return;
        }
        let mut attributes = self.options.attributes.clone();
        if let Some(id) = &self.options.conversation_id {
            attributes.push(KeyValue::new(semconv::attr::CONVERSATION_ID, id.clone()));
        }
        let op = self.session.observer().start_workflow(
            ASR_WORKFLOW,
            StartOptions {
                parent: self.options.parent.clone(),
                trace: self.options.trace,
                attributes,
                ..StartOptions::default()
            },
        );
        *lock(&self.reference) = op.observation_ref();
        *lock(&self.decoder) = None;
        *workflow = Some(op);
    }

    /// Ends the open utterance as cancelled (`reset`); the next feed starts a
    /// new workflow.
    pub(crate) fn reset(&self) {
        if let Some(op) = lock(&self.workflow).take() {
            op.cancel();
        }
        *lock(&self.reference) = None;
        *lock(&self.decoder) = None;
    }

    fn scope(&self) -> RecorderScope {
        let workflow = lock(&self.workflow);
        let (parent, suppressed) = match workflow.as_ref() {
            Some(op) => (op.context(), op.context().is_none()),
            None => (
                self.options.parent.clone(),
                self.options.trace == TraceMode::Suppress,
            ),
        };
        RecorderScope::install(Arc::new(
            StageRecorder::for_session(self.session.clone(), parent, suppressed)
                .with_conversation_id(self.options.conversation_id.clone()),
        ))
    }

    /// Installs the utterance's context for a `feed`, starting the workflow.
    pub(crate) fn enter_feed(&self) -> RecorderScope {
        self.ensure_started();
        self.scope()
    }

    /// Installs the open utterance's context (else the captured parent)
    /// without starting one: `flush` with nothing fed creates no workflow.
    pub(crate) fn enter(&self) -> RecorderScope {
        self.scope()
    }

    /// A warmup span under the open utterance or the captured parent, whose
    /// weight loads are `load_model` operations with trigger `warmup`.
    pub(crate) fn warmup_span(&self) -> (RecorderScope, SpanScope) {
        let scope = self.scope();
        let load = self.model.clone();
        let span = SpanScope::start_loading("warmup", Vec::new(), load);
        (scope, span)
    }

    /// Reference of the last decoder execution that produced a transcript.
    pub(crate) fn decoder_reference(&self) -> Option<ObservationRef> {
        lock(&self.decoder).clone()
    }

    pub(crate) fn reference(&self) -> Option<ObservationRef> {
        lock(&self.reference).clone()
    }

    /// Ends the utterance with the flush outcome. Repeated flushes return the
    /// cached reference; a flush with nothing fed returns `None`.
    pub(crate) fn finish<T>(&self, result: &SdkResult<T>) -> Option<ObservationRef> {
        if let Some(op) = lock(&self.workflow).take() {
            match result {
                Ok(_) => op.end(),
                Err(e) => op.end_with_error(&error_info(e)),
            }
        }
        self.reference()
    }
}

struct DecodeScope<'a> {
    observation: RunObservation,
    resources: xybrid_core::device::RunGuard,
    owner: &'a AsrObservation,
}

impl DecoderObserver for AsrObservation {
    fn start(
        &self,
        metadata: &ModelMetadata,
        input: &Envelope,
    ) -> Box<dyn DecoderObservation + '_> {
        let options = crate::RunOptions::default().with_observation(ObservationOptions {
            session: Some(self.session.clone()),
            trace: self.options.trace,
            conversation_id: self.options.conversation_id.clone(),
            ..ObservationOptions::default()
        });
        let observation = RunObservation::start(RunStart {
            subject: RunSubject::from_metadata(&metadata.model_id, metadata),
            input: Some(input),
            history: &[],
            options: Some(&options),
            generation: None,
            streaming: false,
            location: Location::Local,
            parent: None,
            provider: None,
            server: None,
        });
        Box::new(DecodeScope {
            observation,
            resources: crate::telemetry::begin_resource_run(),
            owner: self,
        })
    }
}

impl DecoderObservation for DecodeScope<'_> {
    fn finish(self: Box<Self>, result: &StreamResult<Envelope>) {
        let this = *self;
        this.observation.resource(this.resources.finish().as_ref());
        match result {
            Ok(output) => {
                let reference = this.observation.succeed(output);
                // Only a decode that produced transcript state replaces the
                // reference partials carry.
                if reference.is_some() {
                    *lock(&this.owner.decoder) = reference;
                }
            }
            Err(error) => {
                this.observation
                    .fail(&SdkError::inference(error.to_string()));
            }
        }
    }
}
