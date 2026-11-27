//! Whole-prompt, CPU-only evaluation and pinned Jinja rendering for decisions.
//!
//! These handles deliberately do not use the causal generation context or its
//! chunked prefill. A renderer owns one immutable AST and creates fresh mutable
//! interpreter state per call. A context exclusively owns each native pass.

use std::ffi::{c_char, c_void, CString};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use llama_cpp_sys::bindings as sys;

use crate::model::ModelHandle;
use crate::{LlamaError, LlamaModel, LlamaResult};

/// Maximum complete input tokens per decision pass, inclusive.
pub const DECISION_CONTEXT_TOKENS: usize = 4096;
/// Maximum bytes exchanged with the pinned renderer.
pub const DECISION_RENDER_BYTES: usize = 8 * 1024 * 1024;
const METADATA_BYTES: usize = 1024 * 1024;
const METADATA_TOTAL_BYTES: usize = 8 * 1024 * 1024;

fn invalid(message: &'static str) -> LlamaError {
    LlamaError::InvalidInput(message.into())
}

fn metadata_budget(used: usize, additional: usize) -> LlamaResult<usize> {
    used.checked_add(additional)
        .filter(|&size| size <= METADATA_TOTAL_BYTES)
        .ok_or_else(|| invalid("decision metadata aggregate byte limit exceeded"))
}

fn read_text(mut read: impl FnMut(*mut c_char, usize) -> i32) -> LlamaResult<Option<String>> {
    let size = read(std::ptr::null_mut(), 0);
    if size == -1 {
        return Ok(None);
    }
    let size = usize::try_from(size).map_err(|_| invalid("decision metadata unavailable"))?;
    if size > METADATA_BYTES {
        return Err(invalid("decision metadata exceeds byte limit"));
    }
    let mut bytes = vec![0u8; size + 1];
    let actual = read(bytes.as_mut_ptr().cast(), bytes.len());
    if usize::try_from(actual).ok() != Some(size) {
        return Err(invalid("decision metadata changed during read"));
    }
    bytes.truncate(size);
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_| invalid("decision metadata is not UTF-8"))
}

/// Whether a GGUF metadata key marks a decision artifact.
///
/// The markers are the named `systemone` template, `systemone.*` settings and
/// upstream's `<architecture>.decision.*` keys. xybrid-core's header sniff
/// (`execution::decision::is_decision_metadata_key`) uses the same rule and a
/// core test compares the two: change them together.
pub fn is_decision_metadata_key(key: &str) -> bool {
    key == "tokenizer.chat_template.systemone"
        || key.starts_with("systemone.")
        || key.contains(".decision.")
}

/// Whether a metadata key holds upstream's decision type,
/// `<architecture>.decision.type`.
pub fn is_decision_type_key(key: &str) -> bool {
    key.ends_with(".decision.type")
}

/// Whether a model of this decision type also generates text.
///
/// The list is `server_model_output_modalities` in the pinned llama.cpp:
/// every decision type serves decisions only, except these, which stay chat
/// models. xybrid-core's header sniff
/// (`execution::decision::decision_type_generates_text`) keeps the same list
/// and a core test compares the two: change them together.
pub fn decision_type_generates_text(decision_type: &str) -> bool {
    decision_type == "granite-switch"
}

/// The chat loader's rule over a model's metadata keys: a decision marker
/// refuses the model, unless its decision type also generates text.
///
/// `value` reads the value of one key, and is called for type keys only.
fn chat_refuses(
    keys: impl IntoIterator<Item = String>,
    mut value: impl FnMut(&str) -> Option<String>,
) -> bool {
    let mut marked = false;
    let mut generates_text = false;
    for key in keys {
        if is_decision_metadata_key(&key) {
            marked = true;
            generates_text |= is_decision_type_key(&key)
                && value(&key).is_some_and(|kind| decision_type_generates_text(&kind));
        }
    }
    marked && !generates_text
}

impl LlamaModel {
    /// Whether this model's metadata mark it as a decision artifact that
    /// cannot generate text.
    ///
    /// Reads every key llama.cpp kept and skips one it cannot read (not
    /// UTF-8, or over the byte limit), so the answer never depends on the
    /// decision admission limits and the scan never fails a load. A marker
    /// key is short ASCII, so skipping cannot hide one. A decision type that
    /// cannot be read counts as one that generates no text.
    pub(crate) fn is_decision_only(&self) -> bool {
        // SAFETY: self retains the live model.
        let count = unsafe { sys::llama_model_metadata_count_c(self.as_ptr()) };
        let keys = (0..count.max(0)).filter_map(|index| {
            read_text(|out, capacity| {
                // SAFETY: index is below the model's metadata count; out is
                // writable for capacity bytes.
                unsafe { sys::llama_model_metadata_key_c(self.as_ptr(), index, out, capacity) }
            })
            .ok()
            .flatten()
        });
        chat_refuses(keys, |key| self.metadata(key).ok().flatten())
    }

    /// Load weights with an explicit empty offload-device list.
    ///
    /// This is the only loader admitted by [`DecisionContext::new`]. Protocol
    /// architecture, calibration and template admission remain caller checks.
    pub fn load_decision_cpu(path: &str) -> LlamaResult<Self> {
        Self::load_decision_inner(path, false)
    }

    fn load_decision_inner(path: &str, vocab_only: bool) -> LlamaResult<Self> {
        crate::backend_init();
        let path = crate::ffi::cstring(path, "model path")?;
        // SAFETY: path is NUL terminated and live for this synchronous call.
        let ptr = unsafe { sys::llama_load_decision_model_c(path.as_ptr(), vocab_only) };
        if ptr.is_null() {
            return Err(invalid("decision model load failed"));
        }
        Ok(Self::from_decision_ptr(ptr, !vocab_only))
    }

    /// Read a metadata value with upstream formatting, without an f32 round trip.
    pub fn metadata(&self, key: &str) -> LlamaResult<Option<String>> {
        let key = CString::new(key).map_err(|_| invalid("invalid metadata key"))?;
        read_text(|out, capacity| {
            // SAFETY: self retains the model; output is writable for capacity.
            unsafe { sys::llama_model_metadata_c(self.as_ptr(), key.as_ptr(), out, capacity) }
        })
    }

    /// Enumerate metadata keys without allocating the values.
    pub fn metadata_keys(&self) -> LlamaResult<Vec<String>> {
        // SAFETY: self retains the live model.
        let count = unsafe { sys::llama_model_metadata_count_c(self.as_ptr()) };
        if !(0..=65536).contains(&count) {
            return Err(invalid("decision metadata count exceeds limit"));
        }
        let mut keys = Vec::with_capacity(count as usize);
        let mut total_bytes = 0usize;
        for index in 0..count {
            let key = read_text(|out, capacity| {
                // SAFETY: index is bounded by the model's metadata count.
                unsafe { sys::llama_model_metadata_key_c(self.as_ptr(), index, out, capacity) }
            })?
            .ok_or_else(|| invalid("decision metadata key unavailable"))?;
            total_bytes = metadata_budget(total_bytes, key.len())?;
            keys.push(key);
        }
        Ok(keys)
    }

    /// Enumerate metadata keys and their upstream-formatted values in order.
    pub fn metadata_entries(&self) -> LlamaResult<Vec<(String, String)>> {
        let keys = self.metadata_keys()?;
        let mut total_bytes = keys.iter().map(String::len).sum();
        keys.into_iter()
            .map(|key| {
                let value = self
                    .metadata(&key)?
                    .ok_or_else(|| invalid("decision metadata value unavailable"))?;
                total_bytes = metadata_budget(total_bytes, value.len())?;
                Ok((key, value))
            })
            .collect()
    }

    /// Copy an explicitly named template; never fall back to the default chat template.
    pub fn named_chat_template(&self, name: &str) -> LlamaResult<Option<String>> {
        let name = CString::new(name).map_err(|_| invalid("invalid template name"))?;
        read_text(|out, capacity| {
            // SAFETY: model and name are live, out is writable for capacity.
            unsafe {
                sys::llama_model_named_template_c(self.as_ptr(), name.as_ptr(), out, capacity)
            }
        })
    }

    /// Copy the vocabulary spelling of a token, including special spellings.
    pub fn token_text(&self, token: i32) -> LlamaResult<String> {
        read_text(|out, capacity| {
            // SAFETY: native wrapper checks token before indexing the vocabulary.
            unsafe { sys::llama_vocab_text_c(self.as_ptr(), token, out, capacity) }
        })?
        .ok_or_else(|| invalid("decision token spelling unavailable"))
    }

    /// Whether a token carries CONTROL, independently of USER_DEFINED.
    pub fn token_is_control(&self, token: i32) -> bool {
        // SAFETY: wrapper checks token range; model lives through the call.
        unsafe { sys::llama_vocab_is_control_c(self.as_ptr(), token) }
    }
}

/// Metadata and tokenizer only; cannot create an inference context.
pub struct VocabularyModel {
    model: LlamaModel,
}

impl std::fmt::Debug for VocabularyModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VocabularyModel").finish_non_exhaustive()
    }
}

impl VocabularyModel {
    /// Load a metadata-only GGUF without allocating model weights.
    pub fn load(path: &str) -> LlamaResult<Self> {
        Ok(Self {
            model: LlamaModel::load_decision_inner(path, true)?,
        })
    }
    /// Read upstream-formatted metadata.
    pub fn metadata(&self, key: &str) -> LlamaResult<Option<String>> {
        self.model.metadata(key)
    }
    /// Enumerate upstream-formatted metadata.
    pub fn metadata_entries(&self) -> LlamaResult<Vec<(String, String)>> {
        self.model.metadata_entries()
    }
    /// Read the explicitly named template.
    pub fn named_chat_template(&self, name: &str) -> LlamaResult<Option<String>> {
        self.model.named_chat_template(name)
    }
    /// Copy a vocabulary spelling.
    pub fn token_text(&self, token: i32) -> LlamaResult<String> {
        self.model.token_text(token)
    }
    /// Test the CONTROL attribute independently of USER_DEFINED.
    pub fn token_is_control(&self, token: i32) -> bool {
        self.model.token_is_control(token)
    }
    /// Number of vocabulary tokens.
    pub fn n_vocab(&self) -> usize {
        self.model.n_vocab()
    }
    /// Tokenize length-delimited UTF-8 without parsing special spellings.
    pub fn tokenize(&self, text: &str, add_special: bool) -> LlamaResult<Vec<i32>> {
        self.model.tokenize(text, add_special)
    }
    /// Tokenize length-delimited UTF-8 with special spelling recognition.
    pub fn tokenize_special(&self, text: &str, add_special: bool) -> LlamaResult<Vec<i32>> {
        self.model.tokenize_special(text, add_special)
    }
}

/// A compiled pinned-upstream Jinja template, with no shared interpreter state.
pub struct DecisionRenderer {
    ptr: NonNull<c_void>,
}

impl std::fmt::Debug for DecisionRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionRenderer").finish_non_exhaustive()
    }
}

impl DecisionRenderer {
    /// Compile an audited template once. The protocol must admit its hash first.
    pub fn new(template: &str) -> LlamaResult<Self> {
        if template.len() > 65536 {
            return Err(invalid("decision template exceeds byte limit"));
        }
        // SAFETY: source is explicit-length UTF-8, copied by the constructor.
        let ptr =
            unsafe { sys::llama_decision_renderer_new_c(template.as_ptr().cast(), template.len()) };
        Ok(Self {
            ptr: NonNull::new(ptr)
                .ok_or_else(|| invalid("decision template compilation failed"))?,
        })
    }

    /// Render ordered JSON with upstream `global_from_json(..., false)` semantics.
    ///
    /// The protocol must check its conservative expansion bound before this
    /// call. This cap also checks the final output before crossing into Rust.
    pub fn render(&mut self, ordered_json: &str, max_output_bytes: usize) -> LlamaResult<String> {
        if ordered_json.len() > DECISION_RENDER_BYTES || max_output_bytes > DECISION_RENDER_BYTES {
            return Err(invalid("decision renderer byte limit exceeded"));
        }
        let mut output = std::ptr::null_mut();
        let mut output_len = 0;
        // SAFETY: all buffers and output pointers live through the call; &mut
        // self prevents concurrent interpreter invocations on this handle.
        let status = unsafe {
            sys::llama_decision_renderer_render_c(
                self.ptr.as_ptr(),
                ordered_json.as_ptr().cast(),
                ordered_json.len(),
                max_output_bytes,
                &mut output,
                &mut output_len,
            )
        };
        if status != 0 {
            return Err(invalid(if status == -3 {
                "decision rendered prompt exceeds byte limit"
            } else {
                "decision template rendering failed"
            }));
        }
        let Some(output) = NonNull::new(output) else {
            return Err(invalid("decision renderer returned no buffer"));
        };
        // SAFETY: successful native render allocated exactly output_len + 1
        // bytes, transfers ownership, and does not retain the pointer.
        let bytes = unsafe { std::slice::from_raw_parts(output.as_ptr().cast::<u8>(), output_len) };
        let result = std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| invalid("decision rendered prompt is not UTF-8"));
        // SAFETY: this is the sole owner of the returned native allocation.
        unsafe { sys::llama_decision_rendered_free_c(output.as_ptr()) };
        result
    }
}
impl Drop for DecisionRenderer {
    fn drop(&mut self) {
        // SAFETY: sole owner of the handle, no active render during destruction.
        unsafe { sys::llama_decision_renderer_free_c(self.ptr.as_ptr()) };
    }
}
// SAFETY: the AST has no thread affinity; mutable interpreter state is per call.
// It is deliberately !Sync, so sharing requires a Mutex.
unsafe impl Send for DecisionRenderer {}

/// Effective native settings observed after context creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionContextSettings {
    /// Total context length.
    pub n_ctx: u32,
    /// Per-sequence context length.
    pub n_ctx_seq: u32,
    /// Logical batch size.
    pub n_batch: u32,
    /// Physical batch size.
    pub n_ubatch: u32,
    /// Maximum sequences.
    pub n_seq_max: u32,
    /// Generation thread count.
    pub n_threads: i32,
    /// Batch thread count.
    pub n_threads_batch: i32,
    /// Number of model offload devices.
    pub n_devices: i32,
    /// Whether dense attention is causal.
    pub causal_attn: bool,
    /// Whether the context retains sequence memory.
    pub has_memory: bool,
}

/// A dedicated full-attention context that evaluates one complete prompt at a time.
pub struct DecisionContext {
    ptr: Option<NonNull<c_void>>,
    model: Arc<ModelHandle>,
    n_vocab: usize,
}

impl std::fmt::Debug for DecisionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionContext")
            .field("n_ctx", &DECISION_CONTEXT_TOKENS)
            .finish_non_exhaustive()
    }
}

impl DecisionContext {
    /// Create the fixed 4096-token, four-thread CPU reference configuration.
    ///
    /// Rejects causal/memory-bearing models and every effective batch/context
    /// mismatch instead of forcing chat models into full-attention execution.
    pub fn new(model: &LlamaModel) -> LlamaResult<Self> {
        if !model.decision_cpu {
            return Err(invalid(
                "decision context requires an explicit CPU-only model",
            ));
        }
        Ok(Self {
            ptr: Some(Self::create(model.handle())?),
            model: Arc::clone(model.handle()),
            n_vocab: model.n_vocab(),
        })
    }

    fn create(model: &ModelHandle) -> LlamaResult<NonNull<c_void>> {
        // SAFETY: shared model owner remains live until after context deletion.
        NonNull::new(unsafe { sys::llama_decision_context_new_c(model.as_ptr()) })
            .ok_or_else(|| invalid("decision native configuration is unsupported"))
    }

    /// Inspect actual native settings rather than the requested configuration.
    pub fn settings(&self) -> LlamaResult<DecisionContextSettings> {
        let ptr = self
            .ptr
            .ok_or_else(|| invalid("decision context awaits recreation"))?;
        let mut native = std::mem::MaybeUninit::<sys::llama_decision_settings_c>::uninit();
        // SAFETY: context lives for the call; true initializes every field.
        if !unsafe { sys::llama_decision_context_settings_c(ptr.as_ptr(), native.as_mut_ptr()) } {
            return Err(invalid("decision context settings unavailable"));
        }
        // SAFETY: the successful native call initialized every field above.
        let native = unsafe { native.assume_init() };
        Ok(DecisionContextSettings {
            n_ctx: native.n_ctx,
            n_ctx_seq: native.n_ctx_seq,
            n_batch: native.n_batch,
            n_ubatch: native.n_ubatch,
            n_seq_max: native.n_seq_max,
            n_threads: native.n_threads,
            n_threads_batch: native.n_threads_batch,
            n_devices: native.n_devices,
            causal_attn: native.causal_attn,
            has_memory: native.has_memory,
        })
    }

    /// Evaluate exactly one whole prompt and copy only requested label logits.
    ///
    /// Cancellation checks run before compute and in llama.cpp's CPU abort
    /// callback. Native failure/abort replaces context state before reuse.
    pub fn evaluate(
        &mut self,
        tokens: &[i32],
        labels: &[i32],
        abort: Option<&AtomicBool>,
    ) -> LlamaResult<Vec<f32>> {
        if abort.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            return Err(LlamaError::DecisionCancelled);
        }
        validate_pass(tokens, labels, self.n_vocab)?;
        let ptr = match self.ptr {
            Some(ptr) => ptr,
            None => {
                let ptr = Self::create(&self.model)?;
                self.ptr = Some(ptr);
                ptr
            }
        };
        let mut selected = vec![0.0; labels.len()];
        // SAFETY: all lengths/IDs validated, self owns exclusive context access,
        // atomic is shared for the synchronous duration; C clears its callback
        // and synchronizes before returning, retaining no borrowed pointer.
        let status = unsafe {
            sys::llama_decision_evaluate_c(
                ptr.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                labels.as_ptr(),
                labels.len() as i32,
                selected.as_mut_ptr(),
                abort.map(|_| abort_requested as unsafe extern "C" fn(*mut c_void) -> bool),
                abort.map_or(std::ptr::null_mut(), |flag| {
                    (flag as *const AtomicBool).cast_mut().cast()
                }),
            )
        };
        if status != 0 {
            // Never reuse a failed context. Free before recreating so recovery
            // cannot double peak context memory. A failed later allocation
            // leaves None and bars compute until successful recreation.
            self.ptr = None;
            // SAFETY: native call synchronized and detached its callback.
            unsafe { sys::llama_free_c(ptr.as_ptr()) };
            return Err(if status == -3 {
                LlamaError::DecisionCancelled
            } else {
                invalid("decision evaluation failed")
            });
        }
        Ok(selected)
    }
}

unsafe extern "C" fn abort_requested(data: *mut c_void) -> bool {
    // SAFETY: installed only with a live &AtomicBool for the synchronous call.
    unsafe { &*data.cast::<AtomicBool>() }.load(Ordering::Acquire)
}

fn validate_pass(tokens: &[i32], labels: &[i32], n_vocab: usize) -> LlamaResult<()> {
    if tokens.is_empty() || tokens.len() > DECISION_CONTEXT_TOKENS {
        return Err(invalid("decision prompt token limit exceeded"));
    }
    if labels.is_empty() || labels.len() > n_vocab || labels.len() > i32::MAX as usize {
        return Err(invalid("decision labels are invalid"));
    }
    if tokens
        .iter()
        .chain(labels)
        .any(|&token| usize::try_from(token).map_or(true, |id| id >= n_vocab))
    {
        return Err(invalid("decision token is outside the vocabulary"));
    }
    Ok(())
}
impl Drop for DecisionContext {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr.take() {
            // SAFETY: sole owner, no active evaluation; model drops afterwards.
            unsafe { sys::llama_free_c(ptr.as_ptr()) };
        }
    }
}
// SAFETY: llama.cpp context handoff is allowed, concurrent use is not. &mut
// evaluate and !Sync enforce exclusive access, including the abort callback.
unsafe impl Send for DecisionContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_markers_are_the_template_its_settings_and_upstream_keys() {
        for key in [
            "tokenizer.chat_template.systemone",
            "systemone.temperature",
            "lfm3.decision.type",
            "lfm3.decision.temperature.choice.2",
        ] {
            assert!(is_decision_metadata_key(key), "{key}");
        }
        for key in [
            "general.architecture",
            "tokenizer.chat_template",
            "tokenizer.chat_template.tool_use",
            "lfm3.context_length",
            "general.decision",
            "systemone",
        ] {
            assert!(!is_decision_metadata_key(key), "{key}");
        }
    }

    #[test]
    fn a_model_without_metadata_is_not_a_decision_artifact() {
        // The stub has no native model: the scan reads nothing and cannot fail.
        assert!(!LlamaModel::test_stub().is_decision_only());
    }

    #[test]
    fn only_granite_switch_is_a_decision_type_that_generates_text() {
        assert!(decision_type_generates_text("granite-switch"));
        for kind in ["lfm3-d1", "lfm3-d1-omni", "lfm2-d1", "laya", "clef", ""] {
            assert!(!decision_type_generates_text(kind), "{kind}");
        }
        assert!(is_decision_type_key("graniteswitch.decision.type"));
        assert!(!is_decision_type_key("lfm3.decision.temperature.choice"));
    }

    #[test]
    fn the_chat_loader_refuses_a_marked_model_that_generates_no_text() {
        let refuses = |entries: &[(&str, &str)]| {
            chat_refuses(entries.iter().map(|(key, _)| key.to_string()), |key| {
                let (_, value) = entries.iter().find(|(name, _)| *name == key)?;
                Some(value.to_string())
            })
        };
        let template = ("tokenizer.chat_template.systemone", "{{ questions }}");

        // A chat model has no marker.
        assert!(!refuses(&[
            ("general.architecture", "lfm3"),
            ("tokenizer.chat_template", "{{ messages }}"),
        ]));
        // d1 answers decisions only.
        assert!(refuses(&[
            ("general.architecture", "lfm3"),
            ("lfm3.decision.type", "lfm3-d1"),
            template,
        ]));
        // A marker with no type says nothing about text.
        assert!(refuses(&[template]));
        assert!(refuses(&[("systemone.temperature", "1")]));
        // Granite Switch answers decisions and generates text, whatever the
        // order of its keys: it stays a chat model.
        let granite = [
            ("general.architecture", "graniteswitch"),
            ("graniteswitch.decision.type", "granite-switch"),
            template,
        ];
        assert!(!refuses(&granite));
        assert!(!refuses(&[granite[2], granite[1], granite[0]]));
    }

    #[test]
    fn a_decision_type_that_cannot_be_read_generates_no_text() {
        let keys = ["graniteswitch.decision.type".to_string()];
        assert!(chat_refuses(keys, |_| None));
    }

    #[test]
    fn metadata_aggregate_budget_is_inclusive_and_overflow_checked() {
        assert_eq!(
            metadata_budget(METADATA_TOTAL_BYTES - 1, 1).unwrap(),
            METADATA_TOTAL_BYTES
        );
        assert!(metadata_budget(METADATA_TOTAL_BYTES, 1).is_err());
        assert!(metadata_budget(1, usize::MAX).is_err());
    }

    #[test]
    fn full_context_is_accepted_without_a_next_token_slot() {
        assert!(validate_pass(&[1; 4096], &[2], 3).is_ok());
        assert!(validate_pass(&[1; 4097], &[2], 3).is_err());
        assert!(validate_pass(&[], &[2], 3).is_err());
        assert!(validate_pass(&[1], &[], 3).is_err());
        assert!(validate_pass(&[-1], &[2], 3).is_err());
        assert!(validate_pass(&[1], &[3], 3).is_err());
    }

    #[test]
    fn pinned_renderer_preserves_order_number_identity_and_nul() {
        let mut renderer = DecisionRenderer::new("{{ state | tojson }}").unwrap();
        let output = renderer
            .render(
                r#"{"state":{"z":1,"a":1.0,"s":-0.0,"n":9007199254740993}}"#,
                4096,
            )
            .unwrap();
        assert_eq!(
            output,
            "{\"z\": 1, \"a\": 1.0, \"s\": -0.0, \"n\": 9007199254740993}"
        );
        let mut renderer = DecisionRenderer::new("{{ state }}").unwrap();
        assert_eq!(
            renderer
                .render(r#"{"state":"a\u0000\u200bb"}"#, 4096)
                .unwrap(),
            "a\0\u{200b}b"
        );
    }

    #[test]
    fn renderer_errors_do_not_include_input_or_source() {
        assert!(DecisionRenderer::new("{% private_canary").is_err());
        let mut renderer = DecisionRenderer::new("{{ state }}").unwrap();
        let error = renderer.render(r#"{"private_canary"#, 64).unwrap_err();
        assert!(!error.to_string().contains("private_canary"));
        assert!(renderer.render(r#"{"state":"secret"}"#, 2).is_err());
        assert_eq!(renderer.render(r#"{"state":"ok"}"#, 2).unwrap(), "ok");
    }

    #[test]
    fn chat_model_handles_cannot_create_decision_contexts() {
        assert!(DecisionContext::new(&LlamaModel::test_stub()).is_err());
    }
}
