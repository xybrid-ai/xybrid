//! Stamps LLM cost-attribution and generation metrics onto spans and envelopes.
//!
//! The executor's four LLM execution paths (`execute_llm`, `execute_llm_streaming`,
//! `execute_llm_with_messages`, `execute_llm_streaming_with_messages`) call
//! `adapter.backend().generate(...)` directly, bypassing
//! [`crate::runtime_adapter::llm::LlmRuntimeAdapter::execute`] where the telemetry
//! extraction originally lived. These helpers re-apply that wire contract at the
//! executor layer so every LLM span surfaces the same cost-attribution fields and
//! the 9 generation scalars the consuming analytics backend expects.

use super::template::{
    backend_label_from_template, quantization_label_from_metadata, ModelMetadata,
};
use crate::runtime_adapter::llm::LlmRuntimeAdapter;
use crate::tracing as xybrid_trace;
use std::collections::HashMap;

/// Stamp cost-attribution metadata (`backend`, `quantization`) onto the
/// currently-open span.
///
/// The outer `execute:<model_id>` span set up by `execute_impl` already
/// carries these — but the chat-context entry points
/// (`execute_with_context_impl`, `execute_streaming_with_context_impl`)
/// dispatch directly to the inner `execute_llm*` methods without ever
/// opening that outer span, so the inner LLM spans are the only place the
/// SDK telemetry hoist can read these fields from on a chat-context call.
/// Call this from each inner LLM span site immediately after `SpanGuard`
/// so both the non-context and chat-context flows produce the same wire
/// shape.
pub(crate) fn stamp_llm_span_cost_attribution(metadata: &ModelMetadata) {
    // Reuse the same resolver the outer `execute:<model_id>` span uses
    // so both spans agree on the canonical wire label — including the
    // GGUF-defaults-to-llamacpp behaviour that lights up unannotated
    // bundles in the registry.
    let backend_hint = metadata.metadata.get("backend").and_then(|v| v.as_str());
    if let Some(label) = backend_label_from_template(&metadata.execution_template, backend_hint) {
        xybrid_trace::add_metadata("backend", label);
    }
    if let Some(quant) = quantization_label_from_metadata(metadata) {
        xybrid_trace::add_metadata("quantization", quant);
    }
}

/// Overwrite the currently-open span's `backend` cost-attribution
/// metadata with the actual runtime's wire label.
///
/// `stamp_llm_span_cost_attribution` stamps the *template-derived*
/// default first, but the runtime is selected by cargo feature (see
/// `LlmRuntimeAdapter::new` precedence), so the template-derived
/// label can disagree with the runtime that actually executes — for
/// example, an `llm-mistral`-only build loading an unannotated GGUF
/// bundle runs on mistral.rs but the template default says `llamacpp`.
/// Calling this after the adapter is resolved replaces the default
/// with ground truth (via [`LlmRuntimeAdapter::wire_label`]). Spans
/// without a wire label (mock/test backends) leave the
/// template-derived stamp in place.
pub(crate) fn stamp_llm_runtime_backend(adapter: &LlmRuntimeAdapter) {
    if let Some(label) = adapter.wire_label() {
        xybrid_trace::add_metadata("backend", label);
    }
}

/// Insert the streaming-derived LLM metrics into the response envelope metadata.
pub(crate) fn insert_llm_streaming_metrics(
    response_metadata: &mut HashMap<String, String>,
    output: &crate::runtime_adapter::llm::GenerationOutput,
) {
    if let Some(v) = output.ttft_ms {
        response_metadata.insert("ttft_ms".to_string(), v.to_string());
    }
    if let Some(v) = output.mean_itl_ms {
        response_metadata.insert("mean_itl_ms".to_string(), format!("{:.4}", v));
    }
    if let Some(v) = output.p95_itl_ms {
        response_metadata.insert("p95_itl_ms".to_string(), v.to_string());
    }
    if let Some(v) = output.emitted_chunks {
        response_metadata.insert("emitted_chunks".to_string(), v.to_string());
    }
    if let Some(v) = output.decode_tps {
        response_metadata.insert("decode_tps".to_string(), format!("{:.4}", v));
    }
    if let Some(v) = output.prefill_tps {
        response_metadata.insert("prefill_tps".to_string(), format!("{:.4}", v));
    }
}

/// Mirror the LLM generation metrics onto the currently-open span.
pub(crate) fn mirror_llm_metrics_to_span(
    output: &crate::runtime_adapter::llm::GenerationOutput,
    backend_name: &str,
    cached_prefix_tokens: Option<usize>,
) {
    // Always-present scalars. These reach the platform via
    // `PlatformEvent.stages[].spans[].metadata` (populated by
    // `xybrid_core::tracing::add_metadata` on the currently active span).
    xybrid_trace::add_metadata("tokens_generated", output.tokens_generated.to_string());
    // Canonical `tokens_out` key for the analytics backend's span extractor.
    // Equal to `tokens_generated` by construction — mirrors
    // `LlmRuntimeAdapter::execute` so the local LLM paths (which bypass it and
    // call this helper instead) surface the same column the SDK hoist lifts
    // onto the trace dashboard.
    xybrid_trace::add_metadata("tokens_out", output.tokens_generated.to_string());
    xybrid_trace::add_metadata("generation_time_ms", output.generation_time_ms.to_string());
    xybrid_trace::add_metadata(
        "tokens_per_second",
        format!("{:.2}", output.tokens_per_second),
    );
    xybrid_trace::add_metadata("finish_reason", &output.finish_reason);

    // Resolved execution provider: which on-device engine path actually
    // ran. Cost-attribution telemetry uses this to explain latency
    // variance on the same chip + model. Sourced from build flags via
    // `local_execution_provider` because backend selection is compile-
    // time. Cloud LLMs go through a different adapter and get
    // attribution from the `provider` field instead.
    xybrid_trace::add_metadata(
        "execution_provider",
        crate::runtime_adapter::llm::local_execution_provider(backend_name),
    );

    // Local KV cache hits: how many prompt tokens this call reused from
    // the cache the previous turn left behind. Only emit when positive
    // — `Some(0)` means a first turn or a totally divergent prompt
    // (telemetry should look like a non-cached call), and `None` means
    // the backend doesn't track prefix reuse at all (cloud, mistralrs,
    // mock test backends). The local mirror of cloud's
    // `cache_read_input_tokens` so analytics can stack them on the
    // same axis.
    if let Some(n) = cached_prefix_tokens {
        if n > 0 {
            xybrid_trace::add_metadata("prompt_cached_tokens", n.to_string());
        }
    }

    // Streaming-derived scalars. Only mirror when the backend reported them;
    // the `Option<_>` + `nonzero` filter in mistral keeps misleading zeros
    // out of the dashboard.
    if let Some(v) = output.ttft_ms {
        xybrid_trace::add_metadata("ttft_ms", v.to_string());
    }
    if let Some(v) = output.mean_itl_ms {
        xybrid_trace::add_metadata("mean_itl_ms", format!("{:.4}", v));
    }
    if let Some(v) = output.p95_itl_ms {
        xybrid_trace::add_metadata("p95_itl_ms", v.to_string());
    }
    if let Some(v) = output.emitted_chunks {
        xybrid_trace::add_metadata("emitted_chunks", v.to_string());
    }
    if let Some(v) = output.decode_tps {
        xybrid_trace::add_metadata("decode_tps", format!("{:.4}", v));
    }
    if let Some(v) = output.prefill_tps {
        xybrid_trace::add_metadata("prefill_tps", format!("{:.4}", v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::template::ExecutionTemplate;
    use crate::tracing;
    use std::sync::Mutex;

    // These cases pin the chat-context cost-attribution path: the wire `backend`
    // label that lands on the currently-open span when
    // `execute_with_context_impl` / `execute_streaming_with_context_impl` bypass
    // the outer `execute:<model_id>` span. They share state with the global
    // span collector, so a process-wide mutex serialises them — `cargo test`
    // parallelises within a binary and these would otherwise stomp each other's
    // reset/measure window.
    static GLOBAL_TRACE_LOCK: Mutex<()> = Mutex::new(());

    fn gguf_metadata(backend_hint: Option<&str>) -> ModelMetadata {
        let mut bundle_metadata = HashMap::new();
        if let Some(hint) = backend_hint {
            bundle_metadata.insert("backend".to_string(), serde_json::json!(hint));
        }
        ModelMetadata {
            model_id: "test-gguf".into(),
            version: "1".into(),
            execution_template: ExecutionTemplate::Gguf {
                model_file: "test.gguf".into(),
                chat_template: None,
                context_length: 2048,
                generation_params: None,
            },
            preprocessing: Vec::new(),
            postprocessing: Vec::new(),
            files: Vec::new(),
            vision_encoder: None,
            description: None,
            metadata: bundle_metadata,
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        }
    }

    fn capture_span_metadata(span_name: &str, metadata: &ModelMetadata) -> HashMap<String, String> {
        tracing::init_tracing(true);
        tracing::reset_tracing();
        {
            let _guard = tracing::SpanGuard::new(span_name);
            stamp_llm_span_cost_attribution(metadata);
        }
        let json = tracing::get_stages_json();
        tracing::reset_tracing();

        let span = json["spans"]
            .as_array()
            .and_then(|spans| spans.iter().find(|s| s["name"].as_str() == Some(span_name)))
            .expect("span recorded by SpanGuard must be present in stages json");
        span["metadata"]
            .as_object()
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn unannotated_gguf_stamps_llamacpp_default() {
        let _lock = GLOBAL_TRACE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let captured = capture_span_metadata("execute:test", &gguf_metadata(None));
        assert_eq!(
            captured.get("backend").map(String::as_str),
            Some("llamacpp"),
            "chat-context flow must default unannotated GGUF bundles to llamacpp so PlatformEvent.backend is non-empty"
        );
    }

    #[test]
    fn mistralrs_hint_wins_on_gguf() {
        let _lock = GLOBAL_TRACE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let captured = capture_span_metadata("execute:test", &gguf_metadata(Some("mistralrs")));
        assert_eq!(
            captured.get("backend").map(String::as_str),
            Some("mistralrs")
        );
    }

    #[test]
    fn legacy_mistral_alias_normalises_to_mistralrs() {
        let _lock = GLOBAL_TRACE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let captured = capture_span_metadata("execute:test", &gguf_metadata(Some("mistral")));
        assert_eq!(
            captured.get("backend").map(String::as_str),
            Some("mistralrs"),
            "the legacy `mistral` bundle alias must canonicalise to the wire label"
        );
    }

    #[test]
    fn quantization_stamped_from_gguf_filename() {
        let _lock = GLOBAL_TRACE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut metadata = gguf_metadata(None);
        metadata.execution_template = ExecutionTemplate::Gguf {
            model_file: "tinyllama-1.1b-chat-q4_k_m.gguf".into(),
            chat_template: None,
            context_length: 2048,
            generation_params: None,
        };
        let captured = capture_span_metadata("execute:test", &metadata);
        assert_eq!(
            captured.get("quantization").map(String::as_str),
            Some("q4_k_m"),
            "stamp must surface the filename-inferred quantization alongside backend"
        );
    }

    // A backend stub that lets a test pick the wire label so we can
    // verify `stamp_llm_runtime_backend` overwrites the
    // template-derived default with the runtime's identity. Needed
    // because the runtime is chosen by cargo feature, so the
    // template-derived label can disagree with the runtime that
    // actually executes (e.g. mistralrs on an `llm-mistral`-only
    // build loading an unannotated GGUF bundle).
    struct WireLabelStub(Option<&'static str>);

    impl crate::runtime_adapter::llm::LlmBackend for WireLabelStub {
        fn name(&self) -> &str {
            "wire-label-stub"
        }
        fn wire_label(&self) -> Option<&'static str> {
            self.0
        }
        fn supported_formats(&self) -> Vec<&'static str> {
            vec!["gguf"]
        }
        fn load(
            &mut self,
            _config: &crate::runtime_adapter::llm::LlmConfig,
        ) -> crate::runtime_adapter::llm::LlmResult<()> {
            Ok(())
        }
        fn is_loaded(&self) -> bool {
            true
        }
        fn unload(&mut self) -> crate::runtime_adapter::llm::LlmResult<()> {
            Ok(())
        }
        fn generate(
            &self,
            _messages: &[crate::runtime_adapter::llm::ChatMessage],
            _config: &crate::runtime_adapter::llm::GenerationConfig,
        ) -> crate::runtime_adapter::llm::LlmResult<crate::runtime_adapter::llm::GenerationOutput>
        {
            unreachable!("stub backend should not be invoked for inference in this test")
        }
        fn generate_raw(
            &self,
            _prompt: &str,
            _config: &crate::runtime_adapter::llm::GenerationConfig,
        ) -> crate::runtime_adapter::llm::LlmResult<crate::runtime_adapter::llm::GenerationOutput>
        {
            unreachable!("stub backend should not be invoked for inference in this test")
        }
    }

    fn capture_with_runtime_overwrite(
        metadata: &ModelMetadata,
        wire_label: Option<&'static str>,
    ) -> HashMap<String, String> {
        let adapter = crate::runtime_adapter::llm::LlmRuntimeAdapter::with_backend(Box::new(
            WireLabelStub(wire_label),
        ));
        tracing::init_tracing(true);
        tracing::reset_tracing();
        {
            let _guard = tracing::SpanGuard::new("execute:test");
            stamp_llm_span_cost_attribution(metadata);
            stamp_llm_runtime_backend(&adapter);
        }
        let json = tracing::get_stages_json();
        tracing::reset_tracing();

        let span = json["spans"]
            .as_array()
            .and_then(|spans| {
                spans
                    .iter()
                    .find(|s| s["name"].as_str() == Some("execute:test"))
            })
            .expect("span recorded by SpanGuard must be present in stages json");
        span["metadata"]
            .as_object()
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn runtime_wire_label_overwrites_template_default() {
        let _lock = GLOBAL_TRACE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Template default for unannotated GGUF is `llamacpp`, but
        // the runtime selected by cargo feature is mistral.rs. The
        // overwrite must flip the stamp to ground truth so the
        // dashboard reflects the runtime that actually executed.
        let captured = capture_with_runtime_overwrite(&gguf_metadata(None), Some("mistralrs"));
        assert_eq!(
            captured.get("backend").map(String::as_str),
            Some("mistralrs"),
            "runtime wire label must overwrite the template-derived default"
        );
    }

    #[test]
    fn runtime_overwrite_preserves_template_default_when_label_absent() {
        let _lock = GLOBAL_TRACE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Mock/test backends return None from wire_label so the
        // template-derived stamp survives — anything else would
        // erase ground truth that the template *did* carry.
        let captured = capture_with_runtime_overwrite(&gguf_metadata(None), None);
        assert_eq!(
            captured.get("backend").map(String::as_str),
            Some("llamacpp"),
            "stub backends without a wire label must not erase the template-derived stamp"
        );
    }

    fn sample_output(tokens: usize) -> crate::runtime_adapter::llm::GenerationOutput {
        crate::runtime_adapter::llm::GenerationOutput {
            text: "hi".to_string(),
            tokens_generated: tokens,
            generation_time_ms: 100,
            tokens_per_second: 10.0,
            finish_reason: "stop".to_string(),
            ttft_ms: None,
            mean_itl_ms: None,
            p95_itl_ms: None,
            emitted_chunks: None,
            inter_chunk_ms: Vec::new(),
            decode_tps: None,
            prefill_tps: None,
            image_preprocess_ms: None,
        }
    }

    fn capture_mirror_metadata(
        output: &crate::runtime_adapter::llm::GenerationOutput,
        backend_name: &str,
    ) -> HashMap<String, String> {
        tracing::init_tracing(true);
        tracing::reset_tracing();
        {
            let _guard = tracing::SpanGuard::new("execute:test");
            mirror_llm_metrics_to_span(output, backend_name, None);
        }
        let json = tracing::get_stages_json();
        tracing::reset_tracing();

        let span = json["spans"]
            .as_array()
            .and_then(|spans| {
                spans
                    .iter()
                    .find(|s| s["name"].as_str() == Some("execute:test"))
            })
            .expect("span recorded by SpanGuard must be present in stages json");
        span["metadata"]
            .as_object()
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn mirror_emits_canonical_tokens_out() {
        let _lock = GLOBAL_TRACE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // The local LLM paths bypass `LlmRuntimeAdapter::execute`, so the
        // analytics span extractor's canonical `tokens_out` key must be stamped
        // here too — equal to `tokens_generated` by construction.
        let captured = capture_mirror_metadata(&sample_output(42), "llamacpp");
        assert_eq!(
            captured.get("tokens_generated").map(String::as_str),
            Some("42")
        );
        assert_eq!(
            captured.get("tokens_out").map(String::as_str),
            Some("42"),
            "local LLM telemetry paths must emit the canonical tokens_out span field"
        );
    }
}
