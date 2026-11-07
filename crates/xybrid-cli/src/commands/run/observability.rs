//! Observe the CLI paths that execute core directly rather than through SDK models.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use xybrid_core::execution::{ExecutionTemplate, ModelMetadata};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::tracing::{self, RecorderScope, TraceHeaders, TraceRecorder};
use xybrid_sdk::observability::{
    observer, ErrorInfo, InferenceRequest, InputMessage, KeyValue, Location, MediaMetadata,
    Modality, ObservationContext, Observer, OutputMessage, Part, SpanOperation, StartOptions,
    TraceMode, Usage,
};

fn parent() -> Option<ObservationContext> {
    let headers = tracing::current_trace_headers()?;
    ObservationContext::from_traceparent(&headers.traceparent, headers.tracestate.as_deref()).ok()
}

/// The closure keeps its original return value and error; telemetry never replaces either.
pub(super) fn inference<E>(
    metadata: &ModelMetadata,
    input: &Envelope,
    execute: impl FnOnce() -> Result<Envelope, E>,
) -> Result<Envelope, E> {
    let observer = observer();
    if !observer.is_enabled() {
        return execute();
    }
    let started = observer.now();
    let operation = observer.start_inference(
        request(metadata, input),
        StartOptions::default().with_parent(parent()),
    );
    if let Some(system) = input.metadata.get("system_prompt") {
        operation.record_system_instructions(&[Part::text(system)]);
    }
    operation.record_input_messages(&[InputMessage {
        role: "user".into(),
        parts: parts(input),
        name: None,
    }]);
    let recorder = Arc::new(StageRecorder::new(observer, operation.context()));
    let scope = RecorderScope::install(recorder.clone());
    let result = execute();
    drop(scope);
    match &result {
        Ok(output) => {
            let mut harvest = recorder.harvest();
            // Only measured fields are harvested; arbitrary envelope metadata
            // never becomes span attributes.
            for (key, value) in &output.metadata {
                if let Some(value) = stage_value(key, value) {
                    harvest.insert(key.clone(), value);
                }
            }
            let token = |keys: &[&str]| {
                keys.iter()
                    .find_map(|key| {
                        output
                            .metadata
                            .get(*key)
                            .and_then(|v| v.parse::<u64>().ok())
                    })
                    .or_else(|| {
                        keys.iter()
                            .find_map(|key| harvest.get(*key).and_then(|v| v.parse::<u64>().ok()))
                    })
            };
            let usage = Usage {
                input_tokens: token(&["prompt_tokens", "tokens_in"]),
                output_tokens: token(&["tokens_generated", "completion_tokens", "tokens_out"]),
                cache_read_input_tokens: token(&[
                    "prompt_cached_tokens",
                    "cache_read_input_tokens",
                ]),
                ..Usage::default()
            };
            operation.record_usage(&usage);
            let finish_reason = harvest.get("finish_reason").map(String::as_str);
            if let Some(reason) = finish_reason {
                operation.record_finish_reasons([reason]);
            }
            let mut output_parts = parts(output);
            if let Some(reasoning) = output.metadata.get("reasoning_content") {
                output_parts.insert(
                    0,
                    Part::Reasoning {
                        content: reasoning.clone(),
                    },
                );
            }
            operation.record_output_messages(&[OutputMessage {
                role: "assistant".into(),
                parts: output_parts,
                name: None,
                finish_reason: finish_reason.map(str::to_owned),
            }]);
            if let Some(at) = harvest
                .get("ttft_ms")
                .and_then(|value| value.parse::<f64>().ok())
                .and_then(|ms| Duration::try_from_secs_f64(ms / 1000.0).ok())
                .and_then(|duration| started.checked_add(duration))
            {
                operation.record_first_token_at(at);
            }
            if let Some(seconds) = harvest
                .get("mean_itl_ms")
                .and_then(|value| value.parse::<f64>().ok())
                .map(|ms| ms / 1000.0)
                .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            {
                operation.record_token_timing(seconds);
            }
            operation.end();
        }
        // Diagnostics may contain prompts, signed URLs or local paths. Keep the
        // original error for the caller, and export only its low-cardinality class.
        Err(_) => operation.end_with_error(&ErrorInfo::new("inference_error")),
    }
    result
}

pub(super) fn workflow<T, E>(name: &str, execute: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let observer = observer();
    if !observer.is_enabled() {
        return execute();
    }
    let operation = observer.start_workflow(name, StartOptions::default().with_parent(parent()));
    let scope = RecorderScope::install(Arc::new(StageRecorder::new(observer, operation.context())));
    let result = execute();
    drop(scope);
    match &result {
        Ok(_) => operation.end(),
        Err(_) => operation.end_with_error(&ErrorInfo::new("pipeline_error")),
    }
    result
}

fn runtime(label: &str) -> &str {
    match label {
        "ort" | "onnx" => "onnxruntime",
        "llamacpp" | "llama.cpp" => "llama.cpp",
        "mistralrs" | "mistral" => "mistral.rs",
        "whispercpp" => "whisper.cpp",
        "cloud" | "gateway" => "http",
        other => other,
    }
}

fn request(metadata: &ModelMetadata, input: &Envelope) -> InferenceRequest {
    let task = metadata
        .metadata
        .get("task")
        .and_then(|value| value.as_str());
    let task_lower = task.unwrap_or_default().to_ascii_lowercase();
    let operation = match &metadata.execution_template {
        ExecutionTemplate::Gguf { .. } | ExecutionTemplate::VisionLanguage { .. } => "chat",
        ExecutionTemplate::GgmlWhisper { .. }
        | ExecutionTemplate::ZzzEmbed { .. }
        | ExecutionTemplate::ZzzPhonon { .. } => "generate_content",
        _ if task_lower.contains("embed") => "embeddings",
        _ if task_lower.contains("generation") || task_lower.contains("chat") => "chat",
        _ if task_lower.contains("speech")
            || task_lower.contains("tts")
            || task_lower.contains("asr")
            || task_lower.contains("transcri") =>
        {
            "generate_content"
        }
        _ => "predict",
    };
    let backend = xybrid_core::execution::template::backend_label_from_template(
        &metadata.execution_template,
        metadata
            .metadata
            .get("backend")
            .and_then(|value| value.as_str()),
    );
    let mut request = InferenceRequest::new(operation, backend.map(runtime).unwrap_or("xybrid"))
        .with_model(&metadata.model_id)
        .with_location(Location::Local);
    request.runtime = backend.map(|label| runtime(label).to_owned());
    request.task = task.map(str::to_owned);
    request.max_tokens = input
        .metadata
        .get("max_tokens")
        .and_then(|value| value.parse().ok());
    request.input_modalities = modalities(input);
    request.output_modalities = match &metadata.execution_template {
        ExecutionTemplate::ZzzEmbed { .. } | ExecutionTemplate::ZzzPhonon { .. } => {
            vec![Modality::Audio]
        }
        _ if task_lower.contains("tts")
            || (task_lower.starts_with("text") && task_lower.contains("speech")) =>
        {
            vec![Modality::Audio]
        }
        _ if operation == "chat" || operation == "generate_content" => vec![Modality::Text],
        _ => Vec::new(),
    };
    request
}

fn modalities(envelope: &Envelope) -> Vec<Modality> {
    match &envelope.kind {
        EnvelopeKind::Text(_) | EnvelopeKind::ChoiceRequest(_) => vec![Modality::Text],
        EnvelopeKind::Audio(_) => vec![Modality::Audio],
        EnvelopeKind::Image { .. } => vec![Modality::Image],
        EnvelopeKind::MultiPart(items) => {
            let mut result = Vec::new();
            for modality in items.iter().flat_map(modalities) {
                if !result.contains(&modality) {
                    result.push(modality);
                }
            }
            result
        }
        EnvelopeKind::Embedding(_) | EnvelopeKind::ChoiceScores(_) => Vec::new(),
    }
}

/// Content always goes through portable operation methods and their capture policy.
/// Describe media without serializing envelopes, raw bytes, paths or arbitrary metadata.
fn parts(envelope: &Envelope) -> Vec<Part> {
    match &envelope.kind {
        EnvelopeKind::Text(text) => vec![Part::text(text)],
        EnvelopeKind::Audio(bytes) => vec![Part::Media(MediaMetadata {
            modality: Some(Modality::Audio),
            size_bytes: Some(bytes.len() as u64),
            ..MediaMetadata::default()
        })],
        EnvelopeKind::Image { .. } => vec![Part::Media(MediaMetadata {
            modality: Some(Modality::Image),
            size_bytes: Some(envelope.payload_size() as u64),
            width: envelope
                .image_dimensions()
                .map(|dimensions| dimensions.width),
            height: envelope
                .image_dimensions()
                .map(|dimensions| dimensions.height),
            ..MediaMetadata::default()
        })],
        EnvelopeKind::MultiPart(items) => items.iter().flat_map(parts).collect(),
        EnvelopeKind::ChoiceRequest(request) => {
            let mut result = parts(request.context());
            if let Ok(choices) = serde_json::to_string(request.choices()) {
                result.push(Part::text(choices));
            }
            result
        }
        EnvelopeKind::ChoiceScores(scores) => serde_json::to_string(scores)
            .ok()
            .map(Part::text)
            .into_iter()
            .collect(),
        EnvelopeKind::Embedding(_) => Vec::new(),
    }
}

struct StageRecorder {
    observer: Observer,
    root: Option<ObservationContext>,
    next: AtomicU64,
    spans: Mutex<HashMap<u64, SpanOperation>>,
    metadata: Mutex<HashMap<String, String>>,
}

// Core metadata is untrusted. Keep this boundary aligned with the SDK stage
// recorder; explicit public observation APIs still support custom identities.
fn stage_value(key: &str, value: &str) -> Option<String> {
    let known = match key {
        "backend" => matches!(
            value,
            "ort"
                | "onnx"
                | "llamacpp"
                | "llama.cpp"
                | "mistralrs"
                | "mistral"
                | "whispercpp"
                | "candle"
                | "zzz"
                | "coreml"
                | "cloud"
                | "gateway"
        ),
        "execution_provider" => matches!(
            value,
            "cpu"
                | "metal"
                | "cuda"
                | "vulkan"
                | "coreml"
                | "directml"
                | "nnapi"
                | "webgpu"
                | "wasm"
                | "tensorrt"
                | "rocm"
        ),
        "provider" => matches!(
            value,
            "openai"
                | "anthropic"
                | "google"
                | "groq"
                | "together"
                | "fireworks"
                | "mistral"
                | "deepseek"
                | "openrouter"
                | "elevenlabs"
                | "custom"
                | "xybrid"
        ),
        "stage_kind" => matches!(
            value,
            "asr" | "tts" | "llm" | "translate" | "vision" | "embed" | "audio" | "decision"
        ),
        "span_kind" => matches!(value, "cpu" | "gpu" | "io" | "tool"),
        "task" => matches!(
            value,
            "speech-recognition"
                | "speech-to-text"
                | "asr"
                | "text-to-speech"
                | "tts"
                | "text-generation"
                | "text_generation"
                | "chat"
                | "llm"
                | "translation"
                | "translate"
                | "image-classification"
                | "image-to-text"
                | "vision"
                | "vision-language"
                | "vision_language"
                | "vlm"
                | "embedding"
                | "sentence-embedding"
                | "audio-classification"
                | "vad"
                | "decision"
        ),
        "quantization" => {
            let normalized = value.to_ascii_lowercase();
            return matches!(
                normalized.as_str(),
                "f16"
                    | "f32"
                    | "fp16"
                    | "fp32"
                    | "bf16"
                    | "int8"
                    | "q2_k"
                    | "q2_k_s"
                    | "q3_k"
                    | "q3_k_s"
                    | "q3_k_m"
                    | "q3_k_l"
                    | "q4_0"
                    | "q4_1"
                    | "q4_k"
                    | "q4_k_s"
                    | "q4_k_m"
                    | "q5_0"
                    | "q5_1"
                    | "q5_k"
                    | "q5_k_s"
                    | "q5_k_m"
                    | "q6_k"
                    | "q8_0"
                    | "q8_1"
                    | "q8_k"
                    | "iq1_s"
                    | "iq1_m"
                    | "iq2_xxs"
                    | "iq2_xs"
                    | "iq2_s"
                    | "iq3_xxs"
                    | "iq3_s"
                    | "iq4_xs"
                    | "iq4_nl"
            )
            .then_some(normalized);
        }
        "tokens_in"
        | "tokens_out"
        | "prompt_tokens"
        | "completion_tokens"
        | "tokens_generated"
        | "prompt_cached_tokens"
        | "cache_read_input_tokens"
        | "emitted_chunks"
        | "choice_count"
        | "image_count"
        | "placeholder_tokens"
        | "raw_image_source_vec_allocations"
        | "raw_image_source_extra_pixel_copies"
        | "raw_image_conversion_buffer_allocations"
        | "raw_image_pixel_bytes" => {
            return value.parse::<u64>().ok().map(|n| n.to_string());
        }
        "ttft_ms"
        | "mean_itl_ms"
        | "p95_itl_ms"
        | "generation_time_ms"
        | "image_preprocess_ms"
        | "tokens_per_second"
        | "decode_tps"
        | "prefill_tps" => {
            return value
                .parse::<f64>()
                .ok()
                .filter(|n| n.is_finite() && *n >= 0.0)
                .map(|n| n.to_string());
        }
        "finish_reason" => {
            return Some(
                match value {
                    "stop" | "eos" | "end_turn" | "stop_sequence" => "stop",
                    "length" | "max_tokens" => "length",
                    "tool_calls" | "tool_call" | "tool_use" => "tool_call",
                    "content_filter" => "content_filter",
                    "error" => "error",
                    _ => "_OTHER",
                }
                .to_owned(),
            );
        }
        _ => false,
    };
    known.then(|| value.to_owned())
}

impl StageRecorder {
    fn new(observer: Observer, root: Option<ObservationContext>) -> Self {
        Self {
            observer,
            root,
            next: AtomicU64::new(1),
            spans: Mutex::new(HashMap::new()),
            metadata: Mutex::new(HashMap::new()),
        }
    }

    fn spans(&self) -> MutexGuard<'_, HashMap<u64, SpanOperation>> {
        self.spans
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn harvest(&self) -> HashMap<String, String> {
        self.metadata
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl TraceRecorder for StageRecorder {
    fn start_span(&self, name: &str, parent: Option<u64>) -> u64 {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let parent = parent
            .and_then(|id| self.spans().get(&id).and_then(SpanOperation::context))
            .or_else(|| self.root.clone());
        let options = parent.map_or_else(
            || StartOptions::default().with_trace(TraceMode::Suppress),
            StartOptions::child_of,
        );
        let span = self.observer.start_span(name, Vec::new(), options);
        self.spans().insert(id, span);
        id
    }

    fn end_span(&self, id: u64) {
        if let Some(span) = self.spans().remove(&id) {
            span.end();
        }
    }

    fn add_metadata(&self, id: u64, key: &str, value: &str) {
        // Never attach an unfiltered stage snapshot: core can include local model
        // paths, raw error strings and caller-owned payloads in its metadata.
        let Some(value) = stage_value(key, value) else {
            return;
        };
        self.metadata
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key.to_owned(), value.clone());
        if let Some(span) = self.spans().get(&id) {
            span.set_attribute(KeyValue::new(format!("xybrid.stage.{key}"), value));
        }
    }

    fn trace_headers(&self, id: Option<u64>) -> Option<TraceHeaders> {
        let context = id
            .and_then(|id| self.spans().get(&id).and_then(SpanOperation::context))
            .or_else(|| self.root.clone())?;
        Some(TraceHeaders {
            traceparent: context.to_traceparent(),
            tracestate: context.tracestate().map(str::to_owned),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::time::Instant;
    use xybrid_sdk::observability::{
        self, BatchLimits, ContentPolicy, ObservabilityConfig, OtlpConfig, Protocol,
        SignalEndpoint, SubstringRedactor,
    };

    fn metadata() -> ModelMetadata {
        serde_json::from_value(serde_json::json!({
            "model_id": "cli-observation-test", "version": "1.0",
            "execution_template": { "type": "Gguf", "model_file": "private-model-path.gguf" },
            "files": [],
        }))
        .unwrap()
    }

    /// One bounded local request; the long batch delay lets shutdown export all
    /// test spans together, without sleeps or a dependency on a collector service.
    fn collector() -> (
        OtlpConfig,
        mpsc::Receiver<String>,
        std::thread::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let (tx, rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "no OTLP export received");
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("collector accept: {error}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            let mut chunk = [0_u8; 4096];
            let (start, length) = loop {
                let n = stream.read(&mut chunk).unwrap();
                assert_ne!(n, 0, "collector received an incomplete request");
                request.extend_from_slice(&chunk[..n]);
                if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&request[..end]);
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .expect("OTLP HTTP export has a content length");
                    if request.len() >= end + 4 + length {
                        break (end + 4, length);
                    }
                }
            };
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
            tx.send(String::from_utf8(request[start..start + length].to_vec()).unwrap())
                .unwrap();
        });
        let config = OtlpConfig::default()
            .with_traces(Some(
                SignalEndpoint::new(format!("http://{address}/v1/traces"))
                    .with_protocol(Protocol::HttpJson)
                    .with_timeout(Duration::from_secs(5)),
            ))
            .with_batch(BatchLimits {
                scheduled_delay: Duration::from_secs(60),
                ..BatchLimits::default()
            });
        (config, rx, handle)
    }

    fn spans(body: &serde_json::Value) -> Vec<&serde_json::Value> {
        body["resourceSpans"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r["scopeSpans"].as_array().unwrap())
            .flat_map(|s| s["spans"].as_array().unwrap())
            .collect()
    }

    fn attribute<'a>(span: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
        span["attributes"]
            .as_array()?
            .iter()
            .find(|a| a["key"] == key)
            .map(|a| &a["value"])
    }

    struct Shutdown;
    impl Drop for Shutdown {
        fn drop(&mut self) {
            let _ = observability::shutdown_observability(Duration::from_secs(5));
        }
    }

    // Keep the two global SDK configurations in one test. Other CLI unit tests
    // parse configuration only and never install a process observer.
    #[test]
    fn direct_core_observability_exports_parented_spans_and_respects_content_policy() {
        let _cleanup = Shutdown;
        for capture in [true, false] {
            let (config, exports, collector) = collector();
            let policy = if capture {
                ContentPolicy::default().with_redactor(SubstringRedactor::new(["redact-me"]))
            } else {
                ContentPolicy::metadata_only()
            };
            observability::init_observability(
                ObservabilityConfig::otlp(config)
                    .content(policy)
                    .device_attributes(false),
            )
            .unwrap();
            let metadata = metadata();
            let mut input = Envelope::new(EnvelopeKind::MultiPart(vec![
                Envelope::new(EnvelopeKind::Text("prompt-private redact-me".into())),
                Envelope::new(EnvelopeKind::Audio(b"audio-private".to_vec())),
            ]));
            input
                .metadata
                .insert("system_prompt".into(), "system-private redact-me".into());
            input
                .metadata
                .insert("untrusted".into(), "envelope-private".into());
            let mut output = Envelope::new(EnvelopeKind::Text("output-private redact-me".into()));
            output.metadata.insert(
                "reasoning_content".into(),
                "reasoning-private redact-me".into(),
            );
            output
                .metadata
                .insert("completion_tokens".into(), "4".into());
            output
                .metadata
                .insert("untrusted".into(), "envelope-private".into());
            for key in [
                "backend",
                "execution_provider",
                "provider",
                "quantization",
                "task",
            ] {
                output
                    .metadata
                    .insert(key.into(), "descriptor-private".into());
            }
            output
                .metadata
                .insert("finish_reason".into(), "finish-private".into());

            let local = tracing::capture();
            let previous = tracing::current_recorder().unwrap();
            let mut headers = None;
            let mut workflow_headers = None;
            let result: Result<Envelope, &str> = workflow("cli-test", || {
                let _stage = tracing::SpanGuard::new("pipeline-stage");
                workflow_headers = tracing::current_trace_headers();
                let result = inference(&metadata, &input, || {
                    let _execute = tracing::SpanGuard::new("core-execute");
                    let _nested = tracing::SpanGuard::new("core-decode");
                    headers = tracing::current_trace_headers();
                    tracing::add_metadata("prompt_tokens", "8");
                    tracing::add_metadata("tokens_generated", "3");
                    // Invalid later metadata cannot replace known measurements.
                    tracing::add_metadata("prompt_tokens", "invalid-metric-private");
                    tracing::add_metadata("finish_reason", "max_tokens");
                    tracing::add_metadata("ttft_ms", "2.5");
                    tracing::add_metadata("mean_itl_ms", "1");
                    for (key, value) in [
                        ("backend", "llamacpp"),
                        ("execution_provider", "cpu"),
                        ("provider", "openai"),
                        ("quantization", "Q4_K_M"),
                        ("stage_kind", "llm"),
                        ("task", "chat"),
                        ("p95_itl_ms", "4"),
                        ("emitted_chunks", "3"),
                        ("choice_count", "0005"),
                    ] {
                        tracing::add_metadata(key, value);
                    }
                    for key in [
                        "backend",
                        "execution_provider",
                        "provider",
                        "quantization",
                        "stage_kind",
                        "task",
                    ] {
                        tracing::add_metadata(key, "descriptor-private");
                    }
                    tracing::add_metadata("ttft_ms", "NaN");
                    tracing::add_metadata("choice_count", "choice-count-private");
                    tracing::add_metadata("choice_count", "-1");
                    tracing::add_metadata("choice_count", "18446744073709551616");
                    tracing::add_metadata("mean_itl_ms", "-1");
                    tracing::add_metadata("model_id", "/private/model-id");
                    tracing::add_metadata("version", "version-private");
                    tracing::add_metadata("model", "/private/model-path.gguf");
                    tracing::add_metadata("error", "stage-error-private");
                    tracing::add_metadata("stages", "stage-snapshot-private");
                    Ok(output.clone())
                });
                assert_eq!(tracing::current_trace_headers(), workflow_headers);
                result
            });
            assert_eq!(result.unwrap(), output);
            assert!(Arc::ptr_eq(
                &previous,
                &tracing::current_recorder().unwrap()
            ));

            let failure: Result<Envelope, &str> =
                inference(&metadata, &input, || Err("caller-error-private"));
            assert_eq!(failure.unwrap_err(), "caller-error-private");
            let failed_workflow: Result<(), &str> =
                workflow("failed-pipeline", || Err("pipeline-error-private"));
            assert_eq!(failed_workflow.unwrap_err(), "pipeline-error-private");

            // Exercise run.rs's actual TemplateExecutor boundary too. A missing
            // local model must still yield the original CLI error and an error span.
            let empty_models = tempfile::tempdir().unwrap();
            assert!(
                super::super::run_inference(empty_models.path(), &metadata, &input, false).is_err()
            );
            // A command failure also has its own scoped workflow; replacing the
            // retired global pipeline context must preserve the caller's error.
            let missing_bundle = empty_models.path().join("missing-bundle-private.xyb");
            let error = super::super::run_bundle(
                &missing_bundle,
                None,
                Some("prompt-private"),
                &[],
                &[],
                None,
                None,
                false,
                false,
                false,
                None,
                None,
            )
            .unwrap_err();
            assert_eq!(
                error.to_string(),
                format!("Bundle file not found: {}", missing_bundle.display())
            );
            assert!(Arc::ptr_eq(
                &previous,
                &tracing::current_recorder().unwrap()
            ));
            {
                let _stage = tracing::SpanGuard::new("after-observed-run");
            }
            let local_spans = local.stages_json();
            assert_eq!(local_spans["spans"].as_array().unwrap().len(), 1);
            assert_eq!(local_spans["spans"][0]["name"], "after-observed-run");
            drop(local);

            observability::shutdown_observability(Duration::from_secs(5)).unwrap();
            let raw = exports.recv_timeout(Duration::from_secs(5)).unwrap();
            collector.join().unwrap();
            let body: serde_json::Value = serde_json::from_str(&raw).unwrap();
            let spans = spans(&body);
            assert_eq!(
                spans
                    .iter()
                    .filter(|s| attribute(s, "gen_ai.operation.name")
                        .is_some_and(|v| v["stringValue"] == "chat"))
                    .count(),
                3,
                "the real run_inference boundary must also produce a logical inference span"
            );
            let decode = spans.iter().find(|s| s["name"] == "core-decode").unwrap();
            let execute = spans.iter().find(|s| s["name"] == "core-execute").unwrap();
            assert_eq!(decode["parentSpanId"], execute["spanId"]);
            for (key, value) in [
                ("backend", "llamacpp"),
                ("execution_provider", "cpu"),
                ("provider", "openai"),
                ("quantization", "q4_k_m"),
                ("stage_kind", "llm"),
                ("task", "chat"),
                ("p95_itl_ms", "4"),
                ("emitted_chunks", "3"),
                ("choice_count", "5"),
                ("ttft_ms", "2.5"),
                ("mean_itl_ms", "1"),
            ] {
                assert_eq!(
                    attribute(decode, &format!("xybrid.stage.{key}")).unwrap()["stringValue"],
                    value
                );
            }
            let inference = spans
                .iter()
                .find(|s| s["spanId"] == execute["parentSpanId"])
                .unwrap();
            assert_eq!(
                attribute(inference, "gen_ai.operation.name").unwrap()["stringValue"],
                "chat"
            );
            assert_eq!(
                attribute(inference, "gen_ai.usage.input_tokens").unwrap()["intValue"],
                "8"
            );
            assert_eq!(
                attribute(inference, "gen_ai.usage.output_tokens").unwrap()["intValue"],
                "4"
            );
            assert_eq!(
                attribute(inference, "gen_ai.response.finish_reasons").unwrap()["arrayValue"]
                    ["values"][0]["stringValue"],
                "_OTHER"
            );
            let pipeline_stage = spans
                .iter()
                .find(|s| s["name"] == "pipeline-stage")
                .unwrap();
            assert_eq!(inference["parentSpanId"], pipeline_stage["spanId"]);
            let workflow = spans
                .iter()
                .find(|s| s["spanId"] == pipeline_stage["parentSpanId"])
                .unwrap();
            assert_eq!(
                attribute(workflow, "gen_ai.operation.name").unwrap()["stringValue"],
                "invoke_workflow"
            );
            let headers = headers.unwrap();
            let cx = ObservationContext::from_traceparent(&headers.traceparent, None).unwrap();
            assert_eq!(
                cx.to_traceparent(),
                format!(
                    "00-{}-{}-01",
                    decode["traceId"].as_str().unwrap(),
                    decode["spanId"].as_str().unwrap()
                )
            );
            assert_eq!(
                spans
                    .iter()
                    .filter(|s| attribute(s, "error.type")
                        .is_some_and(|v| v["stringValue"] == "inference_error"))
                    .count(),
                2
            );
            assert!(spans.iter().any(|s| attribute(s, "error.type")
                .is_some_and(|v| v["stringValue"] == "pipeline_error")));
            let bundle_workflow = spans
                .iter()
                .find(|s| {
                    attribute(s, "gen_ai.workflow.name")
                        .is_some_and(|v| v["stringValue"] == "bundle_execution")
                })
                .unwrap();
            assert_eq!(
                attribute(bundle_workflow, "gen_ai.operation.name").unwrap()["stringValue"],
                "invoke_workflow"
            );
            assert_eq!(
                attribute(bundle_workflow, "error.type").unwrap()["stringValue"],
                "pipeline_error"
            );
            for content in [
                "prompt-private",
                "system-private",
                "output-private",
                "reasoning-private",
            ] {
                assert_eq!(
                    raw.contains(content),
                    capture,
                    "capture policy for {content}"
                );
            }
            for excluded in [
                "redact-me",
                "audio-private",
                "envelope-private",
                "invalid-metric-private",
                "stage-error-private",
                "stage-snapshot-private",
                "caller-error-private",
                "pipeline-error-private",
                "private-model-path",
                "/private/model-path",
                "missing-bundle-private",
                "descriptor-private",
                "finish-private",
                "/private/model-id",
                "version-private",
            ] {
                assert!(
                    !raw.contains(excluded),
                    "unfiltered content exported: {excluded}"
                );
            }
        }
        // With export disabled, wrappers preserve the existing core recorder.
        let local = tracing::capture();
        let result: Result<Envelope, ()> = inference(
            &metadata(),
            &Envelope::new(EnvelopeKind::Text("offline".into())),
            || {
                let _stage = tracing::SpanGuard::new("offline-stage");
                Ok(Envelope::new(EnvelopeKind::Text("offline-output".into())))
            },
        );
        assert!(result.is_ok());
        assert_eq!(local.stages_json()["spans"][0]["name"], "offline-stage");
    }
}
