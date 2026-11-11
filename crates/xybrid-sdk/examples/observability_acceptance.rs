//! Step 6 staging acceptance through the Rust SDK: the shared staging
//! scenario (`tools/scripts/observability_staging/README.md`) plus the cases
//! only the Rust API can express. `run_staging.py` writes the config (it
//! holds the staging key: read, deleted at once, never printed) and checks
//! what staging stored against the expectations this writes.
//!
//! ```bash
//! cargo build --release -p xybrid-sdk --features platform-macos,dev-tools \
//!     --example observability_acceptance
//! target/release/examples/observability_acceptance --config CONFIG --expectations OUT
//! ```
//!
//! Sessions, each its own `init_observability` .. `shutdown_observability`:
//!
//! - `direct`, `collector`: the shared scenario. The stopped stream is the
//!   token callback returning an error after three chunks.
//! - `case-pipeline`: a streaming LLM pipeline whose `on_token` fails.
//! - `case-tts`: streaming TTS to completion, then stopped by the chunk
//!   callback after one chunk.
//! - `case-asr`: batch ASR of a WAV file, then the same audio fed to a live
//!   stream and flushed.
//! - `case-frames`: one frame session of several runs with a long live trace
//!   interval: one traced run, every run measured.
//! - `case-fallback`: a local stream aborted for cloud fallback by a staged
//!   memory-pressure signal, finished by the staging gateway.
//!
//! Every case session exports to the Collector. A failing case is recorded
//! in its own session's `outcome`; only a failure outside the cases sets
//! `scenario_error`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Map, Value};
use xybrid_core::device::{MemoryPressure, ResourceSnapshot};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::orchestrator::authority::test_seams::StagedResourceProvider;
use xybrid_core::runtime_adapter::types::{PartialToken, StreamingError};
use xybrid_core::runtime_adapter::CloudRuntimeAdapter;
use xybrid_observability::otlp::{EnvironmentSource, OtlpConfig};
use xybrid_sdk::observability::{
    self, ContentPolicy, Evaluation, Feedback, KeyValue, ObservabilityConfig, ObservationContext,
    ObservationOptions, ObservationRef, PipelineStats, RedactionRule, SignalStats,
};
use xybrid_sdk::run_options::{AbortPolicy, AbortSignal, RunOptions};
use xybrid_sdk::{GenerationConfig, InferenceResult, ModelLoader, StreamConfig, Xybrid};

type Failure = Box<dyn std::error::Error + Send + Sync>;
type Outcome<T> = Result<T, Failure>;

const SCHEMA: &str = "xybrid-staging-scenario/1";
const STANDARD_SESSIONS: [&str; 2] = ["direct", "collector"];
const ASR_MODEL: &str = "whisper-tiny-ggml";
const TTS_MODEL: &str = "kitten-tts-nano-0.8";
const FRAMES: usize = 5;
const FRAME_INTERVAL: Duration = Duration::from_secs(600);
const CLOUD_MODEL: &str = "gpt-4o-mini";

struct Args {
    config: PathBuf,
    expectations: PathBuf,
    cases: Vec<String>,
    audio: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut config = std::env::var_os("XYBRID_STAGING_CONFIG").map(PathBuf::from);
    let mut expectations = std::env::var_os("XYBRID_STAGING_EXPECTATIONS").map(PathBuf::from);
    let mut cases = None;
    let mut audio = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../integration-tests/fixtures/input/jfk.wav");
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--config" => config = Some(PathBuf::from(value()?)),
            "--expectations" => expectations = Some(PathBuf::from(value()?)),
            "--cases" => cases = Some(value()?),
            "--audio" => audio = PathBuf::from(value()?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        config: config.ok_or("--config is required")?,
        expectations: expectations.ok_or("--expectations is required")?,
        cases: cases
            .map(|list| list.split(',').map(str::to_owned).collect())
            .unwrap_or_default(),
        audio,
    })
}

/// The decoded `config.json` (README, config.json).
struct Config {
    raw: Value,
}

impl Config {
    fn str(&self, pointer: &str) -> Outcome<String> {
        self.raw
            .pointer(pointer)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| format!("config has no {pointer}").into())
    }

    fn num(&self, pointer: &str) -> Outcome<f64> {
        self.raw
            .pointer(pointer)
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("config has no {pointer}").into())
    }

    fn template(&self, key: &str, session: &str) -> Outcome<String> {
        Ok(self
            .str(&format!("/constants/per_lane/{key}"))?
            .replace("{binding}", &self.str("/binding")?)
            .replace("{lane}", &self.str("/lane")?)
            .replace("{run}", &self.str("/run")?)
            .replace("{session}", session))
    }

    fn sessions(&self) -> Vec<String> {
        self.raw
            .pointer("/sessions")
            .and_then(Value::as_object)
            .map(|map| map.keys().cloned().collect())
            .unwrap_or_default()
    }
}

fn reference(value: Option<&ObservationRef>) -> Value {
    value.map_or(
        Value::Null,
        |r| json!({"trace_id": r.trace_id, "span_id": r.span_id}),
    )
}

fn signal(s: &SignalStats) -> Value {
    json!({"accepted": s.accepted, "exported": s.exported, "dropped": s.dropped,
           "failed": s.failed, "pending": s.pending})
}

fn snapshot() -> Value {
    let stats: Option<PipelineStats> = observability::stats();
    match stats {
        Some(s) => json!({"enabled": observability::is_enabled(), "traces": signal(&s.traces),
                          "logs": signal(&s.logs), "metrics": signal(&s.metrics)}),
        None => json!({"enabled": observability::is_enabled()}),
    }
}

fn outcome(result: Result<(), xybrid_sdk::SdkError>) -> Value {
    Value::from(match result {
        Ok(()) => "ok".to_owned(),
        Err(error) => format!("error: {error}"),
    })
}

/// `error: <message>: <source>: ...`, the whole source chain.
fn error_text(error: &(dyn std::error::Error + 'static)) -> String {
    let mut text = format!("error: {error}");
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(&format!(": {cause}"));
        source = cause.source();
    }
    text
}

/// Step 1: export, identity and content rules of `session`, then init.
fn configure(config: &Config, session: &str) -> Outcome<()> {
    let values = format!("/sessions/{session}");
    let mut otlp = if session == "direct" {
        OtlpConfig::xybrid_traces_to(
            &config.str(&format!("{values}/traces_endpoint"))?,
            &config.str(&format!("{values}/api_key"))?,
        )
    } else {
        OtlpConfig::collector(&config.str(&format!("{values}/endpoint"))?)
    }
    .with_environment(EnvironmentSource::Disabled);
    for (key, value) in [
        ("run", config.str("/run")?),
        ("lane", config.str("/lane")?),
        ("commit", config.str("/commit")?),
        ("session", session.to_owned()),
    ] {
        otlp =
            otlp.with_resource_attribute(KeyValue::new(format!("xybrid.acceptance.{key}"), value));
    }
    let content = if session == "direct" {
        ContentPolicy::metadata_only()
    } else {
        ContentPolicy::default().with_rule(RedactionRule::replace_literal(
            config.str("/constants/secret")?,
            config.str("/constants/replacement")?,
        )?)
    };
    let mut setup = ObservabilityConfig::otlp(otlp)
        .content(content)
        .device_id(config.template("device_id", session)?)
        .service(
            config.template("service_name", session)?,
            Some(config.str("/constants/per_lane/service_version")?),
        );
    if session == "case-frames" {
        setup = setup.live_trace_interval(FRAME_INTERVAL);
    }
    observability::init_observability(setup)?;
    Ok(())
}

/// Step 2: the workflow continuing the driver's trace T under span P.
fn workflow(
    config: &Config,
    session: &str,
) -> Outcome<(
    observability::ManualObservation<observability::WorkflowOperation>,
    ObservationContext,
)> {
    let traceparent = format!(
        "00-{}-{}-01",
        config.str(&format!("/sessions/{session}/trace_id"))?,
        config.str(&format!("/sessions/{session}/parent_span_id"))?
    );
    let controls = ObservationOptions::from_traceparent(&traceparent, None)?;
    let workflow =
        observability::start_workflow(&config.str("/constants/workflow_name")?, &controls);
    let context = workflow
        .context()
        .ok_or("the workflow has no context: observability is not enabled")?;
    Ok((workflow, context))
}

fn child(context: &ObservationContext, conversation: Option<&str>) -> ObservationOptions {
    let options = ObservationOptions::default().with_parent(context.clone());
    match conversation {
        Some(id) => options.with_conversation_id(id),
        None => options,
    }
}

fn generation(max_tokens: f64) -> GenerationConfig {
    GenerationConfig {
        max_tokens: max_tokens as usize,
        ..GenerationConfig::default()
    }
}

fn text(prompt: String) -> Envelope {
    Envelope::new(EnvelopeKind::Text(prompt))
}

/// Ends a session (step 7): flush, stats, shutdown, stats.
fn finish(record: &mut Map<String, Value>, timeout: Duration) {
    let mut lifecycle = Map::new();
    lifecycle.insert(
        "flush".into(),
        outcome(observability::flush_observability(timeout)),
    );
    let after_flush = snapshot();
    lifecycle.insert(
        "shutdown".into(),
        outcome(observability::shutdown_observability(timeout)),
    );
    record.insert("lifecycle".into(), Value::Object(lifecycle));
    record.insert(
        "stats".into(),
        json!({"after_flush": after_flush, "after_shutdown": snapshot()}),
    );
}

fn begin(config: &Config, session: &str, record: &mut Map<String, Value>) -> Outcome<()> {
    record.insert(
        "trace_id".into(),
        config.str(&format!("/sessions/{session}/trace_id"))?.into(),
    );
    record.insert(
        "parent_span_id".into(),
        config
            .str(&format!("/sessions/{session}/parent_span_id"))?
            .into(),
    );
    let init = configure(config, session);
    record.insert(
        "init".into(),
        Value::from(
            init.as_ref()
                .map_or_else(|e| error_text(&**e), |()| "ok".to_owned()),
        ),
    );
    init
}

// ---------------------------------------------------------------------------
// The shared scenario (direct, collector)
// ---------------------------------------------------------------------------

fn standard(config: &Config, session: &str, record: &mut Map<String, Value>) -> Outcome<()> {
    begin(config, session, record)?;
    let conversation = config.template("conversation_id", session)?;
    let (workflow, context) = workflow(config, session)?;
    let mut references = Map::new();
    references.insert(
        "workflow".into(),
        reference(workflow.observation_ref().as_ref()),
    );

    // Step 3: a real registry model, loaded under the workflow.
    let model = ModelLoader::from_registry(&config.str("/model_id")?)
        .observation(child(&context, Some(&conversation)))
        .load()?;
    references.insert("load".into(), reference(model.load_observation_ref()));
    let options = |max_tokens: f64| {
        RunOptions::new()
            .with_generation_config(generation(max_tokens))
            .with_observation(child(&context, Some(&conversation)))
    };

    // Step 4: the token callback refuses the fourth chunk; the run ends
    // cancelled and counts exactly the three it accepted.
    let stop_after = config.num("/constants/stopped/stop_after_chunks")? as usize;
    let mut accepted = 0usize;
    let stopped = model.run_streaming_with_options(
        &text(config.str("/constants/stopped/prompt")?),
        &options(config.num("/constants/stopped/max_tokens")?),
        |_token: PartialToken| -> Result<(), StreamingError> {
            if accepted >= stop_after {
                return Err("the staging scenario stops after three chunks".into());
            }
            accepted += 1;
            Ok(())
        },
    );
    record.insert(
        "stopped".into(),
        json!({"outcome": if stopped.is_err() { "cancelled" } else { "completed" },
               "chunks_consumed": accepted, "delivery": "callback"}),
    );

    // Step 5: streamed to the end; the result carries the run reference.
    let mut chunks = 0usize;
    let mut finish_reason: Option<String> = None;
    let result: InferenceResult = model.run_streaming_with_options(
        &text(config.str("/constants/completed/prompt")?),
        &options(config.num("/constants/completed/max_tokens")?),
        |token: PartialToken| -> Result<(), StreamingError> {
            chunks += 1;
            if token.finish_reason.is_some() {
                finish_reason = token.finish_reason;
            }
            Ok(())
        },
    )?;
    record.insert(
        "completed".into(),
        json!({"outcome": "ok", "chunks": chunks, "tokens_out": result.metrics().tokens_out,
               "text_chars": result.text().map_or(0, |t| t.chars().count()),
               "finish_reason": finish_reason}),
    );
    let run = result
        .observation_ref()
        .cloned()
        .ok_or("the completed run returned no observation reference")?;
    references.insert("run".into(), reference(Some(&run)));

    // Step 6: quality results on the run reference.
    let feedback = Feedback {
        score: Some(config.num("/constants/feedback/score")?),
        label: Some(config.str("/constants/feedback/label")?),
        comment: Some(config.str("/constants/feedback/comment")?),
    };
    let evaluation = Evaluation {
        name: config.str("/constants/evaluation/name")?,
        score_value: Some(config.num("/constants/evaluation/score_value")?),
        score_label: Some(config.str("/constants/evaluation/score_label")?),
        explanation: Some(config.str("/constants/evaluation/explanation")?),
        response_id: None,
    };
    record.insert(
        "quality".into(),
        json!({"feedback": observability::record_feedback(&run, &feedback),
               "evaluation": observability::record_evaluation(&run, &evaluation)}),
    );
    record.insert("references".into(), Value::Object(references));

    // Step 7: release, complete, then the lifecycle.
    drop(model);
    workflow.end();
    Ok(())
}

// ---------------------------------------------------------------------------
// Rust-only cases
// ---------------------------------------------------------------------------

/// A streaming LLM pipeline whose `on_token` fails after two chunks.
fn case_pipeline(
    config: &Config,
    context: &ObservationContext,
    record: &mut Map<String, Value>,
) -> Outcome<()> {
    let yaml = format!(
        "name: staging-pipeline\nstages:\n  - model: {}\n    target: device\n",
        config.str("/model_id")?
    );
    let mut accepted = 0usize;
    let options = RunOptions::new()
        .with_generation_config(generation(config.num("/constants/stopped/max_tokens")?))
        .with_observation(child(context, None));
    let result = Xybrid::run_pipeline_streaming_with_options(
        &yaml,
        &text(config.str("/constants/stopped/prompt")?),
        &options,
        Box::new(|_token: PartialToken| -> Result<(), StreamingError> {
            if accepted >= 2 {
                return Err("the staging pipeline consumer failed".into());
            }
            accepted += 1;
            Ok(())
        }),
    );
    record.insert(
        "pipeline".into(),
        json!({"outcome": result.as_ref().map_or_else(|e| error_text(e), |_| "ok".to_owned()),
               "chunks_consumed": accepted, "delivery": "callback"}),
    );
    Ok(())
}

/// Streaming TTS to completion, then stopped after one chunk.
fn case_tts(
    context: &ObservationContext,
    record: &mut Map<String, Value>,
    references: &mut Map<String, Value>,
) -> Outcome<()> {
    let model = ModelLoader::from_registry(TTS_MODEL)
        .observation(child(context, None))
        .load()?;
    references.insert("load".into(), reference(model.load_observation_ref()));
    let options = RunOptions::new().with_observation(child(context, None));
    let speech =
        text("The lighthouse keeper said good evening to the sea. The sea answered with a long, slow wave. \
              Then the gulls began to call across the harbour. Night came down over the quiet town."
            .into());
    let mut chunks = 0u32;
    let completed = model.run_tts_streaming(&speech, &options, |_chunk| {
        chunks += 1;
        true
    })?;
    references.insert(
        "completed".into(),
        reference(completed.observation.as_ref()),
    );
    record.insert(
        "completed".into(),
        json!({"outcome": format!("{:?}", completed.summary.status).to_lowercase(), "chunks": chunks,
               "summary_chunks": completed.summary.chunks}),
    );
    let mut delivered = 0u32;
    let stopped = model.run_tts_streaming(&speech, &options, |_chunk| {
        delivered += 1;
        false
    });
    let (status, observation) = match &stopped {
        Ok(result) => (
            format!("{:?}", result.summary.status).to_lowercase(),
            result.observation.clone(),
        ),
        Err(error) => (error_text(error), None),
    };
    references.insert("stopped".into(), reference(observation.as_ref()));
    record.insert(
        "stopped".into(),
        json!({"outcome": status, "chunks_consumed": delivered, "delivery": "callback"}),
    );
    Ok(())
}

/// 16-bit PCM WAV to mono f32 samples, and its sample rate.
fn wav_samples(bytes: &[u8]) -> Outcome<(Vec<f32>, u32)> {
    let le16 = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
    let le32 =
        |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let (mut at, mut channels, mut rate, mut bits) = (12usize, 0u16, 0u32, 0u16);
    while at + 8 <= bytes.len() {
        let size = le32(at + 4) as usize;
        let body = at + 8;
        match &bytes[at..at + 4] {
            b"fmt " => {
                channels = le16(body + 2);
                rate = le32(body + 4);
                bits = le16(body + 14);
            }
            b"data" => {
                if bits != 16 || channels == 0 {
                    return Err(format!("unsupported WAV: {bits}-bit, {channels} channels").into());
                }
                let end = (body + size).min(bytes.len());
                let frame = 2 * usize::from(channels);
                let samples = bytes[body..end]
                    .chunks_exact(frame)
                    .map(|f| f32::from(i16::from_le_bytes([f[0], f[1]])) / 32768.0)
                    .collect();
                return Ok((samples, rate));
            }
            _ => {}
        }
        at = body + size + (size & 1);
    }
    Err("no data chunk".into())
}

/// Batch ASR of the WAV file, then the same audio through a live stream.
fn case_asr(
    audio: &Path,
    context: &ObservationContext,
    record: &mut Map<String, Value>,
    references: &mut Map<String, Value>,
) -> Outcome<()> {
    let model = ModelLoader::from_registry(ASR_MODEL)
        .observation(child(context, None))
        .load()?;
    references.insert("load".into(), reference(model.load_observation_ref()));
    let bytes = std::fs::read(audio)?;
    let batch = model.run_with_options(
        &Envelope::new(EnvelopeKind::Audio(bytes.clone())),
        &RunOptions::new().with_observation(child(context, None)),
    )?;
    references.insert("batch".into(), reference(batch.observation_ref()));
    record.insert(
        "batch".into(),
        json!({"outcome": "ok", "text_chars": batch.text().map_or(0, |t| t.chars().count())}),
    );
    let (samples, rate) = wav_samples(&bytes)?;
    let stream = model.stream(StreamConfig {
        observation: Some(child(context, None)),
        ..StreamConfig::default()
    })?;
    let mut feeds = 0usize;
    for part in samples.chunks(rate as usize) {
        stream.feed(part)?;
        feeds += 1;
    }
    let transcript = stream.flush()?;
    record.insert(
        "live".into(),
        json!({"outcome": "ok", "feeds": feeds, "sample_rate": rate,
               "text_chars": transcript.text.chars().count()}),
    );
    Ok(())
}

/// One frame session: every run measured, one traced per live interval.
fn case_frames(
    config: &Config,
    context: &ObservationContext,
    record: &mut Map<String, Value>,
    references: &mut Map<String, Value>,
) -> Outcome<()> {
    let model = ModelLoader::from_registry(&config.str("/model_id")?)
        .observation(child(context, None))
        .load()?;
    references.insert("load".into(), reference(model.load_observation_ref()));
    let session_id = format!("staging-frames-{}", config.str("/run")?);
    let mut traced = Vec::new();
    for _ in 0..FRAMES {
        let options = RunOptions::new()
            .with_generation_config(generation(4.0))
            .with_frame_session(&session_id)
            .with_observation(child(context, None));
        let result = model.run_with_options(&text("Say one word.".into()), &options)?;
        traced.push(reference(result.observation_ref()));
    }
    record.insert(
        "frames".into(),
        json!({"outcome": "ok", "runs": FRAMES, "frame_session_id": session_id,
               "live_trace_interval_ms": FRAME_INTERVAL.as_millis() as u64, "run_references": traced}),
    );
    Ok(())
}

/// A local stream aborted for cloud fallback, finished by the staging gateway.
fn case_fallback(
    config: &Config,
    context: &ObservationContext,
    record: &mut Map<String, Value>,
    references: &mut Map<String, Value>,
) -> Outcome<()> {
    let api = config.str("/sessions/case-fallback/platform_url")?;
    xybrid_sdk::set_platform_url(&api);
    xybrid_sdk::set_api_key(&config.str("/sessions/case-fallback/api_key")?);
    let model = ModelLoader::from_registry(&config.str("/model_id")?)
        .observation(child(context, None))
        .load()?;
    references.insert("load".into(), reference(model.load_observation_ref()));
    let mut critical = ResourceSnapshot::unknown();
    critical.memory_pressure = MemoryPressure::Critical;
    let options = RunOptions::new()
        .with_generation_config(generation(32.0))
        .with_abort_policy(
            AbortPolicy::default()
                .stop_on(AbortSignal::MemoryPressureCritical)
                .with_cloud_fallback(true)
                .with_max_grace_tokens(0),
        )
        .with_resource_provider(Arc::new(StagedResourceProvider::new(3, critical)))
        .with_observation(child(context, None));
    let mut envelope = text("Write two sentences about a lighthouse.".into());
    envelope.metadata.insert("model".into(), CLOUD_MODEL.into());
    envelope.metadata.insert("max_tokens".into(), "32".into());
    let adapter = CloudRuntimeAdapter::with_gateway(&format!("{}/v1", api.trim_end_matches('/')));
    let mut tokens = 0usize;
    let mut seam: Option<Value> = None;
    let result = model.run_streaming_with_fallback(
        &envelope,
        &options,
        &adapter,
        &mut |_token: PartialToken| -> Result<(), StreamingError> {
            tokens += 1;
            Ok(())
        },
        &mut |info: xybrid_sdk::SeamInfo| {
            seam = Some(
                json!({"reason": info.reason.to_string(), "local_tokens": info.local_tokens,
                               "local_latency_ms": info.local_latency_ms}),
            );
        },
    );
    let (status, observation) = match &result {
        Ok(result) => ("ok".to_owned(), result.observation_ref().cloned()),
        Err(error) => (error_text(error), None),
    };
    references.insert("run".into(), reference(observation.as_ref()));
    record.insert(
        "fallback".into(),
        json!({"outcome": status, "tokens": tokens, "seam": seam, "cloud_model": CLOUD_MODEL}),
    );
    Ok(())
}

fn run_case(
    config: &Config,
    args: &Args,
    session: &str,
    record: &mut Map<String, Value>,
) -> Outcome<()> {
    begin(config, session, record)?;
    let (workflow, context) = workflow(config, session)?;
    let mut references = Map::new();
    references.insert(
        "workflow".into(),
        reference(workflow.observation_ref().as_ref()),
    );
    let result = match session {
        "case-pipeline" => case_pipeline(config, &context, record),
        "case-tts" => case_tts(&context, record, &mut references),
        "case-asr" => case_asr(&args.audio, &context, record, &mut references),
        "case-frames" => case_frames(config, &context, record, &mut references),
        "case-fallback" => case_fallback(config, &context, record, &mut references),
        other => Err(format!("unknown case session {other}").into()),
    };
    record.insert(
        "outcome".into(),
        Value::from(
            result
                .as_ref()
                .map_or_else(|e| error_text(&**e), |()| "ok".to_owned()),
        ),
    );
    record.insert("references".into(), Value::Object(references));
    workflow.end();
    Ok(())
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(2);
        }
    };
    let raw: Value = match std::fs::read_to_string(&args.config)
        .map_err(|e| e.to_string())
        .and_then(|text| serde_json::from_str(&text).map_err(|e| e.to_string()))
    {
        Ok(raw) => raw,
        Err(error) => {
            eprintln!("error: cannot read the config: {error}");
            std::process::exit(2);
        }
    };
    // It holds the staging key; nothing reads it again.
    let _ = std::fs::remove_file(&args.config);
    let config = Config { raw };
    xybrid_sdk::set_binding("rust");
    let timeout =
        Duration::from_millis(config.num("/constants/timeout_ms").unwrap_or(20_000.0) as u64);

    let mut sessions = BTreeMap::new();
    let mut scenario_error = Value::Null;
    let wanted: Vec<String> = config
        .sessions()
        .into_iter()
        .filter(|s| {
            STANDARD_SESSIONS.contains(&s.as_str())
                || args.cases.is_empty()
                || args.cases.contains(s)
        })
        .collect();
    let mut order: Vec<String> = STANDARD_SESSIONS
        .iter()
        .map(|s| (*s).to_owned())
        .filter(|s| wanted.contains(s))
        .collect();
    order.extend(
        wanted
            .iter()
            .filter(|s| !STANDARD_SESSIONS.contains(&s.as_str()))
            .cloned(),
    );
    for session in order {
        let mut record = Map::new();
        let result = if STANDARD_SESSIONS.contains(&session.as_str()) {
            standard(&config, &session, &mut record)
        } else {
            run_case(&config, &args, &session, &mut record)
        };
        finish(&mut record, timeout);
        if let Err(error) = result {
            if scenario_error.is_null() {
                scenario_error = Value::from(format!("{session}: {error}"));
            }
        }
        println!("{session}: {}", Value::Object(record.clone()));
        sessions.insert(session, Value::Object(record));
    }
    let expectations = json!({
        "schema": SCHEMA,
        "binding": config.str("/binding").unwrap_or_default(),
        "lane": config.str("/lane").unwrap_or_default(),
        "run": config.str("/run").unwrap_or_default(),
        "sdk_version": env!("CARGO_PKG_VERSION"),
        "model_id": config.str("/model_id").unwrap_or_default(),
        "device": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH},
        "sessions": sessions,
        "scenario_error": scenario_error,
    });
    drop(config);
    let temporary = args.expectations.with_extension("json.tmp");
    let written = serde_json::to_string_pretty(&expectations)
        .map_err(|e| e.to_string())
        .and_then(|text| std::fs::write(&temporary, text + "\n").map_err(|e| e.to_string()))
        .and_then(|()| std::fs::rename(&temporary, &args.expectations).map_err(|e| e.to_string()));
    if let Err(error) = written {
        eprintln!(
            "error: cannot write {}: {error}",
            args.expectations.display()
        );
        std::process::exit(1);
    }
    if !expectations["scenario_error"].is_null() {
        eprintln!("{}", expectations["scenario_error"]);
        std::process::exit(1);
    }
}
