//! Observability overhead driver: one process runs a local GGUF LLM through
//! the SDK streaming path with telemetry in one mode, and writes what it
//! measured as JSON. `tools/scripts/observability_perf/run_perf.py` builds it
//! once (release) and runs it many times, interleaving modes, under
//! `/usr/bin/time -l` for peak RSS.
//!
//! ```bash
//! cargo build --release -p xybrid-sdk --features llm-llamacpp --example observability_perf
//! target/release/examples/observability_perf \
//!     --model-dir integration-tests/fixtures/models/qwen3.5-0.8b \
//!     --mode content --endpoint http://127.0.0.1:4318 --out run.json
//! ```
//!
//! Modes:
//!
//! - `off`: no session; the SDK observer is disabled.
//! - `metadata`: packaged OTLP pipeline (all signals, default limits) to
//!   `--endpoint`, metadata-only content policy.
//! - `content`: the same with the default content capture (`span_only`).
//! - `blocked`: `content` against an endpoint that never answers.
//!
//! Per measured run it records callback timestamps (time to first callback,
//! the gaps between token callbacks), wall time, and the runtime's own
//! `ttft_ms` / `tokens_per_second` / `tokens_generated`. After the runs it
//! records the pipeline counters, then times shutdown separately so the
//! blocked mode's bounded shutdown never counts as inference time.
//!
//! Arguments: `--model-dir DIR` (required), `--mode MODE` (default `off`),
//! `--endpoint URL`, `--out FILE` (default stdout), `--runs N` (3),
//! `--warmup N` (1), `--max-tokens N` (128), `--prompt-bytes N` (2048),
//! `--shutdown-timeout-ms N` (2000).

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_observability::otlp::EnvironmentSource;
use xybrid_sdk::observability::{
    self, ContentCapture, ContentPolicy, ObservabilityConfig, ObservabilitySession, OtlpConfig,
    PipelineStats, SignalStats,
};
use xybrid_sdk::{GenerationConfig, ModelLoader};

const SYSTEM_PROMPT: &str = "You are a concise travel assistant. Answer in plain English, \
use short numbered lists when listing facts, and never invent opening hours or prices.";

/// Context the user prompt is built from, repeated to `--prompt-bytes`.
const CONTEXT: &str = "Lisbon is the capital of Portugal, built on seven hills above the \
Tagus estuary. Its historic districts include Alfama, Baixa, Chiado and Belem. Tram 28 \
climbs through narrow streets past tiled facades. The city was rebuilt after the 1755 \
earthquake under the Marquis of Pombal, giving Baixa its grid of streets. ";

const QUESTION: &str = "Using the notes above, write a detailed numbered list of facts \
about Lisbon's history, neighbourhoods and transport.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Off,
    Metadata,
    Content,
    Blocked,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "off" => Ok(Self::Off),
            "metadata" => Ok(Self::Metadata),
            "content" => Ok(Self::Content),
            "blocked" => Ok(Self::Blocked),
            other => Err(format!("unknown --mode {other}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Metadata => "metadata",
            Self::Content => "content",
            Self::Blocked => "blocked",
        }
    }
}

struct Args {
    model_dir: PathBuf,
    mode: Mode,
    endpoint: Option<String>,
    out: Option<PathBuf>,
    runs: usize,
    warmup: usize,
    max_tokens: usize,
    prompt_bytes: usize,
    shutdown_timeout: Duration,
}

fn parse_args() -> Result<Args, String> {
    let mut values: HashMap<String, String> = HashMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let Some(name) = flag.strip_prefix("--") else {
            return Err(format!("unexpected argument {flag}"));
        };
        let value = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
        values.insert(name.to_owned(), value);
    }
    let number = |name: &str, default: usize| -> Result<usize, String> {
        values.get(name).map_or(Ok(default), |v| {
            v.parse().map_err(|_| format!("--{name} must be a number"))
        })
    };
    let args = Args {
        model_dir: values
            .get("model-dir")
            .map(PathBuf::from)
            .ok_or("--model-dir is required")?,
        mode: Mode::parse(values.get("mode").map_or("off", String::as_str))?,
        endpoint: values.get("endpoint").cloned(),
        out: values.get("out").map(PathBuf::from),
        runs: number("runs", 3)?,
        warmup: number("warmup", 1)?,
        max_tokens: number("max-tokens", 128)?,
        prompt_bytes: number("prompt-bytes", 2048)?,
        shutdown_timeout: Duration::from_millis(number("shutdown-timeout-ms", 2000)? as u64),
    };
    if args.mode != Mode::Off && args.endpoint.is_none() {
        return Err(format!("--mode {} needs --endpoint", args.mode.as_str()));
    }
    Ok(args)
}

fn prompt(bytes: usize) -> String {
    let mut text = String::with_capacity(bytes + CONTEXT.len());
    while text.len() + QUESTION.len() < bytes {
        text.push_str(CONTEXT);
    }
    text.push('\n');
    text.push_str(QUESTION);
    text
}

fn session(args: &Args) -> Result<Option<ObservabilitySession>, String> {
    let content = match args.mode {
        Mode::Off => return Ok(None),
        Mode::Metadata => ContentPolicy::metadata_only(),
        Mode::Content | Mode::Blocked => {
            ContentPolicy::default().with_capture(ContentCapture::SpanOnly)
        }
    };
    let endpoint = args.endpoint.as_deref().expect("checked in parse_args");
    let otlp = OtlpConfig::collector(endpoint).with_environment(EnvironmentSource::Disabled);
    let config = ObservabilityConfig::otlp(otlp).content(content).service(
        "observability-perf",
        Some(env!("CARGO_PKG_VERSION").to_owned()),
    );
    let session = ObservabilitySession::new(config).map_err(|e| e.to_string())?;
    observability::install_default_session(session.clone());
    Ok(Some(session))
}

fn signal(s: &SignalStats) -> Value {
    json!({
        "accepted": s.accepted, "exported": s.exported, "dropped": s.dropped,
        "failed": s.failed, "pending": s.pending,
    })
}

fn stats(s: Option<PipelineStats>) -> Value {
    s.map_or(Value::Null, |s| {
        json!({ "traces": signal(&s.traces), "logs": signal(&s.logs), "metrics": signal(&s.metrics) })
    })
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn metadata_number(metadata: &HashMap<String, String>, key: &str) -> Value {
    metadata
        .get(key)
        .and_then(|v| v.parse::<f64>().ok())
        .map_or(Value::Null, Value::from)
}

/// Linux peak RSS of this process (the harness measures every platform
/// from outside; this is a cross-check).
fn self_peak_rss_bytes() -> Value {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kib| kib.parse::<u64>().ok())
        })
        .map_or(Value::Null, |kib| Value::from(kib * 1024))
}

fn run(args: &Args) -> Result<Value, Box<dyn std::error::Error>> {
    let process_start = Instant::now();
    let session = session(args)?;
    let load_start = Instant::now();
    let model = ModelLoader::from_directory(&args.model_dir)?.load()?;
    let load_ms = ms(load_start.elapsed());

    let mut metadata = HashMap::new();
    metadata.insert("system_prompt".to_owned(), SYSTEM_PROMPT.to_owned());
    let envelope = Envelope {
        kind: EnvelopeKind::Text(prompt(args.prompt_bytes)),
        metadata,
    };
    let config = GenerationConfig {
        max_tokens: args.max_tokens,
        temperature: 0.0,
        ..GenerationConfig::default()
    };

    let mut runs = Vec::new();
    for index in 0..args.warmup + args.runs {
        let mut stamps: Vec<Instant> = Vec::with_capacity(args.max_tokens + 16);
        let mut finish: Option<String> = None;
        let start = Instant::now();
        let result = model.run_streaming(&envelope, Some(&config), |token| {
            stamps.push(Instant::now());
            if token.finish_reason.is_some() {
                finish = token.finish_reason;
            }
            Ok(())
        })?;
        let wall = start.elapsed();
        if index < args.warmup {
            continue;
        }
        let gaps: Vec<f64> = stamps.windows(2).map(|w| ms(w[1] - w[0])).collect();
        let first = stamps.first().map(|t| ms(*t - start));
        let decode_s = match (stamps.first(), stamps.last()) {
            (Some(a), Some(b)) if stamps.len() > 1 => (*b - *a).as_secs_f64(),
            _ => 0.0,
        };
        let m = &result.envelope().metadata;
        runs.push(json!({
            "wall_ms": ms(wall),
            "callbacks": stamps.len(),
            "first_callback_ms": first,
            "callback_gaps_ms": gaps,
            "decode_callbacks_per_s": if decode_s > 0.0 { (stamps.len() - 1) as f64 / decode_s } else { 0.0 },
            "finish_reason": finish.or_else(|| m.get("finish_reason").cloned()),
            "runtime": {
                "ttft_ms": metadata_number(m, "ttft_ms"),
                "tokens_generated": metadata_number(m, "tokens_generated"),
                "tokens_per_second": metadata_number(m, "tokens_per_second"),
                "generation_time_ms": metadata_number(m, "generation_time_ms"),
                "mean_itl_ms": metadata_number(m, "mean_itl_ms"),
                "p95_itl_ms": metadata_number(m, "p95_itl_ms"),
            },
            "has_observation_ref": result.observation_ref().is_some(),
        }));
    }
    drop(model);

    let before = session.as_ref().and_then(ObservabilitySession::stats);
    let (shutdown, shutdown_ms) = match &session {
        Some(s) => {
            let t = Instant::now();
            let r = s.shutdown(args.shutdown_timeout);
            (Some(format!("{r:?}")), Some(ms(t.elapsed())))
        }
        None => (None, None),
    };
    let after = session.as_ref().and_then(ObservabilitySession::stats);

    Ok(json!({
        "driver": "observability_perf",
        "sdk_version": env!("CARGO_PKG_VERSION"),
        "mode": args.mode.as_str(),
        "model_dir": args.model_dir.display().to_string(),
        "max_tokens": args.max_tokens,
        "prompt_bytes": args.prompt_bytes,
        "warmup": args.warmup,
        "load_ms": load_ms,
        "runs": runs,
        "telemetry": {
            "enabled": session.is_some(),
            "endpoint": args.endpoint,
            "stats_before_shutdown": stats(before),
            "shutdown_result": shutdown,
            "shutdown_ms": shutdown_ms,
            "shutdown_timeout_ms": args.shutdown_timeout.as_millis() as u64,
            "stats_after_shutdown": stats(after),
        },
        "process_ms": ms(process_start.elapsed()),
        "self_peak_rss_bytes": self_peak_rss_bytes(),
    }))
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("observability_perf: {message}");
            std::process::exit(2);
        }
    };
    match run(&args) {
        Ok(report) => {
            let body = serde_json::to_string_pretty(&report).expect("report serializes");
            match &args.out {
                Some(path) => std::fs::write(path, body).expect("write --out"),
                None => println!("{body}"),
            }
        }
        Err(error) => {
            eprintln!("observability_perf: {error}");
            std::process::exit(1);
        }
    }
}
