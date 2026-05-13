//! Cloud fallback demo: a local LLM stream aborts under simulated memory
//! pressure and seamlessly continues on a cloud target as one continuous
//! token stream.
//!
//! # Prerequisites
//!
//! 1. Cache the local model once so `ModelLoader::from_registry` resolves it
//!    without a download:
//!
//!    ```bash
//!    xybrid run --model qwen2.5-0.5b-instruct --input-text "warmup"
//!    ```
//!
//!    **Avoid thinking-mode LLMs** (qwen 3.x, deepseek-r1, etc.) for this demo:
//!    the SDK's streaming filter buffers `<think>…</think>` blocks, so a
//!    thinking model can spend the full token budget reasoning and never emit
//!    a "safe" chunk — which also starves `AbortState::check_before_token`,
//!    so the seam never fires. Tracked separately as a deeper abort-path bug.
//!
//! 2. Have a Xybrid gateway reachable on the OpenAI-compatible
//!    `/v1/chat/completions` endpoint. By default the demo points at
//!    `http://localhost:3001/v1`, which is the local-dev port the Xybrid
//!    gateway listens on. The gateway dispatches to whichever upstream
//!    provider it has credentials for (e.g. DeepSeek, OpenAI, Anthropic);
//!    the demo sends `model = "deepseek-chat"` to match a gateway
//!    configured with `DEEPSEEK_API_KEY`. Pick a different `model` (and
//!    matching `provider` metadata) below if your gateway exposes
//!    different upstream keys.
//!
//!    Authenticate to the gateway with a Xybrid API key — mint one in
//!    your Xybrid account and export it:
//!
//!    ```bash
//!    export XYBRID_API_KEY=sk_live_…
//!    ```
//!
//!    To override the gateway URL (staging / production / your own
//!    OpenAI-compatible endpoint):
//!
//!    ```bash
//!    export XYBRID_CLOUD_URL=https://your-gateway.example.com/v1
//!    ```
//!
//!    The cloud client appends `/chat/completions`, so `XYBRID_CLOUD_URL`
//!    must end at the API version prefix (`…/v1`), not the bare host.
//!
//! # Run
//!
//! ```bash
//! cargo run -p xybrid-sdk --example cloud_fallback_demo \
//!   --features platform-macos,dev-tools,llm-llamacpp
//! ```
//!
//! On Linux/Windows replace `platform-macos` with `platform-desktop`. On
//! Android the demo is not supported (the streaming abort uses Metal-bound
//! types). The `llm-llamacpp` feature is required at the SDK level so the
//! example block's `required-features` resolves; the platform preset already
//! enables it transitively.
//!
//! # What the demo proves
//!
//! - The local LLM streams a few tokens normally.
//! - A staged resource provider flips device state to `MemoryPressure::Critical`
//!   after 3 reads, tripping the configured `AbortPolicy`.
//! - The SDK's `run_streaming_with_fallback` catches the typed
//!   `CloudFallbackAbort` and re-runs the original prompt on the cloud adapter.
//! - The same `on_token` callback receives both legs' tokens, with a clearly
//!   labelled `↘ cloud fallback (<reason>) ↘` seam between them.
//! - `LocalAborted` and `CloudRetry` telemetry events share a single
//!   `correlation_id`, captured in the printed summary.
//!
//! The wrapper records the local abort outcome for `LocalAuthority` routing
//! feedback. A separate routing demo should exercise direct-to-cloud hysteresis;
//! this example focuses on the visible abort-and-retry seam.

use std::env;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use xybrid_core::device::{MemoryPressure, ResourceSnapshot};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::orchestrator::authority::test_seams::StagedResourceProvider;
use xybrid_core::runtime_adapter::types::PartialToken;
use xybrid_core::runtime_adapter::CloudRuntimeAdapter;
use xybrid_sdk::run_options::{AbortPolicy, AbortSignal, RunOptions};
use xybrid_sdk::telemetry::{flush_platform_telemetry, init_platform_telemetry_from_env};
use xybrid_sdk::{ModelLoader, SeamInfo};

const PROMPT: &str = "Write a short story about a lighthouse keeper who learns to brew tea.";
const LOCAL_MODEL_ID: &str = "qwen2.5-0.5b-instruct";
const NORMAL_READS_BEFORE_PRESSURE: usize = 3;

fn critical_snapshot() -> ResourceSnapshot {
    let mut snap = ResourceSnapshot::unknown();
    snap.memory_pressure = MemoryPressure::Critical;
    snap
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Xybrid Cloud Fallback Demo");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  Prompt: {}", PROMPT);
    println!();

    // 0. Wire the telemetry publisher so `LocalAborted` and `CloudRetry`
    //    events from `run_streaming_with_fallback` actually make it to the
    //    ingest service. Requires `XYBRID_API_KEY` to be set; for a local
    //    backend also set `XYBRID_INGEST_URL=http://localhost:8081`. If
    //    init returns false, the events drop silently and the trace will
    //    never appear in the console — print a loud warning instead of
    //    failing the run, since the SDK orchestration itself is independent
    //    of telemetry.
    if init_platform_telemetry_from_env() {
        println!("  Telemetry: publisher initialized");
        if let Ok(url) = env::var("XYBRID_INGEST_URL") {
            println!("    ingest endpoint: {}", url);
        } else {
            println!("    ingest endpoint: https://ingest.xybrid.dev (default)");
        }
    } else {
        eprintln!(
            "  ⚠ Telemetry: XYBRID_API_KEY not set — events will be dropped, no trace will \
             appear in the console. Mint a key in the platform console and export it."
        );
    }
    println!();

    // 1. Load the local model. The user is expected to have run
    //    `xybrid run --model qwen2.5-0.5b-instruct ...` once to populate the cache.
    //    Use a non-thinking-mode LLM here — see the docstring at the top of
    //    this file for why qwen 3.x and similar break the streaming filter.
    let load_start = Instant::now();
    let model = ModelLoader::from_registry(LOCAL_MODEL_ID).load()?;
    println!(
        "  Loaded {} (local) in {:?}",
        LOCAL_MODEL_ID,
        load_start.elapsed()
    );

    // 2. Stress-injecting resource provider: emit `Normal` for the first
    //    `NORMAL_READS_BEFORE_PRESSURE` snapshot reads, then `Critical`
    //    forever. Throttled snapshot reads in `AbortState` mean each read
    //    represents ~500 ms of run time, so the local leg streams a few
    //    visible tokens before the abort fires.
    let provider = Arc::new(StagedResourceProvider::new(
        NORMAL_READS_BEFORE_PRESSURE,
        critical_snapshot(),
    ));

    // 3. Cloud adapter. Default to the local Xybrid gateway's
    //    `/v1/chat/completions` endpoint (port 3001 in local dev).
    //    Override via `XYBRID_CLOUD_URL` for staging, production, or any
    //    OpenAI-compatible endpoint. The cloud client appends
    //    `/chat/completions`, so the URL must end at the API version
    //    prefix (e.g. `…/v1`).
    let cloud_url =
        env::var("XYBRID_CLOUD_URL").unwrap_or_else(|_| "http://localhost:3001/v1".to_string());
    let cloud_adapter = CloudRuntimeAdapter::with_gateway(&cloud_url);

    // 4. Run options: stop the local leg on `MemoryPressure::Critical`,
    //    permit cloud fallback, deny grace tokens (so the abort fires on the
    //    very next token after pressure trips).
    let options = RunOptions::new()
        .with_abort_policy(
            AbortPolicy::default()
                .stop_on(AbortSignal::MemoryPressureCritical)
                .with_cloud_fallback(true)
                .with_max_grace_tokens(0),
        )
        .with_resource_provider(provider);

    // 5. Envelope. Carries the prompt and the cloud-side routing metadata
    //    `CloudRuntimeAdapter::execute_streaming` will read on the retry leg.
    //    The local leg ignores the cloud-only keys.
    let mut envelope = Envelope::new(EnvelopeKind::Text(PROMPT.to_string()));
    // The local platform backend resolves the upstream provider from the
    //    `model` name (`deepseek-chat` → DeepSeek), so the `provider` field is
    //    SDK-side bookkeeping only and never crosses the wire to the gateway.
    //    Match it to the model so traces stay honest.
    envelope
        .metadata
        .insert("provider".to_string(), "deepseek".to_string());
    envelope
        .metadata
        .insert("model".to_string(), "deepseek-chat".to_string());
    envelope
        .metadata
        .insert("max_tokens".to_string(), "200".to_string());
    envelope.metadata.insert(
        "system_prompt".to_string(),
        "You are a concise storyteller.".to_string(),
    );

    // 6. Shared state for terminal output: a target flag (toggled when the
    //    seam fires) and per-leg token counters.
    let on_cloud = Arc::new(AtomicBool::new(false));
    let local_tokens = Arc::new(Mutex::new(0u32));
    let cloud_tokens = Arc::new(Mutex::new(0u32));
    let captured_seam: Arc<Mutex<Option<SeamInfo>>> = Arc::new(Mutex::new(None));

    let on_cloud_for_token = on_cloud.clone();
    let local_tokens_for_token = local_tokens.clone();
    let cloud_tokens_for_token = cloud_tokens.clone();
    let mut on_token =
        move |token: PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            if on_cloud_for_token.load(Ordering::SeqCst) {
                let mut count = cloud_tokens_for_token.lock().unwrap();
                if *count == 0 {
                    print!("[cloud] ");
                }
                *count += 1;
            } else {
                let mut count = local_tokens_for_token.lock().unwrap();
                if *count == 0 {
                    print!("[device] ");
                }
                *count += 1;
            }
            print!("{}", token.token);
            std::io::stdout().flush().ok();
            Ok(())
        };

    let on_cloud_for_seam = on_cloud.clone();
    let captured_seam_for_seam = captured_seam.clone();
    let mut on_seam = move |info: SeamInfo| {
        println!();
        println!();
        println!(
            "  ↘ cloud fallback ({}) — local leg: {} tokens / {} ms",
            info.reason, info.local_tokens, info.local_latency_ms
        );
        println!();
        on_cloud_for_seam.store(true, Ordering::SeqCst);
        *captured_seam_for_seam.lock().unwrap() = Some(info);
    };

    // 7. Run.
    let run_start = Instant::now();
    let result = model.run_streaming_with_fallback(
        &envelope,
        &options,
        &cloud_adapter,
        &mut on_token,
        &mut on_seam,
    )?;
    let run_latency_ms = run_start.elapsed().as_millis() as u32;

    // 8. Summary.
    println!();
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════════════");
    let local = *local_tokens.lock().unwrap();
    let cloud = *cloud_tokens.lock().unwrap();
    if let Some(info) = captured_seam.lock().unwrap().clone() {
        let cloud_ms = run_latency_ms.saturating_sub(info.local_latency_ms);
        println!("  correlation_id : {}", info.correlation_id);
        println!("  abort reason   : {}", info.reason);
        println!("  target chain   : device → cloud");
        println!(
            "  local leg      : {} tokens / {} ms / aborted",
            local, info.local_latency_ms
        );
        println!(
            "  cloud leg      : {} tokens / ~{} ms / completed",
            cloud, cloud_ms
        );
    } else {
        println!(
            "  No fallback fired — the local leg completed in {} ms with {} tokens.",
            run_latency_ms, local
        );
    }
    println!("  total wallclock: {:?}", total_start.elapsed());
    println!(
        "  result text    : {}",
        result.text().unwrap_or("(non-text)")
    );

    // Drain telemetry before exit. The SDK exporter is parked in a process-
    // static (`PLATFORM_EXPORTER`) that never gets dropped, so its
    // `Drop::stop` final-flush never runs. Without an explicit drain the
    // CloudRetry event (published last, after the ~8s cloud round-trip)
    // gets stuck in the exporter's mpsc channel / internal buffer when the
    // demo exits before the periodic 15s flush thread fires next, which
    // shows up in the console as a missing `target=cloud` row. Sleep
    // briefly so the consumer thread can drain channel→buffer, then call
    // `flush_platform_telemetry` to force a buffer→HTTP send.
    println!();
    println!("  Flushing telemetry (channel drain + buffer flush)…");
    // Brief pause lets the exporter's mpsc-consumer thread drain pending
    // events from the channel into the buffer (typically <50 ms in
    // practice — 200 ms is generous). Then `flush_platform_telemetry` is
    // synchronous-blocking on the HTTP send via ureq, so we don't need
    // another long sleep after; the trailing 200 ms only covers the
    // ingest service's `wait=false` async-queue acceptance path.
    std::thread::sleep(std::time::Duration::from_millis(200));
    flush_platform_telemetry();
    std::thread::sleep(std::time::Duration::from_millis(200));

    Ok(())
}
