//! Resource measurements for readout contexts: initialization time, peak
//! memory and readout latency per token budget, with the one-output cap and
//! without it (llama.cpp's default of one output row per batch token).
//!
//! Ignored tests, Linux only (they read `/proc/self/status`). Stage the d1
//! GGUF as for `tests/d1_readout_smoke.rs`, then:
//!
//! ```sh
//! XYBRID_MEASURE=1 cargo test --release -p xybrid-llama --features bindings --lib -- \
//!     --ignored --exact memoryless::measure::readout_resources --nocapture
//! ```
//!
//! A full run takes over an hour on four CPU threads, so it also needs
//! `XYBRID_MEASURE=1`: `--ignored` alone skips it.
//!
//! `readout_resources` runs every configuration in a fresh process (this
//! test binary, re-run on `readout_resources_case`), so each peak is that
//! configuration's alone, and prints one table row per configuration.
//! `XYBRID_MEASURE_BUDGETS` (comma-separated) overrides the budgets and
//! `XYBRID_MEASURE_OUT` names a file for the table.
//!
//! `cancel_latency` (same flags, minutes rather than an hour) measures how
//! long a readout takes to return after its cancel, by pass length:
//! `XYBRID_MEASURE_LENGTHS` (comma-separated) overrides the lengths.

use std::process::Command;
use std::time::{Duration, Instant};

use super::{MemorylessContext, MemorylessParams, OUTPUT_CAP};
use crate::error::LlamaError;
use crate::model::LlamaModel;
use crate::test_models::staged_d1;

const CASE_ENV: &str = "XYBRID_MEASURE_CASE";
const THREADS: u32 = 4;

/// The smoke test's hand-rendered choice prompt.
const PROMPT: &str = "<|startoftext|><|im_start|>user\n\
The customer writes: my card was charged twice for one order.\n\n\n\
QUESTION:\n\
Which team should handle this ticket?\n\n\
Options:\n\
A Billing\n\
B Shipping\n\n\
Reply with the option code only.<|im_end|>\n<|im_start|>assistant\n<think></think>";

/// `/proc/self/status` fields in kB: VmPeak, VmHWM, RssAnon, RssFile.
fn memory_kb() -> [u64; 4] {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let field = |name: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|kb| kb.parse().ok())
            .unwrap_or(0)
    };
    [
        field("VmPeak:"),
        field("VmHWM:"),
        field("RssAnon:"),
        field("RssFile:"),
    ]
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn mib(kb: u64) -> f64 {
    kb as f64 / 1024.0
}

#[test]
#[ignore = "needs the d1-2B GGUF; run in release, see the module docs"]
fn readout_resources() {
    if std::env::var_os("XYBRID_MEASURE").is_none() {
        eprintln!("Skipping: set XYBRID_MEASURE=1 to run the measurements (over an hour)");
        return;
    }
    if staged_d1().is_none() {
        return;
    }
    let budgets: Vec<u32> = std::env::var("XYBRID_MEASURE_BUDGETS")
        .unwrap_or_else(|_| "2048,4096,8192".to_string())
        .split(',')
        .map(|b| b.trim().parse().expect("a budget"))
        .collect();
    let exe = std::env::current_exe().expect("this test binary");
    let mut table = vec![
        "| settings | budget | outputs | compute buffer | init ms | VmPeak / RSS after init MiB | short readout ms | full readout ms | peak RSS MiB | RssAnon MiB | VmPeak MiB |".to_string(),
        "|---|---|---|---|---|---|---|---|---|---|---|".to_string(),
    ];
    for flash_attn in [false, true] {
        for capped in [true, false] {
            for &budget in &budgets {
                let output = Command::new(&exe)
                    .args([
                        "--exact",
                        "memoryless::measure::readout_resources_case",
                        "--ignored",
                        "--nocapture",
                        "--test-threads",
                        "1",
                    ])
                    .env(CASE_ENV, format!("{budget},{capped},{flash_attn}"))
                    .env("XYBRID_LLAMACPP_VERBOSITY", "3")
                    .output()
                    .expect("run a measurement case");
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let buffers: Vec<String> = stderr
                    .lines()
                    .filter(|line| line.contains("compute buffer size"))
                    .map(|line| {
                        line.split_once(':')
                            .map_or(line, |(_, rest)| rest)
                            .trim()
                            .to_string()
                    })
                    .collect();
                let settings = if flash_attn {
                    "cpu-t4-fa-on"
                } else {
                    "cpu-t4-fa-off"
                };
                let outputs = if capped { "1" } else { "n_batch" };
                // libtest prints the test name and the test's own output on
                // one line under --nocapture.
                let result = stdout
                    .lines()
                    .find_map(|line| line.split_once("MEASURE ").map(|(_, rest)| rest));
                let row = match result {
                    Some(result) => {
                        format!(
                            "| {settings} | {budget} | {outputs} | {} | {result} |",
                            buffers.join("; ")
                        )
                    }
                    None => format!(
                        "| {settings} | {budget} | {outputs} | {} | failed: {} | | | | | | |",
                        buffers.join("; "),
                        output.status
                    ),
                };
                eprintln!("{row}");
                table.push(row);
            }
        }
    }
    let table = table.join("\n");
    println!("\n{table}");
    if let Some(path) = std::env::var_os("XYBRID_MEASURE_OUT") {
        std::fs::write(path, format!("{table}\n")).expect("write the table");
    }
}

/// One configuration from `XYBRID_MEASURE_CASE` (`budget,capped,flash_attn`).
#[test]
#[ignore = "one configuration of readout_resources"]
fn readout_resources_case() {
    let Ok(case) = std::env::var(CASE_ENV) else {
        return;
    };
    let mut parts = case.split(',');
    let budget: u32 = parts.next().unwrap().parse().unwrap();
    let capped: bool = parts.next().unwrap().parse().unwrap();
    let flash_attn: bool = parts.next().unwrap().parse().unwrap();
    let path = staged_d1().expect("the model is staged");

    crate::backend_init();
    let model = LlamaModel::load(path.to_str().unwrap(), 0).expect("load the model");
    let short = model.tokenize_special(PROMPT, false).unwrap();
    let full: Vec<i32> = short
        .iter()
        .cycle()
        .take(budget as usize)
        .copied()
        .collect();
    let select: Vec<i32> = ["A", " A", "B", " B"]
        .iter()
        .map(|text| model.tokenize(text, false).unwrap()[0])
        .collect();

    let start = Instant::now();
    let cap = if capped { OUTPUT_CAP } else { 0 };
    let mut ctx = MemorylessContext::create(
        &model,
        MemorylessParams::new(budget, THREADS, flash_attn),
        cap,
    )
    .expect("create the context");
    let init_ms = ms(start);
    // What creating the context reserved, before a readout touches it.
    let [vm_init, hwm_init, _, _] = memory_kb();

    let mut short_ms = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        ctx.read_last(&short, &select).expect("a short readout");
        short_ms.push(ms(start));
    }
    let mut full_ms = Vec::new();
    for _ in 0..2 {
        let start = Instant::now();
        ctx.read_last(&full, &select)
            .expect("a full-budget readout");
        full_ms.push(ms(start));
    }
    let [vm_peak, hwm, anon, _file] = memory_kb();
    let list = |values: &[f64]| {
        values
            .iter()
            .map(|v| format!("{v:.0}"))
            .collect::<Vec<_>>()
            .join(" / ")
    };
    println!(
        "\nMEASURE {init_ms:.0} | {:.0} / {:.0} | {} ({} tokens) | {} | {:.0} | {:.0} | {:.0}",
        mib(vm_init),
        mib(hwm_init),
        list(&short_ms),
        short.len(),
        list(&full_ms),
        mib(hwm),
        mib(anon),
        mib(vm_peak)
    );
}

/// The time from `cancel()` to the readout's return, on settings
/// `cpu-t4-fa-off`, by pass length. llama.cpp checks for the cancel between
/// two graph nodes, so this is what remains of the node running at the
/// cancel, plus the threads' synchronization: one node is a whole-batch
/// matrix product, so the latency grows with the pass length. Each length
/// is cancelled at 10, 30, 50, 70 and 90 % of a full pass, twice.
#[test]
#[ignore = "needs the d1-2B GGUF; run in release, see the module docs"]
fn cancel_latency() {
    if std::env::var_os("XYBRID_MEASURE").is_none() {
        eprintln!("Skipping: set XYBRID_MEASURE=1 to run the measurements");
        return;
    }
    let Some(path) = staged_d1() else { return };
    crate::backend_init();
    let model = LlamaModel::load(path.to_str().expect("a UTF-8 path"), 0).expect("load d1-2B");
    let prompt = model.tokenize_special(PROMPT, false).expect("tokenize");
    let lengths: Vec<usize> = std::env::var("XYBRID_MEASURE_LENGTHS")
        .unwrap_or_else(|_| "256,512,1024,2048,4096".to_string())
        .split(',')
        .map(|length| length.trim().parse().expect("a length"))
        .collect();
    println!("\nCANCEL length | full pass ms | cancel-to-return ms: min / median / max (n)");
    for length in lengths {
        let tokens: Vec<i32> = prompt.iter().copied().cycle().take(length).collect();
        let budget = u32::try_from(length).expect("a u32 length");
        let mut ctx = MemorylessContext::new(&model, MemorylessParams::new(budget, THREADS, false))
            .expect("a readout context");
        ctx.read_last(&tokens, &[0]).expect("a warm-up pass");
        let start = Instant::now();
        ctx.read_last(&tokens, &[0]).expect("a full pass");
        let pass = start.elapsed();

        let mut latencies: Vec<Duration> = Vec::new();
        for fraction in [0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9] {
            ctx.clear_cancel();
            let cancel = ctx.cancel_handle();
            let delay = pass.mul_f64(fraction);
            let canceller = std::thread::spawn(move || {
                std::thread::sleep(delay);
                let at = Instant::now();
                cancel.cancel();
                at
            });
            let result = ctx.read_last(&tokens, &[0]);
            let returned = Instant::now();
            let cancelled_at = canceller.join().expect("the cancelling thread");
            match result {
                Err(LlamaError::Cancelled) => {
                    latencies.push(returned.saturating_duration_since(cancelled_at));
                }
                Ok(_) => {} // finished before the cancel
                Err(error) => panic!("{error}"),
            }
        }
        latencies.sort();
        let ms_of = |d: Duration| d.as_secs_f64() * 1e3;
        let (min, median, max) = match latencies.as_slice() {
            [] => (f64::NAN, f64::NAN, f64::NAN),
            all => (
                ms_of(all[0]),
                ms_of(all[all.len() / 2]),
                ms_of(all[all.len() - 1]),
            ),
        };
        println!(
            "CANCEL {length:>6} | {:>12.0} | {min:>7.1} / {median:>7.1} / {max:>7.1} ({})",
            ms_of(pass),
            latencies.len()
        );
    }
}
