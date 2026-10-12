//! Final-position logits on a real model: Qwen3.5-0.8B F16, the conformance
//! fixture label scoring is checked against.
//!
//! Stage the model, then run:
//!
//! ```sh
//! tools/conformance/prepare.sh qwen3.5-0.8b-f16-cf
//! cargo test -p xybrid-llama --features bindings --test choice_logits
//! ```
//!
//! A missing model skips each test, unless `XYBRID_REQUIRE_MODELS` is set,
//! which turns the skip into a failure. Everything runs on the CPU, so
//! repeated runs are bitwise equal.
//!
//! Two tests need more and skip without it:
//!
//! - `the_harness_reads_the_same_row` needs the C++ harness, linked against
//!   the llama.cpp libraries this test links: set `XYBRID_LLAMA_HARNESS` to
//!   the path `tools/conformance/llama_harness/build.sh FEATURES` prints,
//!   `FEATURES` being the ones this test runs with.
//! - `a_model_without_memory_is_refused` needs an encoder-only GGUF, such as
//!   bge-small-en-v1.5: set `XYBRID_ENCODER_GGUF` to its path.

#![cfg(feature = "bindings")]

use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use xybrid_llama::{backend_init, generate_with_stops, LlamaContext, LlamaError, LlamaModel};

const MODEL_ID: &str = "qwen3.5-0.8b-f16-cf";
const MODEL_FILE: &str = "Qwen3.5-0.8B-F16.gguf";

const QUESTION: &str = "<|im_start|>user\nWhich colour is the sky on a clear day? \
                        Answer with one word.<|im_end|>\n<|im_start|>assistant\n";
const OTHER_QUESTION: &str = "<|im_start|>user\nName a fruit that is yellow.<|im_end|>\n\
                              <|im_start|>assistant\n";

/// The staged conformance GGUF, or `None` (with a note) when it is absent.
///
/// # Panics
///
/// When `XYBRID_REQUIRE_MODELS` is set and the model is not staged.
fn staged_model_path() -> Option<PathBuf> {
    let models = std::env::var_os("XYBRID_TEST_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../integration-tests/fixtures/models")
        });
    let path = models.join(MODEL_ID).join(MODEL_FILE);
    if path.is_file() {
        return Some(path);
    }
    let reason = format!(
        "'{MODEL_ID}/{MODEL_FILE}' is not staged. Run: tools/conformance/prepare.sh {MODEL_ID}"
    );
    if std::env::var_os("XYBRID_REQUIRE_MODELS").is_some() {
        panic!("XYBRID_REQUIRE_MODELS is set but {reason}");
    }
    eprintln!("Skipping test: {reason}");
    None
}

/// The model, loaded once on the CPU and shared by every test.
fn model() -> Option<&'static LlamaModel> {
    static MODEL: OnceLock<Option<LlamaModel>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let path = staged_model_path()?;
            backend_init();
            Some(LlamaModel::load(path.to_str().expect("a UTF-8 path"), 0).expect("load the GGUF"))
        })
        .as_ref()
}

fn context(model: &LlamaModel, n_ctx: usize, n_batch: usize) -> LlamaContext {
    LlamaContext::new(model, n_ctx, 4, n_batch, false).expect("create a context")
}

fn tokens(model: &LlamaModel, prompt: &str) -> Vec<i32> {
    model
        .tokenize_special(prompt, false)
        .expect("tokenize the prompt")
}

/// Every token id, to read the whole row.
fn vocabulary(model: &LlamaModel) -> Vec<i32> {
    (0..i32::try_from(model.n_vocab()).expect("the vocabulary fits i32")).collect()
}

/// The token id and value of the largest logit.
fn argmax(logits: &[f32]) -> (i32, f32) {
    let (index, max) =
        logits
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (i, &logit)| {
                if logit > best.1 {
                    (i, logit)
                } else {
                    best
                }
            });
    (i32::try_from(index).expect("a token id"), max)
}

/// The first token greedy generation picks, continuing from the context's
/// current memory.
fn first_greedy_token(ctx: &LlamaContext, model: &LlamaModel, prompt: &[i32]) -> i32 {
    let generated = generate_with_stops(ctx, model, prompt, 1, 0.0, 1.0, 0.0, 0, 1.0, &[], None)
        .expect("generate one token");
    generated[0]
}

/// The first token greedy generation picks, from an empty state.
fn greedy_next_token(ctx: &LlamaContext, model: &LlamaModel, prompt: &[i32]) -> i32 {
    ctx.kv_cache_clear();
    first_greedy_token(ctx, model, prompt)
}

/// Generation's own prefill cuts the prompt into the same chunks as
/// scoring, so the row's argmax is the token greedy generation picks whether
/// the prompt is decoded whole, in several chunks or one token at a time.
#[test]
fn the_final_row_is_the_next_token_distribution() {
    let Some(model) = model() else { return };
    let prompt = tokens(model, QUESTION);
    let vocabulary = vocabulary(model);
    assert!(
        prompt.len() > 3 * 7,
        "the prompt spans several 7-token chunks"
    );

    for n_batch in [512, 7, 1] {
        let ctx = context(model, 2048, n_batch);
        let row = ctx
            .score_final_position(&prompt, &vocabulary)
            .expect("score the whole vocabulary");
        let (argmax, max) = argmax(&row.label_logits);

        assert_eq!(
            argmax,
            greedy_next_token(&ctx, model, &prompt),
            "n_batch {n_batch}: the row's argmax is the token greedy generation picks"
        );
        assert!(row.logsumexp >= max, "log-sum-exp is at least the maximum");
        assert!(
            f64::from(row.logsumexp - max) <= (vocabulary.len() as f64).ln(),
            "log-sum-exp exceeds the maximum by at most ln(vocabulary)"
        );
    }
}

#[test]
fn scores_do_not_depend_on_what_ran_before() {
    let Some(model) = model() else { return };
    let ctx = context(model, 2048, 512);
    let first = tokens(model, QUESTION);
    let second = tokens(model, OTHER_QUESTION);
    let labels = [tokens(model, "Blue")[0], tokens(model, "Green")[0], 15];

    let a = ctx.score_final_position(&first, &labels).expect("score A");
    let b = ctx.score_final_position(&second, &labels).expect("score B");
    // No clear in between: llama.cpp refuses a prefill from position 0 while
    // the memory still holds a scored prompt.
    let generated = first_greedy_token(&ctx, model, &second);
    let a_again = ctx
        .score_final_position(&first, &labels)
        .expect("score A after generating");
    let fresh = context(model, 2048, 512);

    assert_ne!(a, b, "different prompts score differently");
    assert_eq!(
        a, a_again,
        "scoring B and generating in between change nothing"
    );
    assert_eq!(
        a,
        fresh
            .score_final_position(&first, &labels)
            .expect("score A on a fresh context"),
        "a used context scores like a fresh one"
    );
    assert_eq!(
        generated,
        greedy_next_token(&fresh, model, &second),
        "generation right after scoring matches a fresh context"
    );
}

/// Every return clears the memory, including a refused request's.
#[test]
fn refused_requests_leave_the_memory_empty() {
    let Some(model) = model() else { return };
    let ctx = context(model, 64, 512);
    let prompt = tokens(model, QUESTION);
    let n_vocab = i32::try_from(model.n_vocab()).unwrap();
    let too_long = vec![prompt[0]; ctx.n_ctx()];
    // Generation leaves the prompt in memory.
    let expected = greedy_next_token(&ctx, model, &prompt);

    for (refused, labels) in [(&too_long[..], &[0][..]), (&prompt[..], &[n_vocab][..])] {
        assert!(matches!(
            ctx.score_final_position(refused, labels),
            Err(LlamaError::InvalidInput(_))
        ));
        // No clear in between: llama.cpp refuses a prefill from position 0
        // while the memory still holds a prompt.
        assert_eq!(
            first_greedy_token(&ctx, model, &prompt),
            expected,
            "the refused request left the memory empty"
        );
    }
    ctx.score_final_position(&prompt, &[0, 1])
        .expect("a valid request still scores");
}

/// llama.cpp caps a causal model's decode call at the context size asked
/// for, 64 here, and only then pads the context to 256. A prompt between the
/// two must be cut every 64 tokens; a longer call would abort the process.
#[test]
fn a_small_context_cuts_at_the_size_it_was_asked_for() {
    let Some(model) = model() else { return };
    let ctx = context(model, 64, 512);
    let prompt = tokens(model, QUESTION).repeat(5);
    let vocabulary = vocabulary(model);
    assert!(
        prompt.len() > 64 && prompt.len() < ctx.n_ctx(),
        "the prompt fits only the padded context"
    );

    let small = ctx
        .score_final_position(&prompt, &vocabulary)
        .expect("score on a 64-token context");
    let cut_every_64 = context(model, 2048, 64)
        .score_final_position(&prompt, &vocabulary)
        .expect("score in 64-token chunks");
    assert_eq!(small, cut_every_64, "both cut the prompt every 64 tokens");
}

/// llama.cpp runs encoder-only models without a memory, as encoders, so they
/// have no next-token row: scoring refuses them before any decode.
#[test]
fn a_model_without_memory_is_refused() {
    let Some(path) = std::env::var_os("XYBRID_ENCODER_GGUF") else {
        eprintln!(
            "Skipping test: set XYBRID_ENCODER_GGUF to an encoder-only GGUF, \
             such as bge-small-en-v1.5"
        );
        return;
    };
    backend_init();
    let encoder = LlamaModel::load(Path::new(&path).to_str().expect("a UTF-8 path"), 0)
        .expect("load the encoder");
    let ctx = context(&encoder, 512, 512);
    let prompt = encoder
        .tokenize("Which colour is the sky?", true)
        .expect("tokenize");

    let refused = ctx
        .score_final_position(&prompt, &[0])
        .expect_err("an encoder has no row to score");
    assert!(
        matches!(&refused, LlamaError::InvalidInput(message) if message.contains("no memory")),
        "{refused}"
    );
}

/// Same-artifact parity: the C++ harness, linked against the llama.cpp
/// libraries this test links and given the same model, settings and prompt,
/// reads the same final row, with the prompt decoded whole, in several
/// chunks, or cut at the size a small context was asked for. Every logit and
/// the log-sum-exp must match bit for bit.
#[test]
fn the_harness_reads_the_same_row() {
    let Some(model) = model() else { return };
    let Some(harness) = std::env::var_os("XYBRID_LLAMA_HARNESS") else {
        eprintln!(
            "Skipping test: set XYBRID_LLAMA_HARNESS to the path \
             tools/conformance/llama_harness/build.sh prints"
        );
        return;
    };
    let model_path = staged_model_path().expect("the model is staged");
    let question = tokens(model, QUESTION);
    let long = question.repeat(5);
    let vocabulary = vocabulary(model);

    for (n_ctx, n_batch, prompt) in [
        (2048, 512, &question),
        (2048, 7, &question),
        (64, 512, &long),
    ] {
        let ours = context(model, n_ctx, n_batch)
            .score_final_position(prompt, &vocabulary)
            .expect("score the whole vocabulary");
        let (logsumexp, labels) =
            harness_row(&harness, &model_path, n_ctx, n_batch, prompt, &vocabulary);

        assert_eq!(
            logsumexp,
            ours.logsumexp.to_bits(),
            "n_ctx {n_ctx}, n_batch {n_batch}: log-sum-exp"
        );
        assert_eq!(labels.len(), ours.label_logits.len(), "one value per label");
        let differing = ours
            .label_logits
            .iter()
            .zip(&labels)
            .filter(|(logit, &bits)| logit.to_bits() != bits)
            .count();
        assert_eq!(
            differing,
            0,
            "n_ctx {n_ctx}, n_batch {n_batch}: {differing} of {} logits differ",
            labels.len()
        );
    }
}

/// Runs the harness on one request: the log-sum-exp and each label's logit,
/// as f32 bit patterns.
fn harness_row(
    harness: &OsStr,
    model_path: &Path,
    n_ctx: usize,
    n_batch: usize,
    prompt: &[i32],
    labels: &[i32],
) -> (u32, Vec<u32>) {
    let join = |ids: &[i32]| ids.iter().map(i32::to_string).collect::<Vec<_>>().join(" ");
    let mut child = Command::new(harness)
        .arg(model_path)
        .args([n_ctx.to_string(), n_batch.to_string(), "4".to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start the harness");
    child
        .stdin
        .take()
        .expect("the harness's stdin")
        .write_all(format!("{}\n{}\n", join(prompt), join(labels)).as_bytes())
        .expect("send the request");
    let output = child.wait_with_output().expect("run the harness");
    assert!(
        output.status.success(),
        "the harness failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let text = String::from_utf8(output.stdout).expect("UTF-8 output");
    let values = |line: Option<&str>, name: &str| -> Vec<u32> {
        let line = line.unwrap_or_else(|| panic!("no {name} line"));
        let rest = line
            .strip_prefix(name)
            .unwrap_or_else(|| panic!("expected {name}, got {line:.40}"));
        rest.split_whitespace()
            .map(|bits| bits.parse().expect("an f32 bit pattern"))
            .collect()
    };
    let mut lines = text.lines();
    let logsumexp = values(lines.next(), "logsumexp");
    let labels = values(lines.next(), "labels");
    assert_eq!(logsumexp.len(), 1, "one log-sum-exp");
    (logsumexp[0], labels)
}
