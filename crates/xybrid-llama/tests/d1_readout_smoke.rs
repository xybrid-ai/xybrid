//! LiquidAI's d1-2B (artifact `d1-2b-q4km`): the GGUF facts llama.cpp
//! loads, and memory-less readouts on settings `cpu-t4-fa-off` (CPU, four
//! threads, Flash Attention off, a 4,096-token budget).
//!
//! Stage the pinned file (LiquidAI/d1-2B-GGUF at
//! e968f6575f0d4fad928312d7af15cad90fd365c0, sha256
//! adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b) as
//! `integration-tests/fixtures/models/d1-2b-q4km/d1-2B-Q4_K_M.gguf`, or
//! point `XYBRID_TEST_MODELS` at a directory holding
//! `d1-2b-q4km/d1-2B-Q4_K_M.gguf`, then run:
//!
//! ```sh
//! cargo test --release -p xybrid-llama --features bindings --test d1_readout_smoke -- --nocapture
//! ```
//!
//! One test also needs a second GGUF that is not LFM3 d1: point
//! `XYBRID_OTHER_GGUF` at one (stories260K: see `integration-tests/README.md`).
//!
//! A missing model skips each test, unless `XYBRID_REQUIRE_MODELS` is set,
//! which turns the skip into a failure, the second GGUF's included.
//! `--release` builds llama.cpp optimized; a debug build runs the same tests,
//! slowly.

#![cfg(feature = "bindings")]

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use xybrid_llama::{
    backend_init, LlamaContext, LlamaError, LlamaModel, MemorylessContext, MemorylessParams,
    TokenAttr,
};

const MODEL_ID: &str = "d1-2b-q4km";
const MODEL_FILE: &str = "d1-2B-Q4_K_M.gguf";

/// Settings `cpu-t4-fa-off`.
const BUDGET: u32 = 4096;
const THREADS: u32 = 4;

/// A choice question rendered by hand from the GGUF's `systemone`
/// template: a text state, text instructions, and options `billing` and
/// `shipping` with descriptions, whose codes are `A` and `B`. Nothing in it
/// needs the template's control-token escape.
const CHOICE_PROMPT: &str = "<|startoftext|><|im_start|>user\n\
The customer writes: my card was charged twice for one order.\n\n\n\
QUESTION:\n\
Which team should handle this ticket?\n\n\
Options:\n\
A Billing\n\
B Shipping\n\n\
Reply with the option code only.<|im_end|>\n<|im_start|>assistant\n<think></think>";

/// The staged GGUF, or `None` (with a note) when it is absent.
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
        "'{MODEL_ID}/{MODEL_FILE}' is not staged under {}",
        models.display()
    );
    if std::env::var_os("XYBRID_REQUIRE_MODELS").is_some() {
        panic!("XYBRID_REQUIRE_MODELS is set but {reason}");
    }
    eprintln!("Skipping test: {reason}");
    None
}

/// The second GGUF, from `XYBRID_OTHER_GGUF`, or `None` (with a note).
///
/// # Panics
///
/// When `XYBRID_REQUIRE_MODELS` is set and `XYBRID_OTHER_GGUF` is not.
fn other_gguf_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("XYBRID_OTHER_GGUF") {
        return Some(PathBuf::from(path));
    }
    let reason = "XYBRID_OTHER_GGUF is not set (a GGUF that is not LFM3 d1)";
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

fn readout_context(model: &LlamaModel, budget: u32) -> MemorylessContext {
    MemorylessContext::new(model, MemorylessParams::new(budget, THREADS, false))
        .expect("create a readout context")
}

/// The one token `text` is when tokenized as a label (no special tokens,
/// no special-token parsing).
fn label_token(model: &LlamaModel, text: &str) -> i32 {
    let tokens = model.tokenize(text, false).expect("tokenize a label");
    assert_eq!(tokens.len(), 1, "{text:?} is one token");
    tokens[0]
}

fn prompt_tokens(model: &LlamaModel) -> Vec<i32> {
    // add_special = false, parse_special = true: how llama.cpp tokenizes a
    // rendered decision prompt.
    model
        .tokenize_special(CHOICE_PROMPT, false)
        .expect("tokenize the prompt")
}

#[test]
fn the_gguf_carries_the_lfm3_d1_profile() {
    let Some(model) = model() else { return };
    let meta = |key: &str| model.meta_str(key).expect("readable metadata");
    assert_eq!(meta("general.architecture").as_deref(), Some("lfm3"));
    assert_eq!(meta("lfm3.decision.type").as_deref(), Some("lfm3-d1"));
    assert_eq!(meta("lfm3.attention.causal").as_deref(), Some("false"));
    for name in ["noul", "choice", "score"] {
        // The effective temperature: llama.cpp's std::to_string of the f32.
        assert_eq!(
            meta(&format!("lfm3.decision.temperature.{name}")).as_deref(),
            Some("1.337398")
        );
    }
    assert_eq!(meta("no.such.key"), None);
    assert_eq!(
        meta("tokenizer.ggml.tokens"),
        None,
        "llama.cpp keeps no array-valued key"
    );
    assert!(!model.has_encoder().unwrap());

    let template = model
        .named_chat_template("systemone")
        .expect("a readable template")
        .expect("the systemone template");
    assert_eq!(template.len(), 1380);
    assert!(template.starts_with("<|startoftext|><|im_start|>user\n"));
    assert!(template.ends_with("<|im_start|>assistant\\n' }}<think></think>"));
    assert_eq!(model.named_chat_template("no-such-template").unwrap(), None);
    assert!(matches!(
        model.named_chat_template("nul\0name"),
        Err(LlamaError::InvalidInput(_))
    ));

    assert_eq!(
        model.meta_count().unwrap(),
        37,
        "d1-2B has 45 keys, 8 of them arrays llama.cpp does not keep"
    );
    let keys = [
        "general.architecture",
        "lfm3.decision.type",
        "lfm3.attention.causal",
        "lfm3.decision.temperature.choice",
        "tokenizer.ggml.tokens",
        "no.such.key",
    ];
    let values = model.meta_values(&keys).expect("one bulk lookup");
    let expected: [Option<&[u8]>; 6] = [
        Some(b"lfm3"),
        Some(b"lfm3-d1"),
        Some(b"false"),
        Some(b"1.337398"),
        None,
        None,
    ];
    for ((key, value), expected) in keys.iter().zip(&values).zip(expected) {
        assert_eq!(value.as_deref(), expected, "{key}");
        assert_eq!(
            meta(key).map(String::into_bytes).as_deref(),
            expected,
            "{key}: the bulk and single lookups agree"
        );
    }
}

#[test]
fn control_spellings_are_the_vocabularys_control_tokens() {
    let Some(model) = model() else { return };
    let n_vocab = i32::try_from(model.n_vocab()).unwrap();
    assert_eq!(n_vocab, 128_000);
    let attr = |t: i32| {
        model
            .token_attr(t)
            .expect("readable")
            .expect("in the vocabulary")
    };
    let text = |t: i32| {
        model
            .token_text(t)
            .expect("readable")
            .expect("in the vocabulary")
    };
    let spellings: Vec<Vec<u8>> = (0..n_vocab)
        .filter(|&t| attr(t).contains(TokenAttr::CONTROL))
        .map(text)
        .filter(|text| text.len() >= 2)
        .collect();
    assert_eq!(spellings.len(), 124, "d1-2B's CONTROL spellings");

    let bos = model.token_bos();
    assert_eq!(bos, 124_894);
    assert_eq!(text(bos), b"<|startoftext|>");
    assert!(attr(bos).contains(TokenAttr::CONTROL));
    assert_eq!(model.token_attr(n_vocab).unwrap(), None);
    assert_eq!(model.token_attr(-1).unwrap(), None);
    assert_eq!(model.token_text(n_vocab).unwrap(), None);
    let a = label_token(model, "A");
    assert_eq!(text(a), b"A");
    assert!(attr(a).contains(TokenAttr::NORMAL));
}

#[test]
fn read_last_reads_the_label_logits_of_a_hand_rendered_prompt() {
    let Some(model) = model() else { return };
    let prompt = prompt_tokens(model);
    let labels: Vec<i32> = ["A", " A", "B", " B"]
        .iter()
        .map(|text| label_token(model, text))
        .collect();
    assert_ne!(
        labels[0], labels[1],
        "\"A\" and \" A\" are different tokens"
    );

    let mut ctx = readout_context(model, BUDGET);
    let facts = ctx.facts();
    assert_eq!(
        (facts.n_ctx, facts.n_batch, facts.n_ubatch, facts.n_seq_max),
        (BUDGET, BUDGET, BUDGET, 1)
    );
    assert!(!facts.has_memory && !facts.causal_attn);

    let logits = ctx.read_last(&prompt, &labels).expect("read the labels");
    println!(
        "d1-2b-q4km, cpu-t4-fa-off, {} prompt tokens; labels {labels:?} (A, \" A\", B, \" B\"):",
        prompt.len()
    );
    for (text, logit) in ["A", " A", "B", " B"].iter().zip(&logits) {
        println!(
            "  {text:>4}: {logit:>12.6} (bits {:#010x})",
            logit.to_bits()
        );
    }
    assert!(logits.iter().all(|logit| logit.is_finite()));
    assert!(
        logits[0] > logits[2],
        "a double charge goes to Billing (A), not Shipping (B)"
    );

    // Bit-identical again, on a fresh context, and after a refused call:
    // nothing survives a readout.
    assert_eq!(
        bits(&ctx.read_last(&prompt, &labels).unwrap()),
        bits(&logits)
    );
    assert!(matches!(
        ctx.read_last(&prompt, &[i32::MAX]),
        Err(LlamaError::InvalidInput(_))
    ));
    let other = model
        .tokenize_special("<|startoftext|><|im_start|>user\nHello", false)
        .unwrap();
    let _ = ctx.read_last(&other, &labels).expect("a different prompt");
    assert_eq!(
        bits(&ctx.read_last(&prompt, &labels).unwrap()),
        bits(&logits)
    );
    assert_eq!(
        bits(
            &readout_context(model, BUDGET)
                .read_last(&prompt, &labels)
                .unwrap()
        ),
        bits(&logits)
    );

    // Duplicates and order are kept, one value per selected id.
    let repeated = ctx
        .read_last(&prompt, &[labels[2], labels[0], labels[0]])
        .unwrap();
    assert_eq!(bits(&repeated), bits(&[logits[2], logits[0], logits[0]]));
}

#[test]
fn the_budget_is_an_exact_boundary() {
    let Some(model) = model() else { return };
    let prompt = prompt_tokens(model);
    let budget = u32::try_from(prompt.len()).unwrap();
    let mut ctx = readout_context(model, budget);
    let facts = ctx.facts();
    assert_eq!((facts.n_batch, facts.n_ubatch), (budget, budget));
    assert!(facts.n_ctx >= budget, "llama.cpp pads the context to 256");

    let a = label_token(model, "A");
    let exact = ctx
        .read_last(&prompt, &[a])
        .expect("a prompt of exactly the budget");
    let mut longer = prompt.clone();
    longer.push(prompt[prompt.len() - 1]);
    assert!(matches!(
        ctx.read_last(&longer, &[a]),
        Err(LlamaError::InvalidInput(_))
    ));
    assert_eq!(
        bits(&exact),
        bits(
            &readout_context(model, BUDGET)
                .read_last(&prompt, &[a])
                .unwrap()
        ),
        "the budget does not change the logits"
    );
}

/// A long readout stops soon after its cancel handle is used (a full
/// 2,048-token pass takes tens of seconds on four threads), and the context
/// reads like a fresh one once the request is cleared.
#[test]
fn a_long_readout_can_be_cancelled_from_another_thread() {
    let Some(model) = model() else { return };
    let prompt = prompt_tokens(model);
    let long: Vec<i32> = prompt.iter().copied().cycle().take(2048).collect();
    let a = label_token(model, "A");
    let mut ctx = readout_context(model, 2048);
    let cancel = ctx.cancel_handle();

    let started = Instant::now();
    let worker = std::thread::spawn(move || {
        let result = ctx.read_last(&long, &[a]);
        (ctx, result)
    });
    std::thread::sleep(Duration::from_millis(300));
    cancel.cancel();
    let (mut ctx, result) = worker.join().expect("the readout thread");
    let elapsed = started.elapsed();
    println!("a 2048-token readout cancelled after 300 ms returned after {elapsed:?}");
    assert!(matches!(result, Err(LlamaError::Cancelled)), "{result:?}");
    assert!(
        elapsed < Duration::from_secs(10),
        "the readout stopped after {elapsed:?}"
    );

    assert!(matches!(
        ctx.read_last(&prompt, &[a]),
        Err(LlamaError::Cancelled)
    ));
    ctx.clear_cancel();
    cancel.cancel(); // a handle of the cancelled request: nothing to stop now
    assert_eq!(
        bits(&ctx.read_last(&prompt, &[a]).unwrap()),
        bits(
            &readout_context(model, 2048)
                .read_last(&prompt, &[a])
                .unwrap()
        ),
        "a cancelled context reads like a fresh one"
    );
}

/// The causal scorer keeps refusing a context without memory.
#[test]
fn score_final_position_refuses_the_d1_model() {
    let Some(model) = model() else { return };
    let ctx = LlamaContext::new(model, 512, THREADS as usize, 512, false).expect("a context");
    assert!(
        !ctx.facts().unwrap().has_memory,
        "llama.cpp gives d1 no memory"
    );
    let refused = ctx
        .score_final_position(&prompt_tokens(model), &[label_token(model, "A")])
        .expect_err("no memory, no causal scoring");
    assert!(
        matches!(&refused, LlamaError::InvalidInput(message) if message.contains("no memory")),
        "{refused}"
    );
}

/// A model outside the profile gets no readout context. Needs another GGUF:
/// set `XYBRID_OTHER_GGUF` to one (an LFM3 chat model, say).
#[test]
fn a_model_outside_the_profile_is_refused() {
    let Some(path) = other_gguf_path() else {
        return;
    };
    backend_init();
    let other =
        LlamaModel::load(path.to_str().expect("a UTF-8 path"), 0).expect("load the other GGUF");
    let refused = MemorylessContext::new(&other, MemorylessParams::new(BUDGET, THREADS, false));
    assert!(
        matches!(&refused, Err(LlamaError::InvalidInput(message)) if message.contains("not an lfm3-d1")),
        "{:?}",
        refused.err()
    );
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}
