//! Exceptions inside llama.cpp come back as errors, never as aborts.
//!
//! A C++ exception that reached Rust would abort the process ("Rust cannot
//! catch foreign exceptions"), so each case runs in a fresh copy of this
//! test binary: the child makes one call with an allocation failing under
//! it and prints the result, and the parent checks that the child exited
//! normally with the expected error.
//!
//! Most cases arm the shim's fault hook (`llama_xybrid_fail_next_c`), which
//! throws `std::bad_alloc` inside the named shim's exception boundary; on a
//! model or context with no native handle the hook fires before anything
//! is read, so they need no model. The others need the staged d1 model:
//! paths that only reach the shim on a real model, and two real allocation
//! failures inside llama.cpp, provoked with an address-space limit.

use crate::context::LlamaContext;
use crate::ffi;
use crate::memoryless::{MemorylessContext, MemorylessParams};
use crate::model::LlamaModel;
use crate::test_models::{run_test_in_child, staged_d1};

const CASE_ENV: &str = "XYBRID_FAULT_CASE";
const CHILD: &str = "fault_tests::fault_case";

/// Cases on handles with no native model or context: (case, the shim the
/// hook fails, the start of the error the call must return).
const STUB_CASES: &[(&str, &str, &str)] = &[
    ("meta_str", "llama_model_meta_val_str_c", "Err(Internal("),
    ("meta", "llama_model_meta_count_c", "Err(Internal("),
    (
        "named_chat_template",
        "llama_model_chat_template_named_c",
        "Err(Internal(",
    ),
    ("token_attr", "llama_vocab_get_attr_c", "Err(Internal("),
    ("token_text", "llama_vocab_get_text_c", "Err(Internal("),
    ("has_encoder", "llama_model_has_encoder_c", "Err(Internal("),
    ("tokenize", "llama_tokenize_c", "Err(TokenizationFailed)"),
    ("facts", "llama_context_facts_c", "Err(Internal("),
    (
        "score_final_position",
        "llama_kv_cache_seq_rm_c",
        "Err(Internal(",
    ),
    (
        "read_last",
        "llama_decode_c",
        "Err(DecodeFailed { code: -1000,",
    ),
];

/// Cases on the real d1 model.
const MODEL_CASES: &[(&str, &str)] = &[
    ("readout_context", "Err(ContextCreationFailed("),
    ("readout_facts", "Err(Internal("),
    ("read_last_logits", "Err(Internal("),
    ("read_last_decode", "Err(DecodeFailed { code: -1000,"),
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    ("meta_str_out_of_memory", "Err(Internal("),
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    ("named_chat_template_out_of_memory", "Err(Internal("),
];

/// Runs `case` in a child and returns its report line, failing the test
/// unless the child exited normally.
fn run_case(case: &str) -> String {
    let output = run_test_in_child(CHILD, &[(CASE_ENV, case)]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{case}: the child did not exit cleanly ({}); stderr: {stderr}",
        output.status
    );
    let prefix = format!("FAULT {case}: ");
    stdout
        .lines()
        .find_map(|line| line.split_once(&prefix).map(|(_, rest)| rest.to_string()))
        .unwrap_or_else(|| panic!("{case}: no report in {stdout}"))
}

#[test]
fn an_exception_in_any_guarded_shim_is_an_error() {
    for &(case, _, expected) in STUB_CASES {
        let report = run_case(case);
        assert!(report.starts_with(expected), "{case}: {report}");
    }
}

#[test]
fn allocation_failures_on_a_real_model_are_errors() {
    if staged_d1().is_none() {
        return;
    }
    for &(case, expected) in MODEL_CASES {
        let report = run_case(case);
        assert!(report.starts_with(expected), "{case}: {report}");
        if case == "read_last_decode" {
            assert!(
                report.contains("then Ok(["),
                "the context reads again after the failure: {report}"
            );
        }
    }
}

/// One case, in the child (returns at once in a normal test run).
#[test]
fn fault_case() {
    let Ok(case) = std::env::var(CASE_ENV) else {
        return;
    };
    let report = match STUB_CASES.iter().find(|(name, _, _)| *name == case) {
        Some(&(_, shim, _)) => stub_case(&case, shim),
        None => model_case(&case),
    };
    println!("\nFAULT {case}: {report}");
}

fn stub_case(case: &str, shim: &str) -> String {
    let model = LlamaModel::test_stub();
    ffi::fail_next_native_call(shim);
    match case {
        "meta_str" => format!("{:?}", model.meta_str("lfm3.decision.type")),
        "meta" => format!("{:?}", model.meta()),
        "named_chat_template" => format!("{:?}", model.named_chat_template("systemone")),
        "token_attr" => format!("{:?}", model.token_attr(0)),
        "token_text" => format!("{:?}", model.token_text(0)),
        "has_encoder" => format!("{:?}", model.has_encoder()),
        "tokenize" => format!("{:?}", model.tokenize_special("<|im_start|>user", false)),
        "facts" => format!("{:?}", LlamaContext::test_stub(&model).facts()),
        "score_final_position" => format!(
            "{:?}",
            LlamaContext::test_stub(&model).score_final_position(&[1, 2], &[0])
        ),
        "read_last" => format!(
            "{:?}",
            MemorylessContext::test_stub(&model, 8, 10).read_last(&[1, 2, 3], &[0])
        ),
        other => panic!("unknown case {other}"),
    }
}

fn model_case(case: &str) -> String {
    let path = staged_d1().expect("the parent checked the model is staged");
    crate::backend_init();
    let model = LlamaModel::load(path.to_str().expect("a UTF-8 path"), 0).expect("load d1-2B");
    let params = MemorylessParams::new(256, 4, false);
    let prompt = model
        .tokenize_special("<|startoftext|><|im_start|>user\nHello", false)
        .expect("tokenize");
    match case {
        "readout_context" => {
            ffi::fail_next_native_call("llama_new_readout_context_c");
            format!("{:?}", MemorylessContext::new(&model, params).map(|_| ()))
        }
        "readout_facts" => {
            ffi::fail_next_native_call("llama_context_facts_c");
            format!("{:?}", MemorylessContext::new(&model, params).map(|_| ()))
        }
        "read_last_logits" => {
            let mut ctx = MemorylessContext::new(&model, params).expect("a context");
            ffi::fail_next_native_call("llama_get_logits_ith_c");
            format!("{:?}", ctx.read_last(&prompt, &[0]))
        }
        "read_last_decode" => {
            let mut ctx = MemorylessContext::new(&model, params).expect("a context");
            ffi::fail_next_native_call("llama_decode_c");
            let failed = ctx.read_last(&prompt, &[0]);
            let again = ctx.read_last(&prompt, &[0]);
            format!("{failed:?}, then {again:?}")
        }
        #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
        "meta_str_out_of_memory" => {
            let key = "k".repeat(LARGE);
            out_of_address_space(|| format!("{:?}", model.meta_str(&key)))
        }
        #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
        "named_chat_template_out_of_memory" => {
            let name = "n".repeat(LARGE);
            out_of_address_space(|| format!("{:?}", model.named_chat_template(&name)))
        }
        other => panic!("unknown case {other}"),
    }
}

/// A key or name this long makes llama.cpp's temporary `std::string` a
/// separate mapping (glibc maps every allocation above 32 MiB on its own).
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
const LARGE: usize = 64 << 20;

/// Runs `call` with the address space capped so that the Rust side's copy
/// of a [`LARGE`] argument still fits but llama.cpp's own copy of it does
/// not: the allocation inside llama.cpp fails with `std::bad_alloc`.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn out_of_address_space(call: impl FnOnce() -> String) -> String {
    #[repr(C)]
    struct RLimit {
        cur: u64,
        max: u64,
    }
    extern "C" {
        fn setrlimit(resource: i32, limit: *const RLimit) -> i32;
    }
    const RLIMIT_AS: i32 = 9;
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let vm_size_kb: u64 = status
        .lines()
        .find_map(|line| line.strip_prefix("VmSize:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kb| kb.parse().ok())
        .expect("VmSize");
    // Room for the Rust copy (one LARGE allocation) and small allocations,
    // not for a second LARGE one.
    let cap = vm_size_kb * 1024 + LARGE as u64 + (32 << 20);
    let limit = RLimit {
        cur: cap,
        max: u64::MAX,
    };
    // SAFETY: setrlimit reads the struct during the call.
    assert_eq!(
        unsafe { setrlimit(RLIMIT_AS, &limit) },
        0,
        "cap the address space"
    );
    let report = call();
    let unlimited = RLimit {
        cur: u64::MAX,
        max: u64::MAX,
    };
    // SAFETY: as above.
    unsafe { setrlimit(RLIMIT_AS, &unlimited) };
    report
}
