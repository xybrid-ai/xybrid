//! The staged models the crate's own tests use.

use std::path::PathBuf;
use std::process::{Command, Output};

/// LiquidAI's d1-2B (artifact `d1-2b-q4km`) when staged, as for
/// `tests/d1_readout_smoke.rs`; `None` (with a note) otherwise.
///
/// # Panics
///
/// When `XYBRID_REQUIRE_MODELS` is set and the model is not staged.
pub(crate) fn staged_d1() -> Option<PathBuf> {
    let models = std::env::var_os("XYBRID_TEST_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../integration-tests/fixtures/models")
        });
    let path = models.join("d1-2b-q4km").join("d1-2B-Q4_K_M.gguf");
    if path.is_file() {
        return Some(path);
    }
    let reason = format!("{} is not staged", path.display());
    if std::env::var_os("XYBRID_REQUIRE_MODELS").is_some() {
        panic!("XYBRID_REQUIRE_MODELS is set but {reason}");
    }
    eprintln!("Skipping: {reason}");
    None
}

/// A GGUF other than d1 (a causal model with memory, such as stories260K)
/// from `XYBRID_OTHER_GGUF`; `None` (with a note) when it is not set.
///
/// # Panics
///
/// When `XYBRID_REQUIRE_MODELS` is set and `XYBRID_OTHER_GGUF` is not.
pub(crate) fn other_gguf() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("XYBRID_OTHER_GGUF") {
        return Some(PathBuf::from(path));
    }
    let reason = "XYBRID_OTHER_GGUF is not set (a causal GGUF that is not d1)";
    if std::env::var_os("XYBRID_REQUIRE_MODELS").is_some() {
        panic!("XYBRID_REQUIRE_MODELS is set but {reason}");
    }
    eprintln!("Skipping: {reason}");
    None
}

/// Runs the test `test` of this test binary in a new process, with `env`
/// set, and returns what it did.
pub(crate) fn run_test_in_child(test: &str, env: &[(&str, &str)]) -> Output {
    let exe = std::env::current_exe().expect("this test binary");
    let mut command = Command::new(exe);
    command.args(["--exact", test, "--nocapture", "--test-threads", "1"]);
    for (name, value) in env {
        command.env(name, value);
    }
    command.output().expect("run the test in a child process")
}
