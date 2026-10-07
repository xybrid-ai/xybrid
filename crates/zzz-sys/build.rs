//! Build script for `xybrid-zzz-sys`.
//!
//! Locates, re-verifies, and statically links the private Kitten TTS 2 engine
//! slice (`libzzz_embed.a`) that `tools/scripts/zzz_pull.py` staged from the
//! zzz release pinned in `crates/zzz-sys/natives-manifest.json`.
//!
//! # The invariant
//!
//! **This crate never compiles zzz and never trusts an unverified path.**
//! The engine is consumed only through exactly the bytes pinned in the
//! committed manifest. `zzz_pull.py` is the single source of verification
//! truth — it happens to be the tool that staged the bytes — so this script
//! delegates verification instead of re-implementing archive, receipt and
//! release checks that would drift.
//!
//! # Resolution order
//!
//!   1. `XYBRID_ZZZ_PREBUILT_DIR` — a verified slice path (what
//!      `zzz_pull.py` prints). Re-verified via `--verify-only` on every
//!      invocation.
//!   2. Otherwise the tool's own private staging root. A cache hit is
//!      reverified on every hit; a cache miss attempts an authenticated
//!      fetch and fails loudly when no dedicated `ZZZ_GITHUB_TOKEN` is
//!      configured. Locally stored gh credentials are used only when
//!      `XYBRID_ZZZ_USE_GH_AUTH=1` (never honored in CI) — mirroring the
//!      tool's own explicit opt-in.
//!
//! Enabling `bindings` for an unsupported target, or without a staged or
//! fetchable pinned slice, fails the build: per the pinned contract, an
//! enabled zzz build must never silently degrade to no engine.
//!
//! # Gating
//!
//! If the `bindings` cargo feature is off, the script is a no-op: default
//! workspace builds (and CI lanes without staging) compile an empty crate.

use std::env;
use std::path::{Path, PathBuf};
use std::process;

/// The target/profile pair the committed manifest ships. Everything else
/// fails the enabled build rather than silently linking nothing.
const SUPPORTED_TARGETS: [&str; 3] = [
    "aarch64-apple-darwin",
    "aarch64-linux-android",
    "x86_64-unknown-linux-gnu",
];

fn main() {
    println!("cargo:rerun-if-env-changed=XYBRID_ZZZ_PREBUILT_DIR");
    println!("cargo:rerun-if-env-changed=XYBRID_ZZZ_USE_GH_AUTH");
    println!("cargo:rerun-if-env-changed=ZZZ_RELEASE_REPOSITORY");
    println!("cargo:rerun-if-env-changed=ZZZ_GITHUB_TOKEN");
    println!("cargo:rerun-if-changed=natives-manifest.json");
    println!("cargo:rerun-if-changed=build.rs");

    // Feature gate — keep the crate a no-op for default builds. Mirrors the
    // gating discipline of the other -sys crates so `cargo check --workspace`
    // on runners without a staged zzz slice stays green.
    if env::var_os("CARGO_FEATURE_BINDINGS").is_none() {
        return;
    }

    println!("cargo:rerun-if-changed=tools/scripts/zzz_pull.py");
    let slice = verified_slice();
    emit_link_directives(&slice);
}

/// Fail the enabled build on anything this crate cannot consume.
fn fail(message: &str) -> ! {
    println!("cargo:error=zzz-sys: {message}");
    process::exit(1);
}

/// Run `zzz_pull.py` for the build target and return the verified slice path.
///
/// The tool's contract is: print the verified slice directory on stdout, put
/// privacy-screened errors on stderr, exit nonzero on any failure. The slice
/// contains the pinned archive as the integrity anchor, the library, headers,
/// receipt and embedded metadata — all reverified before this step runs.
fn verified_slice() -> PathBuf {
    let target = env::var("TARGET").unwrap_or_else(|_| fail("cargo TARGET unset"));
    if !SUPPORTED_TARGETS.contains(&target.as_str()) {
        fail(&format!(
            "target '{target}' has no pinned zzz engine slice; the 'tts-zzz' \
             backend builds only on the pinned targets"
        ));
    }

    // Workspace root: this crate lives one level below `crates/`.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| fail("cannot determine the workspace root for zzz_pull.py"));

    let mut command = process::Command::new("python3");
    command
        .arg(root.join("tools/scripts/zzz_pull.py"))
        .arg("--target")
        .arg(&target);
    if env::var_os("XYBRID_ZZZ_PREBUILT_DIR").is_some() {
        command.arg("--verify-only");
    }
    if env::var_os("XYBRID_ZZZ_USE_GH_AUTH").is_some() {
        // Stored gh credentials are an explicit, local-only opt-in; the tool
        // itself refuses this flag in CI.
        command.arg("--use-gh-auth");
    }

    let output = match command
        .stdout(process::Stdio::piped())
        .stderr(process::Stdio::piped())
        .output()
    {
        Ok(output) if output.status.success() => output,
        Ok(output) => {
            let reason = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let missing = if reason.is_empty() { "" } else { ": " };
            fail(&format!(
                "the pinned zzz engine slice for '{target}' is missing or failed \
                 verification{missing}{reason} — stage it with tools/scripts/zzz_pull.py \
                 (see crates/zzz-sys/README.md)"
            ));
        }
        Err(err) => fail(&format!(
            "cannot run python3 to verify the pinned zzz engine slice: {err}"
        )),
    };

    let slice = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    if !slice.is_dir() {
        fail("zzz_pull.py verification did not return a usable slice directory");
    }
    slice
}

/// Emit the link directives for the verified slice.
///
/// Verification is already done; this is mechanical emission. The library
/// name is fixed by the pinned `library` path (`lib/libzzz_embed.a`), and the
/// required system libraries are read from the verified receipt
/// (`dependencies.system_libraries`), which `zzz_pull.py` has checked to
/// match the committed pin exactly.
fn emit_link_directives(slice: &Path) {
    println!(
        "cargo:rustc-link-search=native={}",
        slice.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=zzz_embed");

    let receipt_path = slice.join("RECEIPT.json");
    match std::fs::read_to_string(&receipt_path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
    {
        Some(receipt) => {
            let libraries = receipt
                .get("dependencies")
                .and_then(|d| d.get("system_libraries"))
                .and_then(|v| v.as_array())
                .map(|entries| {
                    entries
                        .iter()
                        .filter_map(|v| v.as_str())
                        .filter(|name| !name.is_empty())
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if libraries.is_empty() {
                fail("verified receipt carries no system library requirement");
            }
            for name in libraries {
                println!("cargo:rustc-link-lib={name}");
            }
        }
        None => fail("verified slice is missing a parsable receipt"),
    }
    // Not for compile-time use today — the adapter reads equivalent constants
    // from this crate. Emitted so a future header-consuming step (or the
    // Bazel lane, via DEP_ZZZ_EMBED_INCLUDE) finds the verified headers.
    println!("cargo:include={}", slice.join("include").display());
    println!("cargo:root={}", slice.display());
}
