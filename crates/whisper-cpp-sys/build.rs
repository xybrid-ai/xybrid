//! Build script for `xybrid-whisper-sys`.
//!
//! Compiles whisper.cpp's library — a single translation unit,
//! `src/whisper.cpp` — against the ggml that `xybrid-llama-sys` has already
//! built, and emits the link directives that resolve `libwhisper`.
//!
//! # The invariant
//!
//! **Exactly one ggml in the binary.** `llm-llamacpp` already links
//! `libggml{,-base,-cpu}.a` on every shipped platform; whisper.cpp must consume
//! that ggml and never configure its own bundled copy. Everything unusual about
//! this script follows from that one rule:
//!
//!   - `cc`, not `cmake`. Upstream's `src/CMakeLists.txt` builds `libwhisper`
//!     from exactly one source file plus headers, linking `ggml` and `Threads`.
//!     Nothing is generated at configure time, so `cc::Build` is sufficient —
//!     and running whisper's CMake would pull in `ggml/` as a subdirectory.
//!   - Headers and archives come from `DEP_LLAMA_SRC` / `DEP_LLAMA_ROOT`,
//!     forwarded by `xybrid-llama-sys` because it declares `links = "llama"`.
//!   - [`assert_no_bundled_ggml`] fails the build if a ggml source file ever
//!     reaches the compile. The compatibility this crate relies on is
//!     *incidental* (whisper v1.9.2 targets ggml 0.18.1 and merely happens to
//!     use nothing newer than the pinned llama's 0.11.0 at the library level),
//!     so a future whisper bump must fail loudly rather than silently
//!     duplicating ggml.
//!
//! # Vendor location
//!
//! Pinned upstream tag `v1.9.2` (commit
//! `306c88f4d1286aec1bf96e544632897886af5501`). Source lives at workspace
//! `vendor/whisper-cpp/`, alongside `vendor/llama-cpp/`. Lookup order mirrors
//! `xybrid-llama-sys`:
//!
//!   1. `<workspace_root>/vendor/whisper-cpp/` — canonical, populated by
//!      `git submodule update --init`.
//!   2. Pinned-commit clone into `$OUT_DIR/whisper.cpp` — fallback for
//!      consumers without submodule access (crates.io tarball, pub cache).
//!
//! # Gating
//!
//! If the `bindings` cargo feature is off, this script is a no-op, so
//! `cargo check --workspace` never invokes a C++ compiler.

use std::env;
use std::path::{Path, PathBuf};
use std::process;

const WHISPER_CPP_REPO: &str = "https://github.com/ggml-org/whisper.cpp";
/// Pinned whisper.cpp upstream — keep in sync with the git submodule SHA in
/// `.gitmodules` / `git submodule status`. This is tag `v1.9.2`.
const WHISPER_CPP_COMMIT: &str = "306c88f4d1286aec1bf96e544632897886af5501";

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=XYBRID_WHISPER_SYS_WORKSPACE_ROOT");

    // Feature gate — keep the crate a no-op for default builds, so a CI runner
    // without a C++ toolchain can still `cargo check --workspace`.
    if env::var_os("CARGO_FEATURE_BINDINGS").is_none() {
        return;
    }

    compile_whisper_cpp();
}

/// Walk up from this crate's manifest dir to the workspace root (the directory
/// whose `Cargo.toml` contains a `[workspace]` table). Mirrors
/// `xybrid-llama-sys`'s resolution, including the explicit env escape hatch for
/// package managers and unusual layouts.
fn workspace_root(manifest_dir: &Path) -> PathBuf {
    if let Ok(root) = env::var("XYBRID_WHISPER_SYS_WORKSPACE_ROOT") {
        let explicit = PathBuf::from(root);
        if !explicit.as_os_str().is_empty() {
            return explicit;
        }
    }

    let mut dir = manifest_dir.to_path_buf();
    for _ in 0..6 {
        let Some(parent) = dir.parent() else { break };
        let candidate = parent.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(content) = std::fs::read_to_string(&candidate) {
                if content.lines().any(|line| line.trim() == "[workspace]") {
                    return parent.to_path_buf();
                }
            }
        }
        dir = parent.to_path_buf();
    }
    manifest_dir.join("..").join("..")
}

/// Abort with a framed, actionable message. Build-script failures are otherwise
/// easy to lose in cargo's output.
fn fail(title: &str, details: &[String]) -> ! {
    const RULE: &str = "========================================================";
    println!("cargo:warning={RULE}");
    println!("cargo:warning=ERROR: {title}");
    println!("cargo:warning={RULE}");
    for line in details {
        println!("cargo:warning={line}");
    }
    println!("cargo:warning={RULE}");
    process::exit(1);
}

fn compile_whisper_cpp() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("cargo sets it"));
    let root = workspace_root(&manifest_dir);

    // `xybrid-llama-sys` forwards these because it declares `links = "llama"`.
    // Their absence means the llama build script did not run its `bindings`
    // path, so there is no ggml to link against — a configuration error, not
    // something to paper over by building our own.
    let llama_src = env::var("DEP_LLAMA_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            fail(
                "xybrid-whisper-sys: no ggml to build against",
                &[
                    "DEP_LLAMA_SRC is unset, which means xybrid-llama-sys did not run".into(),
                    "its native build. whisper.cpp is compiled against the ggml that".into(),
                    "llama.cpp provides — it never builds its own copy.".into(),
                    String::new(),
                    "Enable the `llm-llamacpp` feature alongside `asr-whispercpp`.".into(),
                ],
            )
        });
    let llama_root = env::var("DEP_LLAMA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            fail(
                "xybrid-whisper-sys: no ggml archives to link",
                &["DEP_LLAMA_ROOT is unset; see the DEP_LLAMA_SRC diagnostic above.".into()],
            )
        });

    // Source-lookup order (see header comment).
    let vendored = root.join("vendor").join("whisper-cpp");
    let whisper_dir = if vendored.join("CMakeLists.txt").exists() {
        vendored
    } else {
        clone_pinned_commit(&out_dir)
    };

    println!("cargo:rerun-if-changed={}", whisper_dir.display());

    let ggml_include = llama_src.join("ggml").join("include");
    if !ggml_include.join("ggml.h").exists() {
        fail(
            "xybrid-whisper-sys: ggml headers not found",
            &[
                format!("Looked in: {}", ggml_include.display()),
                "This path comes from xybrid-llama-sys's DEP_LLAMA_SRC metadata.".into(),
            ],
        );
    }

    generate_bindings(&whisper_dir, &ggml_include, &out_dir);

    let source = whisper_dir.join("src").join("whisper.cpp");
    assert_no_bundled_ggml(&whisper_dir, &source);

    // Upstream's CMake injects this as a private compile definition
    // (`WHISPER_VERSION="${PROJECT_VERSION}"`), and `whisper_version()` returns
    // it. Parse it out of the vendored CMakeLists rather than hardcoding, so it
    // can never drift from whatever the submodule is pinned to.
    let version = project_version(&whisper_dir);

    let mut build = cc::Build::new();
    build
        // whisper-arch.h includes C++ headers before whisper.cpp defines this
        // itself. Define it from the command line so Windows CRT headers expose
        // M_PI from their first inclusion. The empty replacement matches the
        // later source definition without producing a redefinition warning.
        .define("_USE_MATH_DEFINES", "")
        .define("WHISPER_VERSION", format!("\"{version}\"").as_str())
        .cpp(true)
        // Upstream pins C++11 for `whisper` with a "don't bump" comment in
        // src/CMakeLists.txt; C++17 is what the rest of our native surface
        // uses and is a superset for this translation unit.
        .std("c++17")
        .file(&source)
        .include(whisper_dir.join("include"))
        .include(whisper_dir.join("src"))
        .include(&ggml_include);

    // Upstream's CMake compiles the library with warnings-as-information only;
    // ours is a vendored third-party TU, so silence the noise rather than let
    // it drown real diagnostics from first-party code.
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        build.flag_if_supported("-Wno-unused-function");
        build.flag_if_supported("-Wno-unused-variable");
        build.flag_if_supported("-Wno-deprecated-declarations");
    }

    build.compile("whisper");

    // `cc::Build::compile` already emitted `-lwhisper` plus its search path.
    // Re-emit the ggml archives AFTER it: GNU ld resolves static archives left
    // to right, and cargo emits a dependency's link flags before its
    // dependents', so xybrid-llama-sys's `-lggml` lands too early to satisfy
    // whisper's references. Repeating an archive on the link line is free;
    // getting the order wrong is a link failure on Linux.
    println!(
        "cargo:rustc-link-search=native={}/lib",
        llama_root.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/lib64",
        llama_root.display()
    );
    println!("cargo:rustc-link-search=native={}", llama_root.display());
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    emit_platform_deps();
}

/// Re-emit the platform libraries and frameworks that ggml's own object files
/// reference.
///
/// Same ordering problem as the ggml archives above, one level down: ggml-cpu
/// calls into Accelerate (`vDSP_*`) on Apple platforms and libstdc++ elsewhere.
/// `xybrid-llama-sys` already emits these, but its flags land before ours, so
/// the ggml archives we re-emit would have nothing after them to resolve
/// against. Must stay in sync with `emit_link_and_wrapper` in
/// `crates/llama-cpp-sys/build.rs`.
fn emit_platform_deps() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "android" => {
            println!("cargo:rustc-link-lib=c++_shared");
            println!("cargo:rustc-link-lib=log");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=pthread");
        }
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=static=ggml-metal");
        }
        // Windows link deps are handled by llama.cpp's cmake export.
        _ => {}
    }
}

/// Read `project("whisper.cpp" VERSION x.y.z)` out of the vendored
/// `CMakeLists.txt`.
///
/// This is the value upstream compiles into the library as `WHISPER_VERSION`
/// and returns from `whisper_version()`. Deriving it from the source tree keeps
/// it correct across submodule bumps without a second thing to remember to
/// update.
fn project_version(whisper_dir: &Path) -> String {
    const FALLBACK: &str = "unknown";
    let Ok(text) = std::fs::read_to_string(whisper_dir.join("CMakeLists.txt")) else {
        return FALLBACK.to_string();
    };

    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("project(") else {
            continue;
        };
        let Some((_, after)) = rest.split_once("VERSION") else {
            continue;
        };
        let version = after.trim().trim_end_matches(')').trim();
        if !version.is_empty() && version.starts_with(|c: char| c.is_ascii_digit()) {
            return version.to_string();
        }
    }

    println!(
        "cargo:warning=xybrid-whisper-sys: could not parse the whisper.cpp project version from {}; \
         whisper_version() will report \"{FALLBACK}\"",
        whisper_dir.join("CMakeLists.txt").display()
    );
    FALLBACK.to_string()
}

/// Fail if anything under the whisper source tree's bundled `ggml/` directory
/// would be compiled, or if the file we do compile went missing.
///
/// The ggml compatibility this crate relies on is incidental, not contractual:
/// whisper v1.9.2 is written against ggml 0.18.1 and merely happens to use
/// nothing newer than the pinned llama's 0.11.0 at the library level. A future
/// whisper bump must break the build loudly instead of quietly configuring a
/// second ggml and shipping two copies.
fn assert_no_bundled_ggml(whisper_dir: &Path, source: &Path) {
    if !source.exists() {
        fail(
            "xybrid-whisper-sys: whisper.cpp source file missing",
            &[
                format!("Expected: {}", source.display()),
                "Upstream may have split the library across multiple translation".into(),
                "units. If so, this build script needs updating — do NOT switch to".into(),
                "whisper's CMake, which would configure its bundled ggml.".into(),
            ],
        );
    }

    let bundled_ggml = whisper_dir.join("ggml");
    if bundled_ggml.exists() {
        // Its presence on disk is expected (it is part of the upstream tree).
        // What must never happen is it reaching the compile — assert that the
        // one file we build lives outside it.
        let inside = source.starts_with(&bundled_ggml);
        if inside {
            fail(
                "xybrid-whisper-sys: would compile whisper's bundled ggml",
                &[
                    format!("Source: {}", source.display()),
                    format!("Bundled ggml: {}", bundled_ggml.display()),
                    "This crate links the ggml that xybrid-llama-sys builds. Two".into(),
                    "ggml copies in one binary is the failure mode the whole design".into(),
                    "exists to prevent.".into(),
                ],
            );
        }
    }
}

/// Generate the FFI surface from `wrapper.h`. whisper.cpp's public header is
/// already plain C behind `extern "C"`, so unlike the llama.cpp side there is
/// no first-party `_c` shim to declare — `wrapper.h` only sets the include
/// order up for bindgen.
fn generate_bindings(whisper_dir: &Path, ggml_include: &Path, out_dir: &Path) {
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", whisper_dir.join("include").display()))
        .clang_arg(format!("-I{}", ggml_include.display()))
        // Keep the surface to whisper's own world. `ggml_.*` is deliberately
        // not allowlisted: no consumer references a ggml symbol directly, and
        // allowlisting it would create a second Rust-side definition of types
        // xybrid-llama-sys already owns.
        .allowlist_function("whisper_.*")
        .allowlist_type("whisper_.*")
        .allowlist_var("WHISPER_.*")
        // This output is also committed for Bazel and compiled for every
        // supported target. Bindgen's layout tests bake in the generator
        // host's pointer width, so a 64-bit snapshot cannot compile for
        // Android armv7. The #[repr(C)] declarations themselves remain
        // target-correct because pointers and usize retain their native size.
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .unwrap_or_else(|e| fail("xybrid-whisper-sys: bindgen failed", &[format!("{e}")]));

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .unwrap_or_else(|e| {
            fail(
                "xybrid-whisper-sys: could not write bindings.rs",
                &[format!("{e}")],
            )
        });
}

/// Clone the pinned commit into `$OUT_DIR` for consumers that don't ship the
/// submodule. Mirrors `xybrid-llama-sys`'s fallback.
fn clone_pinned_commit(out_dir: &Path) -> PathBuf {
    let dir = out_dir.join("whisper.cpp");
    if dir.join("CMakeLists.txt").exists() {
        return dir;
    }

    let run = |args: &[&str]| -> bool {
        process::Command::new("git")
            .args(args)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };

    let dir_str = dir.to_string_lossy().to_string();
    let ok = std::fs::create_dir_all(&dir).is_ok()
        && run(&["-C", &dir_str, "init", "-q"])
        && run(&["-C", &dir_str, "remote", "add", "origin", WHISPER_CPP_REPO])
        && run(&[
            "-C",
            &dir_str,
            "fetch",
            "--depth",
            "1",
            "origin",
            WHISPER_CPP_COMMIT,
        ])
        && run(&["-C", &dir_str, "checkout", "-q", "FETCH_HEAD"]);

    if !ok {
        fail(
            "xybrid-whisper-sys: could not fetch whisper.cpp",
            &[
                format!("Tried to clone {WHISPER_CPP_REPO} at {WHISPER_CPP_COMMIT}"),
                format!("into {}", dir.display()),
                String::new(),
                "Populate the submodule instead:".into(),
                "  git submodule update --init vendor/whisper-cpp".into(),
            ],
        );
    }

    dir
}
