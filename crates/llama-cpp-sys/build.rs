//! Build script for `llama-cpp-sys`.
//!
//! Compiles llama.cpp + the first-party `wrapper.cpp` shim and emits the
//! link directives that resolve `libllama`, `libggml*`, and the platform
//! frameworks llama.cpp depends on.
//!
//! # Vendor location (epic open decision §1, resolved)
//!
//! Pinned upstream commit `b46812de78f8fbcb6cf0154947e8633ebc78d9ac`.
//! Source lives at workspace `vendor/llama-cpp/`, alongside
//! `vendor/mlx-apple/` and `vendor/ort-{ios,android}/`. `.gitmodules`
//! is updated to match.
//!
//! Source-lookup order:
//!
//!   1. `<workspace_root>/vendor/llama-cpp/` — the canonical in-tree
//!      location; populated by `git submodule update --init` on a fresh
//!      checkout.
//!   2. Pinned-commit clone into `$OUT_DIR/llama.cpp` — fallback for
//!      consumers that don't ship the submodule (crates.io tarball,
//!      Flutter pub cache git deps, container builds without submodule
//!      access).
//!
//! `wrapper.cpp` lives crate-local at `crates/llama-cpp-sys/wrapper.cpp`.
//!
//! # Gating
//!
//!   - If the `bindings` cargo feature is off, the script is a no-op.
//!     `cargo check --workspace` (no `llm-llamacpp` / no `bindings`) runs
//!     this branch and never invokes cmake.
//!   - If the feature is on, we resolve the source tree, run cmake, compile
//!     the wrapper shim via `cc`, and emit link directives.

use std::env;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Duration;

const LLAMA_CPP_REPO: &str = "https://github.com/ggml-org/llama.cpp";
// Pinned llama.cpp upstream — keep in sync with the git submodule SHA in
// .gitmodules / `git submodule status`. The fallback clone below uses this
// exact commit so consumers without submodule support (e.g. Flutter pub
// cache git deps, crates.io tarballs) get a reproducible build instead of
// upstream HEAD.
const LLAMA_CPP_COMMIT: &str = "b46812de78f8fbcb6cf0154947e8633ebc78d9ac";

fn main() {
    println!("cargo:rerun-if-changed=wrapper.cpp");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build.rs");
    // NDK lookup pulls from several env vars; declare them so flipping any
    // one (e.g. switching between two NDK installs) properly invalidates
    // the build-script cache. Declared unconditionally so the cache-busting
    // signal stays consistent across feature toggles.
    for var in [
        "ANDROID_NDK_HOME",
        "NDK_HOME",
        "ANDROID_HOME",
        "ANDROID_SDK_ROOT",
        "CC_aarch64-linux-android",
        "CC_aarch64_linux_android",
        "TARGET_CC",
        "CC",
        "XYBRID_LLAMA_CPP_VULKAN",
        "VULKAN_SDK",
        "LLAMA_CPP_SYS_WORKSPACE_ROOT",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    // Feature gate — keep the crate a no-op for default builds. Mirrors the
    // `mlx-c-sys` gating discipline so `cargo check --workspace` stays cheap
    // on CI runners without a C++ toolchain.
    if env::var_os("CARGO_FEATURE_BINDINGS").is_none() {
        return;
    }

    compile_llama_cpp();
}

/// Phase 5: invoke bindgen against `wrapper.h` to generate the FFI
/// surface that previously lived as a hand-written extern block in
/// `src/lib.rs::bindings`.
///
/// The allowlist is intentionally narrow: only our `llama_*_c` wrapper
/// symbols plus the upstream `llama_*` types/functions they reference.
/// Per brief §5: no `ggml_*` allowlist (nothing in `xybrid-llama` or
/// `xybrid-core` references a `ggml_*` symbol directly — wrapper.cpp
/// handles all ggml interop).
fn generate_bindings(llama_cpp_dir: &Path, out_dir: &Path, ndk_root: Option<&str>) {
    let include_dir = llama_cpp_dir.join("include");
    let ggml_include = llama_cpp_dir.join("ggml").join("include");
    let mtmd_include = llama_cpp_dir.join("tools").join("mtmd");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    let vision_enabled = env::var_os("CARGO_FEATURE_VISION").is_some();

    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir.display()))
        // ggml/include is needed only for transitive types pulled in by
        // llama.h; we still don't allowlist any `ggml_*` symbol below.
        .clang_arg(format!("-I{}", ggml_include.display()))
        // Match the upstream C ABI 1:1 — handle types are tiny and copying
        // them is the documented pattern.
        .derive_copy(true)
        .derive_default(false)
        .layout_tests(false)
        // Tightened allowlist per brief §5.5: a survey of the
        // broader `llama_.*` allowlist showed 230+ native `llama_*`
        // symbols from the upstream API, none of which are consumed
        // by `xybrid-llama` or `xybrid-core`. The wrapper exposes
        // exactly the `llama_.*_c` surface that the safe layer
        // calls into, so the narrower allowlist gives a 1:1 match
        // with the prior hand-written 26-symbol list.
        // Native llama.cpp constants / opaque types come along
        // transitively via `wrapper.h`'s reference to them in our
        // `_c` signatures.
        .allowlist_function("llama_.*_c")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*");

    if vision_enabled {
        builder = builder
            .clang_arg(format!("-I{}", mtmd_include.display()))
            .clang_arg("-DXYBRID_LLAMA_VISION")
            .allowlist_function("mtmd_.*_c")
            .allowlist_type("mtmd_.*");
    }

    // Cross-compile sysroot/target plumbing. bindgen drives libclang,
    // which resolves headers (`<stdio.h>` etc.) relative to its own
    // sysroot — wrong on cross-builds without explicit overrides.
    // Mirrors the pattern `mlx-c-sys/build.rs` uses for the macOS slice.
    // clang 21+ rejects Rust's `-sim` simulator triples (e.g.
    // `aarch64-apple-ios-sim`) verbatim ("error: version 'sim' in target triple
    // ... is invalid") and wants the canonical `<arch>-apple-<os>-simulator`
    // form. Older clang (the Xcode on the CI runners) accepted `-sim`, which is
    // why this only bites on newer Xcode locally. Translate any `-sim` triple
    // for the bindgen clang arg (`arm64` is clang's spelling of `aarch64`);
    // device and other triples (incl. the x86_64 iOS simulator, which Rust
    // spells `x86_64-apple-ios` with no `-sim` and clang parses fine) pass
    // through unchanged. The deployment version + codegen target are set
    // elsewhere (rustflags / cc), so the bindgen triple only needs to parse.
    let clang_target = match target.strip_suffix("-sim") {
        Some(base) => format!("{}-simulator", base.replace("aarch64", "arm64")),
        None => target.clone(),
    };
    builder = builder.clang_arg(format!("--target={clang_target}"));
    if target_os == "macos" || target_os == "ios" {
        let sdk = if target_os == "ios" {
            if target.contains("sim") {
                "iphonesimulator"
            } else {
                "iphoneos"
            }
        } else {
            "macosx"
        };
        if let Ok(out) = process::Command::new("xcrun")
            .args(["--show-sdk-path", "--sdk", sdk])
            .output()
        {
            if out.status.success() {
                let sdk_path = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !sdk_path.is_empty() {
                    builder = builder.clang_arg(format!("-isysroot{}", sdk_path));
                }
            }
        }
    } else if target_os == "android" {
        // Resolve sysroot from the detected NDK. Without this, libclang
        // can't find <stdio.h> on cross-builds.
        if let Some(ndk) = ndk_root {
            let host_tag = if cfg!(target_os = "macos") {
                "darwin-x86_64"
            } else if cfg!(target_os = "linux") {
                "linux-x86_64"
            } else {
                "windows-x86_64"
            };
            let sysroot = format!("{ndk}/toolchains/llvm/prebuilt/{host_tag}/sysroot");
            if Path::new(&sysroot).is_dir() {
                builder = builder.clang_arg(format!("--sysroot={}", sysroot));
            }
        }
    }

    let bindings = builder
        .generate()
        .expect("llama-cpp-sys: bindgen failed to generate bindings");

    let out_path = out_dir.join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .unwrap_or_else(|e| panic!("llama-cpp-sys: failed to write {}: {e}", out_path.display()));

    // Drift guard for the committed snapshot consumed under the
    // `committed-bindings` feature (the Bazel path, which cannot run bindgen).
    // The snapshot is generated WITH `vision` (the Bazel llama chain is always
    // vision-capable — it builds libmtmd), so only a vision run produces
    // comparable output. Warning (not error): bindgen output is not
    // guaranteed byte-stable across bindgen/toolchain versions.
    if vision_enabled {
        let committed = Path::new(&env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"))
            .join("src")
            .join("bindings.rs");
        let fresh = std::fs::read_to_string(&out_path).ok();
        if fresh.is_some() && fresh != std::fs::read_to_string(&committed).ok() {
            println!(
                "cargo:warning=llama-cpp-sys: committed src/bindings.rs is stale — \
                 copy {} over it to resync the `committed-bindings` (Bazel) path",
                out_path.display()
            );
        }
    }
}

/// Check if CMake is available in PATH
fn check_cmake_available() -> bool {
    process::Command::new("cmake")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Get platform-specific CMake installation instructions
fn cmake_install_instructions() -> &'static str {
    if cfg!(target_os = "macos") {
        "Install CMake:\n  brew install cmake"
    } else if cfg!(target_os = "linux") {
        "Install CMake:\n  Ubuntu/Debian: sudo apt install cmake\n  Fedora: sudo dnf install cmake\n  Arch: sudo pacman -S cmake"
    } else if cfg!(target_os = "windows") {
        "Install CMake:\n  choco install cmake\n  or download from https://cmake.org/download/"
    } else {
        "Install CMake from https://cmake.org/download/"
    }
}

/// Result of NDK detection with both found path and list of tried paths
struct NdkDetectionResult {
    /// The found NDK path, if any
    ndk_path: Option<String>,
    /// All paths that were tried during detection
    tried_paths: Vec<String>,
}

struct BuildContext {
    manifest_dir: PathBuf,
    out_dir: PathBuf,
    workspace_root: PathBuf,
    target: String,
    target_os: String,
    target_arch: String,
    android_ndk: Option<NdkDetectionResult>,
}

impl BuildContext {
    fn from_env() -> Self {
        let manifest_dir = PathBuf::from(
            env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set by cargo"),
        );
        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is always set by cargo"));
        let target = env::var("TARGET").expect("TARGET is always set by cargo");
        let target_os =
            env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is always set by cargo");
        let target_arch =
            env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "aarch64".to_string());
        let workspace_root = workspace_root(&manifest_dir);
        let android_ndk = if target_os == "android" {
            Some(find_android_ndk())
        } else {
            None
        };

        Self {
            manifest_dir,
            out_dir,
            workspace_root,
            target,
            target_os,
            target_arch,
            android_ndk,
        }
    }

    fn android_ndk_path(&self) -> Option<&str> {
        self.android_ndk
            .as_ref()
            .and_then(|result| result.ndk_path.as_deref())
    }
}

/// The framing rule shared by every build-failure banner.
const ERROR_RULE: &str = "=================================================================";

/// Print a framed `cargo:warning=` error banner and abort the build.
///
/// Centralizes the three build-failure surfaces (missing CMake, missing
/// NDK, clone failure) so the framing and the single `process::exit(1)`
/// policy live in one place instead of being hand-rolled at each site.
fn fatal(title: &str, body: &[String]) -> ! {
    println!("cargo:warning={ERROR_RULE}");
    println!("cargo:warning=ERROR: {title}");
    println!("cargo:warning={ERROR_RULE}");
    for line in body {
        println!("cargo:warning={line}");
    }
    println!("cargo:warning={ERROR_RULE}");
    process::exit(1);
}

/// Highest-versioned subdirectory of `dir` — used to pick the newest
/// installed NDK under `<sdk>/ndk`. Version components are compared
/// numerically (so `9.0` < `21.0`), not lexicographically. Returns `None`
/// when `dir` is absent or has no subdirectories.
fn latest_versioned_subdir(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max_by(|a, b| version_key(a).cmp(&version_key(b)))
}

/// Numeric version key for a path's final component, e.g.
/// `"21.4.7075529"` → `[21, 4, 7075529]`. Non-numeric components map to
/// `0`, so directories that aren't version-shaped sort lowest.
fn version_key(path: &Path) -> Vec<u64> {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|name| {
            name.split('.')
                .map(|c| c.parse::<u64>().unwrap_or(0))
                .collect()
        })
        .unwrap_or_default()
}

/// Find the Android NDK path from various sources
fn find_android_ndk() -> NdkDetectionResult {
    let mut tried_paths = Vec::new();

    // Helper to expand ~ in paths
    let expand_tilde = |path: String| -> String {
        if path.starts_with("~") {
            env::var("HOME")
                .map(|home| path.replacen("~", &home, 1))
                .unwrap_or(path)
        } else {
            path
        }
    };

    // 1. Try ANDROID_NDK_HOME and NDK_HOME first
    for var in ["ANDROID_NDK_HOME", "NDK_HOME"] {
        if let Ok(ndk) = env::var(var) {
            let expanded = expand_tilde(ndk);
            tried_paths.push(format!("${} = {}", var, expanded));
            if Path::new(&expanded).exists() {
                return NdkDetectionResult {
                    ndk_path: Some(expanded),
                    tried_paths,
                };
            }
        }
    }

    // 2. Try to extract from CC environment variable (set by cargo/cmake)
    // e.g., CC=/path/to/ndk/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang
    for var in [
        "CC_aarch64-linux-android",
        "CC_aarch64_linux_android",
        "TARGET_CC",
        "CC",
    ] {
        if let Ok(cc_path) = env::var(var) {
            // Extract NDK path: go up from .../toolchains/llvm/prebuilt/.../bin/clang
            if cc_path.contains("/ndk/") {
                if let Some(ndk_end) = cc_path.find("/toolchains/") {
                    let ndk = &cc_path[..ndk_end];
                    tried_paths.push(format!("${} -> extracted: {}", var, ndk));
                    if Path::new(ndk).exists() {
                        return NdkDetectionResult {
                            ndk_path: Some(ndk.to_string()),
                            tried_paths,
                        };
                    }
                }
            }
        }
    }

    // 3. Try ANDROID_HOME/ANDROID_SDK_ROOT with common NDK locations
    for sdk_var in ["ANDROID_HOME", "ANDROID_SDK_ROOT"] {
        if let Ok(sdk) = env::var(sdk_var) {
            let sdk_expanded = expand_tilde(sdk);
            let ndk_dir = Path::new(&sdk_expanded).join("ndk");
            let ndk_path_str = ndk_dir.to_string_lossy().to_string();
            tried_paths.push(format!("${}/ndk = {}", sdk_var, ndk_path_str));
            if let Some(latest) = latest_versioned_subdir(&ndk_dir) {
                return NdkDetectionResult {
                    ndk_path: Some(latest.to_string_lossy().to_string()),
                    tried_paths,
                };
            }
        }
    }

    // 4. Try common locations. `/opt/homebrew/share/android-ndk` is the
    //    symlink the `android-ndk` Homebrew cask installs on Apple
    //    Silicon Macs and points directly at the NDK root (no `ndk/`
    //    subdirectory). The trailing `**/{ndk}` discovery in the
    //    Android Studio install layout doesn't apply there.
    let home = env::var("HOME").unwrap_or_default();
    let common_locations = [
        format!("{}/Library/Android/sdk/ndk", home),
        format!("{}/Android/Sdk/ndk", home),
        "/opt/android-sdk/ndk".to_string(),
    ];
    let direct_locations = [
        "/opt/homebrew/share/android-ndk".to_string(),
        "/usr/local/share/android-ndk".to_string(),
    ];

    for location in &direct_locations {
        tried_paths.push(format!("brew cask: {}", location));
        let p = Path::new(location);
        if p.is_dir() && p.join("toolchains/llvm/prebuilt").is_dir() {
            return NdkDetectionResult {
                ndk_path: Some(location.clone()),
                tried_paths,
            };
        }
    }

    for location in &common_locations {
        tried_paths.push(format!("common: {}", location));
        let ndk_dir = Path::new(location);
        if let Some(latest) = latest_versioned_subdir(ndk_dir) {
            return NdkDetectionResult {
                ndk_path: Some(latest.to_string_lossy().to_string()),
                tried_paths,
            };
        }
    }

    NdkDetectionResult {
        ndk_path: None,
        tried_paths,
    }
}

/// Walk up from this crate's manifest dir to find the workspace root (the
/// directory containing the top-level `Cargo.toml` with `[workspace]`).
/// If `LLAMA_CPP_SYS_WORKSPACE_ROOT` is set, use it directly; this gives
/// package managers and unusual workspace layouts an explicit escape hatch.
/// Falls back to `..` if the marker can't be located — should never happen
/// in normal cargo invocations but kept defensive so the build script
/// errors loudly rather than panicking with a confusing path.
fn workspace_root(manifest_dir: &Path) -> PathBuf {
    if let Ok(root) = env::var("LLAMA_CPP_SYS_WORKSPACE_ROOT") {
        let explicit = PathBuf::from(root);
        if !explicit.as_os_str().is_empty() {
            return explicit;
        }
    }

    let mut dir = manifest_dir.to_path_buf();
    for _ in 0..6 {
        if let Some(parent) = dir.parent() {
            let candidate = parent.join("Cargo.toml");
            if candidate.exists() {
                if let Ok(content) = std::fs::read_to_string(&candidate) {
                    if content.lines().any(|line| line.trim() == "[workspace]") {
                        return parent.to_path_buf();
                    }
                }
            }
            dir = parent.to_path_buf();
        } else {
            break;
        }
    }
    manifest_dir.join("..").join("..")
}

fn compile_llama_cpp() {
    let ctx = BuildContext::from_env();
    let wrapper_path = ctx.manifest_dir.join("wrapper.cpp");
    let vision_enabled = env::var_os("CARGO_FEATURE_VISION").is_some();
    let vulkan_enabled = env_flag("XYBRID_LLAMA_CPP_VULKAN");

    // Windows is deliberately excluded. ggml builds its GLSL compiler
    // (`vulkan-shaders-gen`) as a nested CMake ExternalProject, and under
    // cargo's `target/<profile>/build/<crate>-<hash>/out/` prefix the paths
    // MSBuild's FileTracker generates for it exceed Windows' 260-character
    // MAX_PATH (`error FTK1011`, surfacing as a bogus "no working C
    // compiler"). Only ~17 characters remain for the repo root, so no
    // realistic checkout location builds. Re-enabling needs the Ninja
    // generator, which does not use FileTracker at all.
    if vulkan_enabled && ctx.target_os != "linux" {
        fatal(
            "llama.cpp Vulkan backend is only supported by xybrid on Linux.",
            &[
                format!(
                    "Target `{}` does not support `XYBRID_LLAMA_CPP_VULKAN=1`.",
                    ctx.target
                ),
                String::new(),
                "Unset `XYBRID_LLAMA_CPP_VULKAN` or set it to `0` for this target.".to_string(),
            ],
        );
    }

    // Source-lookup order (see header comment for rationale):
    //   1. workspace/vendor/llama-cpp (canonical, declared in `.gitmodules`)
    //   2. $OUT_DIR/llama.cpp pinned-commit clone (consumer fallback)
    let workspace_vendor = ctx.workspace_root.join("vendor").join("llama-cpp");

    let llama_cpp_dir = if workspace_vendor.join("CMakeLists.txt").exists() {
        workspace_vendor
    } else {
        clone_pinned_commit(&ctx.out_dir)
    };

    // Phase 5: generate the FFI surface from wrapper.h before the cmake
    // build runs. Bindgen needs the llama.cpp source for include paths,
    // so this lives after llama_cpp_dir is resolved. NDK detection
    // happens here too because Android cross-builds need libclang to
    // resolve `<stdio.h>` through the NDK sysroot.
    generate_bindings(&llama_cpp_dir, &ctx.out_dir, ctx.android_ndk_path());

    // rerun signals (apply to both the prebuilt fast path and the source
    // build, so they live before the branch below).
    println!("cargo:rerun-if-changed={}", llama_cpp_dir.display());
    println!("cargo:rerun-if-changed={}", wrapper_path.display());
    if vision_enabled {
        println!(
            "cargo:rerun-if-changed={}",
            llama_cpp_dir.join("tools/mtmd/mtmd.h").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            llama_cpp_dir.join("tools/mtmd/mtmd-helper.h").display()
        );
    }
    // The prebuilt fast paths and the publisher export hook are selected by
    // these env vars; declare them so toggling any of them re-runs the script.
    for var in [
        "XYBRID_NATIVES_PREBUILT_DIR",
        "XYBRID_NATIVES_EXPORT_DIR",
        "XYBRID_NATIVES_FORCE_SOURCE",
        "XYBRID_NATIVES_CACHE_DIR",
        "XYBRID_NATIVES_PKG",
        "XYBRID_NATIVES_TOKEN",
        "CARGO_NET_OFFLINE",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    // Republishing changes the manifest, which changes which slice (if any)
    // the download path resolves.
    println!(
        "cargo:rerun-if-changed={}",
        ctx.manifest_dir.join(NATIVES_MANIFEST_FILE).display()
    );

    // Resolve the install prefix (`dst`) one of three ways, in order:
    //   1. Staged slice: a complete prebuilt staged for this target+feature in
    //      `XYBRID_NATIVES_PREBUILT_DIR/<target>`. Explicit, so it wins — this
    //      is how our own CI jobs pin exactly what they pulled.
    //   2. Downloaded slice: a published slice resolved from
    //      `natives-manifest.txt` and fetched over HTTPS. This is what makes
    //      a plain `cargo build` cheap for an external consumer.
    //   3. Source path: today's cmake build, also the fallback whenever
    //      either fast path misses for any reason.
    let dst = match resolve_prebuilt(&ctx, vision_enabled, vulkan_enabled) {
        Some(prebuilt) => {
            println!(
                "cargo:warning=llama.cpp: using prebuilt natives for {} ({})",
                ctx.target,
                prebuilt.display()
            );
            prebuilt
        }
        None => resolve_downloaded(&ctx, vision_enabled, vulkan_enabled).unwrap_or_else(|| {
            build_from_source(&ctx, &llama_cpp_dir, vision_enabled, vulkan_enabled)
        }),
    };

    emit_link_and_wrapper(
        &ctx,
        &llama_cpp_dir,
        &wrapper_path,
        &dst,
        vision_enabled,
        vulkan_enabled,
    );
}

fn env_flag(name: &str) -> bool {
    match env::var(name) {
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        Err(env::VarError::NotPresent) => false,
        Ok(value) => fatal(
            &format!("Invalid value for `{name}`."),
            &[
                format!("Expected `0` or `1`, but received `{value}`."),
                String::new(),
                format!("Set `{name}=1` to enable it, or unset it to disable it."),
            ],
        ),
        Err(env::VarError::NotUnicode(_)) => fatal(
            &format!("Invalid value for `{name}`."),
            &["The value must be valid UTF-8 and either `0` or `1`.".to_string()],
        ),
    }
}

/// Compile llama.cpp (+ the mtmd vision libs when enabled) from source via
/// cmake and return the install prefix. This is the original build path; it is
/// taken whenever the prebuilt fast path misses. When
/// `XYBRID_NATIVES_EXPORT_DIR` is set, the produced archives + headers are also
/// copied out so a publisher job can upload them as a reusable slice.
fn build_from_source(
    ctx: &BuildContext,
    llama_cpp_dir: &Path,
    vision_enabled: bool,
    vulkan_enabled: bool,
) -> PathBuf {
    if !check_cmake_available() {
        fatal(
            "CMake not found!",
            &[
                "llama.cpp requires CMake to build.".to_string(),
                String::new(),
                cmake_install_instructions().to_string(),
                String::new(),
                "Or disable the llm-llamacpp feature:".to_string(),
                "  cargo build --no-default-features".to_string(),
            ],
        );
    }

    let mut metal_enabled = false;
    let mut ndk_path_used: Option<String> = None;

    let mut cmake_config = cmake::Config::new(llama_cpp_dir);
    cmake_config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define(
            "LLAMA_BUILD_TOOLS",
            if vision_enabled { "ON" } else { "OFF" },
        )
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_CURL", "OFF")
        // Default-ON upstream, but never let vendored cpp-httplib pick up a
        // host OpenSSL: we only ship the static archives, so tool-binary TLS
        // is dead weight — and on the arm64 macOS runners cross-building
        // x86_64 it finds the arm64 homebrew libcrypto and the vision tools
        // fail to link ("symbol(s) not found for architecture x86_64").
        .define("LLAMA_OPENSSL", "OFF")
        .define("GGML_OPENMP", "OFF")
        // Distribution baseline: never optimize for the build machine's CPU.
        // GGML_NATIVE defaults ON for non-cross builds and bakes
        // -march=native / -mcpu=native into the archives. The published
        // 2026-08-24 x86_64-linux vision slice picked up AVX-512 + AMX from a
        // Sapphire-Rapids runner and SIGILLed every consumer without them,
        // and the aarch64-linux slice carried Graviton SVE. Slices must run
        // on any reasonable consumer CPU; publishers and consumers do not
        // share hardware.
        .define("GGML_NATIVE", "OFF");

    // GGML_NATIVE=OFF alone is not a baseline: llama.cpp then defaults every
    // x86 SIMD family (SSE4.2/AVX/AVX2/FMA/F16C/BMI2) to OFF for cross builds
    // and scalar ggml is 3-5x slower. Pin x86-64-v3 (Haswell 2013 / Zen1
    // 2017) explicitly — every GitHub runner and effectively all consumer
    // x86_64 hardware clears it. Android x86_64 is excluded: its ABI only
    // guarantees SSE4.2, and configure_android keeps today's conservative
    // flags. Pre-Haswell hosts can XYBRID_NATIVES_FORCE_SOURCE=1.
    if ctx.target_arch == "x86_64" && ctx.target_os != "android" {
        cmake_config
            .define("GGML_SSE42", "ON")
            .define("GGML_AVX", "ON")
            .define("GGML_AVX2", "ON")
            .define("GGML_BMI2", "ON")
            .define("GGML_FMA", "ON")
            .define("GGML_F16C", "ON");
    }
    // aarch64 stays on the compiler's plain armv8-a default: adding dotprod
    // would break Cortex-A72 (Raspberry Pi 4) consumers with a runtime
    // SIGILL, the one failure mode a prebuilt slice must never have. Apple
    // arm64 is unaffected (the M1 baseline already includes dotprod).

    if ctx.target_os == "android" {
        ndk_path_used = configure_android(&mut cmake_config, ctx);
    } else if ctx.target_os == "macos" || ctx.target_os == "ios" {
        cmake_config
            .define("GGML_METAL", "ON")
            .define("GGML_VULKAN", "OFF")
            .define("GGML_ACCELERATE", "ON")
            .define("GGML_BLAS", "OFF");
        metal_enabled = true;
    } else if ctx.target_os == "linux" {
        cmake_config
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "OFF")
            .define("GGML_VULKAN", if vulkan_enabled { "ON" } else { "OFF" });
    } else if ctx.target_os == "windows" {
        // GGML_VULKAN stays OFF here: `vulkan_enabled` is rejected above for
        // every non-Linux target, so this arm is only ever reached without it.
        cmake_config
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "OFF")
            .define("GGML_VULKAN", "OFF");

        // Force CMake Release build on Windows to match the cc crate's CRT choice.
        // The cc crate always emits /MD (release CRT) — it never emits /MDd, even in
        // debug cargo builds. CMake defaults to Debug (/MDd) for `cargo test`, creating
        // a CRT mismatch (LNK2038). Forcing Release ensures both CMake and cc use /MD.
        cmake_config.profile("Release");
    }

    println!(
        "cargo:warning=llama.cpp build: target={}, metal={}, vulkan={}, ndk={}",
        ctx.target,
        if metal_enabled { "yes" } else { "no" },
        if vulkan_enabled { "yes" } else { "no" },
        ndk_path_used.as_deref().unwrap_or("N/A")
    );

    let dst = cmake_config.build();
    export_prebuilt(ctx, &dst);
    dst
}

/// Emit the link directives and compile the first-party `wrapper.cpp` shim
/// against `dst`. Shared by both the prebuilt and source paths so the linked
/// surface is identical regardless of how `dst` was produced.
fn emit_link_and_wrapper(
    ctx: &BuildContext,
    llama_cpp_dir: &Path,
    wrapper_path: &Path,
    dst: &Path,
    vision_enabled: bool,
    vulkan_enabled: bool,
) {
    // Metadata for direct dependents. Because this crate declares
    // `links = "llama"`, cargo forwards these as `DEP_LLAMA_ROOT` /
    // `DEP_LLAMA_SRC`. `whisper-cpp-sys` consumes both: `SRC` for the
    // `ggml/include` headers it compiles whisper.cpp against, `ROOT` for the
    // ggml archives it must re-emit AFTER `-lwhisper` (GNU ld resolves static
    // archives left to right, and cargo emits a dependency's flags first).
    // This is what keeps exactly one ggml in the binary.
    println!("cargo:root={}", dst.display());
    println!("cargo:src={}", llama_cpp_dir.display());
    // The resolved NDK, so a dependent's bindgen can point libclang at the
    // same sysroot. Without it, a cross-build's libclang resolves headers
    // against the HOST sysroot and fails on `<stdio.h>`. Emitted only when the
    // detection above actually found one, so its absence is meaningful.
    if let Some(ndk) = ctx.android_ndk_path() {
        println!("cargo:ndk={ndk}");
    }

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-search=native={}", dst.display());

    if vision_enabled {
        println!("cargo:rustc-link-lib=static=mtmd");
    }
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    if vulkan_enabled {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
        emit_vulkan_sdk_link_search(&ctx.target_os);
    }

    // Build our C++ wrapper (C++17 required by llama.cpp headers)
    // Note: The cc crate always uses /MD (release CRT) on MSVC — it never emits /MDd.
    // CMake is forced to Release on Windows above to match (see LNK2038 comment).
    let mut wrapper_build = cc::Build::new();
    wrapper_build
        .cpp(true)
        .std("c++17")
        .file(wrapper_path)
        .include(llama_cpp_dir.join("include"))
        .include(llama_cpp_dir.join("ggml/include"));
    if vision_enabled {
        wrapper_build
            .include(llama_cpp_dir.join("tools/mtmd"))
            .define("XYBRID_LLAMA_VISION", None);
    }
    wrapper_build.include(dst.join("include"));

    // Windows MSVC CRT: Do NOT call static_crt() — let the cc crate auto-detect from
    // CARGO_CFG_TARGET_FEATURE. When crt-static is set (CLI via RUSTFLAGS), cc uses /MT.
    // When not set (Flutter cdylib default), cc uses /MD. This keeps wrapper in sync with
    // both llama.cpp (CMake) and esaxx-rs automatically.

    wrapper_build.compile("llama_wrapper");

    if ctx.target_os == "android" {
        println!("cargo:rustc-link-lib=c++_shared");
        println!("cargo:rustc-link-lib=log");
    } else if ctx.target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
        if vulkan_enabled {
            println!("cargo:rustc-link-lib=vulkan");
        }
    } else if ctx.target_os == "macos" || ctx.target_os == "ios" {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");

        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
}

fn emit_vulkan_sdk_link_search(target_os: &str) {
    let Some(sdk) = env::var_os("VULKAN_SDK") else {
        return;
    };
    let sdk = PathBuf::from(sdk);
    let dirs: &[&str] = if target_os == "windows" {
        &["Lib"]
    } else {
        &["lib"]
    };

    for dir in dirs {
        let path = sdk.join(dir);
        if path.is_dir() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
}

/// The static archives the link step requires for a target + feature set —
/// must stay in sync with the `rustc-link-lib=static=` directives in
/// [`emit_link_and_wrapper`]. Used to validate a prebuilt slice before
/// trusting it.
fn required_archives(target_os: &str, vision_enabled: bool, vulkan_enabled: bool) -> Vec<String> {
    // MSVC names static libs `<name>.lib` (no `lib` prefix); every other target
    // we build for is Unix-style `lib<name>.a`.
    let (prefix, suffix) = if target_os == "windows" {
        ("", ".lib")
    } else {
        ("lib", ".a")
    };
    let mut libs = vec![
        format!("{prefix}llama{suffix}"),
        format!("{prefix}ggml{suffix}"),
        format!("{prefix}ggml-base{suffix}"),
        format!("{prefix}ggml-cpu{suffix}"),
    ];
    if target_os == "macos" || target_os == "ios" {
        // Apple links ggml-metal unconditionally (see emit_link_and_wrapper).
        libs.push(format!("{prefix}ggml-metal{suffix}"));
    }
    if vulkan_enabled {
        libs.push(format!("{prefix}ggml-vulkan{suffix}"));
    }
    if vision_enabled {
        libs.push(format!("{prefix}mtmd{suffix}"));
    }
    libs
}

/// True if static archive `name` exists and is non-empty under any of the
/// link-search roots [`emit_link_and_wrapper`] emits (`<dir>/lib`,
/// `<dir>/lib64`, `<dir>`).
fn archive_present(dir: &Path, name: &str) -> bool {
    ["lib", "lib64", ""].iter().any(|sub| {
        std::fs::metadata(dir.join(sub).join(name)).is_ok_and(|m| m.is_file() && m.len() > 0)
    })
}

/// Fast path: if `XYBRID_NATIVES_PREBUILT_DIR` is set and holds a *complete*
/// install prefix for this exact target under `<dir>/<target-triple>`, return
/// it to be linked in place of a cmake build.
///
/// Returns `None` — falling through to a source build — on any miss: env unset,
/// the per-target slice absent, a required archive missing/empty, or no
/// `include/` dir. The fast path therefore never fails the build; a cold or
/// partial cache degrades silently to compiling from source. Keying by the
/// full target triple lets one base dir serve a multi-ABI build (the Android
/// build compiles every ABI from one cargo invocation, each running this
/// script for its own `TARGET`).
fn resolve_prebuilt(
    ctx: &BuildContext,
    vision_enabled: bool,
    vulkan_enabled: bool,
) -> Option<PathBuf> {
    let base = env::var_os("XYBRID_NATIVES_PREBUILT_DIR")?;
    let base = PathBuf::from(base);
    if base.as_os_str().is_empty() {
        return None;
    }
    let dir = base.join(&ctx.target);
    // A missing per-target slice dir is a normal cold miss (e.g. the other
    // Android ABIs when only arm64 was pulled) — degrade silently. The
    // per-archive warnings below are reserved for a *present but incomplete*
    // slice, which is the case actually worth surfacing.
    if !dir.is_dir() {
        return None;
    }

    for archive in required_archives(&ctx.target_os, vision_enabled, vulkan_enabled) {
        if !archive_present(&dir, &archive) {
            println!(
                "cargo:warning=llama.cpp: prebuilt slice for {} incomplete (missing {archive}); compiling from source",
                ctx.target
            );
            return None;
        }
    }
    // wrapper.cpp includes `dst/include` for cmake-generated headers.
    if !dir.join("include").is_dir() {
        println!(
            "cargo:warning=llama.cpp: prebuilt slice for {} has no include/; compiling from source",
            ctx.target
        );
        return None;
    }
    Some(dir)
}

// =========================================================================
// Prebuilt natives — automatic download
// =========================================================================
//
// `XYBRID_NATIVES_PREBUILT_DIR` (above) requires the consumer to stage a
// slice themselves. That works for our own CI, but an external cargo user who
// merely runs `cargo build --features llm-llamacpp` still pays the full cmake
// compile. The functions below close that gap: build.rs resolves a published
// slice on its own, over plain HTTPS, with no `oras`, no env var, and no
// CMake on the machine.
//
// Selection is driven by `natives-manifest.txt`, a generated file committed
// next to this script and therefore present in the crates.io tarball. It maps
// (target triple, feature set) to a content digest published at
// `ghcr.io/xybrid-ai/llama-natives`, and it pins the source identity the
// slices were built from.
//
// Deliberately, the CONSUMER never recomputes the publisher's fingerprint
// (`tools/scripts/natives-fingerprint.sh`). That fingerprint folds in the
// *local* cmake/cc/NDK versions, which is right for build-cache parity and
// wrong for binary distribution: a user without cmake would compute a
// different key and always miss. Here the publisher declares the digest and
// the consumer selects by target + features + ABI constraints instead.
//
// Every failure mode — absent manifest, stale manifest, no row for this
// target, offline, HTTP error, digest mismatch, corrupt archive — returns
// `None` and falls through to the source build. The fast path can never fail
// a build, only fail to accelerate one.

/// Generated manifest mapping (target, feature-set) to a published slice.
/// Lives next to this build script so it ships inside the crate tarball.
const NATIVES_MANIFEST_FILE: &str = "natives-manifest.txt";

/// Hard cap on a downloaded slice so a malformed or hostile response cannot
/// balloon build-script memory. Real slices are tens of megabytes.
const MAX_SLICE_BYTES: u64 = 512 * 1024 * 1024;

/// Parsed `natives-manifest.txt`.
struct NativesManifest {
    /// OCI repository the slices are published to, e.g.
    /// `ghcr.io/xybrid-ai/llama-natives`.
    registry: String,
    /// Source identity the published slices were built from. All four must
    /// match the local files or the manifest is stale and is ignored.
    llama_commit: String,
    wrapper_cpp: String,
    wrapper_h: String,
    build_rs: String,
    slices: Vec<ManifestSlice>,
}

/// One published slice row.
struct ManifestSlice {
    target: String,
    /// `base` | `vision` | `vulkan` | `vision-vulkan` — mirrors the
    /// feature-set names used by `tools/scripts/natives-*.sh`.
    features: String,
    /// `sha256:<hex>` of the `native.tar.gz` layer blob.
    digest: String,
    /// Oldest glibc the archives link against (linux-gnu rows only).
    min_glibc: Option<String>,
    /// MSVC CRT flavour the archives were compiled with (windows rows only).
    crt: Option<String>,
}

/// The feature-set name used by the publisher scripts and the manifest.
fn feature_set_name(vision_enabled: bool, vulkan_enabled: bool) -> &'static str {
    match (vision_enabled, vulkan_enabled) {
        (true, true) => "vision-vulkan",
        (true, false) => "vision",
        (false, true) => "vulkan",
        (false, false) => "base",
    }
}

/// Read + parse `natives-manifest.txt`. Returns `None` when the file is
/// absent, unreadable, a future format version, or missing a required
/// header field — all of which mean "no fast path", never "fail the build".
///
/// Unknown keys and unknown `key=value` slice attributes are ignored so a
/// newer manifest stays readable by an older build script.
fn load_natives_manifest(manifest_dir: &Path) -> Option<NativesManifest> {
    let text = std::fs::read_to_string(manifest_dir.join(NATIVES_MANIFEST_FILE)).ok()?;

    let mut registry = None;
    let mut llama_commit = None;
    let mut wrapper_cpp = None;
    let mut wrapper_h = None;
    let mut build_rs = None;
    let mut slices = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let Some(key) = fields.next() else { continue };
        match key {
            // A bumped format version means this script cannot be trusted to
            // read the rows correctly — decline the fast path outright.
            "version" => {
                let format_version = fields.next();
                if format_version != Some("1") {
                    return None;
                }
            }
            "registry" => registry = fields.next().map(str::to_string),
            "llama_commit" => llama_commit = fields.next().map(str::to_string),
            "wrapper_cpp" => wrapper_cpp = fields.next().map(str::to_string),
            "wrapper_h" => wrapper_h = fields.next().map(str::to_string),
            "build_rs" => build_rs = fields.next().map(str::to_string),
            "slice" => {
                let (Some(target), Some(features), Some(digest)) =
                    (fields.next(), fields.next(), fields.next())
                else {
                    continue;
                };
                let mut min_glibc = None;
                let mut crt = None;
                for attr in fields {
                    if let Some(value) = attr.strip_prefix("min_glibc=") {
                        min_glibc = Some(value.to_string());
                    } else if let Some(value) = attr.strip_prefix("crt=") {
                        crt = Some(value.to_string());
                    }
                }
                slices.push(ManifestSlice {
                    target: target.to_string(),
                    features: features.to_string(),
                    digest: digest.to_string(),
                    min_glibc,
                    crt,
                });
            }
            _ => {}
        }
    }

    Some(NativesManifest {
        registry: registry?,
        llama_commit: llama_commit?,
        wrapper_cpp: wrapper_cpp?,
        wrapper_h: wrapper_h?,
        build_rs: build_rs?,
        slices,
    })
}

/// True when the published slices were built from exactly the sources present
/// on this machine.
///
/// This is the guard that makes a *local* edit safe: change `wrapper.cpp`,
/// `wrapper.h`, this script, or the pinned llama.cpp commit without
/// republishing, and the manifest no longer describes what you are building,
/// so the download is skipped and cmake runs. Each field is a plain hash of a
/// single file — no derived formula — so the publisher script and this
/// function cannot drift apart.
fn manifest_matches_source(manifest: &NativesManifest, manifest_dir: &Path) -> bool {
    manifest.llama_commit == LLAMA_CPP_COMMIT
        && sha256_file(&manifest_dir.join("wrapper.cpp")).as_deref() == Some(&manifest.wrapper_cpp)
        && sha256_file(&manifest_dir.join("wrapper.h")).as_deref() == Some(&manifest.wrapper_h)
        && sha256_file(&manifest_dir.join("build.rs")).as_deref() == Some(&manifest.build_rs)
}

/// Quiet completeness check: all archives the link step needs, plus headers.
/// Mirrors [`resolve_prebuilt`]'s validation without its warnings, which are
/// meant for a hand-staged directory rather than a cache entry.
fn slice_complete(dir: &Path, target_os: &str, vision_enabled: bool, vulkan_enabled: bool) -> bool {
    dir.join("include").is_dir()
        && required_archives(target_os, vision_enabled, vulkan_enabled)
            .iter()
            .all(|archive| archive_present(dir, archive))
}

/// Reject a slice whose ABI cannot link into *this* build even though the
/// triple matches. These are the two dimensions the target triple does not
/// capture; getting them wrong is a link error, not a graceful miss, so they
/// are checked before the download rather than after.
fn slice_abi_compatible(ctx: &BuildContext, slice: &ManifestSlice) -> bool {
    // MSVC: mixing a /MD (dynamic CRT) archive into a /MT (`crt-static`)
    // build is LNK2038. The publisher builds /MD; a crt-static consumer must
    // compile from source until a /MT slice exists.
    if ctx.target_os == "windows" {
        let crt_static = env::var("CARGO_CFG_TARGET_FEATURE")
            .map(|features| features.split(',').any(|f| f == "crt-static"))
            .unwrap_or(false);
        let wanted = if crt_static { "MT" } else { "MD" };
        if slice.crt.as_deref().unwrap_or("MD") != wanted {
            println!(
                "cargo:warning=llama.cpp: prebuilt slice for {} is CRT {} but this build needs {wanted}; compiling from source",
                ctx.target,
                slice.crt.as_deref().unwrap_or("MD")
            );
            return false;
        }
    }

    // glibc: static archives carry versioned symbol references, so archives
    // built against a NEWER glibc fail to link on an older host. Only
    // meaningful when a linux-gnu target is being built on a linux host —
    // elsewhere `ldd` is absent and the check silently passes.
    let host_glibc = (ctx.target_os == "linux" && cfg!(target_os = "linux"))
        .then(host_glibc_version)
        .flatten();
    if let (Some(min), Some(host)) = (slice.min_glibc.as_deref(), host_glibc.as_deref()) {
        if version_lt(host, min) {
            println!(
                "cargo:warning=llama.cpp: prebuilt slice for {} needs glibc >= {min} (host has {host}); compiling from source",
                ctx.target
            );
            return false;
        }
    }

    true
}

/// Host glibc version parsed from `ldd --version`'s first line, whose last
/// whitespace-separated token is the version (e.g. `... GLIBC 2.39-...) 2.39`).
/// `None` on any non-glibc or non-Linux host.
fn host_glibc_version() -> Option<String> {
    let output = process::Command::new("ldd")
        .arg("--version")
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let token = stdout.lines().next()?.split_whitespace().last()?;
    token
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit())
        .then(|| token.to_string())
}

/// Dotted-numeric `<` comparison (`2.35 < 2.39`). Non-numeric components
/// compare as 0, which is fine for the glibc `major.minor` strings this sees.
fn version_lt(lhs: &str, rhs: &str) -> bool {
    let parse = |v: &str| -> Vec<u64> {
        v.split(['.', '-'])
            .map(|part| part.parse::<u64>().unwrap_or(0))
            .collect()
    };
    let (lhs, rhs) = (parse(lhs), parse(rhs));
    for index in 0..lhs.len().max(rhs.len()) {
        let (l, r) = (
            lhs.get(index).copied().unwrap_or(0),
            rhs.get(index).copied().unwrap_or(0),
        );
        if l != r {
            return l < r;
        }
    }
    false
}

/// Where downloaded slices are unpacked. Keyed by content digest, so entries
/// are immutable and shared across every crate, target, and profile that
/// resolves the same slice.
///
/// `$CARGO_HOME` is preferred so the cache survives `cargo clean`; `$OUT_DIR`
/// is the last resort and merely scopes the cache to one build directory.
fn natives_cache_dir(ctx: &BuildContext) -> PathBuf {
    let dir_from = |var: &str| {
        env::var_os(var)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    if let Some(explicit) = dir_from("XYBRID_NATIVES_CACHE_DIR") {
        return explicit;
    }
    if let Some(cargo_home) = dir_from("CARGO_HOME") {
        return cargo_home.join("xybrid-natives");
    }
    ctx.out_dir.join("xybrid-natives")
}

/// Fast path #2: resolve a published slice for this target + feature set,
/// downloading it if it is not already cached, and return the install prefix
/// to link. `None` on any miss — the caller compiles from source.
fn resolve_downloaded(
    ctx: &BuildContext,
    vision_enabled: bool,
    vulkan_enabled: bool,
) -> Option<PathBuf> {
    if env_flag("XYBRID_NATIVES_FORCE_SOURCE") {
        return None;
    }
    // Respect the user's global "no network during builds" switch rather than
    // hanging on a connect timeout in a sandboxed or air-gapped build.
    if env::var("CARGO_NET_OFFLINE").is_ok_and(|value| value == "true") {
        return None;
    }

    let manifest = load_natives_manifest(&ctx.manifest_dir)?;
    if !manifest_matches_source(&manifest, &ctx.manifest_dir) {
        println!(
            "cargo:warning=llama.cpp: natives manifest does not match the local sources (edited wrapper/build.rs, or a bumped llama.cpp pin); compiling from source"
        );
        return None;
    }

    let features = feature_set_name(vision_enabled, vulkan_enabled);
    // A target with no published row is the common case for exotic triples —
    // stay silent, this is not a problem worth a warning.
    let slice = manifest
        .slices
        .iter()
        .find(|slice| slice.target == ctx.target && slice.features == features)?;
    if !slice_abi_compatible(ctx, slice) {
        return None;
    }

    let hex = slice.digest.strip_prefix("sha256:")?;
    let cache_root = natives_cache_dir(ctx);
    let dir = cache_root.join(hex);

    if slice_complete(&dir, &ctx.target_os, vision_enabled, vulkan_enabled) {
        println!(
            "cargo:warning=llama.cpp: using cached prebuilt natives for {} ({features})",
            ctx.target
        );
        return Some(dir);
    }

    let registry = env::var("XYBRID_NATIVES_PKG").unwrap_or_else(|_| manifest.registry.clone());
    println!(
        "cargo:warning=llama.cpp: downloading prebuilt natives for {} ({features}) from {registry} — set XYBRID_NATIVES_FORCE_SOURCE=1 to build from source instead",
        ctx.target
    );

    let bytes = fetch_slice_blob(&registry, &slice.digest)?;
    let actual = sha256_hex(&bytes);
    if actual != hex {
        println!(
            "cargo:warning=llama.cpp: prebuilt slice digest mismatch (expected {hex}, got {actual}); compiling from source"
        );
        return None;
    }
    unpack_slice(&bytes, &cache_root, &dir)?;

    if !slice_complete(&dir, &ctx.target_os, vision_enabled, vulkan_enabled) {
        println!(
            "cargo:warning=llama.cpp: downloaded slice for {} is incomplete; compiling from source",
            ctx.target
        );
        return None;
    }
    println!(
        "cargo:warning=llama.cpp: linked prebuilt natives for {} ({features}); skipped the cmake build",
        ctx.target
    );
    Some(dir)
}

/// Download one layer blob by digest from an OCI registry, anonymously.
///
/// Two requests: a pull-scoped token, then the content-addressed blob. Going
/// straight to the blob by digest skips the tag/manifest lookup entirely,
/// which is also what makes the download tamper-evident — the digest is
/// pinned in the committed manifest and verified by the caller.
fn fetch_slice_blob(registry: &str, digest: &str) -> Option<Vec<u8>> {
    use std::io::Read;

    let (host, repository) = registry.split_once('/')?;
    let url = format!("https://{host}/v2/{repository}/blobs/{digest}");
    let token = registry_pull_token(host, repository);

    // Explicit timeouts: ureq applies none by default, and a build script that
    // hangs on a wedged connection is far worse than one that gives up and
    // compiles. The read timeout is generous because slices are tens of MB.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(600))
        .build();

    // Two attempts: losing a 20-minute cmake compile to one transient blip is
    // a bad trade, and a genuinely dead network costs only the connect timeout
    // twice before falling through.
    let mut last_error = None;
    for _ in 0..2 {
        let mut request = agent.get(&url).set("Accept", "application/octet-stream");
        if let Some(token) = &token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        let response = match request.call() {
            Ok(response) => response,
            Err(e) => {
                last_error = Some(e.to_string());
                continue;
            }
        };
        let mut buffer = Vec::new();
        match response
            .into_reader()
            .take(MAX_SLICE_BYTES)
            .read_to_end(&mut buffer)
        {
            Ok(_) => return Some(buffer),
            Err(e) => last_error = Some(e.to_string()),
        }
    }

    println!(
        "cargo:warning=llama.cpp: prebuilt natives download failed ({}); compiling from source",
        last_error.as_deref().unwrap_or("unknown error")
    );
    None
}

/// Pull token for the slice registry.
///
/// `XYBRID_NATIVES_TOKEN` wins when set — that is the escape hatch for a
/// PRIVATE mirror (or for our own package before its visibility is flipped).
/// Otherwise an anonymous pull token is requested, which only a public
/// repository will issue; `None` degrades to an unauthenticated request.
fn registry_pull_token(host: &str, repository: &str) -> Option<String> {
    if let Some(token) = env::var("XYBRID_NATIVES_TOKEN")
        .ok()
        .filter(|token| !token.is_empty())
    {
        return Some(token);
    }
    let url = format!("https://{host}/token?scope=repository:{repository}:pull&service={host}");
    let body = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(30))
        .build()
        .get(&url)
        .call()
        .ok()?
        .into_string()
        .ok()?;
    json_string_field(&body, "token").or_else(|| json_string_field(&body, "access_token"))
}

/// Minimal `{"key":"value"}` extractor, so the build script needs no JSON
/// dependency. Registry tokens are JWT/base64url text with no escapes; a
/// value containing a backslash yields `None` rather than being mis-decoded.
fn json_string_field(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let rest = &body[body.find(&needle)? + needle.len()..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let value = &rest[..rest.find('"')?];
    (!value.is_empty() && !value.contains('\\')).then(|| value.to_string())
}

/// Unpack `native.tar.gz` bytes into `dst` atomically: extract to a
/// process-private temp dir, then rename into place. A concurrent build that
/// wins the race leaves a complete `dst`, which we adopt.
///
/// `tar::Archive::unpack` rejects absolute and `..` paths, so a hostile
/// archive cannot escape the cache directory.
fn unpack_slice(bytes: &[u8], cache_root: &Path, dst: &Path) -> Option<()> {
    std::fs::create_dir_all(cache_root).ok()?;
    let tmp = cache_root.join(format!(".tmp-{}", process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).ok()?;

    let unpacked = tar::Archive::new(flate2::read::GzDecoder::new(bytes)).unpack(&tmp);
    if let Err(e) = unpacked {
        println!("cargo:warning=llama.cpp: prebuilt natives archive is corrupt: {e}");
        let _ = std::fs::remove_dir_all(&tmp);
        return None;
    }

    match std::fs::rename(&tmp, dst) {
        Ok(()) => Some(()),
        Err(_) => {
            // Either another build populated `dst` first (fine — content is
            // digest-addressed, so it is the same bytes) or the rename failed
            // for real, in which case `dst` is absent and we fall to source.
            let _ = std::fs::remove_dir_all(&tmp);
            dst.is_dir().then_some(())
        }
    }
}

/// Lowercase hex SHA-256 of a byte slice.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Lowercase hex SHA-256 of a file; `None` when it cannot be read.
fn sha256_file(path: &Path) -> Option<String> {
    std::fs::read(path).ok().map(|bytes| sha256_hex(&bytes))
}

/// Publisher hook: when `XYBRID_NATIVES_EXPORT_DIR` is set, copy the freshly
/// built install prefix (`lib/`, `lib64/`, `include/`) into
/// `<export>/<target-triple>/` so a CI job can tar + upload it as a reusable
/// slice that [`resolve_prebuilt`] later consumes. No-op when the env var is
/// unset (the normal build). Best-effort: a copy failure warns but does not
/// fail the build, which already succeeded.
fn export_prebuilt(ctx: &BuildContext, dst: &Path) {
    let Some(base) = env::var_os("XYBRID_NATIVES_EXPORT_DIR") else {
        return;
    };
    let base = PathBuf::from(base);
    if base.as_os_str().is_empty() {
        return;
    }
    let out = base.join(&ctx.target);
    let mut exported_any = false;
    let mut had_error = false;
    for sub in ["lib", "lib64", "include"] {
        let src = dst.join(sub);
        if src.is_dir() {
            if let Err(e) = copy_dir(&src, &out.join(sub)) {
                println!(
                    "cargo:warning=llama.cpp: failed to export {sub} for {}: {e}",
                    ctx.target
                );
                had_error = true;
            } else {
                exported_any = true;
            }
        }
    }
    if exported_any && !had_error {
        println!(
            "cargo:warning=llama.cpp: exported prebuilt slice for {} to {}",
            ctx.target,
            out.display()
        );
    }
}

/// Recursively copy a directory tree (portable; no extra deps). Used only by
/// the export hook in CI.
fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

fn configure_android(cmake_config: &mut cmake::Config, ctx: &BuildContext) -> Option<String> {
    cmake_config
        .define("GGML_NATIVE", "OFF")
        .define("GGML_METAL", "OFF")
        .define("GGML_CUDA", "OFF")
        .define("GGML_VULKAN", "OFF")
        .define("GGML_CPU_HBM", "OFF")
        // Disable llamafile SGEMM — its FP16 NEON intrinsics (vld1q_f16) require
        // armv8.2-a+fp16 which the NDK doesn't enable by default.
        .define("GGML_LLAMAFILE", "OFF");

    let ndk_result = ctx
        .android_ndk
        .as_ref()
        .expect("android target should resolve NDK detection once");

    if let Some(ref ndk) = ndk_result.ndk_path {
        println!("cargo:warning=Android NDK detected: {}", ndk);

        let toolchain_file = format!("{}/build/cmake/android.toolchain.cmake", ndk);
        if Path::new(&toolchain_file).exists() {
            cmake_config.define("CMAKE_TOOLCHAIN_FILE", &toolchain_file);
        }

        let android_abi = match ctx.target_arch.as_str() {
            "aarch64" => "arm64-v8a",
            "arm" => "armeabi-v7a",
            "x86_64" => "x86_64",
            "x86" => "x86",
            _ => "arm64-v8a",
        };
        cmake_config.define("ANDROID_ABI", android_abi);

        // Enable ARMv8.2-A dotprod for arm64 Android targets.
        // The new llama.cpp (b541241+) relies on dotprod-optimized GEMM
        // microkernels in repack.cpp. Without this, quantized models
        // (Q4_K_M, Q5_K, etc.) fall back to generic NEON paths that are
        // 3-5x slower. dotprod is available on all Cortex-A76+ cores
        // (2019+): Snapdragon 855+, Tensor G1+, Dimensity 1000+.
        if android_abi == "arm64-v8a" {
            cmake_config.define("GGML_CPU_ARM_ARCH", "armv8.2-a+dotprod");
        }

        cmake_config.define("ANDROID_PLATFORM", "android-28");
        cmake_config.define("ANDROID_STL", "c++_shared");
        cmake_config.define("ANDROID_NDK", ndk);
        Some(ndk.clone())
    } else {
        let mut body = vec!["Paths tried:".to_string()];
        for path in &ndk_result.tried_paths {
            body.push(format!("  - {}", path));
        }
        body.extend([
            String::new(),
            "To fix this, set one of these environment variables:".to_string(),
            "  export ANDROID_NDK_HOME=/path/to/android-ndk".to_string(),
            "  export ANDROID_HOME=/path/to/android-sdk  (with ndk/ subdirectory)".to_string(),
            String::new(),
            "Or install Android Studio which sets up the NDK automatically.".to_string(),
        ]);
        fatal("Android NDK not found!", &body);
    }
}

/// Pinned-commit clone into $OUT_DIR. Consumer fallback for crates.io
/// tarball and Flutter pub cache git-dep scenarios. Init + fetch + checkout
/// at depth 1 — `git clone --depth 1` cannot target an arbitrary commit,
/// so we do it in three steps. Idempotent: re-using an existing OUT_DIR
/// clone is fine because the checked-out commit is pinned.
fn clone_pinned_commit(out_dir: &Path) -> PathBuf {
    let cloned = out_dir.join("llama.cpp");
    println!(
        "cargo:warning=llama.cpp not vendored in-tree, cloning {}@{} into OUT_DIR...",
        LLAMA_CPP_REPO, LLAMA_CPP_COMMIT
    );

    let dir_str = cloned.to_string_lossy().to_string();
    let run = |args: &[&str]| -> bool {
        process::Command::new("git")
            .args(args)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };

    let already_initialized =
        cloned.join(".git").exists() && cloned.join("CMakeLists.txt").exists();
    let needs_clone = !already_initialized;
    if needs_clone && cloned.exists() {
        let _ = std::fs::remove_dir_all(&cloned);
    }

    let ok = if needs_clone {
        std::fs::create_dir_all(&cloned).is_ok()
            && run(&["-C", &dir_str, "init", "-q"])
            && run(&["-C", &dir_str, "remote", "add", "origin", LLAMA_CPP_REPO])
            && run(&[
                "-C",
                &dir_str,
                "fetch",
                "--depth",
                "1",
                "origin",
                LLAMA_CPP_COMMIT,
            ])
            && run(&["-C", &dir_str, "checkout", "--detach", "FETCH_HEAD"])
    } else {
        true
    };

    if ok {
        println!(
            "cargo:warning=llama.cpp ready at {} ({})",
            cloned.display(),
            LLAMA_CPP_COMMIT
        );
        cloned
    } else {
        fatal(
            "Failed to clone llama.cpp!",
            &[
                format!("Expected location: {}", cloned.display()),
                String::new(),
                "To fix this manually, run:".to_string(),
                format!("  git clone {} {} && \\", LLAMA_CPP_REPO, cloned.display()),
                format!(
                    "    git -C {} checkout {}",
                    cloned.display(),
                    LLAMA_CPP_COMMIT
                ),
                String::new(),
                "Or disable the llm-llamacpp feature:".to_string(),
                "  cargo build --no-default-features".to_string(),
            ],
        );
    }
}
