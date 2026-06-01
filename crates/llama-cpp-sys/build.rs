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
    builder = builder.clang_arg(format!("--target={target}"));
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

    let mut cmake_config = cmake::Config::new(&llama_cpp_dir);
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
        .define("GGML_OPENMP", "OFF");

    if ctx.target_os == "android" {
        ndk_path_used = configure_android(&mut cmake_config, &ctx);
    } else if ctx.target_os == "macos" || ctx.target_os == "ios" {
        cmake_config
            .define("GGML_METAL", "ON")
            .define("GGML_ACCELERATE", "ON")
            .define("GGML_BLAS", "OFF");
        metal_enabled = true;
    } else if ctx.target_os == "linux" {
        cmake_config
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "OFF");
    } else if ctx.target_os == "windows" {
        cmake_config
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "OFF");

        // Force CMake Release build on Windows to match the cc crate's CRT choice.
        // The cc crate always emits /MD (release CRT) — it never emits /MDd, even in
        // debug cargo builds. CMake defaults to Debug (/MDd) for `cargo test`, creating
        // a CRT mismatch (LNK2038). Forcing Release ensures both CMake and cc use /MD.
        cmake_config.profile("Release");
    }

    println!(
        "cargo:warning=llama.cpp build: target={}, metal={}, ndk={}",
        ctx.target,
        if metal_enabled { "yes" } else { "no" },
        ndk_path_used.as_deref().unwrap_or("N/A")
    );

    let dst = cmake_config.build();

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

    // Build our C++ wrapper (C++17 required by llama.cpp headers)
    // Note: The cc crate always uses /MD (release CRT) on MSVC — it never emits /MDd.
    // CMake is forced to Release on Windows above to match (see LNK2038 comment).
    let mut wrapper_build = cc::Build::new();
    wrapper_build
        .cpp(true)
        .std("c++17")
        .file(&wrapper_path)
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
    } else if ctx.target_os == "macos" || ctx.target_os == "ios" {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");

        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=static=ggml-metal");
    } else if ctx.target_os == "windows" {
        // Windows linking handled by CMake
    }
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
