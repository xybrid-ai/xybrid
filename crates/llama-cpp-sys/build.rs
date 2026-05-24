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

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();

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
            if ndk_dir.exists() {
                // Find the latest NDK version
                if let Ok(entries) = std::fs::read_dir(&ndk_dir) {
                    let mut versions: Vec<_> = entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().is_dir())
                        .map(|e| e.path())
                        .collect();
                    versions.sort();
                    if let Some(latest) = versions.last() {
                        return NdkDetectionResult {
                            ndk_path: Some(latest.to_string_lossy().to_string()),
                            tried_paths,
                        };
                    }
                }
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
        if ndk_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(ndk_dir) {
                let mut versions: Vec<_> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path())
                    .collect();
                versions.sort();
                if let Some(latest) = versions.last() {
                    return NdkDetectionResult {
                        ndk_path: Some(latest.to_string_lossy().to_string()),
                        tried_paths,
                    };
                }
            }
        }
    }

    NdkDetectionResult {
        ndk_path: None,
        tried_paths,
    }
}

/// Walk up from this crate's manifest dir to find the workspace root (the
/// directory containing the top-level `Cargo.toml` with `[workspace]`).
/// Falls back to `..` if the marker can't be located — should never happen
/// in normal cargo invocations but kept defensive so the build script
/// errors loudly rather than panicking with a confusing path.
fn workspace_root(manifest_dir: &Path) -> PathBuf {
    let mut dir = manifest_dir.to_path_buf();
    for _ in 0..6 {
        if let Some(parent) = dir.parent() {
            let candidate = parent.join("Cargo.toml");
            if candidate.exists() {
                if let Ok(content) = std::fs::read_to_string(&candidate) {
                    if content.contains("[workspace]") {
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
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let wrapper_path = manifest_dir.join("wrapper.cpp");

    let workspace = workspace_root(&manifest_dir);

    // Source-lookup order (see header comment for rationale):
    //   1. workspace/vendor/llama-cpp (canonical, declared in `.gitmodules`)
    //   2. $OUT_DIR/llama.cpp pinned-commit clone (consumer fallback)
    let workspace_vendor = workspace.join("vendor").join("llama-cpp");

    let llama_cpp_dir = if workspace_vendor.join("CMakeLists.txt").exists() {
        workspace_vendor
    } else {
        clone_pinned_commit(&out_dir)
    };

    // Phase 5: generate the FFI surface from wrapper.h before the cmake
    // build runs. Bindgen needs the llama.cpp source for include paths,
    // so this lives after llama_cpp_dir is resolved. NDK detection
    // happens here too because Android cross-builds need libclang to
    // resolve `<stdio.h>` through the NDK sysroot.
    let target_os_early = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let ndk_for_bindgen = if target_os_early == "android" {
        find_android_ndk().ndk_path
    } else {
        None
    };
    generate_bindings(&llama_cpp_dir, &out_dir, ndk_for_bindgen.as_deref());

    if !check_cmake_available() {
        println!("cargo:warning=================================================================");
        println!("cargo:warning=ERROR: CMake not found!");
        println!("cargo:warning=================================================================");
        println!("cargo:warning=llama.cpp requires CMake to build.");
        println!("cargo:warning=");
        println!("cargo:warning={}", cmake_install_instructions());
        println!("cargo:warning=");
        println!("cargo:warning=Or disable the llm-llamacpp feature:");
        println!("cargo:warning=  cargo build --no-default-features");
        println!("cargo:warning=================================================================");
        process::exit(1);
    }

    let target = env::var("TARGET").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    let mut metal_enabled = false;
    let mut ndk_path_used: Option<String> = None;

    println!("cargo:rerun-if-changed={}", llama_cpp_dir.display());
    println!("cargo:rerun-if-changed={}", wrapper_path.display());

    let mut cmake_config = cmake::Config::new(&llama_cpp_dir);
    cmake_config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("GGML_OPENMP", "OFF");

    if target_os == "android" {
        cmake_config
            .define("GGML_NATIVE", "OFF")
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "OFF")
            .define("GGML_VULKAN", "OFF")
            .define("GGML_CPU_HBM", "OFF")
            // Disable llamafile SGEMM — its FP16 NEON intrinsics (vld1q_f16) require
            // armv8.2-a+fp16 which the NDK doesn't enable by default
            .define("GGML_LLAMAFILE", "OFF");

        let ndk_result = find_android_ndk();

        if let Some(ref ndk) = ndk_result.ndk_path {
            println!("cargo:warning=Android NDK detected: {}", ndk);
            ndk_path_used = Some(ndk.clone());

            let toolchain_file = format!("{}/build/cmake/android.toolchain.cmake", ndk);
            if Path::new(&toolchain_file).exists() {
                cmake_config.define("CMAKE_TOOLCHAIN_FILE", &toolchain_file);
            }

            let target_arch =
                env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "aarch64".to_string());
            let android_abi = match target_arch.as_str() {
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
        } else {
            println!(
                "cargo:warning================================================================="
            );
            println!("cargo:warning=ERROR: Android NDK not found!");
            println!(
                "cargo:warning================================================================="
            );
            println!("cargo:warning=Paths tried:");
            for path in &ndk_result.tried_paths {
                println!("cargo:warning=  - {}", path);
            }
            println!("cargo:warning=");
            println!("cargo:warning=To fix this, set one of these environment variables:");
            println!("cargo:warning=  export ANDROID_NDK_HOME=/path/to/android-ndk");
            println!("cargo:warning=  export ANDROID_HOME=/path/to/android-sdk  (with ndk/ subdirectory)");
            println!("cargo:warning=");
            println!(
                "cargo:warning=Or install Android Studio which sets up the NDK automatically."
            );
            println!(
                "cargo:warning================================================================="
            );
            process::exit(1);
        }
    } else if target_os == "macos" || target_os == "ios" {
        cmake_config
            .define("GGML_METAL", "ON")
            .define("GGML_ACCELERATE", "ON")
            .define("GGML_BLAS", "OFF");
        metal_enabled = true;
    } else if target.contains("linux") {
        cmake_config
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "OFF");
    } else if target.contains("windows") {
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
        target,
        if metal_enabled { "yes" } else { "no" },
        ndk_path_used.as_deref().unwrap_or("N/A")
    );

    let dst = cmake_config.build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-search=native={}", dst.display());

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
        .include(llama_cpp_dir.join("ggml/include"))
        .include(dst.join("include"));

    // Windows MSVC CRT: Do NOT call static_crt() — let the cc crate auto-detect from
    // CARGO_CFG_TARGET_FEATURE. When crt-static is set (CLI via RUSTFLAGS), cc uses /MT.
    // When not set (Flutter cdylib default), cc uses /MD. This keeps wrapper in sync with
    // both llama.cpp (CMake) and esaxx-rs automatically.

    wrapper_build.compile("llama_wrapper");

    if target_os == "android" {
        println!("cargo:rustc-link-lib=c++_shared");
        println!("cargo:rustc-link-lib=log");
    } else if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
    } else if target_os == "macos" || target_os == "ios" {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");

        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=static=ggml-metal");
    } else if target.contains("windows") {
        // Windows linking handled by CMake
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
        println!("cargo:warning=================================================================");
        println!("cargo:warning=ERROR: Failed to clone llama.cpp!");
        println!("cargo:warning=================================================================");
        println!("cargo:warning=Expected location: {}", cloned.display());
        println!("cargo:warning=");
        println!("cargo:warning=To fix this manually, run:");
        println!(
            "cargo:warning=  git clone {} {} && \\",
            LLAMA_CPP_REPO,
            cloned.display()
        );
        println!(
            "cargo:warning=    git -C {} checkout {}",
            cloned.display(),
            LLAMA_CPP_COMMIT
        );
        println!("cargo:warning=");
        println!("cargo:warning=Or disable the llm-llamacpp feature:");
        println!("cargo:warning=  cargo build --no-default-features");
        println!("cargo:warning=================================================================");
        process::exit(1);
    }
}
