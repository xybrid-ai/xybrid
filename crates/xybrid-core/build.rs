//! Build script for xybrid-core
//!
//! Handles conditional compilation of llama.cpp when the `llm-llamacpp` feature is enabled.
//! Uses CMake for building llama.cpp to properly handle its complex build system.

fn main() {
    // Only compile llama.cpp when the feature is enabled
    #[cfg(feature = "llm-llamacpp")]
    compile_llama_cpp();
}

/// Check if CMake is available in PATH
#[cfg(feature = "llm-llamacpp")]
fn check_cmake_available() -> bool {
    std::process::Command::new("cmake")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Get platform-specific CMake installation instructions
/// Note: Uses #[cfg] based on the build machine (not target) which is correct for build scripts
#[cfg(feature = "llm-llamacpp")]
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
#[cfg(feature = "llm-llamacpp")]
struct NdkDetectionResult {
    /// The found NDK path, if any
    ndk_path: Option<String>,
    /// All paths that were tried during detection
    tried_paths: Vec<String>,
}

/// Find the Android NDK path from various sources
#[cfg(feature = "llm-llamacpp")]
fn find_android_ndk() -> NdkDetectionResult {
    use std::env;
    use std::path::Path;

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

    // 4. Try common locations
    let home = env::var("HOME").unwrap_or_default();
    let common_locations = [
        format!("{}/Library/Android/sdk/ndk", home),
        format!("{}/Android/Sdk/ndk", home),
        "/opt/android-sdk/ndk".to_string(),
    ];

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

#[cfg(feature = "llm-llamacpp")]
fn compile_llama_cpp() {
    use std::env;
    use std::path::PathBuf;
    use std::process;

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let wrapper_path = manifest_dir.join("vendor/llama_wrapper.cpp");

    // Pinned llama.cpp upstream — keep in sync with the git submodule SHA in
    // .gitmodules / `git submodule status`. The fallback clone below uses this
    // exact commit so consumers without submodule support (e.g. Flutter pub
    // cache git deps, crates.io tarballs) get a reproducible build instead of
    // upstream HEAD.
    const LLAMA_CPP_REPO: &str = "https://github.com/ggml-org/llama.cpp";
    const LLAMA_CPP_COMMIT: &str = "b46812de78f8fbcb6cf0154947e8633ebc78d9ac";

    // Prefer the in-tree submodule when present (developer workflow). Fall back
    // to a clone in OUT_DIR for consumers that don't ship it: crates.io tarball
    // (no vendor/), Flutter pub cache (empty submodule placeholder), etc.
    // Writing to OUT_DIR (not the source tree) is required by cargo — `cargo
    // publish` rejects build.rs scripts that modify $CARGO_MANIFEST_DIR.
    let in_tree = manifest_dir.join("vendor/llama.cpp");
    let llama_cpp_dir = if in_tree.join("CMakeLists.txt").exists() {
        in_tree
    } else {
        let cloned = out_dir.join("llama.cpp");
        println!(
            "cargo:warning=llama.cpp not vendored in-tree, cloning {}@{} into OUT_DIR...",
            LLAMA_CPP_REPO, LLAMA_CPP_COMMIT
        );

        // Pinned-commit clone: init empty repo, fetch the exact SHA at depth 1,
        // then check it out. `git clone --depth 1` cannot target an arbitrary
        // commit, so we do it in three steps. Idempotent: re-using an existing
        // OUT_DIR clone is fine because the checked-out commit is pinned.
        let dir_str = cloned.to_string_lossy().to_string();
        let run = |args: &[&str]| -> bool {
            process::Command::new("git")
                .args(args)
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        };

        // If the OUT_DIR clone already exists from a previous build, re-use it
        // when it has the expected commit checked out; otherwise wipe and
        // re-clone so we don't end up mixing two states.
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
        } else {
            println!(
                "cargo:warning================================================================="
            );
            println!("cargo:warning=ERROR: Failed to clone llama.cpp!");
            println!(
                "cargo:warning================================================================="
            );
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
            println!(
                "cargo:warning================================================================="
            );
            process::exit(1);
        }

        cloned
    };

    // Check if CMake is available
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

    // Detect target platform
    let target = env::var("TARGET").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    // Track build configuration for summary
    let mut metal_enabled = false;
    let mut ndk_path_used: Option<String> = None;

    println!("cargo:rerun-if-changed=vendor/llama.cpp");
    println!("cargo:rerun-if-changed=vendor/llama_wrapper.cpp");

    // Configure CMake
    let mut cmake_config = cmake::Config::new(&llama_cpp_dir);

    // Disable building examples, tests, and server
    cmake_config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("GGML_OPENMP", "OFF");

    // Platform-specific configuration
    if target_os == "android" {
        // Android: CPU only with runtime SIMD detection
        cmake_config
            .define("GGML_NATIVE", "OFF") // Don't optimize for build machine
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "OFF")
            .define("GGML_VULKAN", "OFF")
            .define("GGML_CPU_HBM", "OFF")
            // Disable llamafile SGEMM — its FP16 NEON intrinsics (vld1q_f16) require
            // armv8.2-a+fp16 which the NDK doesn't enable by default
            .define("GGML_LLAMAFILE", "OFF");

        // Find NDK path from multiple sources
        let ndk_result = find_android_ndk();

        if let Some(ref ndk) = ndk_result.ndk_path {
            println!("cargo:warning=Android NDK detected: {}", ndk);
            ndk_path_used = Some(ndk.clone());

            // Use Android NDK's CMake toolchain file for proper cross-compilation
            let toolchain_file = format!("{}/build/cmake/android.toolchain.cmake", ndk);
            if std::path::Path::new(&toolchain_file).exists() {
                cmake_config.define("CMAKE_TOOLCHAIN_FILE", &toolchain_file);
            }

            // Set Android-specific CMake variables
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
        // Apple: Enable Metal and Accelerate, disable BLAS (use Accelerate directly)
        cmake_config
            .define("GGML_METAL", "ON")
            .define("GGML_ACCELERATE", "ON")
            .define("GGML_BLAS", "OFF");
        metal_enabled = true;
    } else if target.contains("linux") {
        // Linux: CPU only (can enable CUDA later)
        cmake_config
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "OFF");
    } else if target.contains("windows") {
        // Windows: CPU only
        cmake_config
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "OFF");

        // Force CMake Release build on Windows to match the cc crate's CRT choice.
        // The cc crate always emits /MD (release CRT) — it never emits /MDd, even in
        // debug cargo builds. CMake defaults to Debug (/MDd) for `cargo test`, creating
        // a CRT mismatch (LNK2038). Forcing Release ensures both CMake and cc use /MD.
        cmake_config.profile("Release");
    }

    // Output build summary
    println!(
        "cargo:warning=llama.cpp build: target={}, metal={}, ndk={}",
        target,
        if metal_enabled { "yes" } else { "no" },
        ndk_path_used.as_deref().unwrap_or("N/A")
    );

    // Build llama.cpp
    let dst = cmake_config.build();

    // Link directories - all paths should use the CMake output directory (dst)
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-search=native={}", dst.display());

    // Link llama.cpp static libraries
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

    // Platform-specific linking
    if target_os == "android" {
        println!("cargo:rustc-link-lib=c++_shared");
        println!("cargo:rustc-link-lib=log");
    } else if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
    } else if target_os == "macos" || target_os == "ios" {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");

        // Metal framework
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=static=ggml-metal");
    } else if target.contains("windows") {
        // Windows linking handled by CMake
    }
}
