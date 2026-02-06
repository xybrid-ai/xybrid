//! Build script for xybrid-core
//!
//! Handles conditional compilation of llama.cpp when the `llm-llamacpp` feature is enabled.
//! Uses CMake for building llama.cpp to properly handle its complex build system.

fn main() {
    // Only compile llama.cpp when the feature is enabled
    #[cfg(feature = "llm-llamacpp")]
    compile_llama_cpp();
}

/// Find the Android NDK path from various sources
#[cfg(feature = "llm-llamacpp")]
fn find_android_ndk() -> Option<String> {
    use std::env;
    use std::path::Path;

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
    if let Ok(ndk) = env::var("ANDROID_NDK_HOME").or_else(|_| env::var("NDK_HOME")) {
        let expanded = expand_tilde(ndk);
        if Path::new(&expanded).exists() {
            return Some(expanded);
        }
    }

    // 2. Try to extract from CC environment variable (set by cargo/cmake)
    // e.g., CC=/path/to/ndk/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang
    for var in ["CC_aarch64-linux-android", "CC_aarch64_linux_android", "TARGET_CC", "CC"] {
        if let Ok(cc_path) = env::var(var) {
            // Extract NDK path: go up from .../toolchains/llvm/prebuilt/.../bin/clang
            if cc_path.contains("/ndk/") {
                if let Some(ndk_end) = cc_path.find("/toolchains/") {
                    let ndk = &cc_path[..ndk_end];
                    if Path::new(ndk).exists() {
                        return Some(ndk.to_string());
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
                        return Some(latest.to_string_lossy().to_string());
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
                    return Some(latest.to_string_lossy().to_string());
                }
            }
        }
    }

    None
}

#[cfg(feature = "llm-llamacpp")]
fn compile_llama_cpp() {
    use std::env;
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama_cpp_dir = manifest_dir.join("vendor/llama.cpp");
    let wrapper_path = manifest_dir.join("vendor/llama_wrapper.cpp");

    // Check if llama.cpp is vendored
    if !llama_cpp_dir.exists() {
        panic!(
            "llama.cpp not found at {:?}. \n\
            Run: git clone --depth 1 https://github.com/ggerganov/llama.cpp {:?}",
            llama_cpp_dir, llama_cpp_dir
        );
    }

    // Detect target platform
    let target = env::var("TARGET").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

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
            .define("GGML_VULKAN", "OFF");

        // Find NDK path from multiple sources
        let ndk_path = find_android_ndk();

        if let Some(ref ndk) = ndk_path {
            // Use Android NDK's CMake toolchain file for proper cross-compilation
            let toolchain_file = format!("{}/build/cmake/android.toolchain.cmake", ndk);
            if std::path::Path::new(&toolchain_file).exists() {
                cmake_config.define("CMAKE_TOOLCHAIN_FILE", &toolchain_file);
            }

            // Set Android-specific CMake variables
            let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "aarch64".to_string());
            let android_abi = match target_arch.as_str() {
                "aarch64" => "arm64-v8a",
                "arm" => "armeabi-v7a",
                "x86_64" => "x86_64",
                "x86" => "x86",
                _ => "arm64-v8a",
            };
            cmake_config.define("ANDROID_ABI", android_abi);
            cmake_config.define("ANDROID_PLATFORM", "android-28");
            cmake_config.define("ANDROID_STL", "c++_shared");
            cmake_config.define("ANDROID_NDK", ndk);
        } else {
            panic!("Android NDK not found. Set ANDROID_NDK_HOME or ANDROID_HOME environment variable.");
        }
    } else if target_os == "macos" || target_os == "ios" {
        // Apple: Enable Metal and Accelerate, disable BLAS (use Accelerate directly)
        cmake_config
            .define("GGML_METAL", "ON")
            .define("GGML_ACCELERATE", "ON")
            .define("GGML_BLAS", "OFF");
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
    }

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

    // Build our C++ wrapper
    cc::Build::new()
        .cpp(true)
        .file(&wrapper_path)
        .include(llama_cpp_dir.join("include"))
        .include(llama_cpp_dir.join("ggml/include"))
        .include(dst.join("include"))
        .opt_level(3)
        .compile("llama_wrapper");

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
