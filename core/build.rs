//! Build script for xybrid-core
//!
//! Handles conditional compilation of llama.cpp when the `local-llm-llamacpp` feature is enabled.
//! Uses CMake for building llama.cpp to properly handle its complex build system.

fn main() {
    // Only compile llama.cpp when the feature is enabled
    #[cfg(feature = "local-llm-llamacpp")]
    compile_llama_cpp();
}

#[cfg(feature = "local-llm-llamacpp")]
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

        // Get the Android NDK path - expand ~ if needed
        let ndk_path = env::var("ANDROID_NDK_HOME")
            .or_else(|_| env::var("NDK_HOME"))
            .map(|ndk| {
                if ndk.starts_with("~") {
                    env::var("HOME")
                        .map(|home| ndk.replacen("~", &home, 1))
                        .unwrap_or(ndk)
                } else {
                    ndk
                }
            });

        if let Ok(ref ndk) = ndk_path {
            // Use Android NDK's CMake toolchain file for proper cross-compilation
            let toolchain_file = format!("{}/build/cmake/android.toolchain.cmake", ndk);
            cmake_config.define("CMAKE_TOOLCHAIN_FILE", &toolchain_file);

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
        }
    } else if target_os == "macos" || target_os == "ios" {
        // Apple: Enable Metal
        cmake_config
            .define("GGML_METAL", "ON")
            .define("GGML_ACCELERATE", "ON");
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

    // Link directories
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!(
        "cargo:rustc-link-search=native={}/build",
        llama_cpp_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/build/ggml/src",
        llama_cpp_dir.display()
    );

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
