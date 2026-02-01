fn main() {
    // Link against C++ standard library for macOS/iOS
    // This is required because ONNX Runtime (ort) is a C++ library
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=c++");
    }
    
    #[cfg(target_os = "ios")]
    {
        println!("cargo:rustc-link-lib=c++");
    }
}

