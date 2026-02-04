//! Build script for xybrid-ffi
//!
//! This generates:
//! - C header file (`include/xybrid.h`) using cbindgen (always)
//! - C# bindings using csbindgen (when `csharp` feature enabled)
//!
//! ## Usage
//!
//! ```bash
//! # Build with C header only
//! cargo build -p xybrid-ffi
//!
//! # Build with C# bindings for Unity
//! cargo build -p xybrid-ffi --features csharp
//! ```

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let crate_path = PathBuf::from(&crate_dir);

    // Always generate C header
    generate_c_header(&crate_path);

    // Generate C# bindings if feature enabled
    #[cfg(feature = "csharp")]
    generate_csharp_bindings(&crate_path);

    // Rerun triggers
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");
}

/// Generate C header using cbindgen
fn generate_c_header(crate_path: &PathBuf) {
    let include_dir = crate_path.join("include");
    std::fs::create_dir_all(&include_dir).expect("Failed to create include directory");

    let output_path = include_dir.join("xybrid.h");
    let config_path = crate_path.join("cbindgen.toml");

    let result = cbindgen::Builder::new()
        .with_crate(crate_path)
        .with_config(
            cbindgen::Config::from_file(&config_path).expect("Failed to read cbindgen.toml"),
        )
        .generate();

    match result {
        Ok(bindings) => {
            bindings.write_to_file(&output_path);
            println!("cargo:warning=Generated C header: {}", output_path.display());
        }
        Err(e) => {
            eprintln!("Warning: cbindgen failed: {}", e);
            eprintln!("The C header will not be generated.");
        }
    }
}

/// Generate C# bindings using csbindgen
#[cfg(feature = "csharp")]
fn generate_csharp_bindings(crate_path: &PathBuf) {
    // Determine output path
    // Default: Unity package in the bindings directory
    let out_dir = env::var("XYBRID_CSHARP_OUT_DIR").unwrap_or_else(|_| {
        crate_path
            .join("../../bindings/unity/Runtime/Native")
            .to_string_lossy()
            .to_string()
    });

    let out_path = PathBuf::from(&out_dir);

    // Create output directory if it doesn't exist
    if let Err(e) = std::fs::create_dir_all(&out_path) {
        eprintln!("Warning: Could not create C# output directory: {}", e);
        return;
    }

    let output_file = out_path.join("NativeMethods.g.cs");

    println!(
        "cargo:warning=Generating C# bindings: {}",
        output_file.display()
    );

    csbindgen::Builder::default()
        // Input: the lib.rs with all our extern "C" functions
        .input_extern_file(crate_path.join("src/lib.rs").to_str().unwrap())
        // C# configuration
        .csharp_dll_name("xybrid_ffi")
        .csharp_namespace("Xybrid.Native")
        .csharp_class_name("NativeMethods")
        .csharp_class_accessibility("internal")
        // Unity compatibility: use Func/Action instead of function pointers
        // Required for MonoPInvokeCallback to work
        .csharp_use_function_pointer(false)
        // Generate the file
        .generate_csharp_file(&output_file)
        .expect("Failed to generate C# bindings");

    println!("cargo:warning=C# bindings generated successfully!");
}
