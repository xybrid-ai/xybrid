//! Build script for xybrid-ffi
//!
//! This generates the C header file (`include/xybrid.h`) using cbindgen.

use std::env;
use std::path::PathBuf;

fn main() {
    // Get the crate directory
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let crate_path = PathBuf::from(&crate_dir);

    // Create the include directory if it doesn't exist
    let include_dir = crate_path.join("include");
    std::fs::create_dir_all(&include_dir).expect("Failed to create include directory");

    // Output path for the header
    let output_path = include_dir.join("xybrid.h");

    // Generate the header using cbindgen
    let config_path = crate_path.join("cbindgen.toml");

    let result = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(cbindgen::Config::from_file(&config_path).expect("Failed to read cbindgen.toml"))
        .generate();

    match result {
        Ok(bindings) => {
            bindings.write_to_file(&output_path);
            println!("cargo:rerun-if-changed=src/lib.rs");
            println!("cargo:rerun-if-changed=cbindgen.toml");
        }
        Err(e) => {
            // Print a warning but don't fail the build
            // This allows building even if cbindgen fails (e.g., during initial setup)
            eprintln!("Warning: cbindgen failed: {}", e);
            eprintln!("The C header will not be generated.");
        }
    }
}
