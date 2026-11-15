//! Build script for `xybrid-mlx`.
//!
//! When the real MLX bindings are enabled, expose the resolved vendored
//! `mlx.metallib` path to the safe wrapper. The runtime helper uses it to
//! place a lightweight resource link next to CLI/test binaries before MLX
//! initialises Metal.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MLX_XCFRAMEWORK_PATH");

    if env::var_os("CARGO_FEATURE_BINDINGS").is_none() {
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS set by cargo");
    let target_arch =
        env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH set by cargo");
    if target_os != "macos" || target_arch != "aarch64" {
        return;
    }

    let metallib = resolve_metallib_path().unwrap_or_else(|e| panic!("xybrid-mlx: {e}"));
    println!("cargo:rerun-if-changed={}", metallib.display());
    println!(
        "cargo:rustc-env=XYBRID_MLX_METALLIB_PATH={}",
        metallib.display()
    );
}

fn resolve_metallib_path() -> Result<PathBuf, String> {
    let xcframework = if let Some(path) = env::var_os("MLX_XCFRAMEWORK_PATH") {
        PathBuf::from(path)
    } else {
        let manifest_dir =
            PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
        manifest_dir
            .join("..")
            .join("..")
            .join("vendor")
            .join("mlx-apple")
            .join("mlx.xcframework")
    };

    let metallib = xcframework
        .join("macos-arm64")
        .join("Resources")
        .join("mlx.metallib");
    if metallib.is_file() {
        Ok(metallib)
    } else {
        Err(format!(
            "expected MLX Metal library at {}. Run \
             `tools/scripts/fetch-mlx-xcframework.sh`, run \
             `tools/scripts/build-local-mlx-xcframework.sh`, or set \
             $MLX_XCFRAMEWORK_PATH to a complete mlx.xcframework.",
            metallib.display()
        ))
    }
}
