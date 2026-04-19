//! Build script for `mlx-c-sys`.
//!
//! Runs `bindgen` against the mlx-c public headers bundled inside
//! `vendor/mlx-apple/mlx.xcframework`, and emits the static link directives
//! that resolve `libmlx-combined.a` plus the Apple system frameworks MLX
//! depends on.
//!
//! Gated end-to-end behaviour:
//!   - If the `bindings` cargo feature is off, the script is a no-op.
//!     `cargo check --workspace` on Linux CI runs this branch.
//!   - If the feature is on but the host target is not Apple, the lib.rs
//!     `compile_error!` fires. This script still runs but short-circuits
//!     so the error message comes from rustc and not from a missing file.
//!   - If the feature is on and the target is Apple, we locate the
//!     xcframework slice, run bindgen, and emit link flags.
//!
//! The xcframework itself is CI-built (see `.github/workflows/build-mlx-xcframework.yml`)
//! and materialised locally by `tools/scripts/fetch-mlx-xcframework.sh`.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Watch triggers. Keep these unconditional so toggling the feature
    // re-runs the script.
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MLX_XCFRAMEWORK_PATH");

    // Feature gate — keep the crate a no-op for default builds.
    if env::var_os("CARGO_FEATURE_BINDINGS").is_none() {
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS set by cargo");
    let target_arch =
        env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH set by cargo");

    // Non-Apple targets: leave linking to lib.rs's compile_error!.
    if target_os != "macos" && target_os != "ios" {
        return;
    }

    let slice =
        resolve_slice(&target_os, &target_arch).unwrap_or_else(|e| panic!("mlx-c-sys: {e}"));

    let xcframework = resolve_xcframework_dir().unwrap_or_else(|e| panic!("mlx-c-sys: {e}"));

    let slice_dir = xcframework.join(&slice);
    if !slice_dir.is_dir() {
        panic!(
            "mlx-c-sys: xcframework slice `{slice}` not found at {}. \
             Run `./tools/scripts/fetch-mlx-xcframework.sh` or point \
             $MLX_XCFRAMEWORK_PATH at a prebuilt mlx.xcframework.",
            slice_dir.display()
        );
    }

    let headers = slice_dir.join("Headers");
    let lib_name = "mlx-combined";
    let lib_file = slice_dir.join(format!("lib{lib_name}.a"));
    if !lib_file.is_file() {
        panic!(
            "mlx-c-sys: expected static library at {} — the xcframework layout \
             may have drifted. Rebuild it via .github/workflows/build-mlx-xcframework.yml.",
            lib_file.display()
        );
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    generate_bindings(&headers, &slice, &out_dir);
    emit_link_directives(&slice_dir, lib_name);
}

fn resolve_slice(target_os: &str, target_arch: &str) -> Result<String, String> {
    match (target_os, target_arch) {
        ("macos", "aarch64") => Ok("macos-arm64".into()),
        ("ios", "aarch64") => {
            // The `aarch64-apple-ios-sim` target triple sets target_os=ios,
            // target_arch=aarch64, and target_abi=sim. Distinguish via
            // target_abi (propagated as CARGO_CFG_TARGET_ABI) so the device
            // vs simulator slice is picked correctly.
            let abi = env::var("CARGO_CFG_TARGET_ABI").unwrap_or_default();
            if abi == "sim" {
                Ok("ios-arm64-simulator".into())
            } else {
                Ok("ios-arm64".into())
            }
        }
        (os, arch) => Err(format!(
            "no mlx.xcframework slice available for target_os={os} target_arch={arch}"
        )),
    }
}

fn resolve_xcframework_dir() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("MLX_XCFRAMEWORK_PATH") {
        let p = PathBuf::from(path);
        if p.is_dir() {
            return Ok(p);
        }
        return Err(format!(
            "$MLX_XCFRAMEWORK_PATH is set to {} but that directory does not exist",
            p.display()
        ));
    }

    // Default: repo-root vendor/mlx-apple/mlx.xcframework. The crate lives at
    // repos/xybrid/crates/mlx-c-sys, so two `..` reach repos/xybrid.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let default = manifest_dir
        .join("..")
        .join("..")
        .join("vendor")
        .join("mlx-apple")
        .join("mlx.xcframework");
    if default.is_dir() {
        return Ok(default);
    }
    Err(format!(
        "mlx.xcframework not found at {}. Run `tools/scripts/fetch-mlx-xcframework.sh` \
         or set $MLX_XCFRAMEWORK_PATH.",
        default.display()
    ))
}

fn generate_bindings(headers: &Path, slice: &str, out_dir: &Path) {
    let (sdk, target_triple) = match slice {
        "macos-arm64" => ("macosx", "arm64-apple-macos13.3"),
        "ios-arm64" => ("iphoneos", "arm64-apple-ios15.0"),
        "ios-arm64-simulator" => ("iphonesimulator", "arm64-apple-ios15.0-simulator"),
        other => panic!("mlx-c-sys: unsupported slice `{other}`"),
    };

    let sdk_path = Command::new("xcrun")
        .args(["--show-sdk-path", "--sdk", sdk])
        .output()
        .unwrap_or_else(|e| panic!("mlx-c-sys: failed to invoke xcrun --sdk {sdk}: {e}"));
    if !sdk_path.status.success() {
        panic!(
            "mlx-c-sys: xcrun --sdk {sdk} exited with {}: {}",
            sdk_path.status,
            String::from_utf8_lossy(&sdk_path.stderr)
        );
    }
    let sdk_path = String::from_utf8(sdk_path.stdout)
        .expect("xcrun sdk path is utf-8")
        .trim()
        .to_string();

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-xc")
        .clang_arg(format!("-isysroot{sdk_path}"))
        .clang_arg(format!("--target={target_triple}"))
        .clang_arg(format!("-I{}", headers.display()))
        // mlx-c's `*_` struct tags carry no payload beyond `void *ctx`; letting
        // bindgen derive Copy on them makes handle-returning FFI functions
        // ergonomic. We still uphold ownership manually via drop glue in
        // xybrid-mlx (US-005), which owns the safe wrappers.
        .derive_copy(true)
        .derive_default(false)
        // Keep the generated surface small and focused on mlx_* identifiers.
        // This avoids dragging in macOS system types bindgen would otherwise
        // emit from transitive includes.
        .allowlist_function("mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        .layout_tests(false)
        .generate()
        .expect("mlx-c-sys: bindgen failed to generate bindings");

    let out_path = out_dir.join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .unwrap_or_else(|e| panic!("mlx-c-sys: failed to write {}: {e}", out_path.display()));
}

fn emit_link_directives(slice_dir: &Path, lib_name: &str) {
    println!("cargo:rustc-link-search=native={}", slice_dir.display());
    println!("cargo:rustc-link-lib=static={lib_name}");

    // C++ standard library — mlx core is C++20. On Apple platforms this is
    // libc++ (clang's default), not libstdc++.
    println!("cargo:rustc-link-lib=c++");

    // System frameworks. Accelerate carries BLAS/LAPACK used on CPU; Metal
    // and MetalPerformanceShaders are the GPU path; Foundation is pulled in
    // by Metal for NSString/NSDictionary interop inside MLX.
    for fw in [
        "Metal",
        "MetalPerformanceShaders",
        "Foundation",
        "Accelerate",
    ] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }
}
