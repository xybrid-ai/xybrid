//! Runtime resource placement for the MLX Metal shader library.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::error::{MlxError, MlxResult};

static METALLIB_RESOURCE: OnceLock<Result<(), String>> = OnceLock::new();

/// Ensure MLX can discover `mlx.metallib` through its normal bundle/resource
/// lookup before the first FFI call touches Metal.
///
/// The upstream static library also contains a CMake build-directory fallback,
/// but that path points at a disposable source-build directory. Xybrid binaries
/// instead link or copy the vendored xcframework resource next to the current
/// executable, matching MLX's `Resources/mlx.metallib` lookup for plain CLI
/// and test binaries.
pub(crate) fn ensure_metallib_resource() -> MlxResult<()> {
    METALLIB_RESOURCE
        .get_or_init(install_metallib_resource)
        .as_ref()
        .map(|_| ())
        .map_err(|e| MlxError::MetalCompileFailure(e.clone()))
}

fn install_metallib_resource() -> Result<(), String> {
    let source = PathBuf::from(option_env!("XYBRID_MLX_METALLIB_PATH").ok_or(
        "XYBRID_MLX_METALLIB_PATH was not set at compile time; rebuild with a complete mlx.xcframework",
    )?);
    if !source.is_file() {
        return Err(format!(
            "compiled MLX metallib source does not exist: {}",
            source.display()
        ));
    }

    let resource_dirs = candidate_resource_dirs()?;
    let mut last_error = None;
    for dir in resource_dirs {
        match ensure_link(&source, &dir.join("mlx.metallib")) {
            Ok(()) => return Ok(()),
            Err(e) => last_error = Some(e),
        }
    }

    Err(last_error.unwrap_or_else(|| {
        "no candidate resource directory was available for mlx.metallib".to_string()
    }))
}

fn candidate_resource_dirs() -> Result<Vec<PathBuf>, String> {
    let exe = std::env::current_exe()
        .map_err(|e| format!("failed to resolve current executable path: {e}"))?;
    let exe_dir = exe.parent().ok_or_else(|| {
        format!(
            "current executable has no parent directory: {}",
            exe.display()
        )
    })?;

    let mut dirs = Vec::new();
    if let Some(contents_dir) = macos_app_contents_dir(exe_dir) {
        dirs.push(contents_dir.join("Resources"));
    }
    dirs.push(exe_dir.join("Resources"));

    if let Some(parent) = exe_dir.parent() {
        dirs.push(parent.join("Resources"));
    }

    let mut unique = Vec::new();
    for dir in dirs {
        if !unique.contains(&dir) {
            unique.push(dir);
        }
    }
    Ok(unique)
}

fn macos_app_contents_dir(exe_dir: &Path) -> Option<PathBuf> {
    if exe_dir.file_name()? != "MacOS" {
        return None;
    }
    let contents_dir = exe_dir.parent()?;
    if contents_dir.file_name()? == "Contents" {
        Some(contents_dir.to_path_buf())
    } else {
        None
    }
}

fn ensure_link(source: &Path, target: &Path) -> Result<(), String> {
    if target.is_file() {
        return Ok(());
    }

    let parent = target.parent().ok_or_else(|| {
        format!(
            "resource target has no parent directory: {}",
            target.display()
        )
    })?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;

    #[cfg(unix)]
    {
        match std::os::unix::fs::symlink(source, target) {
            Ok(()) => return Ok(()),
            Err(_) if target.is_file() => return Ok(()),
            Err(_) => {}
        }
    }

    std::fs::copy(source, target)
        .map(|_| ())
        .map_err(|e| format!("failed to install {}: {e}", target.display()))
}
