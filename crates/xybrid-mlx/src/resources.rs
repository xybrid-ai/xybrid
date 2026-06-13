//! Runtime resource placement for the MLX Metal shader library.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::error::{MlxError, MlxResult};

static METALLIB_RESOURCE: OnceLock<Result<(), String>> = OnceLock::new();
const RUNTIME_METALLIB_ENV: &str = "XYBRID_MLX_METALLIB";
const PACKAGING_CONTRACT: &str =
    "ship Resources/mlx.metallib next to the binary or set XYBRID_MLX_METALLIB";

#[derive(Debug, Clone, PartialEq, Eq)]
enum MetallibResolution {
    AlreadyInstalled(PathBuf),
    Source(PathBuf),
}

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
    let resource_dirs = candidate_resource_dirs()?;
    let runtime_override = std::env::var_os(RUNTIME_METALLIB_ENV).map(PathBuf::from);
    let compile_time = option_env!("XYBRID_MLX_METALLIB_PATH").map(PathBuf::from);
    let resolution = resolve_metallib_resource(runtime_override, &resource_dirs, compile_time)?;
    let MetallibResolution::Source(source) = resolution else {
        return Ok(());
    };

    let mut last_error = None;
    for dir in resource_dirs {
        match ensure_link(&source, &dir.join("mlx.metallib")) {
            Ok(()) => return Ok(()),
            Err(e) => last_error = Some(e),
        }
    }

    Err(last_error.unwrap_or_else(|| {
        format!(
            "no candidate resource directory was available for mlx.metallib; {PACKAGING_CONTRACT}"
        )
    }))
}

fn resolve_metallib_resource(
    env_override: Option<PathBuf>,
    candidates: &[PathBuf],
    compile_time: Option<PathBuf>,
) -> Result<MetallibResolution, String> {
    if let Some(source) = env_override {
        if source.is_file() {
            return Ok(MetallibResolution::Source(source));
        }
        return Err(format!(
            "{RUNTIME_METALLIB_ENV} is set to {} but that file does not exist; {PACKAGING_CONTRACT}",
            source.display()
        ));
    }

    for dir in candidates {
        let installed = dir.join("mlx.metallib");
        if installed.is_file() {
            return Ok(MetallibResolution::AlreadyInstalled(installed));
        }
    }

    if let Some(source) = compile_time {
        if source.is_file() {
            return Ok(MetallibResolution::Source(source));
        }
        return Err(format!(
            "compiled MLX metallib source does not exist: {}; {PACKAGING_CONTRACT}",
            source.display()
        ));
    }

    Err(format!(
        "no MLX metallib resource found; {PACKAGING_CONTRACT}"
    ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "xybrid-mlx-resources-{}-{}-{name}",
                std::process::id(),
                NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_file(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(path, b"metallib").expect("write file");
    }

    #[test]
    fn resolve_prefers_runtime_override() {
        let temp = TempDir::new("runtime-override");
        let override_source = temp.path().join("override.metallib");
        let candidate = temp.path().join("Resources");
        let installed = candidate.join("mlx.metallib");
        let compile_time = temp.path().join("compile.metallib");
        write_file(&override_source);
        write_file(&installed);
        write_file(&compile_time);

        let resolved = resolve_metallib_resource(
            Some(override_source.clone()),
            &[candidate],
            Some(compile_time),
        )
        .expect("resolve override");

        assert_eq!(resolved, MetallibResolution::Source(override_source));
    }

    #[test]
    fn resolve_rejects_missing_runtime_override() {
        let temp = TempDir::new("missing-override");
        let missing = temp.path().join("missing.metallib");
        let candidate = temp.path().join("Resources");
        write_file(&candidate.join("mlx.metallib"));

        let err = resolve_metallib_resource(Some(missing), &[candidate], None)
            .expect_err("missing override should fail");

        assert!(err.contains("XYBRID_MLX_METALLIB"));
    }

    #[test]
    fn resolve_short_circuits_installed_resource() {
        let temp = TempDir::new("installed-resource");
        let candidate = temp.path().join("Resources");
        let installed = candidate.join("mlx.metallib");
        let missing_compile_time = temp.path().join("missing-compile.metallib");
        write_file(&installed);

        let resolved = resolve_metallib_resource(None, &[candidate], Some(missing_compile_time))
            .expect("resolve installed resource");

        assert_eq!(resolved, MetallibResolution::AlreadyInstalled(installed));
    }

    #[test]
    fn resolve_falls_back_to_compile_time_source() {
        let temp = TempDir::new("compile-time");
        let candidate = temp.path().join("Resources");
        let compile_time = temp.path().join("compile.metallib");
        write_file(&compile_time);

        let resolved = resolve_metallib_resource(None, &[candidate], Some(compile_time.clone()))
            .expect("resolve compile-time source");

        assert_eq!(resolved, MetallibResolution::Source(compile_time));
    }

    #[test]
    fn resolve_missing_all_tiers_describes_packaging_contract() {
        let temp = TempDir::new("missing-all");
        let candidate = temp.path().join("Resources");

        let err = resolve_metallib_resource(None, &[candidate], None)
            .expect_err("missing all tiers should fail");

        assert!(err.contains("Resources/mlx.metallib"));
        assert!(err.contains("XYBRID_MLX_METALLIB"));
    }
}
