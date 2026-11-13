//! Runtime placement of the MLX Metal shader library.
//!
//! MLX loads `mlx.metallib` when it first touches the Metal device, and looks
//! for it relative to the binary image MLX is linked into (the executable, or
//! the dynamic library that embeds it): `mlx.metallib` beside that image, then
//! `Resources/mlx.metallib`. Its last resort is a path inside the MLX build
//! directory, which is gone on every machine but the one that built it.
//!
//! So a distributed binary must ship the library at one of those two places.
//! This module handles the remaining cases before the first FFI call that
//! reaches Metal:
//!
//! - a library is already there: nothing to do;
//! - `XYBRID_MLX_METALLIB` names one: link it into `Resources/`;
//! - a development build (`cargo run`, `cargo test`) whose vendored
//!   xcframework still exists at the path recorded at compile time: link that.
//!
//! Anything else is an error that states the packaging contract and the
//! directory MLX searches.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::error::{MlxError, MlxResult};
use crate::ffi;

static METALLIB_RESOURCE: OnceLock<Result<(), String>> = OnceLock::new();
const RUNTIME_METALLIB_ENV: &str = "XYBRID_MLX_METALLIB";
const METALLIB_NAME: &str = "mlx.metallib";

/// Ensure MLX can find `mlx.metallib` through its own lookup before the first
/// FFI call touches Metal. Runs once per process; the outcome is cached.
pub(crate) fn ensure_metallib_resource() -> MlxResult<()> {
    METALLIB_RESOURCE
        .get_or_init(|| {
            let image_dir = ffi::image_dir().ok_or_else(|| {
                "could not locate the binary image that links MLX (dladdr failed)".to_string()
            })?;
            install_metallib(
                &image_dir,
                std::env::var_os(RUNTIME_METALLIB_ENV).map(PathBuf::from),
                option_env!("XYBRID_MLX_METALLIB_PATH").map(PathBuf::from),
            )
        })
        .as_ref()
        .map(|_| ())
        .map_err(|e| MlxError::MetalCompileFailure(e.clone()))
}

/// The two places MLX looks for a colocated library, in its order.
fn probe_paths(image_dir: &Path) -> [PathBuf; 2] {
    [
        image_dir.join(METALLIB_NAME),
        image_dir.join("Resources").join(METALLIB_NAME),
    ]
}

fn packaging_contract(image_dir: &Path) -> String {
    format!(
        "ship {METALLIB_NAME} or Resources/{METALLIB_NAME} in {} (the directory of the binary \
         that links MLX), or set {RUNTIME_METALLIB_ENV} to a copy of it",
        image_dir.display()
    )
}

fn install_metallib(
    image_dir: &Path,
    env_override: Option<PathBuf>,
    compile_time: Option<PathBuf>,
) -> Result<(), String> {
    let [beside, in_resources] = probe_paths(image_dir);

    let source = match env_override {
        Some(source) if source.is_file() => Some(source),
        Some(missing) => {
            return Err(format!(
                "{RUNTIME_METALLIB_ENV} is set to {} but that file does not exist; {}",
                missing.display(),
                packaging_contract(image_dir)
            ));
        }
        // The compile-time path exists only where the xcframework was built
        // or fetched: a development checkout. Elsewhere it is simply absent.
        None => compile_time.filter(|source| source.is_file()),
    };

    // A library shipped beside the image is what MLX loads first; there is
    // nothing to add, and nothing this module may replace.
    if beside.is_file() {
        return Ok(());
    }

    match source {
        Some(source) => link(&source, &in_resources).map_err(|e| {
            format!(
                "could not place {} at {}: {e}; {}",
                source.display(),
                in_resources.display(),
                packaging_contract(image_dir)
            )
        }),
        None if in_resources.is_file() => Ok(()),
        None => Err(format!(
            "no MLX Metal library found; {}",
            packaging_contract(image_dir)
        )),
    }
}

/// Make `target` resolve to `source`.
///
/// A regular file at `target` was shipped by whoever packaged the binary and
/// is left alone. A symlink is this module's own from an earlier run: it is
/// kept when it already points at `source`, and replaced when it is stale
/// (the source moved, or `XYBRID_MLX_METALLIB` changed) or dangling. Replacing
/// rather than writing through it matters: copying onto a dangling link would
/// create a file at the link's old destination.
fn link(source: &Path, target: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(target) {
        Ok(existing) if existing.file_type().is_symlink() => {
            if std::fs::read_link(target).is_ok_and(|current| current == source) {
                return Ok(());
            }
            std::fs::remove_file(target)?;
        }
        Ok(_) => return Ok(()),
        Err(_) => {}
    }

    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    match std::os::unix::fs::symlink(source, target) {
        Ok(()) => Ok(()),
        // Another thread or process placed it between the check and here.
        Err(_) if target.is_file() => Ok(()),
        // A file system without symlinks: fall back to a copy.
        Err(_) => std::fs::copy(source, target).map(|_| ()),
    }
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

    fn write_file(path: &Path, contents: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(path, contents).expect("write file");
    }

    fn image_dir(temp: &TempDir) -> PathBuf {
        let dir = temp.path().join("bin");
        fs::create_dir_all(&dir).expect("create image dir");
        dir
    }

    #[test]
    fn the_image_dir_is_where_this_binary_lives() {
        // MLX resolves its colocated library against the image it is linked
        // into. For this test binary that is the executable itself.
        let dir = ffi::image_dir().expect("dladdr resolves this image");
        let exe = std::env::current_exe().expect("current exe");
        assert_eq!(
            dir.canonicalize().expect("image dir"),
            exe.parent()
                .expect("exe dir")
                .canonicalize()
                .expect("exe dir")
        );
    }

    #[test]
    fn a_library_shipped_where_mlx_looks_needs_nothing() {
        for shipped in [METALLIB_NAME, "Resources/mlx.metallib"] {
            let temp = TempDir::new("shipped");
            let dir = image_dir(&temp);
            write_file(&dir.join(shipped), b"shipped");

            install_metallib(&dir, None, None).expect("shipped library");

            assert_eq!(fs::read(dir.join(shipped)).unwrap(), b"shipped");
        }
    }

    #[test]
    fn the_runtime_override_is_linked_into_resources() {
        let temp = TempDir::new("override");
        let dir = image_dir(&temp);
        let source = temp.path().join("elsewhere").join(METALLIB_NAME);
        write_file(&source, b"override");
        let compile_time = temp.path().join("vendor").join(METALLIB_NAME);
        write_file(&compile_time, b"compile-time");

        install_metallib(&dir, Some(source.clone()), Some(compile_time)).expect("install");

        let [_, in_resources] = probe_paths(&dir);
        assert_eq!(fs::read_link(&in_resources).unwrap(), source);
        assert_eq!(fs::read(&in_resources).unwrap(), b"override");
    }

    #[test]
    fn a_missing_runtime_override_is_an_error_even_with_a_library_in_place() {
        let temp = TempDir::new("missing-override");
        let dir = image_dir(&temp);
        write_file(&dir.join("Resources").join(METALLIB_NAME), b"shipped");

        let err = install_metallib(&dir, Some(temp.path().join("nope.metallib")), None)
            .expect_err("a mistyped override must not pass silently");

        assert!(err.contains(RUNTIME_METALLIB_ENV), "{err}");
    }

    #[test]
    fn a_development_build_links_the_vendored_library() {
        let temp = TempDir::new("dev");
        let dir = image_dir(&temp);
        let compile_time = temp.path().join("vendor").join(METALLIB_NAME);
        write_file(&compile_time, b"vendored");

        install_metallib(&dir, None, Some(compile_time.clone())).expect("install");

        let [_, in_resources] = probe_paths(&dir);
        assert_eq!(fs::read_link(&in_resources).unwrap(), compile_time);
    }

    #[test]
    fn a_stale_or_dangling_link_is_replaced_not_written_through() {
        let temp = TempDir::new("stale");
        let dir = image_dir(&temp);
        let [_, in_resources] = probe_paths(&dir);
        let old = temp.path().join("old-checkout").join(METALLIB_NAME);
        let new = temp.path().join("new-checkout").join(METALLIB_NAME);
        write_file(&new, b"new");
        fs::create_dir_all(in_resources.parent().unwrap()).unwrap();
        fs::create_dir_all(old.parent().unwrap()).unwrap();
        // The link an earlier build left, whose xcframework has since moved.
        std::os::unix::fs::symlink(&old, &in_resources).unwrap();

        install_metallib(&dir, None, Some(new.clone())).expect("repair");

        assert_eq!(fs::read_link(&in_resources).unwrap(), new);
        assert!(
            !old.exists(),
            "nothing may be created at the stale link's destination"
        );
    }

    #[test]
    fn a_shipped_file_in_resources_is_never_replaced() {
        let temp = TempDir::new("keep-shipped");
        let dir = image_dir(&temp);
        let [_, in_resources] = probe_paths(&dir);
        write_file(&in_resources, b"shipped");
        let source = temp.path().join("elsewhere").join(METALLIB_NAME);
        write_file(&source, b"override");

        install_metallib(&dir, Some(source), None).expect("install");

        assert!(!fs::symlink_metadata(&in_resources)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(&in_resources).unwrap(), b"shipped");
    }

    #[test]
    fn a_distributed_binary_without_a_library_states_the_packaging_contract() {
        let temp = TempDir::new("nothing");
        let dir = image_dir(&temp);
        // A compile-time path from the build machine does not exist here.
        let build_machine = temp.path().join("builder").join(METALLIB_NAME);

        let err =
            install_metallib(&dir, None, Some(build_machine)).expect_err("nothing to install");

        assert!(err.contains("Resources/mlx.metallib"), "{err}");
        assert!(err.contains(RUNTIME_METALLIB_ENV), "{err}");
        assert!(err.contains(&dir.display().to_string()), "{err}");
        assert!(!err.contains("builder"), "{err}");
    }

    #[test]
    fn an_unwritable_image_dir_is_an_error_that_names_the_contract() {
        use std::os::unix::fs::PermissionsExt;

        let temp = TempDir::new("read-only");
        let dir = image_dir(&temp);
        let source = temp.path().join("elsewhere").join(METALLIB_NAME);
        write_file(&source, b"override");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();

        let result = install_metallib(&dir, Some(source), None);
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();

        // Root ignores directory permissions; the contract only matters when
        // the write is refused.
        if let Err(err) = result {
            assert!(err.contains("could not place"), "{err}");
            assert!(err.contains(RUNTIME_METALLIB_ENV), "{err}");
        }
    }
}
