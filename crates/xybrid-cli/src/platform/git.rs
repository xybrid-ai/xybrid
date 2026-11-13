//! Git checks for project files that will hold a key.
//!
//! A key must never reach a commit. Before writing, `connect` refuses a file
//! Git already tracks (`.gitignore` does nothing for those), then adds an
//! escaped, root-anchored entry to the repository's `.gitignore` and confirms
//! it with `git check-ignore`. Outside a repository nothing is claimed:
//! callers report `gitignored: false`.

use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output};

use super::CliError;

/// `path` with symlinks resolved. On Windows the `\\?\` prefix of a plain
/// drive path is dropped, so stored roots read naturally and `git -C`
/// accepts them.
pub(crate) fn canonicalize(path: &Path) -> std::io::Result<PathBuf> {
    let canonical = path.canonicalize()?;
    #[cfg(windows)]
    {
        if let Some(rest) = canonical.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
            let bytes = rest.as_bytes();
            if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
                return Ok(PathBuf::from(rest));
            }
        }
    }
    Ok(canonical)
}

/// The nearest ancestor of `start` holding `.git` (a directory, or a file for
/// worktrees and submodules), canonicalized. No `git` binary needed.
pub(crate) fn repository_root(start: &Path) -> Option<PathBuf> {
    let start = canonicalize(start).ok()?;
    start
        .ancestors()
        .find(|dir| dir.join(".git").exists())
        .map(Path::to_path_buf)
}

/// The key under which credentials are stored for the current directory: its
/// repository root, or the directory itself outside a repository.
pub(crate) fn project_root(cwd: &Path) -> Result<PathBuf, CliError> {
    if let Some(root) = repository_root(cwd) {
        return Ok(root);
    }
    canonicalize(cwd).map_err(|err| {
        CliError::new(
            "project_root_unavailable",
            format!("Could not resolve the current directory: {err}"),
        )
    })
}

/// A Git working tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Repository {
    root: PathBuf,
}

impl Repository {
    pub(crate) fn at(root: PathBuf) -> Self {
        Self { root }
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    /// Whether Git tracks `relative` (staged or committed).
    pub(crate) fn is_tracked(&self, relative: &str) -> Result<bool, CliError> {
        // `:(literal)`: the path is a name, never a glob.
        let pathspec = format!(":(literal){relative}");
        let output = self.git(&["ls-files", "-z", "--cached", "--", &pathspec])?;
        if !output.status.success() {
            return Err(git_failed("ls-files", &output));
        }
        Ok(!output.stdout.is_empty())
    }

    /// Whether `relative` is ignored (for an untracked path).
    pub(crate) fn is_ignored(&self, relative: &str) -> Result<bool, CliError> {
        let output = self.git(&["check-ignore", "-q", "--", relative])?;
        match output.status.code() {
            Some(0) => Ok(true),
            Some(1) => Ok(false),
            _ => Err(git_failed("check-ignore", &output)),
        }
    }

    /// Make sure `relative` is ignored, appending a root-anchored entry to
    /// `<root>/.gitignore` if needed. Returns whether an entry was added.
    pub(crate) fn ensure_ignored(&self, relative: &str) -> Result<bool, CliError> {
        if self.is_ignored(relative)? {
            return Ok(false);
        }
        let pattern = ignore_pattern(relative).ok_or_else(|| {
            CliError::new(
                "env_file_not_ignorable",
                format!(
                    "Can't write a .gitignore entry for {relative:?}; choose a simpler file name"
                ),
            )
        })?;
        let gitignore = self.root.join(".gitignore");
        let existing = match std::fs::read(&gitignore) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(err) => return Err(unwritable(&gitignore, &err)),
        };
        let newline = if existing.windows(2).any(|w| w == b"\r\n") {
            "\r\n"
        } else {
            "\n"
        };
        let mut addition = String::new();
        if !existing.is_empty() && !existing.ends_with(b"\n") {
            addition.push_str(newline);
        }
        addition.push_str(&format!(
            "# Xybrid project key written by `xybrid connect`; keep it out of Git{newline}{pattern}{newline}"
        ));
        use std::io::Write;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&gitignore)
            .and_then(|mut file| file.write_all(addition.as_bytes()))
            .map_err(|err| unwritable(&gitignore, &err))?;
        if !self.is_ignored(relative)? {
            return Err(CliError::new(
                "env_file_not_ignorable",
                format!(
                    "{relative} is still not ignored after adding {pattern} to .gitignore \
                     (another rule re-includes it); choose another file"
                ),
            ));
        }
        Ok(true)
    }

    fn git(&self, args: &[&str]) -> Result<Output, CliError> {
        Command::new("git")
            .arg("-C")
            .arg(&self.root)
            .args(args)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .output()
            .map_err(|err| {
                CliError::new(
                    "git_unavailable",
                    format!(
                        "{} is a Git repository, but git could not run ({err}); \
                         it is needed to keep the key out of commits",
                        self.root.display()
                    ),
                )
            })
    }
}

/// `relative` as forward-slash components, or `None` if it leaves the root.
pub(crate) fn relative_slash_path(root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root).ok()?;
    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(part) => parts.push(part.to_str()?.to_string()),
            Component::CurDir => {}
            _ => return None,
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

/// A `.gitignore` line matching exactly `relative` from the repository root:
/// leading `/`, glob characters and trailing spaces escaped. `None` for names
/// a pattern cannot express (control characters such as newlines).
pub(crate) fn ignore_pattern(relative: &str) -> Option<String> {
    if relative.is_empty() || relative.chars().any(char::is_control) {
        return None;
    }
    let trimmed = relative.trim_end_matches(' ');
    let trailing = relative.len() - trimmed.len();
    let mut pattern = String::with_capacity(relative.len() + 8);
    pattern.push('/');
    for ch in trimmed.chars() {
        if matches!(ch, '\\' | '*' | '?' | '[') {
            pattern.push('\\');
        }
        pattern.push(ch);
    }
    for _ in 0..trailing {
        pattern.push_str("\\ ");
    }
    Some(pattern)
}

fn git_failed(command: &str, output: &Output) -> CliError {
    CliError::new(
        "git_failed",
        format!(
            "git {command} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ),
    )
}

fn unwritable(path: &Path, err: &std::io::Error) -> CliError {
    CliError::new(
        "env_file_unwritable",
        format!("Could not update {}: {err}", path.display()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git_available() -> bool {
        Command::new("git")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    }

    fn init_repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let status = Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir.path())
            .env("GIT_CONFIG_GLOBAL", dir.path().join("no-global-config"))
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .status()
            .unwrap();
        assert!(status.success());
        // A developer's global excludes file must not decide these tests.
        let empty = dir.path().join(".git").join("no-excludes");
        std::fs::write(&empty, "").unwrap();
        let status = Command::new("git")
            .args(["config", "core.excludesFile"])
            .arg(&empty)
            .current_dir(dir.path())
            .status()
            .unwrap();
        assert!(status.success());
        dir
    }

    #[test]
    fn patterns_are_anchored_and_escaped() {
        assert_eq!(ignore_pattern(".env").unwrap(), "/.env");
        assert_eq!(ignore_pattern("app/.env.local").unwrap(), "/app/.env.local");
        assert_eq!(ignore_pattern("#notes").unwrap(), "/#notes");
        assert_eq!(ignore_pattern("!keep").unwrap(), "/!keep");
        assert_eq!(ignore_pattern("a*b?[c]").unwrap(), "/a\\*b\\?\\[c]");
        assert_eq!(ignore_pattern("back\\slash").unwrap(), "/back\\\\slash");
        assert_eq!(ignore_pattern("trailing  ").unwrap(), "/trailing\\ \\ ");
        assert_eq!(ignore_pattern("new\nline"), None);
        assert_eq!(ignore_pattern(""), None);
    }

    #[test]
    fn relative_paths_stay_inside_the_root() {
        let root = Path::new("/repo");
        assert_eq!(
            relative_slash_path(root, Path::new("/repo/app/.env")).unwrap(),
            "app/.env"
        );
        assert_eq!(relative_slash_path(root, Path::new("/other/.env")), None);
        assert_eq!(relative_slash_path(root, Path::new("/repo")), None);
    }

    #[test]
    fn finds_the_repository_root_from_a_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("a/b")).unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let root = canonicalize(dir.path()).unwrap();
        assert_eq!(repository_root(&dir.path().join("a/b")).unwrap(), root);
        assert_eq!(project_root(&dir.path().join("a")).unwrap(), root);
    }

    #[test]
    fn tracked_and_ignored_files_are_detected() {
        if !git_available() {
            eprintln!("skipping: git is not installed");
            return;
        }
        let dir = init_repo();
        let repo = Repository::at(canonicalize(dir.path()).unwrap());
        std::fs::write(dir.path().join("tracked.env"), "X=1\n").unwrap();
        let status = Command::new("git")
            .args(["add", "tracked.env"])
            .current_dir(dir.path())
            .status()
            .unwrap();
        assert!(status.success());
        assert!(repo.is_tracked("tracked.env").unwrap());
        assert!(!repo.is_tracked(".env").unwrap());
        // Pathspecs are literal: a glob-looking name only matches itself.
        assert!(!repo.is_tracked("*.env").unwrap());

        std::fs::write(dir.path().join(".gitignore"), "node_modules").unwrap();
        assert!(!repo.is_ignored(".env").unwrap());
        assert!(repo.ensure_ignored(".env").unwrap());
        assert!(repo.is_ignored(".env").unwrap());
        assert!(!repo.ensure_ignored(".env").unwrap(), "already ignored");
        let gitignore = std::fs::read_to_string(dir.path().join(".gitignore")).unwrap();
        assert!(gitignore.starts_with("node_modules\n"), "{gitignore}");
        assert_eq!(gitignore.matches("/.env\n").count(), 1, "{gitignore}");
        // The anchored entry does not ignore same-named files elsewhere.
        assert!(!repo.is_ignored("sub/.env").unwrap());
    }

    #[test]
    fn a_reincluding_rule_is_reported() {
        if !git_available() {
            eprintln!("skipping: git is not installed");
            return;
        }
        let dir = init_repo();
        let repo = Repository::at(canonicalize(dir.path()).unwrap());
        std::fs::create_dir(dir.path().join("app")).unwrap();
        std::fs::write(dir.path().join("app/.gitignore"), "!.env\n").unwrap();
        assert_eq!(
            repo.ensure_ignored("app/.env").unwrap_err().code,
            "env_file_not_ignorable"
        );
    }
}
