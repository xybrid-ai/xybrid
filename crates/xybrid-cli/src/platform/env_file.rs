//! Writers for the project file that configures the key.
//!
//! | Format | Chosen for | Default variable | Comment |
//! |---|---|---|---|
//! | dotenv | `.env`, `.env.*` (`.env.local`, …), `*.env` | `XYBRID_API_KEY` | `#` |
//! | Java properties | `local.properties`, `*.properties` | `xybrid.apiKey` | `#` |
//! | xcconfig | `*.xcconfig` | `XYBRID_API_KEY` | `//` |
//!
//! Expo reads `EXPO_PUBLIC_*` variables from dotenv files: pass
//! `--env-name EXPO_PUBLIC_XYBRID_API_KEY`. Unrelated lines are preserved
//! byte for byte, a nonempty value is never replaced, and the comment above
//! the key carries the claim command and dates, never the claim token or link.
//! Any other file name is refused: a raw-key resource (for example a Unity
//! text asset) is not an assignment file, and writing `NAME=value` into one
//! would only look configured.

use std::path::Path;

use serde::Serialize;

use super::CliError;

/// The syntax of a project file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EnvFormat {
    Dotenv,
    Properties,
    Xcconfig,
}

impl EnvFormat {
    /// The format a file name implies, or `None` for a file this module
    /// does not write.
    pub(crate) fn for_path(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
        if name.ends_with(".xcconfig") {
            Some(EnvFormat::Xcconfig)
        } else if name.ends_with(".properties") {
            Some(EnvFormat::Properties)
        } else if name == ".env" || name.starts_with(".env.") || name.ends_with(".env") {
            Some(EnvFormat::Dotenv)
        } else {
            None
        }
    }

    /// [`EnvFormat::for_path`], or an error naming the supported files.
    pub(crate) fn require_for_path(path: &Path) -> Result<Self, CliError> {
        Self::for_path(path).ok_or_else(|| {
            CliError::new(
                "unsupported_env_file",
                format!(
                    "--write-env {} is not a file xybrid knows how to write. Use a dotenv file \
                     (.env, .env.local, ...), Java properties (local.properties) or an xcconfig \
                     (Secrets.xcconfig). A raw-key resource, such as a Unity text asset, needs \
                     its own recipe: copy the key from the private credentials instead",
                    path.display()
                ),
            )
        })
    }

    pub(crate) fn default_variable(self) -> &'static str {
        match self {
            EnvFormat::Dotenv | EnvFormat::Xcconfig => "XYBRID_API_KEY",
            EnvFormat::Properties => "xybrid.apiKey",
        }
    }

    fn comment_prefix(self) -> &'static str {
        match self {
            EnvFormat::Dotenv | EnvFormat::Properties => "#",
            EnvFormat::Xcconfig => "//",
        }
    }

    /// Reject a variable name the format cannot hold safely.
    pub(crate) fn validate_variable(self, name: &str) -> Result<(), CliError> {
        let first_ok = name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
        let valid = !name.is_empty()
            && name.len() <= 128
            && match self {
                EnvFormat::Dotenv | EnvFormat::Xcconfig => {
                    first_ok && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                }
                EnvFormat::Properties => name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')),
            };
        if valid {
            Ok(())
        } else {
            Err(CliError::new(
                "invalid_env_name",
                format!("{name:?} is not a valid variable name for a {self:?} file"),
            ))
        }
    }

    fn assignment(self, name: &str, key: &str) -> String {
        match self {
            EnvFormat::Dotenv | EnvFormat::Properties => format!("{name}={key}"),
            EnvFormat::Xcconfig => format!("{name} = {key}"),
        }
    }

    /// The name and raw value assigned on `line`, if it is an assignment.
    fn parse_assignment(self, line: &str) -> Option<(&str, &str)> {
        let line = line.trim_start();
        match self {
            EnvFormat::Dotenv => {
                if line.starts_with('#') {
                    return None;
                }
                let line = strip_export(line).unwrap_or(line);
                let (name, value) = line.split_once('=')?;
                Some((name.trim(), value))
            }
            EnvFormat::Properties => {
                if line.starts_with('#') || line.starts_with('!') {
                    return None;
                }
                let end = line.find(['=', ':', ' ', '\t'])?;
                let (name, rest) = line.split_at(end);
                let rest = rest.trim_start();
                let rest = rest.strip_prefix(['=', ':']).map_or(rest, str::trim_start);
                Some((name, rest))
            }
            EnvFormat::Xcconfig => {
                if line.starts_with("//") || line.starts_with('#') {
                    return None;
                }
                // `NAME[config=Debug] = value` sets NAME conditionally; the
                // condition may itself contain `=`.
                let search_from = match (line.find('['), line.find('=')) {
                    (Some(open), Some(eq)) if open < eq => line[open..].find(']')? + open,
                    _ => 0,
                };
                let eq = line[search_from..].find('=')? + search_from;
                let (name, value) = (&line[..eq], &line[eq + 1..]);
                let name = name.split('[').next().unwrap_or(name).trim();
                Some((name, value.split("//").next().unwrap_or(value)))
            }
        }
    }

    fn value_is_empty(self, raw: &str) -> bool {
        let value = raw.trim();
        match self {
            EnvFormat::Dotenv => {
                for quote in ['"', '\''] {
                    if let Some(inner) = value.strip_prefix(quote) {
                        return inner.split(quote).next().unwrap_or("").trim().is_empty();
                    }
                }
                let unquoted = match value.find(" #") {
                    Some(at) => &value[..at],
                    None if value.starts_with('#') => "",
                    None => value,
                };
                unquoted.trim().is_empty()
            }
            EnvFormat::Properties | EnvFormat::Xcconfig => value.is_empty(),
        }
    }
}

/// The rest of a dotenv line after an `export ` prefix.
fn strip_export(line: &str) -> Option<&str> {
    line.strip_prefix("export")
        .filter(|rest| rest.starts_with([' ', '\t']))
        .map(str::trim_start)
}

/// Whether a project file already configures the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lookup {
    /// The variable is not assigned.
    Absent,
    /// Assigned an empty value: it may be filled in.
    Empty,
    /// Assigned a nonempty value: never replaced.
    Present,
}

/// How `text` assigns `name`. Values are never returned.
pub(crate) fn lookup(format: EnvFormat, text: &str, name: &str) -> Lookup {
    let mut result = Lookup::Absent;
    for line in text.lines() {
        if let Some((assigned, value)) = format.parse_assignment(line) {
            if assigned == name {
                if !format.value_is_empty(value) {
                    return Lookup::Present;
                }
                result = Lookup::Empty;
            }
        }
    }
    result
}

/// `text` with `name` set to `key` and `comment` above it, everything else
/// unchanged. An empty assignment is filled in place; otherwise the block is
/// appended. Fails if a nonempty value exists.
pub(crate) fn upsert(
    format: EnvFormat,
    text: &str,
    name: &str,
    key: &str,
    comment: &[String],
) -> Result<String, CliError> {
    format.validate_variable(name)?;
    validate_key(key)?;
    let newline = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let mut block = String::new();
    for line in comment {
        let clean: String = line
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        block.push_str(format.comment_prefix());
        block.push(' ');
        block.push_str(clean.trim());
        block.push_str(newline);
    }
    let assignment = format.assignment(name, key);

    match lookup(format, text, name) {
        Lookup::Present => Err(CliError::new(
            "env_file_has_key",
            format!("{name} already has a value; it is never replaced"),
        )),
        Lookup::Absent => {
            let mut out = text.to_string();
            if !out.is_empty() && !out.ends_with('\n') {
                out.push_str(newline);
            }
            if !out.trim().is_empty() {
                out.push_str(newline);
            }
            out.push_str(&block);
            out.push_str(&assignment);
            out.push_str(newline);
            Ok(out)
        }
        Lookup::Empty => {
            let mut out = String::with_capacity(text.len() + block.len() + key.len());
            let mut replaced = false;
            for line in text.split_inclusive('\n') {
                let content = line.trim_end_matches(['\n', '\r']);
                let ending = &line[content.len()..];
                let is_target = !replaced
                    && format
                        .parse_assignment(content)
                        .is_some_and(|(n, v)| n == name && format.value_is_empty(v));
                if is_target {
                    replaced = true;
                    let export =
                        format == EnvFormat::Dotenv && strip_export(content.trim_start()).is_some();
                    out.push_str(&block);
                    if export {
                        out.push_str("export ");
                    }
                    out.push_str(&assignment);
                    out.push_str(if ending.is_empty() { newline } else { ending });
                } else {
                    out.push_str(line);
                }
            }
            Ok(out)
        }
    }
}

/// Reject anything but a plain token: a hostile server must not be able to
/// inject lines or syntax into a project file.
pub(crate) fn validate_key(key: &str) -> Result<(), CliError> {
    let ok = (8..=256).contains(&key.len())
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if ok {
        Ok(())
    } else {
        Err(CliError::new(
            "invalid_api_key_format",
            "The platform returned a key with unexpected characters; nothing was written",
        ))
    }
}

/// The file's text, or `None` if it does not exist. Symlinks are refused:
/// replacing one atomically would silently detach it.
pub(crate) fn read(path: &Path) -> Result<Option<String>, CliError> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => {
            return Err(CliError::new(
                "env_file_symlink",
                format!(
                    "{} is a symbolic link; point --write-env at the real file",
                    path.display()
                ),
            ))
        }
        Ok(meta) if !meta.is_file() => {
            return Err(CliError::new(
                "env_file_unwritable",
                format!("{} is not a regular file", path.display()),
            ))
        }
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(unwritable(path, &err)),
    }
    let bytes = std::fs::read(path).map_err(|err| unwritable(path, &err))?;
    String::from_utf8(bytes).map(Some).map_err(|_| {
        CliError::new(
            "env_file_unwritable",
            format!("{} is not UTF-8 text", path.display()),
        )
    })
}

/// Check that `path`'s directory exists and accepts new files, without
/// leaving anything behind.
pub(crate) fn check_writable(path: &Path) -> Result<(), CliError> {
    let directory = path.parent().filter(|p| !p.as_os_str().is_empty());
    let directory = directory.unwrap_or_else(|| Path::new("."));
    tempfile::Builder::new()
        .prefix(".xybrid-write-check-")
        .tempfile_in(directory)
        .map(drop)
        .map_err(|err| unwritable(path, &err))
}

/// Atomically replace (or create) `path` with `contents`. A new file is
/// `0600` on Unix; an existing file keeps its permissions.
pub(crate) fn write_atomic(path: &Path, contents: &str) -> Result<(), CliError> {
    let directory = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let existing = std::fs::metadata(path).ok().map(|m| m.permissions());
    let write = || -> std::io::Result<()> {
        use std::io::Write;
        let mut temp = tempfile::Builder::new()
            .prefix(".xybrid-env-")
            .tempfile_in(directory)?;
        temp.write_all(contents.as_bytes())?;
        temp.as_file().sync_all()?;
        if let Some(permissions) = existing.clone() {
            temp.as_file().set_permissions(permissions)?;
        }
        temp.persist(path).map_err(|err| err.error)?;
        Ok(())
    };
    write().map_err(|err| unwritable(path, &err))
}

fn unwritable(path: &Path, err: &std::io::Error) -> CliError {
    CliError::new(
        "env_file_unwritable",
        format!("Could not write {}: {err}", path.display()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "sk_live_AbCd1234EfGh5678";

    fn comment() -> Vec<String> {
        vec![
            "Xybrid provisional workspace \"my-app\".".into(),
            "Claim it by 2026-11-15: run `xybrid auth claim`.".into(),
        ]
    }

    #[test]
    fn formats_follow_the_file_name() {
        for name in [
            ".env",
            ".env.local",
            "app/.env.development.local",
            "dev.env",
        ] {
            assert_eq!(
                EnvFormat::for_path(Path::new(name)),
                Some(EnvFormat::Dotenv),
                "{name}"
            );
        }
        assert_eq!(
            EnvFormat::for_path(Path::new("android/local.properties")),
            Some(EnvFormat::Properties)
        );
        assert_eq!(
            EnvFormat::for_path(Path::new("ios/Secrets.XCCONFIG")),
            Some(EnvFormat::Xcconfig)
        );
        assert_eq!(EnvFormat::Properties.default_variable(), "xybrid.apiKey");
        // A raw-key resource is not an assignment file.
        for name in [
            "Assets/Resources/xybrid_key.txt",
            "config.json",
            "settings.yaml",
            "environment",
        ] {
            assert_eq!(EnvFormat::for_path(Path::new(name)), None, "{name}");
            assert_eq!(
                EnvFormat::require_for_path(Path::new(name))
                    .unwrap_err()
                    .code,
                "unsupported_env_file"
            );
        }
    }

    #[test]
    fn dotenv_appends_and_preserves_unrelated_content() {
        let text = "# app config\nDATABASE_URL=postgres://x\nexport OTHER=1";
        let out = upsert(EnvFormat::Dotenv, text, "XYBRID_API_KEY", KEY, &comment()).unwrap();
        assert!(out.starts_with("# app config\nDATABASE_URL=postgres://x\nexport OTHER=1\n\n"));
        assert!(out.ends_with(&format!(
            "# Xybrid provisional workspace \"my-app\".\n\
             # Claim it by 2026-11-15: run `xybrid auth claim`.\n\
             XYBRID_API_KEY={KEY}\n"
        )));
        assert_eq!(
            lookup(EnvFormat::Dotenv, &out, "XYBRID_API_KEY"),
            Lookup::Present
        );
        // Empty file: no leading blank line.
        let fresh = upsert(EnvFormat::Dotenv, "", "XYBRID_API_KEY", KEY, &[]).unwrap();
        assert_eq!(fresh, format!("XYBRID_API_KEY={KEY}\n"));
    }

    #[test]
    fn expo_variables_are_plain_dotenv() {
        let out = upsert(
            EnvFormat::Dotenv,
            "",
            "EXPO_PUBLIC_XYBRID_API_KEY",
            KEY,
            &comment(),
        )
        .unwrap();
        assert!(out.ends_with(&format!("EXPO_PUBLIC_XYBRID_API_KEY={KEY}\n")));
    }

    #[test]
    fn existing_values_are_detected_in_every_dotenv_shape() {
        for text in [
            "XYBRID_API_KEY=abc",
            "export XYBRID_API_KEY=abc",
            "  XYBRID_API_KEY = \"abc\"",
            "XYBRID_API_KEY='abc' # comment",
        ] {
            assert_eq!(
                lookup(EnvFormat::Dotenv, text, "XYBRID_API_KEY"),
                Lookup::Present,
                "{text}"
            );
            assert_eq!(
                upsert(EnvFormat::Dotenv, text, "XYBRID_API_KEY", KEY, &[])
                    .unwrap_err()
                    .code,
                "env_file_has_key"
            );
        }
        for text in [
            "XYBRID_API_KEY=",
            "XYBRID_API_KEY=\"\"",
            "XYBRID_API_KEY= # later",
        ] {
            assert_eq!(
                lookup(EnvFormat::Dotenv, text, "XYBRID_API_KEY"),
                Lookup::Empty,
                "{text}"
            );
        }
        for text in [
            "# XYBRID_API_KEY=abc",
            "XYBRID_API_KEY_OLD=abc",
            "MY_XYBRID_API_KEY=abc",
        ] {
            assert_eq!(
                lookup(EnvFormat::Dotenv, text, "XYBRID_API_KEY"),
                Lookup::Absent,
                "{text}"
            );
        }
    }

    #[test]
    fn an_empty_assignment_is_filled_in_place() {
        let text = "A=1\r\nexport XYBRID_API_KEY=\r\nB=2\r\n";
        let out = upsert(EnvFormat::Dotenv, text, "XYBRID_API_KEY", KEY, &comment()).unwrap();
        assert_eq!(
            out,
            format!(
                "A=1\r\n# Xybrid provisional workspace \"my-app\".\r\n\
                 # Claim it by 2026-11-15: run `xybrid auth claim`.\r\n\
                 export XYBRID_API_KEY={KEY}\r\nB=2\r\n"
            )
        );
    }

    #[test]
    fn properties_use_hash_comments() {
        let text = "sdk.dir=/Users/me/Library/Android/sdk\n";
        let out = upsert(
            EnvFormat::Properties,
            text,
            "xybrid.apiKey",
            KEY,
            &comment(),
        )
        .unwrap();
        assert!(out.starts_with(text));
        assert!(out.contains("# Claim it by 2026-11-15"));
        assert!(out.ends_with(&format!("xybrid.apiKey={KEY}\n")));
        for text in [
            "xybrid.apiKey=abc",
            "xybrid.apiKey : abc",
            "xybrid.apiKey abc",
        ] {
            assert_eq!(
                lookup(EnvFormat::Properties, text, "xybrid.apiKey"),
                Lookup::Present,
                "{text}"
            );
        }
        assert_eq!(
            lookup(
                EnvFormat::Properties,
                "! xybrid.apiKey=abc",
                "xybrid.apiKey"
            ),
            Lookup::Absent
        );
        assert_eq!(
            lookup(EnvFormat::Properties, "xybrid.apiKey=", "xybrid.apiKey"),
            Lookup::Empty
        );
    }

    #[test]
    fn xcconfig_uses_slash_comments() {
        let text = "#include \"Base.xcconfig\"\nPRODUCT_NAME = MyApp\n";
        let out = upsert(EnvFormat::Xcconfig, text, "XYBRID_API_KEY", KEY, &comment()).unwrap();
        assert!(out.starts_with(text));
        assert!(out.contains("// Xybrid provisional workspace"));
        assert!(!out.contains("\n# Xybrid"));
        assert!(out.ends_with(&format!("XYBRID_API_KEY = {KEY}\n")));
        assert_eq!(
            lookup(
                EnvFormat::Xcconfig,
                "XYBRID_API_KEY[config=Debug] = abc // note",
                "XYBRID_API_KEY"
            ),
            Lookup::Present
        );
        assert_eq!(
            lookup(
                EnvFormat::Xcconfig,
                "XYBRID_API_KEY = // filled in later",
                "XYBRID_API_KEY"
            ),
            Lookup::Empty
        );
        assert_eq!(
            lookup(
                EnvFormat::Xcconfig,
                "XYBRID_API_KEY[config=Debug] =",
                "XYBRID_API_KEY"
            ),
            Lookup::Empty
        );
        assert_eq!(
            lookup(
                EnvFormat::Xcconfig,
                "// XYBRID_API_KEY = abc",
                "XYBRID_API_KEY"
            ),
            Lookup::Absent
        );
    }

    #[test]
    fn names_and_keys_are_validated() {
        assert!(EnvFormat::Dotenv
            .validate_variable("EXPO_PUBLIC_XYBRID_API_KEY")
            .is_ok());
        assert!(EnvFormat::Properties
            .validate_variable("xybrid.apiKey")
            .is_ok());
        for bad in ["", "1ABC", "A B", "A=B", "A\nB", "xybrid.apiKey"] {
            assert!(EnvFormat::Dotenv.validate_variable(bad).is_err(), "{bad:?}");
        }
        assert!(EnvFormat::Properties.validate_variable("a=b").is_err());
        for bad in [
            "short",
            "sk_live_abc\nEVIL=1",
            "sk_live_abc def",
            "sk_live_\"quoted\"",
        ] {
            assert_eq!(
                validate_key(bad).unwrap_err().code,
                "invalid_api_key_format",
                "{bad:?}"
            );
        }
        // Comments cannot smuggle extra lines in.
        let out = upsert(
            EnvFormat::Dotenv,
            "",
            "XYBRID_API_KEY",
            KEY,
            &["name\nEVIL=1".to_string()],
        )
        .unwrap();
        assert_eq!(out.lines().count(), 2, "{out}");
    }

    #[test]
    fn atomic_writes_keep_existing_permissions_and_create_private_files() {
        let dir = tempfile::tempdir().unwrap();
        let fresh = dir.path().join(".env");
        check_writable(&fresh).unwrap();
        assert_eq!(read(&fresh).unwrap(), None);
        write_atomic(&fresh, "A=1\n").unwrap();
        assert_eq!(read(&fresh).unwrap().as_deref(), Some("A=1\n"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&fresh).unwrap().permissions().mode() & 0o777,
                0o600
            );
            let shared = dir.path().join("local.properties");
            std::fs::write(&shared, "x=1\n").unwrap();
            std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o644)).unwrap();
            write_atomic(&shared, "x=1\ny=2\n").unwrap();
            assert_eq!(
                std::fs::metadata(&shared).unwrap().permissions().mode() & 0o777,
                0o644
            );

            let link = dir.path().join("linked.env");
            std::os::unix::fs::symlink(&fresh, &link).unwrap();
            assert_eq!(read(&link).unwrap_err().code, "env_file_symlink");
        }
        let names: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.iter().all(|n| !n.starts_with(".xybrid-")),
            "{names:?}"
        );
    }
}
