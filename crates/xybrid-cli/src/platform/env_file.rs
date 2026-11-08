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
//! A single empty assignment is filled in place, keeping an xcconfig
//! condition such as `[config=Debug]`; several empty assignments are refused,
//! since which one an application reads depends on its loader.
//! Any other file name is refused: a raw-key resource (for example a Unity
//! text asset) is not an assignment file, and writing `NAME=value` into one
//! would only look configured.
//!
//! A line that loaders read differently is neither filled nor trusted. When
//! the variable is set in or at the edge of one of these multi-line
//! constructs, or the file ends inside one (an appended key would land in
//! it), the file is refused as ambiguous:
//!
//! | Format | Construct | How readings differ |
//! |---|---|---|
//! | dotenv | a quoted value left open at the end of its line | Node's and Python's dotenv run it to the closing quote on a later line; line-based loaders end it with the line |
//! | Java properties | a line ending in an odd number of backslashes | `java.util.Properties` joins the next line to it; a line reader does not |
//! | xcconfig | a value ending in a backslash | Xcode's current parser joins the next line to it; earlier parsers and line readers do not |
//! | xcconfig | a `/* … */` block | Xcode has no block comments: it rejects the lines starting with `/*` and reads the assignments up to `*/` as live, which the block marks as commented out |
//!
//! The xcconfig rules follow Xcode's parser (swift-build's
//! `MacroConfigFileParser`): only `//` starts a comment, and a value whose
//! last character is `\` continues on the next line.

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

    /// `name` set to `key`, under an xcconfig `condition` (empty for none).
    fn assignment(self, name: &str, condition: &str, key: &str) -> String {
        match self {
            EnvFormat::Dotenv | EnvFormat::Properties => format!("{name}={key}"),
            EnvFormat::Xcconfig => format!("{name}{condition} = {key}"),
        }
    }

    /// The assignment on `line`, if it is one.
    fn parse_assignment(self, line: &str) -> Option<Assignment<'_>> {
        let line = line.trim_start();
        let plain = |name, value| Assignment {
            name,
            condition: "",
            value,
        };
        match self {
            EnvFormat::Dotenv => {
                if line.starts_with('#') {
                    return None;
                }
                let line = strip_export(line).unwrap_or(line);
                let (name, value) = line.split_once('=')?;
                Some(plain(name.trim(), value))
            }
            EnvFormat::Properties => {
                if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
                    return None;
                }
                // As `java.util.Properties` reads it: the key ends at the
                // first `=`, `:` or blank, and a line holding only a key sets
                // it to the empty string.
                let end = line
                    .find(['=', ':', ' ', '\t', '\x0c'])
                    .unwrap_or(line.len());
                let (name, rest) = line.split_at(end);
                let rest = rest.trim_start();
                let rest = rest.strip_prefix(['=', ':']).map_or(rest, str::trim_start);
                Some(plain(name, rest))
            }
            EnvFormat::Xcconfig => {
                if line.starts_with("//") || line.starts_with('#') {
                    return None;
                }
                // `NAME[config=Debug][sdk=iphoneos*] = value` sets NAME
                // conditionally; each condition may itself contain `=`.
                let mut from = 0;
                let eq = loop {
                    let eq = line[from..].find('=')? + from;
                    match line[from..eq].find('[') {
                        Some(open) => from = line[from + open..].find(']')? + from + open + 1,
                        None => break eq,
                    }
                };
                let (left, value) = (&line[..eq], &line[eq + 1..]);
                let (name, condition) = match left.find('[') {
                    Some(open) => (left[..open].trim(), left[open..].trim()),
                    None => (left.trim(), ""),
                };
                Some(Assignment {
                    name,
                    condition,
                    value: value.split("//").next().unwrap_or(value),
                })
            }
        }
    }

    /// The value of an assignment as an application would read it: quotes,
    /// inline comments and surrounding blanks removed.
    fn value(self, raw: &str) -> &str {
        let value = raw.trim();
        match self {
            EnvFormat::Dotenv => {
                for quote in ['"', '\''] {
                    if let Some(inner) = value.strip_prefix(quote) {
                        return inner.split(quote).next().unwrap_or("").trim();
                    }
                }
                match value.find(" #") {
                    Some(at) => value[..at].trim(),
                    None if value.starts_with('#') => "",
                    None => value,
                }
            }
            EnvFormat::Properties | EnvFormat::Xcconfig => value,
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

/// One assignment line.
struct Assignment<'a> {
    name: &'a str,
    /// An xcconfig build-setting condition such as `[config=Debug]`,
    /// verbatim (empty elsewhere): filling the line keeps it, so the key
    /// reaches only the builds the line was written for.
    condition: &'a str,
    /// The raw value: quotes and comments not yet removed.
    value: &'a str,
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
    /// Assigned an empty value on one line: it may be filled in.
    Empty,
    /// No line can be filled, nor the file appended to, safely.
    Ambiguous(Ambiguity),
    /// Assigned a nonempty value: never replaced.
    Present,
}

/// Why a project file is [`Lookup::Ambiguous`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ambiguity {
    /// The variable is assigned an empty value on more than one line, and
    /// no value elsewhere: which one an application reads depends on its
    /// loader (most take the last), so none is filled.
    SeveralEmpty,
    /// It is set in or at the edge of a dotenv quoted value spanning lines,
    /// or the file ends inside one.
    QuotedLines,
    /// It is set on a line a trailing backslash joins to the next, or on a
    /// line so joined, or the file ends with such a backslash.
    Continuation,
    /// It is set in or at the edge of an xcconfig `/* … */` block, or the
    /// file ends inside one.
    BlockComment,
}

/// How `text` assigns `name`. Values are never returned.
///
/// A line in a multi-line construct that could assign `name` makes the file
/// ambiguous even when another line holds a value: which one wins depends on
/// how the construct is read.
pub(crate) fn lookup(format: EnvFormat, text: &str, name: &str) -> Lookup {
    let mut scanner = Scanner::new(format);
    let (mut empty, mut present, mut unclear) = (0_usize, false, None);
    for line in lines(text) {
        match scanner.read(line) {
            Line::Clear(Some(assignment)) if assignment.name == name => {
                if format.value_is_empty(assignment.value) {
                    empty += 1;
                } else {
                    present = true;
                }
            }
            Line::Unclear(why, readings) if readings.iter().flatten().any(|a| a.name == name) => {
                unclear.get_or_insert(why);
            }
            _ => {}
        }
    }
    match (unclear, present, empty) {
        (Some(why), _, _) => Lookup::Ambiguous(why),
        (None, true, _) => Lookup::Present,
        // Appending after an open construct would land inside it.
        (None, false, 0) => scanner.open().map_or(Lookup::Absent, Lookup::Ambiguous),
        (None, false, 1) => Lookup::Empty,
        (None, false, _) => Lookup::Ambiguous(Ambiguity::SeveralEmpty),
    }
}

/// The error for a [`Lookup::Ambiguous`] file, named `file`.
pub(crate) fn ambiguous(file: &str, name: &str, why: Ambiguity) -> CliError {
    let message = match why {
        Ambiguity::SeveralEmpty => format!(
            "{file} assigns {name} an empty value on more than one line, and which one the app \
             reads depends on its loader; keep a single {name} line or choose another \
             --write-env file"
        ),
        Ambiguity::QuotedLines => format!(
            "{file} has a quoted value spanning several lines that holds or sets {name}, or that \
             never closes, and dotenv loaders disagree on where such a value ends; set {name} \
             on a line of its own outside any quoted value, or choose another --write-env file"
        ),
        Ambiguity::Continuation => format!(
            "{file} sets {name} on a line that a trailing backslash joins to another, or ends \
             with such a backslash, and loaders disagree on which lines that joins; set {name} \
             on a line of its own, or choose another --write-env file"
        ),
        Ambiguity::BlockComment => format!(
            "{file} sets {name} in or next to a /* */ block, or leaves one open. Xcode has no \
             block comments: it reads an assignment inside one as live although the block \
             marks it as commented out. Comment lines out with // instead, or choose another \
             --write-env file"
        ),
    };
    CliError::new("env_file_ambiguous", message)
}

/// Whether `text` assigns exactly `value` to `name` (for example, the key a
/// previous `xybrid connect` wrote) on a line every loader reads the same
/// way. The value itself is never returned.
pub(crate) fn assigns(format: EnvFormat, text: &str, name: &str, value: &str) -> bool {
    let mut scanner = Scanner::new(format);
    lines(text).any(|line| {
        matches!(
            scanner.read(line),
            Line::Clear(Some(a)) if a.name == name && format.value(a.value) == value
        )
    })
}

/// Whether some reading of `text` may give `name` a nonempty value: a line
/// every loader reads the same way assigns one, or a line in a multi-line
/// construct could assign it anything at all. Unlike [`lookup`], this never
/// takes an ambiguous file as unset (for telling the user a file may still
/// configure a key).
pub(crate) fn may_assign(format: EnvFormat, text: &str, name: &str) -> bool {
    let mut scanner = Scanner::new(format);
    lines(text).any(|line| match scanner.read(line) {
        Line::Clear(reading) => {
            reading.is_some_and(|a| a.name == name && !format.value_is_empty(a.value))
        }
        Line::Unclear(_, readings) => readings
            .iter()
            .flatten()
            .any(|a| a.name == name && !a.value.trim().is_empty()),
    })
}

/// The lines of `text` without their endings, as [`upsert`] rewrites them.
fn lines(text: &str) -> impl Iterator<Item = &str> {
    text.split_inclusive('\n')
        .map(|line| line.trim_end_matches(['\n', '\r']))
}

/// How one line reads.
enum Line<'a> {
    /// The same for every loader: this assignment, or none.
    Clear(Option<Assignment<'a>>),
    /// In or at the edge of a multi-line construct: the assignments the line
    /// could be, depending on how a loader reads the construct.
    Unclear(Ambiguity, [Option<Assignment<'a>>; 2]),
}

/// The construct a line starts in, after the lines before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Plain,
    /// Inside a dotenv value opened by this quote.
    Quoted(char),
    /// Joined to the line before by its trailing backslash.
    Continued,
    /// Inside an xcconfig `/* … */` block.
    BlockComment,
}

/// Reads a file's lines in order, following its format's multi-line
/// constructs (see the module documentation).
struct Scanner {
    format: EnvFormat,
    state: State,
}

impl Scanner {
    fn new(format: EnvFormat) -> Self {
        Self {
            format,
            state: State::Plain,
        }
    }

    /// The construct the lines read so far end inside, if any.
    fn open(&self) -> Option<Ambiguity> {
        match self.state {
            State::Plain => None,
            State::Quoted(_) => Some(Ambiguity::QuotedLines),
            State::Continued => Some(Ambiguity::Continuation),
            State::BlockComment => Some(Ambiguity::BlockComment),
        }
    }

    /// The next line, without its ending.
    fn read<'a>(&mut self, line: &'a str) -> Line<'a> {
        match self.format {
            EnvFormat::Dotenv => self.read_dotenv(line),
            EnvFormat::Properties => self.read_properties(line),
            EnvFormat::Xcconfig => self.read_xcconfig(line),
        }
    }

    /// A value opened by `"`, `'` or a backtick and not closed on its line
    /// runs to the next unescaped matching quote for Node's and Python's
    /// dotenv; a line-based loader ends it with the line.
    fn read_dotenv<'a>(&mut self, line: &'a str) -> Line<'a> {
        let reading = EnvFormat::Dotenv.parse_assignment(line);
        if let State::Quoted(quote) = self.state {
            if closing_quote(line, quote).is_some() {
                self.state = State::Plain;
            }
            return Line::Unclear(Ambiguity::QuotedLines, [reading, None]);
        }
        let Some(assignment) = reading else {
            return Line::Clear(None);
        };
        let value = assignment.value.trim_start();
        match value.chars().next() {
            Some(quote @ ('"' | '\'' | '`')) if closing_quote(&value[1..], quote).is_none() => {
                self.state = State::Quoted(quote);
                Line::Unclear(Ambiguity::QuotedLines, [Some(assignment), None])
            }
            _ => Line::Clear(Some(assignment)),
        }
    }

    /// A line ending in an odd number of backslashes continues on the next,
    /// as `java.util.Properties` reads it: the next line's leading blanks are
    /// skipped, a blank line ends the continuation, and a comment line never
    /// continues.
    fn read_properties<'a>(&mut self, line: &'a str) -> Line<'a> {
        let reading = EnvFormat::Properties.parse_assignment(line);
        let rest = line.trim_start_matches([' ', '\t', '\x0c']);
        let continues = (line.len() - line.trim_end_matches('\\').len()) % 2 == 1;
        if self.state == State::Continued {
            if rest.is_empty() {
                self.state = State::Plain;
                return Line::Clear(None);
            }
            if !continues {
                self.state = State::Plain;
            }
            return Line::Unclear(Ambiguity::Continuation, [reading, None]);
        }
        if rest.is_empty() || rest.starts_with(['#', '!']) {
            return Line::Clear(None);
        }
        if continues {
            self.state = State::Continued;
            return Line::Unclear(Ambiguity::Continuation, [reading, None]);
        }
        Line::Clear(reading)
    }

    /// Xcode joins a value ending in a backslash to the next line, up to a
    /// line holding only blanks or a `//` comment; earlier parsers and line
    /// readers don't. Xcode has no block comments: it rejects a line starting
    /// with `/*` and reads the lines after it as usual, while the author
    /// meant everything up to `*/` as commented out.
    fn read_xcconfig<'a>(&mut self, line: &'a str) -> Line<'a> {
        let format = EnvFormat::Xcconfig;
        let reading = format.parse_assignment(line);
        match self.state {
            State::BlockComment => {
                // What follows `*/` is what the author meant to be live.
                let after = line.find("*/").map(|at| {
                    self.state = State::Plain;
                    &line[at + 2..]
                });
                let meant = after.and_then(|rest| format.parse_assignment(rest));
                return Line::Unclear(Ambiguity::BlockComment, [reading, meant]);
            }
            State::Continued => {
                let rest = line.split("//").next().unwrap_or(line);
                if rest.trim().is_empty() {
                    self.state = State::Plain;
                    return Line::Clear(None);
                }
                if !continues_xcconfig(rest) {
                    self.state = State::Plain;
                }
                return Line::Unclear(Ambiguity::Continuation, [reading, None]);
            }
            State::Plain | State::Quoted(_) => {}
        }
        if let Some(comment) = line.trim_start().strip_prefix("/*") {
            let after = match comment.find("*/") {
                Some(at) => &comment[at + 2..],
                None => {
                    self.state = State::BlockComment;
                    ""
                }
            };
            return Line::Unclear(
                Ambiguity::BlockComment,
                [format.parse_assignment(after), None],
            );
        }
        let Some(assignment) = reading else {
            return Line::Clear(None);
        };
        if continues_xcconfig(assignment.value) {
            self.state = State::Continued;
            return Line::Unclear(Ambiguity::Continuation, [Some(assignment), None]);
        }
        // Xcode reads `/* … */` in a value as part of it.
        if assignment.value.contains("/*") || assignment.value.contains("*/") {
            return Line::Unclear(Ambiguity::BlockComment, [Some(assignment), None]);
        }
        Line::Clear(Some(assignment))
    }
}

/// Whether an xcconfig value (its `//` comment removed) continues on the
/// next line: after trailing blanks and one optional `;`, it ends in `\`.
fn continues_xcconfig(value: &str) -> bool {
    let value = value.trim_end();
    value
        .strip_suffix(';')
        .map_or(value, str::trim_end)
        .ends_with('\\')
}

/// Where the first `quote` in `text` that no backslash escapes is.
fn closing_quote(text: &str, quote: char) -> Option<usize> {
    let mut chars = text.char_indices();
    while let Some((at, c)) = chars.next() {
        if c == quote {
            return Some(at);
        }
        // `\"` is an escaped quote; any other backslash stands for itself.
        if c == '\\' && text[at + 1..].starts_with(quote) {
            chars.next();
        }
    }
    None
}

/// `text` with `name` set to `key` and `comment` above it, everything else
/// unchanged. A single empty assignment is filled in place (keeping an
/// xcconfig condition); otherwise the block is appended. Fails if a nonempty
/// value exists, or if several empty assignments leave no single line to fill.
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
    match lookup(format, text, name) {
        Lookup::Present => Err(CliError::new(
            "env_file_has_key",
            format!("{name} already has a value; it is never replaced"),
        )),
        Lookup::Ambiguous(why) => Err(ambiguous("the file", name, why)),
        Lookup::Absent => {
            let mut out = text.to_string();
            if !out.is_empty() && !out.ends_with('\n') {
                out.push_str(newline);
            }
            if !out.trim().is_empty() {
                out.push_str(newline);
            }
            out.push_str(&block);
            out.push_str(&format.assignment(name, "", key));
            out.push_str(newline);
            Ok(out)
        }
        Lookup::Empty => {
            // `lookup` found exactly one empty assignment, on a line every
            // loader reads the same way: fill that line.
            let mut scanner = Scanner::new(format);
            let mut out = String::with_capacity(text.len() + block.len() + key.len());
            for line in text.split_inclusive('\n') {
                let content = line.trim_end_matches(['\n', '\r']);
                let ending = &line[content.len()..];
                let target = match scanner.read(content) {
                    Line::Clear(Some(a)) if a.name == name && format.value_is_empty(a.value) => {
                        Some(a)
                    }
                    _ => None,
                };
                if let Some(target) = target {
                    let export =
                        format == EnvFormat::Dotenv && strip_export(content.trim_start()).is_some();
                    out.push_str(&block);
                    if export {
                        out.push_str("export ");
                    }
                    out.push_str(&format.assignment(name, target.condition, key));
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
    fn assignments_are_compared_as_applications_read_them() {
        for (format, text) in [
            (EnvFormat::Dotenv, format!("XYBRID_API_KEY={KEY}")),
            (
                EnvFormat::Dotenv,
                format!("export XYBRID_API_KEY=\"{KEY}\" # note"),
            ),
            (EnvFormat::Dotenv, format!("XYBRID_API_KEY='{KEY}'")),
            (EnvFormat::Properties, format!("xybrid.apiKey = {KEY}")),
            (
                EnvFormat::Xcconfig,
                format!("XYBRID_API_KEY = {KEY} // note"),
            ),
        ] {
            let name = format.default_variable();
            assert!(assigns(format, &text, name, KEY), "{text}");
            assert!(!assigns(format, &text, name, "sk_live_other"), "{text}");
            assert!(!assigns(format, &text, "OTHER", KEY), "{text}");
        }
        assert!(!assigns(
            EnvFormat::Dotenv,
            &format!("# XYBRID_API_KEY={KEY}"),
            "XYBRID_API_KEY",
            KEY
        ));
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

    /// Loaders differ on which repeated assignment wins (most take the last),
    /// so filling one could leave the app reading an empty one: refused in
    /// every format, the file left as it was.
    #[test]
    fn several_empty_assignments_are_refused_in_every_format() {
        for (format, text) in [
            (
                EnvFormat::Dotenv,
                "XYBRID_API_KEY=\nA=1\nexport XYBRID_API_KEY=\"\"\n",
            ),
            (EnvFormat::Properties, "xybrid.apiKey=\nsdk.dir=/sdk\nxybrid.apiKey :\n"),
            // A bare key is an empty assignment, and the one Java reads last.
            (EnvFormat::Properties, "xybrid.apiKey=\nsdk.dir=/sdk\nxybrid.apiKey\n"),
            (
                EnvFormat::Xcconfig,
                "XYBRID_API_KEY =\nPRODUCT_NAME = MyApp\nXYBRID_API_KEY[config=Release] = // later\n",
            ),
        ] {
            let name = format.default_variable();
            assert_eq!(
                lookup(format, text, name),
                Lookup::Ambiguous(Ambiguity::SeveralEmpty),
                "{text}"
            );
            let err = upsert(format, text, name, KEY, &comment()).unwrap_err();
            assert_eq!(err.code, "env_file_ambiguous", "{text}");
            assert!(err.message.contains(name), "{}", err.message);
            assert!(!err.message.contains(KEY));
        }
        // A value anywhere is still a configured key, never replaced.
        assert_eq!(
            lookup(
                EnvFormat::Dotenv,
                "XYBRID_API_KEY=\nXYBRID_API_KEY=abc\nXYBRID_API_KEY=\n",
                "XYBRID_API_KEY"
            ),
            Lookup::Present
        );
        assert_eq!(
            ambiguous(".env", "XYBRID_API_KEY", Ambiguity::SeveralEmpty).code,
            "env_file_ambiguous"
        );
    }

    /// `NAME[config=Debug] =` is meant for Debug builds only: filling it keeps
    /// the condition, so the key never reaches Release.
    #[test]
    fn a_conditional_xcconfig_assignment_keeps_its_condition() {
        for (line, filled) in [
            (
                "XYBRID_API_KEY[config=Debug] =",
                format!("XYBRID_API_KEY[config=Debug] = {KEY}"),
            ),
            (
                "  XYBRID_API_KEY[config=Debug][sdk=iphoneos*] = // filled in later",
                format!("XYBRID_API_KEY[config=Debug][sdk=iphoneos*] = {KEY}"),
            ),
            ("XYBRID_API_KEY =", format!("XYBRID_API_KEY = {KEY}")),
        ] {
            let text = format!("PRODUCT_NAME = MyApp\n{line}\nOTHER = 1\n");
            let out = upsert(EnvFormat::Xcconfig, &text, "XYBRID_API_KEY", KEY, &[]).unwrap();
            assert_eq!(
                out,
                format!("PRODUCT_NAME = MyApp\n{filled}\nOTHER = 1\n"),
                "{line}"
            );
            assert!(assigns(EnvFormat::Xcconfig, &out, "XYBRID_API_KEY", KEY));
        }
        // Every condition is skipped before the assignment's own `=`.
        let line = "XYBRID_API_KEY[config=Debug][sdk=iphoneos*] = abc // note";
        assert_eq!(
            lookup(EnvFormat::Xcconfig, line, "XYBRID_API_KEY"),
            Lookup::Present
        );
        assert!(assigns(EnvFormat::Xcconfig, line, "XYBRID_API_KEY", "abc"));
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

    /// `java.util.Properties` reads a line holding only a key (blanks around
    /// it allowed) as that key set to the empty string.
    #[test]
    fn a_bare_properties_key_is_an_empty_assignment() {
        for line in [
            "xybrid.apiKey",
            "  xybrid.apiKey",
            "xybrid.apiKey\t",
            "\x0cxybrid.apiKey",
        ] {
            assert_eq!(
                lookup(EnvFormat::Properties, line, "xybrid.apiKey"),
                Lookup::Empty,
                "{line:?}"
            );
            let text = format!("sdk.dir=/sdk\n{line}\nother=1\n");
            let out = upsert(EnvFormat::Properties, &text, "xybrid.apiKey", KEY, &[]).unwrap();
            assert_eq!(out, format!("sdk.dir=/sdk\nxybrid.apiKey={KEY}\nother=1\n"));
        }
        // Blank lines and other bare keys assign nothing to it.
        for text in ["", "   ", "\n\t\n", "xybrid.apiKeyOld", "xybrid"] {
            assert_eq!(
                lookup(EnvFormat::Properties, text, "xybrid.apiKey"),
                Lookup::Absent,
                "{text:?}"
            );
        }
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

    /// Asserts that `format` reads `text` as `expected` for its default
    /// variable, and that `upsert` refuses an ambiguous file untouched.
    fn assert_lookup(format: EnvFormat, text: &str, expected: Lookup) {
        let name = format.default_variable();
        assert_eq!(lookup(format, text, name), expected, "{text:?}");
        if let Lookup::Ambiguous(why) = expected {
            let err = upsert(format, text, name, KEY, &comment()).unwrap_err();
            assert_eq!(err, ambiguous("the file", name, why), "{text:?}");
        }
    }

    /// Xcode has no block comments: it rejects a line starting with `/*` and
    /// reads the lines after it as live. A key line in such a block is meant
    /// as commented out but read as set, so it is neither filled nor taken
    /// as a configured key.
    #[test]
    fn xcconfig_block_comments_around_the_key_are_refused() {
        let blocked = Lookup::Ambiguous(Ambiguity::BlockComment);
        for text in [
            "/*\nXYBRID_API_KEY =\n*/\n",
            "/*\nXYBRID_API_KEY = old\n*/\n",
            "PRODUCT_NAME = MyApp\n  /* disabled:\n  XYBRID_API_KEY[config=Debug] =\n  */\n",
            "/*\n * notes\n */ XYBRID_API_KEY =\n",
            "/* note */ XYBRID_API_KEY = old\n",
            "XYBRID_API_KEY = /* set me */\n",
            // A value elsewhere doesn't settle it: Xcode reads both lines.
            "XYBRID_API_KEY = abc\n/*\nXYBRID_API_KEY =\n*/\n",
            // Appending after an open block would put the key inside it.
            "PRODUCT_NAME = MyApp\n/* notes\n",
        ] {
            assert_lookup(EnvFormat::Xcconfig, text, blocked);
        }
        // A one-line block is not an assignment for Xcode or its author.
        let text = "/* XYBRID_API_KEY = old */\nPRODUCT_NAME = MyApp\n";
        assert_lookup(EnvFormat::Xcconfig, text, Lookup::Absent);
        let out = upsert(EnvFormat::Xcconfig, text, "XYBRID_API_KEY", KEY, &[]).unwrap();
        assert_eq!(out, format!("{text}\nXYBRID_API_KEY = {KEY}\n"));
        // A block elsewhere leaves a plain line to fill.
        let text = "/*\nPRODUCT_NAME = Old\n*/\nXYBRID_API_KEY =\n";
        assert_lookup(EnvFormat::Xcconfig, text, Lookup::Empty);
        let out = upsert(EnvFormat::Xcconfig, text, "XYBRID_API_KEY", KEY, &[]).unwrap();
        assert_eq!(
            out,
            format!("/*\nPRODUCT_NAME = Old\n*/\nXYBRID_API_KEY = {KEY}\n")
        );
        // `/*` inside another setting's value is part of that value.
        let text = "HEADER_SEARCH_PATHS = $(SRCROOT)/**\nXYBRID_API_KEY =\n";
        assert_lookup(EnvFormat::Xcconfig, text, Lookup::Empty);
        assert_lookup(
            EnvFormat::Xcconfig,
            "HEADER_SEARCH_PATHS = $(SRCROOT)/**\n",
            Lookup::Absent,
        );
    }

    /// Xcode joins a value ending in `\` to the next line; earlier parsers
    /// and line readers don't, so a key line on either side is refused.
    #[test]
    fn continued_xcconfig_values_are_refused_around_the_key() {
        let continued = Lookup::Ambiguous(Ambiguity::Continuation);
        for text in [
            "OTHER_FLAGS = -a \\\nXYBRID_API_KEY =\n",
            "OTHER_FLAGS = -a \\ // note\n  XYBRID_API_KEY = abc\n",
            "OTHER_FLAGS = -a \\;\nXYBRID_API_KEY =\n",
            "OTHER_FLAGS = -a \\\n  -b \\\n  XYBRID_API_KEY =\n",
            "XYBRID_API_KEY = \\\n  $(SECRET)\n",
            "XYBRID_API_KEY = abc\nOTHER_FLAGS = -a \\\nXYBRID_API_KEY =\n",
            "OTHER_FLAGS = -a \\",
        ] {
            assert_lookup(EnvFormat::Xcconfig, text, continued);
        }
        // The join ends at a line without `\`, or at a blank or comment line.
        for gap in ["  -b\n", "\n", "// end\n"] {
            let text = format!("OTHER_FLAGS = -a \\\n{gap}XYBRID_API_KEY =\n");
            assert_lookup(EnvFormat::Xcconfig, &text, Lookup::Empty);
            let out = upsert(EnvFormat::Xcconfig, &text, "XYBRID_API_KEY", KEY, &[]).unwrap();
            assert_eq!(
                out,
                format!("OTHER_FLAGS = -a \\\n{gap}XYBRID_API_KEY = {KEY}\n")
            );
        }
    }

    /// `java.util.Properties` joins a line ending in an odd number of
    /// backslashes to the next one; a line reader doesn't.
    #[test]
    fn continued_properties_lines_are_refused_around_the_key() {
        let continued = Lookup::Ambiguous(Ambiguity::Continuation);
        for text in [
            "other = foo \\\nxybrid.apiKey=\n",
            "other = foo \\\n  xybrid.apiKey=abc\n",
            "other = a \\\\\\\nxybrid.apiKey=\n",
            "xybrid.apiKey = \\\n  abc\n",
            "xybrid.apiKey = abc\nother = \\\nxybrid.apiKey\n",
            "other = foo \\",
        ] {
            assert_lookup(EnvFormat::Properties, text, continued);
        }
        // An escaped backslash, a comment and a blank line don't continue.
        for (before, after) in [
            ("sdk.dir = C:\\\\\n", ""),
            ("# note \\\n", ""),
            ("other = foo \\\n", "\n"),
        ] {
            let text = format!("{before}{after}xybrid.apiKey=\n");
            assert_lookup(EnvFormat::Properties, &text, Lookup::Empty);
            let out = upsert(EnvFormat::Properties, &text, "xybrid.apiKey", KEY, &[]).unwrap();
            assert_eq!(out, format!("{before}{after}xybrid.apiKey={KEY}\n"));
        }
    }

    /// Node's and Python's dotenv run a quoted value to its closing quote on
    /// a later line; line-based loaders end it with the line.
    #[test]
    fn dotenv_quoted_values_spanning_lines_are_refused_around_the_key() {
        let quoted = Lookup::Ambiguous(Ambiguity::QuotedLines);
        for text in [
            "CERT=\"-----BEGIN\nXYBRID_API_KEY=\n-----END\"\n",
            "CERT='-----BEGIN\nexport XYBRID_API_KEY=abc\n-----END'\n",
            "CERT=`one\nXYBRID_API_KEY=\n`\n",
            "A=\"x\\\"\nXYBRID_API_KEY=\n\"\n",
            "XYBRID_API_KEY=\"\nabc\"\n",
            "CERT=\"open\nXYBRID_API_KEY=\"x\"\n",
            "XYBRID_API_KEY=abc\nCERT=\"\nXYBRID_API_KEY=\n\"\n",
            "CERT=\"-----BEGIN\n",
        ] {
            assert_lookup(EnvFormat::Dotenv, text, quoted);
        }
        // A value closed on its line, a commented quote, or a closed block
        // elsewhere leaves a plain line to fill.
        for before in [
            "A=\"one line\" # \"quoted\"\n",
            "A='it''s'\n",
            "# CERT=\"open\n",
            "CERT=\"-----BEGIN\nMIIE==\n-----END\"\n",
        ] {
            let text = format!("{before}XYBRID_API_KEY=\n");
            assert_lookup(EnvFormat::Dotenv, &text, Lookup::Empty);
            let out = upsert(EnvFormat::Dotenv, &text, "XYBRID_API_KEY", KEY, &[]).unwrap();
            assert_eq!(out, format!("{before}XYBRID_API_KEY={KEY}\n"));
        }
        // A line inside a quoted value never counts as the saved key.
        let text = format!("CERT=\"\nXYBRID_API_KEY={KEY}\n\"\n");
        assert!(!assigns(EnvFormat::Dotenv, &text, "XYBRID_API_KEY", KEY));
    }

    /// Logout asks whether a file may still configure the key: a file too
    /// ambiguous to write into is not taken as unset.
    #[test]
    fn a_key_may_still_be_assigned_in_an_ambiguous_file() {
        for (format, text) in [
            (EnvFormat::Dotenv, format!("XYBRID_API_KEY={KEY}\n")),
            (EnvFormat::Dotenv, "XYBRID_API_KEY=\"\nabc\"\n".into()),
            (
                EnvFormat::Dotenv,
                format!("XYBRID_API_KEY={KEY}\nCERT=\"\n"),
            ),
            (
                EnvFormat::Properties,
                "other = \\\nxybrid.apiKey=abc\n".into(),
            ),
            (EnvFormat::Xcconfig, "/*\nXYBRID_API_KEY = abc\n*/\n".into()),
            (
                EnvFormat::Xcconfig,
                "XYBRID_API_KEY = \\\n  $(SECRET)\n".into(),
            ),
        ] {
            let name = format.default_variable();
            assert!(may_assign(format, &text, name), "{text:?}");
        }
        for (format, text) in [
            (EnvFormat::Dotenv, "XYBRID_API_KEY=\n# XYBRID_API_KEY=abc\n"),
            (EnvFormat::Dotenv, "OTHER=1\nCERT=\"\n"),
            (EnvFormat::Xcconfig, "/*\nXYBRID_API_KEY =\n*/\n"),
            (EnvFormat::Xcconfig, "/* XYBRID_API_KEY = abc */\n"),
        ] {
            let name = format.default_variable();
            assert!(!may_assign(format, text, name), "{text:?}");
        }
    }

    #[test]
    fn ambiguity_errors_name_the_file_and_variable() {
        for why in [
            Ambiguity::SeveralEmpty,
            Ambiguity::QuotedLines,
            Ambiguity::Continuation,
            Ambiguity::BlockComment,
        ] {
            let err = ambiguous("ios/Secrets.xcconfig", "XYBRID_API_KEY", why);
            assert_eq!(err.code, "env_file_ambiguous");
            assert!(err.message.contains("ios/Secrets.xcconfig"), "{why:?}");
            assert!(err.message.contains("XYBRID_API_KEY"), "{why:?}");
        }
        assert!(ambiguous("a", "B", Ambiguity::BlockComment)
            .message
            .contains("Xcode has no block comments"));
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
