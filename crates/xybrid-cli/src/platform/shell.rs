//! Shell words for the commands the CLI suggests.
//!
//! A suggested command (a rerun of `connect`, a `chmod` fix) is pasted into
//! the shell of the machine the CLI runs on, so every value in it is written
//! as one word of that shell: POSIX `sh` on Unix, and on Windows a form that
//! PowerShell and `cmd.exe` both read the same way. Plain values stay as they
//! are, so ordinary commands remain readable.
//!
//! A value that can't be written safely is refused (`None`), and the caller
//! says so instead of printing a command that would run something else: on
//! every platform, a control or invisible character (a shown command must be
//! the pasted one) or a backtick (commands are shown in backticks); on
//! Windows also a value needing quotes that holds `"`, `%`, `$`, `!` or a
//! typographic quote, or ends in a backslash, since PowerShell or `cmd.exe`
//! would still read those inside double quotes.

/// `value` as one word of a command for this platform's shell, or `None`
/// when it can't be written safely there.
fn quote(value: &str) -> Option<String> {
    if cfg!(windows) {
        quote_windows(value)
    } else {
        quote_posix(value)
    }
}

/// `flag value` as words of a command for this platform's shell, or `None`
/// when the value can't be written safely there. A value starting with `-`
/// is attached as `flag=value`, so it isn't read as another flag.
pub(crate) fn flag(flag: &str, value: &str) -> Option<String> {
    if value.starts_with('-') {
        quote(&format!("{flag}={value}"))
    } else {
        quote(value).map(|word| format!("{flag} {word}"))
    }
}

/// `value` as one POSIX shell word: unchanged when plain, else in single
/// quotes, each `'` written as `'\''`. Anything is quoted, including
/// characters [`quote`] refuses to show.
pub(crate) fn posix_quote(value: &str) -> String {
    if is_plain(value, b"/._-+:") {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn quote_posix(value: &str) -> Option<String> {
    (!value.chars().any(never_shown)).then(|| posix_quote(value))
}

/// One word that PowerShell and `cmd.exe` both pass on unchanged: plain, or
/// in double quotes when nothing inside them is special to either shell.
fn quote_windows(value: &str) -> Option<String> {
    if value.chars().any(never_shown) {
        return None;
    }
    if is_plain(value, b"/._-:\\") {
        return Some(value.to_string());
    }
    // Inside double quotes `cmd.exe` still expands `%VAR%` (and `!VAR!` with
    // delayed expansion), PowerShell expands `$var` and ends the string at a
    // typographic double quote, and a closing `\"` is an escaped quote to
    // the program's argument parser.
    let special = |c: char| matches!(c, '"' | '%' | '$' | '!' | '\u{2018}'..='\u{201e}');
    if value.chars().any(special) || value.ends_with('\\') {
        return None;
    }
    Some(format!("\"{value}\""))
}

/// A character no suggested command shows: control and invisible or
/// reordering characters (the shown command must be the one pasted), and a
/// backtick, which ends the code span the command is shown in.
fn never_shown(c: char) -> bool {
    c.is_control()
        || c == '`'
        || matches!(
            c,
            '\u{ad}'
                | '\u{61c}'
                | '\u{200b}'..='\u{200f}'
                | '\u{2028}'..='\u{202e}'
                | '\u{2060}'..='\u{206f}'
                | '\u{feff}'
        )
}

/// Nonempty, and only ASCII letters, digits and `extra`.
fn is_plain(value: &str, extra: &[u8]) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || extra.contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_values_stay_readable() {
        for value in [
            "config/.env",
            "apps/web/.env.local",
            "http://127.0.0.1:8080",
            "EXPO_PUBLIC_XYBRID_API_KEY",
        ] {
            assert_eq!(quote_posix(value).as_deref(), Some(value), "{value}");
            assert_eq!(quote_windows(value).as_deref(), Some(value), "{value}");
        }
        assert_eq!(
            quote_windows(r"ios\Secrets.xcconfig").unwrap(),
            r"ios\Secrets.xcconfig"
        );
    }

    #[test]
    fn posix_words_are_single_quoted() {
        for (value, word) in [
            (
                "ios/My App/Secrets.xcconfig",
                "'ios/My App/Secrets.xcconfig'",
            ),
            ("it's $HOME/.env", r"'it'\''s $HOME/.env'"),
            ("http://[::1]:8080", "'http://[::1]:8080'"),
            ("https://a$(id).example", "'https://a$(id).example'"),
            ("", "''"),
        ] {
            assert_eq!(quote_posix(value).as_deref(), Some(word), "{value}");
        }
    }

    /// The quoted word reads back as the value in a real POSIX shell.
    #[cfg(unix)]
    #[test]
    fn posix_words_round_trip_through_sh() {
        let values = [
            "ios/My App/Secrets.xcconfig",
            "it's $HOME/`x` \"q\" \\n *.env",
            "a;b&c|d>e<f(g)h{i}~j!k#l",
        ];
        let script = values
            .iter()
            .map(|value| format!("printf '%s\\0' {}", posix_quote(value)))
            .collect::<Vec<_>>()
            .join("; ");
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(script)
            .output()
            .expect("sh runs");
        assert!(output.status.success(), "{output:?}");
        let read: Vec<_> = output
            .stdout
            .split(|&b| b == 0)
            .filter(|w| !w.is_empty())
            .map(|w| String::from_utf8_lossy(w).into_owned())
            .collect();
        assert_eq!(read, values);
    }

    #[test]
    fn windows_words_are_double_quoted_or_refused() {
        assert_eq!(
            quote_windows("ios/My App/Secrets.xcconfig").unwrap(),
            "\"ios/My App/Secrets.xcconfig\""
        );
        assert_eq!(
            quote_windows(r"C:\Users\me\My App\.env").unwrap(),
            "\"C:\\Users\\me\\My App\\.env\""
        );
        assert_eq!(quote_windows("it's.env").unwrap(), "\"it's.env\"");
        for value in [
            "a\"b.env",
            "%USERPROFILE%/.env",
            "$HOME/.env",
            "wow!/.env",
            "\u{201c}quoted\u{201d}/.env",
            r"My Dir\",
        ] {
            assert_eq!(quote_windows(value), None, "{value}");
        }
    }

    #[test]
    fn hidden_characters_and_backticks_are_never_shown() {
        for value in [
            "a\u{ad}b.env",
            "a\nb.env",
            "a\tb.env",
            "a\u{1b}[31m.env",
            "a`b.env",
            "a\u{202e}vne.",
            "a\u{200b}b",
        ] {
            assert_eq!(quote_posix(value), None, "{value:?}");
            assert_eq!(quote_windows(value), None, "{value:?}");
        }
        // A fix that is only ever quoted, never refused, still quotes them.
        assert_eq!(posix_quote("a`b"), "'a`b'");
    }

    #[test]
    fn a_value_starting_with_a_dash_is_attached_to_its_flag() {
        assert_eq!(
            flag("--write-env", "config/.env").as_deref(),
            Some("--write-env config/.env")
        );
        assert_eq!(
            flag("--write-env", "My App/.env").unwrap(),
            if cfg!(windows) {
                "--write-env \"My App/.env\""
            } else {
                "--write-env 'My App/.env'"
            }
        );
        assert_eq!(
            flag("--write-env", "-x.env").unwrap(),
            if cfg!(windows) {
                "\"--write-env=-x.env\""
            } else {
                "'--write-env=-x.env'"
            }
        );
        assert_eq!(flag("--write-env", "a`b"), None);
    }
}
