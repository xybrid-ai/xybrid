//! CI detection for registry call telemetry.
//!
//! Decides whether this process runs under continuous integration, so the
//! `X-Xybrid-Client` header can carry `ci=1` and registry analytics can tell
//! test runs from real use (see `docs/telemetry/registry.md`).
//!
//! `XYBRID_CI` decides when it holds `1`/`true` (CI) or `0`/`false` (not CI);
//! any other value is ignored. Otherwise `CI`, which GitHub Actions, GitLab
//! CI, CircleCI, Buildkite and most other CI services set, marks a CI run
//! unless it is empty, `0` or `false`. Values are compared case-insensitively,
//! surrounding whitespace ignored.
//!
//! Like the opt-out in [`crate::telemetry_optout`], the decision is made on
//! first call and cached for the process lifetime: a mid-process change to
//! either variable is not observed.

use std::sync::OnceLock;

/// Explicit override, for runners that do not set `CI` (coding-agent
/// sandboxes, some build services) or CI jobs that should count as real use.
const OVERRIDE_ENV_VAR: &str = "XYBRID_CI";

/// The variable most CI services set.
const CI_ENV_VAR: &str = "CI";

static IS_CI: OnceLock<bool> = OnceLock::new();

/// Returns `true` when this process runs under CI.
///
/// The result is cached on first call; later calls read no environment.
pub(crate) fn is_ci() -> bool {
    *IS_CI.get_or_init(|| detect_ci(process_env))
}

/// Reads a variable from the process environment. A value that is not valid
/// UTF-8 is still a set variable, so it is converted lossily, not dropped.
fn process_env(name: &str) -> Option<String> {
    std::env::var_os(name).map(|value| value.to_string_lossy().into_owned())
}

/// Pure decision behind [`is_ci`], reading variables through `env`.
///
/// Takes the lookup as a parameter so the matrix can be unit-tested without
/// touching the process environment or the [`IS_CI`] cache.
fn detect_ci(env: impl Fn(&str) -> Option<String>) -> bool {
    let normalized = |name: &str| env(name).map(|value| value.trim().to_ascii_lowercase());
    match normalized(OVERRIDE_ENV_VAR).as_deref() {
        Some("1" | "true") => return true,
        Some("0" | "false") => return false,
        _ => {}
    }
    !matches!(
        normalized(CI_ENV_VAR).as_deref(),
        None | Some("" | "0" | "false")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lookup over a fixed set of variables.
    fn env_of<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        }
    }

    #[test]
    fn neither_variable_set_is_not_ci() {
        assert!(!detect_ci(env_of(&[])));
    }

    #[test]
    fn ci_variable_marks_a_ci_run_unless_empty_zero_or_false() {
        for value in ["true", "TRUE", "1", "yes", "anything", " true "] {
            assert!(detect_ci(env_of(&[("CI", value)])), "CI={value:?}");
        }
        for value in ["", "   ", "0", "false", "FALSE", "False", " false\n"] {
            assert!(!detect_ci(env_of(&[("CI", value)])), "CI={value:?}");
        }
    }

    #[test]
    fn xybrid_ci_turns_the_label_on_without_ci() {
        for value in ["1", "true", "TRUE", "True", " 1 "] {
            assert!(
                detect_ci(env_of(&[("XYBRID_CI", value)])),
                "XYBRID_CI={value:?}"
            );
        }
    }

    #[test]
    fn xybrid_ci_turns_the_label_off_under_ci() {
        for value in ["0", "false", "FALSE", "False", "\tfalse "] {
            assert!(
                !detect_ci(env_of(&[("CI", "true"), ("XYBRID_CI", value)])),
                "XYBRID_CI={value:?}"
            );
        }
    }

    #[test]
    fn unrecognized_xybrid_ci_defers_to_ci() {
        for value in ["", "yes", "no", "on", "off", "2"] {
            assert!(
                detect_ci(env_of(&[("CI", "true"), ("XYBRID_CI", value)])),
                "XYBRID_CI={value:?} under CI=true"
            );
            assert!(
                !detect_ci(env_of(&[("XYBRID_CI", value)])),
                "XYBRID_CI={value:?} without CI"
            );
            assert!(
                !detect_ci(env_of(&[("CI", "false"), ("XYBRID_CI", value)])),
                "XYBRID_CI={value:?} under CI=false"
            );
        }
    }

    #[test]
    fn cached_value_matches_the_process_environment() {
        // No test in this crate sets CI or XYBRID_CI, so the cached decision
        // and a fresh read of the same variables agree.
        assert_eq!(is_ci(), detect_ci(process_env));
        assert_eq!(is_ci(), is_ci(), "the decision is cached");
    }
}
