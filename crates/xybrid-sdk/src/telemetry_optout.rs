//! Registry telemetry opt-out helper.
//!
//! Single source of truth for whether the user has opted out of registry call
//! telemetry via the `XYBRID_TELEMETRY_OPTOUT` environment variable.
//!
//! The check is performed once on first call and cached in a `OnceLock<bool>`,
//! so every call site reads the same value with no further env access. A
//! mid-process change to the env var is intentionally not observed — opt-out
//! state is established at SDK startup and held for the process lifetime.

use std::sync::OnceLock;

const ENV_VAR: &str = "XYBRID_TELEMETRY_OPTOUT";

static OPTED_OUT: OnceLock<bool> = OnceLock::new();

/// Returns `true` when the user has opted out of registry call telemetry.
///
/// Truthy values (case-insensitive): `"1"`, `"true"`, `"yes"`. Any other value
/// — including unset — returns `false`.
///
/// The result is cached on first call; subsequent calls return the cached
/// value with no environment access. This is intentional: a mid-process env
/// change is not observed.
pub fn is_telemetry_opted_out() -> bool {
    *OPTED_OUT.get_or_init(|| parse_optout(std::env::var(ENV_VAR).ok().as_deref()))
}

/// Parse a raw env-var value into a boolean opt-out flag.
///
/// Pure function so the parsing matrix can be unit-tested without touching
/// the process environment (which would otherwise interact with the
/// [`OPTED_OUT`] cache used by [`is_telemetry_opted_out`]).
fn parse_optout(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_unset_returns_false() {
        assert!(!parse_optout(None));
    }

    #[test]
    fn parse_truthy_values_return_true() {
        assert!(parse_optout(Some("1")));
        assert!(parse_optout(Some("true")));
        assert!(parse_optout(Some("TRUE")));
        assert!(parse_optout(Some("True")));
        assert!(parse_optout(Some("yes")));
        assert!(parse_optout(Some("YES")));
        assert!(parse_optout(Some("Yes")));
    }

    #[test]
    fn parse_truthy_values_tolerate_surrounding_whitespace() {
        assert!(parse_optout(Some(" 1 ")));
        assert!(parse_optout(Some("\ttrue\n")));
    }

    #[test]
    fn parse_falsy_values_return_false() {
        assert!(!parse_optout(Some("")));
        assert!(!parse_optout(Some("0")));
        assert!(!parse_optout(Some("false")));
        assert!(!parse_optout(Some("FALSE")));
        assert!(!parse_optout(Some("no")));
        assert!(!parse_optout(Some("off")));
        assert!(!parse_optout(Some("anything-else")));
    }

    #[test]
    fn cached_value_ignores_mid_process_env_changes() {
        // Seed the cache with whatever the environment currently says.
        let initial = is_telemetry_opted_out();

        // Flip the env var to the opposite value. The cached result must
        // not change — that's the documented contract.
        let prev = std::env::var(ENV_VAR).ok();
        let toggle = if initial { "" } else { "1" };
        std::env::set_var(ENV_VAR, toggle);
        let after_change = is_telemetry_opted_out();

        // Restore prior env state so other tests see a clean environment.
        match prev {
            Some(v) => std::env::set_var(ENV_VAR, v),
            None => std::env::remove_var(ENV_VAR),
        }

        assert_eq!(
            initial, after_change,
            "cache must not observe mid-process env changes"
        );
    }
}
