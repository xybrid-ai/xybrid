//! Resolved-execution-provider capture via ORT session profiling.
//!
//! ONNX Runtime's C API has no session-level "what providers did this session
//! resolve to" getter. EPs are appended in fallback order and per-op
//! resolution is opaque to the API. The only authoritative source is the
//! per-session profiling JSON, which records the EP under
//! `args.provider` on every Node event (older / external profiling
//! tooling sometimes uses `args.provider_type`; we accept either).
//!
//! This module exposes:
//! - [`ResolvedExecutionProviders`] — the parsed summary surfaced to callers.
//! - [`parse_profile_json`] — Chrome-trace-format profile parser used after
//!   `Session::end_profiling()` finalizes the file.
//!
//! The lifecycle (enable profiling → run first inference → end profiling →
//! parse → cache) lives on [`super::ONNXSession`]; this module is the pure
//! parsing + summary side.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Per-EP op-count breakdown for one ONNX session, computed from the
/// session's profile JSON after the first inference.
///
/// `primary` is the EP that ran the most ops in the trace — the one most
/// representative of "which engine actually executed this model". A model
/// that requested CoreML but got 90 % CPU fallback will surface as
/// `primary = "cpu"` with a breakdown showing both.
///
/// EP names are normalised (lower-cased and stripped of the
/// `"ExecutionProvider"` suffix ORT bakes into its profile output), so
/// callers see e.g. `"coreml"` and `"cpu"` rather than
/// `"CoreMLExecutionProvider"`. This matches the wire format the
/// telemetry layer emits on events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedExecutionProviders {
    /// EP that ran the most ops. Suffix-stripped, lower-cased.
    pub primary: String,
    /// `(ep_name, op_count)` pairs sorted by count descending. Only EPs
    /// with at least one Node event appear; empty list is impossible
    /// because [`parse_profile_json`] errors out when no Node events
    /// are found.
    pub breakdown: Vec<(String, usize)>,
}

/// Errors the profile parser can produce. Bubbled up to the caller so the
/// session can fall back to "no resolved info" rather than poisoning state
/// when ORT writes something unexpected.
#[derive(Debug, thiserror::Error)]
pub enum ProfileParseError {
    #[error("failed to read profile file {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse profile JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error(
        "profile contained no Node events; session may not have run any inference. \
         Categories seen: {seen_categories:?}; sample event keys: {sample_keys:?}"
    )]
    NoNodeEvents {
        seen_categories: Vec<String>,
        sample_keys: Vec<String>,
    },
}

/// Read an ORT-emitted profile JSON file and aggregate the per-Node EP
/// values into a [`ResolvedExecutionProviders`] summary.
///
/// ORT writes Chrome-trace-format JSON: an array of event objects, each
/// with a `cat` and an `args` map. Node-execution events have
/// `cat == "Node"` and carry the EP under `args.provider` (preferred,
/// what ORT 2.x emits) or `args.provider_type` (older / external
/// tooling). Fence / kernel-internal events use other categories and
/// are ignored, as are Node events missing both keys — they don't
/// represent actual op execution.
pub fn parse_profile_json(path: &Path) -> Result<ResolvedExecutionProviders, ProfileParseError> {
    let raw = fs::read_to_string(path).map_err(|source| ProfileParseError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let events: Vec<serde_json::Value> = serde_json::from_str(&raw)?;

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut seen_categories: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut sample_keys: Vec<String> = Vec::new();
    for event in &events {
        if let Some(cat) = event.get("cat").and_then(|c| c.as_str()) {
            seen_categories.insert(cat.to_string());
        }
        if event.get("cat").and_then(|c| c.as_str()) != Some("Node") {
            continue;
        }
        // Sample the first Node event's full shape (top-level keys + args
        // keys) so a parse failure surfaces what ORT actually wrote.
        if sample_keys.is_empty() {
            if let Some(obj) = event.as_object() {
                let top: Vec<String> = obj.keys().cloned().collect();
                let args: Vec<String> = event
                    .get("args")
                    .and_then(|a| a.as_object())
                    .map(|m| m.keys().cloned().collect())
                    .unwrap_or_default();
                sample_keys = top;
                sample_keys.push(format!("args=[{}]", args.join(",")));
            }
        }
        let args = event.get("args");
        let Some(provider) = args
            .and_then(|a| a.get("provider"))
            .and_then(|p| p.as_str())
            .or_else(|| {
                args.and_then(|a| a.get("provider_type"))
                    .and_then(|p| p.as_str())
            })
        else {
            continue;
        };
        *counts.entry(normalise_provider(provider)).or_insert(0) += 1;
    }

    if counts.is_empty() {
        return Err(ProfileParseError::NoNodeEvents {
            seen_categories: seen_categories.into_iter().collect(),
            sample_keys,
        });
    }

    let mut breakdown: Vec<(String, usize)> = counts.into_iter().collect();
    // Stable secondary sort on EP name so the breakdown is deterministic when
    // two EPs tie on op count — useful for snapshot tests.
    breakdown.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let primary = breakdown[0].0.clone();
    Ok(ResolvedExecutionProviders { primary, breakdown })
}

/// Strip ORT's `"ExecutionProvider"` suffix and lower-case the result so
/// `"CoreMLExecutionProvider"` becomes `"coreml"`. Unknown shapes are
/// returned lower-cased verbatim — the wire format is open-string.
fn normalise_provider(raw: &str) -> String {
    let trimmed = raw.strip_suffix("ExecutionProvider").unwrap_or(raw);
    trimmed.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_profile(events: &[serde_json::Value]) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("tempfile");
        let body = serde_json::to_string(events).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        f
    }

    #[test]
    fn parse_picks_majority_ep_as_primary() {
        let events = vec![
            serde_json::json!({"cat": "Node", "args": {"provider": "CPUExecutionProvider"}}),
            serde_json::json!({"cat": "Node", "args": {"provider": "CPUExecutionProvider"}}),
            serde_json::json!({"cat": "Node", "args": {"provider": "CoreMLExecutionProvider"}}),
        ];
        let f = write_profile(&events);

        let summary = parse_profile_json(f.path()).unwrap();
        assert_eq!(summary.primary, "cpu");
        assert_eq!(
            summary.breakdown,
            vec![("cpu".to_string(), 2), ("coreml".to_string(), 1)]
        );
    }

    #[test]
    fn parse_ignores_non_node_events() {
        let events = vec![
            serde_json::json!({"cat": "Session", "args": {"provider": "CPUExecutionProvider"}}),
            serde_json::json!({"cat": "Node", "args": {"provider": "CoreMLExecutionProvider"}}),
        ];
        let f = write_profile(&events);

        let summary = parse_profile_json(f.path()).unwrap();
        assert_eq!(summary.primary, "coreml");
        assert_eq!(summary.breakdown.len(), 1);
    }

    #[test]
    fn parse_skips_events_without_provider_type() {
        let events = vec![
            serde_json::json!({"cat": "Node", "args": {}}),
            serde_json::json!({"cat": "Node", "args": {"provider": "CoreMLExecutionProvider"}}),
        ];
        let f = write_profile(&events);

        let summary = parse_profile_json(f.path()).unwrap();
        assert_eq!(summary.breakdown, vec![("coreml".to_string(), 1)]);
    }

    #[test]
    fn parse_errors_when_no_node_events() {
        let events = vec![serde_json::json!({"cat": "Session"})];
        let f = write_profile(&events);

        let err = parse_profile_json(f.path()).unwrap_err();
        assert!(matches!(err, ProfileParseError::NoNodeEvents { .. }));
    }

    #[test]
    fn parse_ties_break_alphabetically_for_determinism() {
        let events = vec![
            serde_json::json!({"cat": "Node", "args": {"provider": "CoreMLExecutionProvider"}}),
            serde_json::json!({"cat": "Node", "args": {"provider": "CPUExecutionProvider"}}),
        ];
        let f = write_profile(&events);

        let summary = parse_profile_json(f.path()).unwrap();
        // Both have count = 1; "coreml" sorts before "cpu" alphabetically.
        assert_eq!(summary.primary, "coreml");
    }

    #[test]
    fn parse_falls_back_to_legacy_provider_type_key() {
        // Some external profiling tooling (and pre-2.x ORT) wrote
        // `args.provider_type` instead of `args.provider`. We accept
        // both so we don't strand callers on the older shape.
        let events = vec![
            serde_json::json!({"cat": "Node", "args": {"provider_type": "CoreMLExecutionProvider"}}),
        ];
        let f = write_profile(&events);

        let summary = parse_profile_json(f.path()).unwrap();
        assert_eq!(summary.primary, "coreml");
    }

    #[test]
    fn parse_prefers_provider_over_provider_type_when_both_present() {
        // Forward-compat guard: if a future ORT writes both, the
        // canonical key wins. (Today only one is set per event, but
        // the precedence is documented behaviour.)
        let events = vec![serde_json::json!({
            "cat": "Node",
            "args": {
                "provider": "CPUExecutionProvider",
                "provider_type": "CoreMLExecutionProvider",
            }
        })];
        let f = write_profile(&events);

        let summary = parse_profile_json(f.path()).unwrap();
        assert_eq!(summary.primary, "cpu");
    }

    #[test]
    fn normalise_strips_suffix() {
        assert_eq!(normalise_provider("CoreMLExecutionProvider"), "coreml");
        assert_eq!(normalise_provider("CPUExecutionProvider"), "cpu");
        assert_eq!(normalise_provider("CUDAExecutionProvider"), "cuda");
        assert_eq!(normalise_provider("UnknownThing"), "unknownthing");
    }
}
