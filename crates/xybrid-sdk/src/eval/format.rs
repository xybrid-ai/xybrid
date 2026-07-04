//! Evalset file format — the on-disk schema for eval-driven development.
//!
//! An evalset lives in `evals/<name>/` as a manifest (`evalset.yaml`) plus a
//! line-delimited case file (`cases.jsonl`) and optional binary payload sidecar
//! files referenced by relative path. The file is the source of truth; the
//! platform holds a synced copy.
//!
//! This module owns the **frozen schema** (the primitives and trust layer) and a
//! loader that validates structure and — critically — refuses payload references
//! that escape the evalset directory (path-traversal / exfiltration defense).
//!
//! Schema-stability notes:
//! - Fields use `#[serde(default)]` + `skip_serializing_if` so older/newer files
//!   round-trip without breaking. New optional fields are additive.
//! - Date-ish fields are stored as plain ISO strings, not `chrono` types, so the
//!   wire format never depends on a serialization detail of a dependency.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

/// Manifest filename inside an evalset directory.
pub const MANIFEST_FILE: &str = "evalset.yaml";
/// Case file (line-delimited JSON) inside an evalset directory.
pub const CASES_FILE: &str = "cases.jsonl";

/// The `file:` scheme prefix used by payload references in a case input.
const FILE_SCHEME: &str = "file:";

const MAX_MANIFEST_FILE_BYTES: u64 = 1024 * 1024;
/// Maximum `cases.jsonl` size accepted on load (DoS guard).
const MAX_CASES_FILE_BYTES: u64 = 64 * 1024 * 1024;

// ============================================================================
// Task & kind
// ============================================================================

/// The task verb that *implies* the default grader. Tier 1 never
/// chooses a grader — the task type selects it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskType {
    /// Label classification — graded by normalized label match.
    Classify,
    /// Open chat — graded by LLM-as-judge with a task rubric.
    Chat,
    /// Summarization — judge with a summarization rubric.
    Summarize,
    /// Structured extraction — JSON-schema validity + per-field match.
    Extract,
    /// Speech recognition — graded by Word Error Rate.
    Asr,
    /// Speech synthesis — golden-output + duration/RTF sanity (v1).
    Tts,
    /// Embedding — recall@k over labeled query→doc pairs.
    Embedding,
    /// Vision-language — judge / golden mode.
    Vlm,
}

impl TaskType {
    /// The manifest spelling for this task.
    pub const fn as_str(self) -> &'static str {
        match self {
            TaskType::Classify => "classify",
            TaskType::Chat => "chat",
            TaskType::Summarize => "summarize",
            TaskType::Extract => "extract",
            TaskType::Asr => "asr",
            TaskType::Tts => "tts",
            TaskType::Embedding => "embedding",
            TaskType::Vlm => "vlm",
        }
    }

    /// Whether the default grader for this task requires an `expected` field.
    ///
    /// `chat`/`summarize`/`vlm` can run reference-free (judge reads input+output)
    /// or fall to golden mode; `tts` uses golden mode. The rest need a reference.
    pub fn requires_expected(self) -> bool {
        matches!(
            self,
            TaskType::Classify | TaskType::Extract | TaskType::Asr | TaskType::Embedding
        )
    }
}

/// Sibling evalset kinds. All three share the case/run/gate
/// primitives so safety/performance are part of the loop, not a late audit. The
/// slot is reserved now; safety and performance graders land later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalsetKind {
    /// Task-implied quality graders (the default kind).
    #[default]
    Quality,
    /// Prompt injection, refusal correctness, jailbreak, data-leakage, etc.
    Safety,
    /// On-device SLOs (cold-start, TTFT/ITL, energy, memory) as thresholds.
    Performance,
}

// ============================================================================
// Case provenance & governance (trust layer — data trust)
// ============================================================================

/// Where a case came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaseSource {
    /// Minted from a flagged production result (carries `trace_id`).
    Flagged,
    /// Hand-authored by the developer.
    #[default]
    Authored,
    /// Imported from an external evalset.
    Imported,
    /// A blessed golden case (output pinned as the reference).
    Golden,
}

/// Curation lifecycle state of a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReviewStatus {
    /// Freshly ingested, not yet triaged.
    #[default]
    Unreviewed,
    /// A human confirmed the case (and its `expected`, if any).
    Reviewed,
    /// Output blessed as the golden reference.
    Golden,
}

/// Which split a case belongs to. Auto-assigned, never asked of a Tier 1 dev
/// (flagged → regression wall; authored/pasted cold-start → dev).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    /// Iteration set.
    #[default]
    Dev,
    /// The regression wall — what a gate replays.
    Regression,
}

/// Triage severity of a failure case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Visible privacy state of a case's payload (data trust). The field is
/// declared here; population/behavior is owned by the payload-privacy layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrivacyClass {
    /// Full input/output payload captured.
    Captured,
    /// Payload captured after redaction.
    Redacted,
    /// No payload — counters only.
    MetadataOnly,
}

// ============================================================================
// Case input / expected
// ============================================================================

/// A case input. Mirrors `Envelope` kinds. Binary payloads are sibling files
/// referenced by relative path (`file:clips/x.wav`) so an evalset stays a normal
/// git directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaseInput {
    /// Inline text input.
    Text(String),
    /// Audio payload reference (`file:…` relative path) or inline-encoded data.
    Audio(String),
    /// Image payload reference (`file:…` relative path) or inline-encoded data.
    Image(String),
}

impl CaseInput {
    /// The `file:`-scheme reference carried by this input, if any (audio/image).
    pub fn payload_ref(&self) -> Option<&str> {
        match self {
            CaseInput::Audio(v) | CaseInput::Image(v) => v.strip_prefix(FILE_SCHEME),
            CaseInput::Text(_) => None,
        }
    }
}

/// The task-shaped reference output. Optional on a `Case` — a case with no
/// `expected` runs in golden mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Expected {
    /// A classification label.
    Label(String),
    /// A reference string (ASR transcript, chat reference answer).
    Text(String),
    /// A structured reference object (extraction) or any Tier-3 shape.
    Json(serde_json::Value),
}

// ============================================================================
// Case
// ============================================================================

fn default_weight() -> f64 {
    1.0
}

/// One eval case: an input plus, optionally, what should have happened.
///
/// The `expected` field is optional: a case without it runs in **golden mode**
/// (the first blessed run pins its output, later runs diff against it).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Case {
    /// Stable case id.
    pub id: String,
    /// The input envelope.
    pub input: CaseInput,
    /// The reference output, if known. Absent ⇒ golden mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<Expected>,
    /// Provenance.
    #[serde(default)]
    pub source: CaseSource,
    /// Originating inference trace, when `source = flagged`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// ISO-8601 date the case was added.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub added: Option<String>,

    // ---- governance (trust layer — data trust) ----
    /// Curation status.
    #[serde(default)]
    pub review_status: ReviewStatus,
    /// Triage severity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub severity: Option<Severity>,
    /// Scoring weight (default 1.0).
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Near-duplicate cluster id assigned by the inbox.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<String>,
    /// Exact-duplicate hash used to drop dupes on ingest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dedupe_hash: Option<String>,
    /// Visible payload privacy state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy_class: Option<PrivacyClass>,
    /// Confidence that the case is correctly labeled (0..1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_confidence: Option<f64>,
    /// dev / regression split.
    #[serde(default)]
    pub split: Split,
    /// Owning reviewer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// ISO-8601 date after which the case is stale and excluded from gates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    /// If set, the case is quarantined (excluded from gates, retained for audit).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quarantine_reason: Option<String>,
}

impl Case {
    /// Minimal constructor for an authored case.
    pub fn new(id: impl Into<String>, input: CaseInput) -> Self {
        Self {
            id: id.into(),
            input,
            expected: None,
            source: CaseSource::Authored,
            trace_id: None,
            added: None,
            review_status: ReviewStatus::Unreviewed,
            severity: None,
            weight: 1.0,
            cluster_id: None,
            dedupe_hash: None,
            privacy_class: None,
            source_confidence: None,
            split: Split::Dev,
            owner: None,
            expires_at: None,
            quarantine_reason: None,
        }
    }

    /// Builder: set the expected reference output.
    pub fn with_expected(mut self, expected: Expected) -> Self {
        self.expected = Some(expected);
        self
    }

    /// Whether this case is quarantined (excluded from gate runs).
    pub fn is_quarantined(&self) -> bool {
        self.quarantine_reason.is_some()
    }

    /// Whether this case is expired relative to `today` (an ISO `YYYY-MM-DD`).
    ///
    /// A case with no `expires_at`, or with an unparseable date, is never
    /// considered expired (fail-open on the date, not on inclusion).
    pub fn is_expired_on(&self, today: &str) -> bool {
        match (&self.expires_at, parse_date(today)) {
            (Some(exp), Some(today)) => match parse_date(exp) {
                Some(exp) => exp < today,
                None => false,
            },
            _ => false,
        }
    }

    /// Whether this case counts toward a gate run on `today`: not quarantined,
    /// not expired, and on the regression split.
    pub fn counts_for_gate(&self, today: &str) -> bool {
        !self.is_quarantined() && !self.is_expired_on(today) && self.split == Split::Regression
    }
}

/// Parse an ISO `YYYY-MM-DD` date into a comparable `(year, month, day)` tuple.
/// Lenient: returns `None` on any malformed input.
fn parse_date(s: &str) -> Option<(i32, u32, u32)> {
    let mut it = s.split('-');
    let y = it.next()?.parse::<i32>().ok()?;
    let m = it.next()?.parse::<u32>().ok()?;
    let d = it.next()?.parse::<u32>().ok()?;
    if it.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some((y, m, d))
}

// ============================================================================
// Gate & grader config (manifest)
// ============================================================================

/// Optional gate thresholds + statistical policy carried on the manifest and
/// consumed by `eval gate` / OTA promotion.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Gate {
    /// Minimum aggregate quality (0..1) to pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_quality: Option<f64>,
    /// Maximum allowed p95 latency in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_p95_latency_ms: Option<f64>,
    /// Minimum case count below which the gate is `inconclusive`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_cases: Option<usize>,
    /// Non-inferiority margin: a delta within ± this resolves to a tie.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_inferiority_margin: Option<f64>,
    /// Repeat count for nondeterministic candidates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeats: Option<u32>,
}

/// Tier-3 grader override (judge model + rubric, or a custom scorer). Tier 1
/// never writes this — the task type implies the grader.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GraderConfig {
    /// Override judge model id (chat/summarize).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_model: Option<String>,
    /// Override rubric text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rubric: Option<String>,
    /// Custom scorer reference (e.g. `wasm:./graders/my_metric.wasm`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom: Option<String>,
}

// ============================================================================
// Evalset manifest
// ============================================================================

fn default_version() -> u32 {
    1
}

/// The evalset manifest (`evalset.yaml`). Generated and maintained by
/// `xybrid eval init` / `pull`; rarely hand-edited.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evalset {
    /// Evalset name (matches the directory name).
    pub name: String,
    /// Task type — implies the default grader.
    pub task: TaskType,
    /// Bumped when cases change; every run records it.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Quality / safety / performance sibling kind.
    #[serde(default)]
    pub kind: EvalsetKind,
    /// Allowed labels (classify) + alias source for label normalization.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
    /// Optional gate thresholds + statistical policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate: Option<Gate>,
    /// Tier-3 grader override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grader: Option<GraderConfig>,
}

impl Evalset {
    /// A fresh manifest for `task`, with sensible defaults.
    pub fn new(name: impl Into<String>, task: TaskType) -> Self {
        Self {
            name: name.into(),
            task,
            version: 1,
            kind: EvalsetKind::Quality,
            labels: Vec::new(),
            gate: None,
            grader: None,
        }
    }
}

// ============================================================================
// Loaded evalset
// ============================================================================

/// A loaded evalset: manifest + cases + the on-disk root used to resolve
/// payload references.
#[derive(Debug, Clone)]
pub struct LoadedEvalset {
    /// Parsed manifest.
    pub manifest: Evalset,
    /// All cases, in file order.
    pub cases: Vec<Case>,
    /// The `evals/<name>/` directory the set was loaded from.
    pub root: PathBuf,
}

impl LoadedEvalset {
    /// Load and validate an evalset from a directory.
    ///
    /// Validates manifest + case syntax and ensures every `file:` payload
    /// reference resolves *inside* `dir` (path-traversal defense). Malformed
    /// cases produce a precise `EvalError::Case { line }`.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, EvalError> {
        let dir = dir.as_ref();
        let manifest_path = dir.join(MANIFEST_FILE);
        let manifest_meta = match std::fs::metadata(&manifest_path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(EvalError::ManifestNotFound(manifest_path));
            }
            Err(e) => return Err(EvalError::Io(format!("{}: {e}", manifest_path.display()))),
        };
        if !manifest_meta.is_file() {
            return Err(EvalError::Invalid(format!(
                "{} is not a regular file",
                manifest_path.display()
            )));
        }
        if manifest_meta.len() > MAX_MANIFEST_FILE_BYTES {
            return Err(EvalError::Invalid(format!(
                "manifest is {} bytes (max {MAX_MANIFEST_FILE_BYTES})",
                manifest_meta.len()
            )));
        }
        let manifest_src = std::fs::read_to_string(&manifest_path)
            .map_err(|e| EvalError::Io(format!("{}: {e}", manifest_path.display())))?;
        let manifest: Evalset =
            serde_yaml::from_str(&manifest_src).map_err(|e| EvalError::Manifest(e.to_string()))?;
        if manifest.name.trim().is_empty() {
            return Err(EvalError::Invalid("evalset name is empty".into()));
        }
        validate_gate(manifest.gate.as_ref())?;

        let cases_path = dir.join(CASES_FILE);
        let parsed_cases = if cases_path.exists() {
            // DoS guard: refuse a pathologically large cases file before reading
            // it into memory.
            let len = std::fs::metadata(&cases_path)
                .map_err(|e| EvalError::Io(format!("{}: {e}", cases_path.display())))?;
            if !len.is_file() {
                return Err(EvalError::Invalid(format!(
                    "{} is not a regular file",
                    cases_path.display()
                )));
            }
            let len = len.len();
            if len > MAX_CASES_FILE_BYTES {
                return Err(EvalError::Invalid(format!(
                    "cases file is {len} bytes (max {MAX_CASES_FILE_BYTES})"
                )));
            }
            let src = std::fs::read_to_string(&cases_path)
                .map_err(|e| EvalError::Io(format!("{}: {e}", cases_path.display())))?;
            parse_cases(&src, &cases_path)?
        } else {
            Vec::new()
        };

        validate_required_expected(&manifest, &parsed_cases, &cases_path)?;

        // Security: refuse any payload reference that escapes the evalset dir.
        for parsed in &parsed_cases {
            if let Some(rel) = parsed.case.input.payload_ref() {
                validate_payload_file(dir, &parsed.case.id, rel)?;
            }
        }
        let cases = parsed_cases.into_iter().map(|parsed| parsed.case).collect();

        Ok(Self {
            manifest,
            cases,
            root: dir.to_path_buf(),
        })
    }

    /// Cases that count toward a gate run on `today` (regression split, not
    /// quarantined, not expired).
    pub fn gate_cases(&self, today: &str) -> Vec<&Case> {
        self.cases
            .iter()
            .filter(|c| c.counts_for_gate(today))
            .collect()
    }

    /// Resolve a case's payload reference to an absolute path under the root.
    /// Returns `None` for inline (non-`file:`) inputs. Re-validates containment.
    pub fn resolve_payload(&self, case: &Case) -> Option<Result<PathBuf, EvalError>> {
        let rel = case.input.payload_ref()?;
        Some(validate_payload_file(&self.root, &case.id, rel))
    }
}

#[derive(Debug)]
struct ParsedCase {
    case: Case,
    line: usize,
}

fn validate_gate(gate: Option<&Gate>) -> Result<(), EvalError> {
    let Some(gate) = gate else {
        return Ok(());
    };
    if let Some(value) = gate.min_quality {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(EvalError::Invalid(format!(
                "gate.min_quality must be finite in 0.0..=1.0, got {value}"
            )));
        }
    }
    if let Some(value) = gate.max_p95_latency_ms {
        if !value.is_finite() || value <= 0.0 {
            return Err(EvalError::Invalid(format!(
                "gate.max_p95_latency_ms must be finite and > 0.0, got {value}"
            )));
        }
    }
    if let Some(value) = gate.min_cases {
        if value == 0 {
            return Err(EvalError::Invalid(format!(
                "gate.min_cases must be >= 1, got {value}"
            )));
        }
    }
    if let Some(value) = gate.repeats {
        if value == 0 {
            return Err(EvalError::Invalid(format!(
                "gate.repeats must be >= 1, got {value}"
            )));
        }
    }
    if let Some(value) = gate.non_inferiority_margin {
        if !value.is_finite() || value < 0.0 {
            return Err(EvalError::Invalid(format!(
                "gate.non_inferiority_margin must be finite and >= 0.0, got {value}"
            )));
        }
    }
    Ok(())
}

fn validate_required_expected(
    manifest: &Evalset,
    cases: &[ParsedCase],
    path: &Path,
) -> Result<(), EvalError> {
    if !manifest.task.requires_expected() {
        return Ok(());
    }
    for parsed in cases {
        if parsed.case.expected.is_none() {
            return Err(EvalError::Case {
                path: path.to_path_buf(),
                line: parsed.line,
                reason: format!(
                    "case '{}' is missing expected for {} task",
                    parsed.case.id,
                    manifest.task.as_str()
                ),
            });
        }
    }
    Ok(())
}

/// Parse a `cases.jsonl` body into cases, attaching the 1-based line number to
/// any parse error. Blank lines are skipped.
fn parse_cases(src: &str, path: &Path) -> Result<Vec<ParsedCase>, EvalError> {
    let mut cases = Vec::new();
    let mut seen_ids = HashMap::new();
    for (idx, line) in src.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let case: Case = serde_json::from_str(trimmed).map_err(|e| EvalError::Case {
            path: path.to_path_buf(),
            line: idx + 1,
            reason: e.to_string(),
        })?;
        let line = idx + 1;
        if let Some(first_line) = seen_ids.insert(case.id.clone(), line) {
            return Err(EvalError::Case {
                path: path.to_path_buf(),
                line,
                reason: format!(
                    "duplicate case id '{}' first appears at line {first_line} and repeats at line {line}",
                    case.id
                ),
            });
        }
        cases.push(ParsedCase { case, line });
    }
    Ok(cases)
}

/// Validate that `rel` (the part after `file:`) resolves *inside* `root`.
///
/// Defense is two-layered:
/// 1. **Lexical** — reject absolute paths and any `..`/root/prefix component.
///    This blocks `../../etc/passwd`, `/etc/shadow`, `C:\…`, `\\UNC\…` without
///    touching the filesystem (works even if the file doesn't exist yet).
/// 2. **Canonical** — if the target exists, canonicalize both sides and require
///    containment, catching symlink escapes (`clips/evil -> /etc`). The
///    **canonical** (symlink-resolved) path is returned so callers read the
///    real, contained target rather than the lexical join.
fn validate_payload_file(root: &Path, case_id: &str, rel: &str) -> Result<PathBuf, EvalError> {
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() {
        return Err(EvalError::PathEscape(rel.to_string()));
    }
    for comp in rel_path.components() {
        match comp {
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(EvalError::PathEscape(rel.to_string()));
            }
            // `.` and normal components are fine.
            Component::CurDir | Component::Normal(_) => {}
        }
    }
    let joined = root.join(rel_path);
    let meta = match std::fs::metadata(&joined) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(EvalError::Invalid(format!(
                "case {case_id}: payload file not found: {}",
                joined.display()
            )));
        }
        Err(e) => return Err(EvalError::Io(format!("{}: {e}", joined.display()))),
    };
    if !meta.is_file() {
        return Err(EvalError::Invalid(format!(
            "case {case_id}: payload is not a regular file: {}",
            joined.display()
        )));
    }
    let canon_root = root
        .canonicalize()
        .map_err(|e| EvalError::Io(format!("{}: {e}", root.display())))?;
    let canon = joined
        .canonicalize()
        .map_err(|e| EvalError::Io(format!("{}: {e}", joined.display())))?;
    if !canon.starts_with(&canon_root) {
        return Err(EvalError::PathEscape(rel.to_string()));
    }
    // Return the symlink-resolved path so the caller reads the contained target
    // (not the lexical join, which a symlink could still redirect).
    Ok(canon)
}

// ============================================================================
// Errors
// ============================================================================

/// Errors from loading / validating an evalset.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// No `evalset.yaml` at the expected path.
    #[error("evalset manifest not found: {0}")]
    ManifestNotFound(PathBuf),
    /// The manifest failed to parse.
    #[error("invalid evalset manifest: {0}")]
    Manifest(String),
    /// A case line failed to parse.
    #[error("invalid case at {path}:{line}: {reason}")]
    Case {
        /// The cases file.
        path: PathBuf,
        /// 1-based line number.
        line: usize,
        /// Underlying parse error.
        reason: String,
    },
    /// A payload reference escaped the evalset directory.
    #[error("payload reference escapes evalset directory: {0}")]
    PathEscape(String),
    /// Filesystem error.
    #[error("io error: {0}")]
    Io(String),
    /// Structurally invalid evalset.
    #[error("invalid evalset: {0}")]
    Invalid(String),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Write a manifest + cases into a fresh temp evalset dir.
    fn write_evalset(manifest: &str, cases: &str) -> TempDir {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join(MANIFEST_FILE), manifest).unwrap();
        fs::write(dir.path().join(CASES_FILE), cases).unwrap();
        dir
    }

    #[test]
    fn loads_classify_evalset() {
        let manifest = "name: intent-classifier\ntask: classify\nversion: 3\nlabels: [refund, cancel, question, other]\n";
        let cases = r#"{"id":"c1","input":{"text":"I want my money back"},"expected":{"label":"refund"},"source":"flagged","trace_id":"tr_9a31"}
{"id":"c2","input":{"text":"cancel please"},"expected":{"label":"cancel"}}
"#;
        let dir = write_evalset(manifest, cases);
        let set = LoadedEvalset::load(dir.path()).unwrap();
        assert_eq!(set.manifest.name, "intent-classifier");
        assert_eq!(set.manifest.task, TaskType::Classify);
        assert_eq!(set.manifest.version, 3);
        assert_eq!(set.manifest.kind, EvalsetKind::Quality);
        assert_eq!(set.manifest.labels.len(), 4);
        assert_eq!(set.cases.len(), 2);
        assert_eq!(set.cases[0].source, CaseSource::Flagged);
        assert_eq!(set.cases[0].trace_id.as_deref(), Some("tr_9a31"));
        // defaults applied
        assert_eq!(set.cases[1].source, CaseSource::Authored);
        assert_eq!(set.cases[1].weight, 1.0);
        assert_eq!(set.cases[1].split, Split::Dev);
        assert_eq!(set.cases[1].review_status, ReviewStatus::Unreviewed);
    }

    #[test]
    fn manifest_defaults_version_and_kind() {
        let dir = write_evalset("name: s\ntask: chat\n", "");
        let set = LoadedEvalset::load(dir.path()).unwrap();
        assert_eq!(set.manifest.version, 1);
        assert_eq!(set.manifest.kind, EvalsetKind::Quality);
        assert!(set.cases.is_empty());
    }

    #[test]
    fn performance_and_safety_kinds_round_trip() {
        for (kind_str, kind) in [
            ("safety", EvalsetKind::Safety),
            ("performance", EvalsetKind::Performance),
        ] {
            let dir = write_evalset(&format!("name: s\ntask: chat\nkind: {kind_str}\n"), "");
            let set = LoadedEvalset::load(dir.path()).unwrap();
            assert_eq!(set.manifest.kind, kind);
        }
    }

    #[test]
    fn golden_case_has_no_expected() {
        let cases = r#"{"id":"g1","input":{"text":"summarize this"}}
"#;
        let dir = write_evalset("name: s\ntask: summarize\n", cases);
        let set = LoadedEvalset::load(dir.path()).unwrap();
        assert!(set.cases[0].expected.is_none());
    }

    #[test]
    fn reference_required_task_rejects_missing_expected() {
        let cases = r#"{"id":"c1","input":{"text":"refund please"}}
"#;
        let dir = write_evalset("name: s\ntask: classify\n", cases);
        let err = LoadedEvalset::load(dir.path()).unwrap_err();
        match err {
            EvalError::Case { line, reason, .. } => {
                assert_eq!(line, 1);
                assert!(reason.contains("c1"));
                assert!(reason.contains("classify"));
                assert!(reason.contains("missing expected"));
            }
            other => panic!("expected Case error, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_case_id_reports_both_lines() {
        let cases = r#"{"id":"dup","input":{"text":"a"}}
{"id":"ok","input":{"text":"b"}}
{"id":"dup","input":{"text":"c"}}
"#;
        let dir = write_evalset("name: s\ntask: chat\n", cases);
        let err = LoadedEvalset::load(dir.path()).unwrap_err();
        match err {
            EvalError::Case { line, reason, .. } => {
                assert_eq!(line, 3);
                assert!(reason.contains("dup"));
                assert!(reason.contains("line 1"));
                assert!(reason.contains("line 3"));
            }
            other => panic!("expected Case error, got {other:?}"),
        }
    }

    #[test]
    fn gate_numeric_ranges_are_validated() {
        for (gate_field, field_name, value) in [
            ("min_quality: 1.1", "gate.min_quality", "1.1"),
            ("max_p95_latency_ms: 0", "gate.max_p95_latency_ms", "0"),
            ("min_cases: 0", "gate.min_cases", "0"),
            ("repeats: 0", "gate.repeats", "0"),
            (
                "non_inferiority_margin: -0.1",
                "gate.non_inferiority_margin",
                "-0.1",
            ),
            ("min_quality: .nan", "gate.min_quality", "NaN"),
        ] {
            let manifest = format!("name: s\ntask: chat\ngate:\n  {gate_field}\n");
            let dir = write_evalset(&manifest, "");
            let err = LoadedEvalset::load(dir.path()).unwrap_err();
            let message = err.to_string();
            assert!(message.contains(field_name), "{message}");
            assert!(message.contains(value), "{message}");
        }
    }

    #[test]
    fn valid_gate_numeric_ranges_load() {
        let manifest = "\
name: s
task: chat
gate:
  min_quality: 0.0
  max_p95_latency_ms: 1.0
  min_cases: 1
  repeats: 1
  non_inferiority_margin: 0.0
";
        let dir = write_evalset(manifest, "");
        let set = LoadedEvalset::load(dir.path()).unwrap();
        assert!(set.manifest.gate.is_some());
    }

    #[test]
    fn case_round_trips_through_serde() {
        let case = Case::new("c1", CaseInput::Text("hi".into()))
            .with_expected(Expected::Label("refund".into()));
        let json = serde_json::to_string(&case).unwrap();
        let back: Case = serde_json::from_str(&json).unwrap();
        assert_eq!(case, back);
        // Optional empty fields are omitted from the wire form.
        assert!(!json.contains("cluster_id"));
        assert!(!json.contains("quarantine_reason"));
    }

    #[test]
    fn missing_manifest_is_an_error() {
        let dir = TempDir::new().unwrap();
        let err = LoadedEvalset::load(dir.path()).unwrap_err();
        assert!(matches!(err, EvalError::ManifestNotFound(_)));
    }

    #[test]
    fn malformed_manifest_is_an_error() {
        let dir = write_evalset("name: s\ntask: : : not yaml\n", "");
        let err = LoadedEvalset::load(dir.path()).unwrap_err();
        assert!(matches!(err, EvalError::Manifest(_)));
    }

    #[test]
    fn unsupported_task_is_an_error() {
        let dir = write_evalset("name: s\ntask: telekinesis\n", "");
        let err = LoadedEvalset::load(dir.path()).unwrap_err();
        assert!(matches!(err, EvalError::Manifest(_)), "got {err:?}");
    }

    #[test]
    fn malformed_case_reports_line_number() {
        let cases = "{\"id\":\"ok\",\"input\":{\"text\":\"a\"}}\nnot json at all\n";
        let dir = write_evalset("name: s\ntask: chat\n", cases);
        let err = LoadedEvalset::load(dir.path()).unwrap_err();
        match err {
            EvalError::Case { line, .. } => assert_eq!(line, 2),
            other => panic!("expected Case error, got {other:?}"),
        }
    }

    #[test]
    fn blank_case_lines_are_skipped() {
        let cases = "\n{\"id\":\"a\",\"input\":{\"text\":\"x\"}}\n\n";
        let dir = write_evalset("name: s\ntask: chat\n", cases);
        let set = LoadedEvalset::load(dir.path()).unwrap();
        assert_eq!(set.cases.len(), 1);
    }

    // ---- path-traversal security (the critical defense) ----

    #[test]
    fn rejects_parent_dir_payload_ref() {
        let cases = r#"{"id":"evil","input":{"audio":"file:../../etc/passwd"},"expected":{"text":"transcript"}}
"#;
        let dir = write_evalset("name: s\ntask: asr\n", cases);
        let err = LoadedEvalset::load(dir.path()).unwrap_err();
        assert!(matches!(err, EvalError::PathEscape(_)), "got {err:?}");
    }

    #[test]
    fn rejects_absolute_payload_ref() {
        let cases = r#"{"id":"evil","input":{"audio":"file:/etc/shadow"},"expected":{"text":"transcript"}}
"#;
        let dir = write_evalset("name: s\ntask: asr\n", cases);
        let err = LoadedEvalset::load(dir.path()).unwrap_err();
        assert!(matches!(err, EvalError::PathEscape(_)), "got {err:?}");
    }

    #[test]
    fn rejects_nested_parent_escape() {
        let cases = r#"{"id":"evil","input":{"image":"file:clips/../../secret"}}
"#;
        let dir = write_evalset("name: s\ntask: vlm\n", cases);
        let err = LoadedEvalset::load(dir.path()).unwrap_err();
        assert!(matches!(err, EvalError::PathEscape(_)), "got {err:?}");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        use std::os::unix::fs::symlink;
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join(MANIFEST_FILE), "name: s\ntask: asr\n").unwrap();
        // A real secret outside the evalset dir.
        let outside = TempDir::new().unwrap();
        let secret = outside.path().join("secret.wav");
        fs::write(&secret, b"RIFF....").unwrap();
        // clips/leak -> /tmp/.../secret.wav (escaping symlink), referenced lexically-clean.
        fs::create_dir(dir.path().join("clips")).unwrap();
        symlink(&secret, dir.path().join("clips/leak.wav")).unwrap();
        fs::write(
            dir.path().join(CASES_FILE),
            "{\"id\":\"x\",\"input\":{\"audio\":\"file:clips/leak.wav\"},\"expected\":{\"text\":\"transcript\"}}\n",
        )
        .unwrap();
        let err = LoadedEvalset::load(dir.path()).unwrap_err();
        assert!(matches!(err, EvalError::PathEscape(_)), "got {err:?}");
    }

    #[test]
    fn accepts_contained_payload_ref() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join(MANIFEST_FILE), "name: s\ntask: asr\n").unwrap();
        fs::create_dir(dir.path().join("clips")).unwrap();
        fs::write(dir.path().join("clips/a.wav"), b"RIFF").unwrap();
        fs::write(
            dir.path().join(CASES_FILE),
            "{\"id\":\"x\",\"input\":{\"audio\":\"file:clips/a.wav\"},\"expected\":{\"text\":\"transcript\"}}\n",
        )
        .unwrap();
        let set = LoadedEvalset::load(dir.path()).unwrap();
        let resolved = set.resolve_payload(&set.cases[0]).unwrap().unwrap();
        assert!(resolved.ends_with("clips/a.wav"));
    }

    #[test]
    fn rejects_missing_payload_ref() {
        let cases = r#"{"id":"missing","input":{"audio":"file:clips/missing.wav"},"expected":{"text":"transcript"}}
"#;
        let dir = write_evalset("name: s\ntask: asr\n", cases);
        let err = LoadedEvalset::load(dir.path()).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("case missing"), "{message}");
        assert!(message.contains("clips/missing.wav"), "{message}");
        assert!(message.contains("not found"), "{message}");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_fifo_cases_file_without_reading() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join(MANIFEST_FILE), "name: s\ntask: chat\n").unwrap();
        let cases_path = dir.path().join(CASES_FILE);
        let status = std::process::Command::new("mkfifo")
            .arg(&cases_path)
            .status()
            .unwrap();
        assert!(status.success());
        let err = LoadedEvalset::load(dir.path()).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("cases.jsonl"), "{message}");
        assert!(message.contains("not a regular file"), "{message}");
    }

    // ---- governance helpers ----

    #[test]
    fn quarantined_and_expired_excluded_from_gate() {
        let cases = r#"{"id":"ok","input":{"text":"a"},"split":"regression"}
{"id":"q","input":{"text":"b"},"split":"regression","quarantine_reason":"bad label"}
{"id":"old","input":{"text":"c"},"split":"regression","expires_at":"2020-01-01"}
{"id":"dev","input":{"text":"d"},"split":"dev"}
"#;
        let dir = write_evalset("name: s\ntask: chat\n", cases);
        let set = LoadedEvalset::load(dir.path()).unwrap();
        let gate: Vec<_> = set
            .gate_cases("2026-06-14")
            .iter()
            .map(|c| c.id.clone())
            .collect();
        assert_eq!(gate, vec!["ok".to_string()]);
    }

    #[test]
    fn expiry_is_fail_open_on_bad_dates() {
        let case = Case {
            expires_at: Some("not-a-date".into()),
            ..Case::new("x", CaseInput::Text("a".into()))
        };
        assert!(!case.is_expired_on("2026-06-14"));
        assert!(!case.is_expired_on("garbage"));
    }

    #[test]
    fn requires_expected_matches_prd_table() {
        assert!(TaskType::Classify.requires_expected());
        assert!(TaskType::Asr.requires_expected());
        assert!(TaskType::Extract.requires_expected());
        assert!(TaskType::Embedding.requires_expected());
        assert!(!TaskType::Chat.requires_expected());
        assert!(!TaskType::Tts.requires_expected());
    }
}
