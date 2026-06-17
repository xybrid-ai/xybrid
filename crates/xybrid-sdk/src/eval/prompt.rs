//! Prompt optimization — model-aware lint + rewrite suggestions.
//!
//! The harness's second capability: given a prompt and the target model's
//! profile, surface actionable, **rule-cited** suggestions (token budget, system
//! prompt support, output-format clarity, …). Suggestions are only trustworthy
//! once *scored against an evalset* — that scoring rides the normal runner; this
//! module is the pure lint/rewrite engine.
//!
//! The hosted **prompt library** + write-back of an applied suggestion as a new
//! prompt version is a server-side concern; this produces the suggestions it
//! would store.

use serde::{Deserialize, Serialize};

/// Per-model facts that drive the lint rules (sourced from registry metadata in
/// production; a small built-in table stands in here).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelProfile {
    /// Model id.
    pub model_id: String,
    /// Context window in tokens.
    pub ctx_window: usize,
    /// Whether the model honors a separate system prompt.
    pub supports_system_prompt: bool,
    /// Chat template family (informational).
    pub chat_template: String,
}

impl ModelProfile {
    /// A reasonable default profile for an unknown model.
    pub fn generic(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ctx_window: 4096,
            supports_system_prompt: true,
            chat_template: "chatml".to_string(),
        }
    }
}

/// Look up a built-in profile by model id, falling back to a generic one. (A
/// production build resolves this from registry metadata.)
pub fn model_profile(model_id: &str) -> ModelProfile {
    let id = model_id.to_lowercase();
    let (ctx, sys, tmpl) = if id.contains("gemma") {
        (8192, false, "gemma") // Gemma folds the system role into the first user turn
    } else if id.contains("qwen") || id.contains("lfm") || id.contains("liquid") {
        (32768, true, "chatml")
    } else if id.contains("llama") {
        (8192, true, "llama")
    } else {
        return ModelProfile::generic(model_id);
    };
    ModelProfile {
        model_id: model_id.to_string(),
        ctx_window: ctx,
        supports_system_prompt: sys,
        chat_template: tmpl.to_string(),
    }
}

/// How strongly a suggestion is recommended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SuggestionLevel {
    /// Style / minor efficiency.
    Hint,
    /// Likely to affect quality.
    Warn,
    /// Will break or badly degrade the prompt on this model.
    Error,
}

/// One actionable, rule-cited prompt suggestion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptSuggestion {
    /// The lint rule that fired (cited to the developer).
    pub rule: String,
    /// Severity.
    pub level: SuggestionLevel,
    /// What's wrong.
    pub message: String,
    /// How to fix it.
    pub fix: String,
}

/// Rough token estimate (~4 chars/token) — good enough for a budget lint without
/// a tokenizer dependency.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

/// Task verbs a well-formed instruction usually opens with.
const TASK_VERBS: &[&str] = &[
    "summarize",
    "classify",
    "extract",
    "translate",
    "rewrite",
    "write",
    "answer",
    "list",
    "explain",
    "generate",
    "analyze",
    "compare",
    "label",
    "transcribe",
];

/// Politeness padding that wastes tokens with no effect on most models.
const POLITENESS: &[&str] = &[
    "please",
    "thank you",
    "thanks",
    "could you",
    "would you kindly",
];

/// Markers that an output format is specified.
const FORMAT_MARKERS: &[&str] = &[
    "json",
    "format",
    "bullet",
    "list",
    "table",
    "one sentence",
    "one word",
    "markdown",
    "yaml",
    "schema",
];

/// Lint a prompt against a model profile, returning rule-cited suggestions
/// (highest severity first). `system` is the separate system prompt, if any.
pub fn lint_prompt(
    prompt: &str,
    system: Option<&str>,
    profile: &ModelProfile,
) -> Vec<PromptSuggestion> {
    let mut out = Vec::new();
    let lower = prompt.to_lowercase();
    let trimmed = prompt.trim();

    // 1. Token budget.
    let total = estimate_tokens(prompt) + system.map(estimate_tokens).unwrap_or(0);
    if total > profile.ctx_window {
        out.push(PromptSuggestion {
            rule: "token-budget".into(),
            level: SuggestionLevel::Error,
            message: format!(
                "prompt ≈{total} tokens exceeds {}'s {}-token context",
                profile.model_id, profile.ctx_window
            ),
            fix: "shorten the prompt or move bulky context into retrieval".into(),
        });
    } else if total * 2 > profile.ctx_window {
        out.push(PromptSuggestion {
            rule: "token-budget".into(),
            level: SuggestionLevel::Warn,
            message: format!(
                "prompt ≈{total} tokens uses over half of {}'s {}-token context, leaving little room for output",
                profile.model_id, profile.ctx_window
            ),
            fix: "trim the prompt to leave headroom for the response".into(),
        });
    }

    // 2. System-prompt handling.
    match (profile.supports_system_prompt, system) {
        (true, None) | (true, Some("")) => out.push(PromptSuggestion {
            rule: "system-prompt-missing".into(),
            level: SuggestionLevel::Warn,
            message: format!(
                "{} supports a system prompt but none is set; role/persona is mixed into the user turn",
                profile.model_id
            ),
            fix: "move role/behavior instructions into a dedicated system prompt".into(),
        }),
        (false, Some(s)) if !s.is_empty() => out.push(PromptSuggestion {
            rule: "system-prompt-unsupported".into(),
            level: SuggestionLevel::Warn,
            message: format!(
                "{} ({}) does not honor a separate system prompt; it will be ignored or merged",
                profile.model_id, profile.chat_template
            ),
            fix: "fold the system instructions into the start of the user prompt".into(),
        }),
        _ => {}
    }

    // 3. No clear task verb.
    if !TASK_VERBS.iter().any(|v| lower.contains(v)) {
        out.push(PromptSuggestion {
            rule: "no-task-verb".into(),
            level: SuggestionLevel::Warn,
            message: "no explicit task verb (summarize / classify / extract / …) — the instruction is vague".into(),
            fix: "open with an imperative that names the task".into(),
        });
    }

    // 4. No output-format constraint.
    if !FORMAT_MARKERS.iter().any(|m| lower.contains(m)) {
        out.push(PromptSuggestion {
            rule: "no-output-format".into(),
            level: SuggestionLevel::Hint,
            message: "no output format/length specified — outputs will vary in shape".into(),
            fix: "state the expected format (e.g. 'reply with one of: refund, cancel')".into(),
        });
    }

    // 5. Politeness padding.
    if POLITENESS.iter().any(|p| lower.contains(p)) {
        out.push(PromptSuggestion {
            rule: "politeness-padding".into(),
            level: SuggestionLevel::Hint,
            message:
                "politeness padding ('please' / 'thank you') costs tokens without improving output"
                    .into(),
            fix: "drop conversational padding for a tighter instruction".into(),
        });
    }

    // 6. No examples for a complex prompt.
    if estimate_tokens(prompt) > 60 && !lower.contains("example") && !lower.contains("e.g.") {
        out.push(PromptSuggestion {
            rule: "no-examples".into(),
            level: SuggestionLevel::Hint,
            message: "a long instruction with no examples — few-shot examples sharpen behavior"
                .into(),
            fix: "add 1–3 input→output examples".into(),
        });
    }

    // 7. Trailing whitespace / empty.
    if trimmed.is_empty() {
        out.insert(
            0,
            PromptSuggestion {
                rule: "empty-prompt".into(),
                level: SuggestionLevel::Error,
                message: "the prompt is empty".into(),
                fix: "write an instruction".into(),
            },
        );
    } else if prompt != trimmed {
        out.push(PromptSuggestion {
            rule: "trailing-whitespace".into(),
            level: SuggestionLevel::Hint,
            message: "leading/trailing whitespace can leak into some chat templates".into(),
            fix: "trim the prompt".into(),
        });
    }

    // Highest severity first, stable within a level.
    out.sort_by_key(|s| match s.level {
        SuggestionLevel::Error => 0,
        SuggestionLevel::Warn => 1,
        SuggestionLevel::Hint => 2,
    });
    out
}

// ============================================================================
// Prompt library — versioned prompt store (the write-back target)
// ============================================================================

use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};

use crate::eval::format::EvalError;

/// Max prompt-version file size accepted on load (DoS guard; mirrors the run /
/// promotion caps).
const MAX_PROMPT_FILE_BYTES: u64 = 4 * 1024 * 1024;

/// Retry budget when racing a concurrent writer for the next version slot.
const MAX_WRITE_ATTEMPTS: usize = 8;

/// One stored version of a prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptVersion {
    /// Prompt id (a stable slug).
    pub prompt_id: String,
    /// Version number (monotonic, starts at 1).
    pub version: u32,
    /// The prompt text.
    pub text: String,
    /// The separate system prompt, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// What changed (e.g. an applied prompt-opt suggestion).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// RFC3339 creation timestamp (injected).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
}

/// A versioned, on-disk prompt store — the local stand-in for the hosted
/// **prompt library** that `prompt-opt --write` commits a new version to. A
/// `RemoteAuthority` swaps the backing without changing this interface.
/// Dependency-injected base dir; production resolves `~/.xybrid/prompts`.
#[derive(Debug, Clone)]
pub struct PromptLibrary {
    base: PathBuf,
}

/// Validate a prompt id is a single safe path component (no traversal, no
/// control chars — the latter only fail deep in the OS otherwise).
fn validate_prompt_id(id: &str) -> Result<&str, EvalError> {
    let mut comps = Path::new(id).components();
    match (comps.next(), comps.next()) {
        (Some(Component::Normal(c)), None)
            if c == OsStr::new(id) && !id.is_empty() && !id.chars().any(|ch| ch.is_control()) =>
        {
            Ok(id)
        }
        _ => Err(EvalError::Invalid(format!(
            "invalid prompt id {id:?}: must be a single path component"
        ))),
    }
}

impl PromptLibrary {
    /// A library rooted at an explicit base directory.
    pub fn with_dir(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    /// The default library at `~/.xybrid/prompts`.
    pub fn default_location() -> Result<Self, EvalError> {
        let home = dirs::home_dir()
            .ok_or_else(|| EvalError::Io("could not resolve home directory".into()))?;
        Ok(Self::with_dir(home.join(".xybrid").join("prompts")))
    }

    /// Existing version numbers for a prompt, ascending.
    pub fn versions(&self, prompt_id: &str) -> Result<Vec<u32>, EvalError> {
        let id = validate_prompt_id(prompt_id)?;
        let dir = self.base.join(id);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in
            std::fs::read_dir(&dir).map_err(|e| EvalError::Io(format!("{}: {e}", dir.display())))?
        {
            let entry = entry.map_err(|e| EvalError::Io(e.to_string()))?;
            // Regular files only — a dir or FIFO named `vN.json` must not be
            // counted (or `load` would stat/block on it).
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            if let Some(name) = entry.file_name().to_str() {
                if let Some(v) = name.strip_prefix('v').and_then(|s| s.strip_suffix(".json")) {
                    if let Ok(n) = v.parse::<u32>() {
                        out.push(n);
                    }
                }
            }
        }
        out.sort_unstable();
        Ok(out)
    }

    /// The latest version of a prompt, if any.
    pub fn latest(&self, prompt_id: &str) -> Result<Option<PromptVersion>, EvalError> {
        match self.versions(prompt_id)?.last() {
            Some(&v) => self.load(prompt_id, v).map(Some),
            None => Ok(None),
        }
    }

    /// Load a specific prompt version (size-capped, regular-file only).
    pub fn load(&self, prompt_id: &str, version: u32) -> Result<PromptVersion, EvalError> {
        let id = validate_prompt_id(prompt_id)?;
        let path = self.base.join(id).join(format!("v{version}.json"));
        let meta = std::fs::metadata(&path)
            .map_err(|e| EvalError::Io(format!("{}: {e}", path.display())))?;
        if !meta.is_file() || meta.len() > MAX_PROMPT_FILE_BYTES {
            return Err(EvalError::Invalid(format!(
                "prompt version {} is not a regular file or is too large",
                path.display()
            )));
        }
        let src = std::fs::read_to_string(&path)
            .map_err(|e| EvalError::Io(format!("{}: {e}", path.display())))?;
        serde_json::from_str(&src)
            .map_err(|e| EvalError::Invalid(format!("{prompt_id} v{version}: {e}")))
    }

    /// Write a new version of a prompt (the write-back). The version is
    /// `latest + 1`; returns the stored [`PromptVersion`]. `created` is injected.
    ///
    /// The slot is claimed atomically: the content is written to a private temp
    /// and hard-linked into `vN.json`, so a concurrent writer cannot silently
    /// clobber a version (the link fails if the slot is taken — we recompute and
    /// retry) and a reader never observes a half-written file.
    pub fn write(
        &self,
        prompt_id: &str,
        text: &str,
        system: Option<&str>,
        note: Option<&str>,
        created: Option<String>,
    ) -> Result<PromptVersion, EvalError> {
        let id = validate_prompt_id(prompt_id)?;
        let dir = self.base.join(id);
        std::fs::create_dir_all(&dir)
            .map_err(|e| EvalError::Io(format!("{}: {e}", dir.display())))?;
        for _ in 0..MAX_WRITE_ATTEMPTS {
            let next = self
                .versions(id)?
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| EvalError::Invalid("prompt version counter exhausted".into()))?;
            let version = PromptVersion {
                prompt_id: id.to_string(),
                version: next,
                text: text.to_string(),
                system: system.map(str::to_string),
                note: note.map(str::to_string),
                created: created.clone(),
            };
            let json = serde_json::to_string_pretty(&version)
                .map_err(|e| EvalError::Io(format!("serialize prompt: {e}")))?;
            let tmp = dir.join(format!(".v{next}.{}.tmp", std::process::id()));
            std::fs::write(&tmp, &json)
                .map_err(|e| EvalError::Io(format!("{}: {e}", tmp.display())))?;
            let path = dir.join(format!("v{next}.json"));
            match std::fs::hard_link(&tmp, &path) {
                Ok(()) => {
                    let _ = std::fs::remove_file(&tmp);
                    return Ok(version);
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // A concurrent writer took this slot — recompute and retry.
                    let _ = std::fs::remove_file(&tmp);
                    continue;
                }
                Err(e) => {
                    let _ = std::fs::remove_file(&tmp);
                    return Err(EvalError::Io(format!("{}: {e}", path.display())));
                }
            }
        }
        Err(EvalError::Io(
            "prompt library write contention — retry".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_resolve_with_fallback() {
        assert!(!model_profile("gemma-4-e2b").supports_system_prompt);
        assert!(model_profile("qwen3.5-0.8b").supports_system_prompt);
        let g = model_profile("totally-unknown-model");
        assert_eq!(g.ctx_window, 4096);
    }

    #[test]
    fn vague_chat_prompt_yields_at_least_three_cited_suggestions() {
        // A typical weak chat prompt on a known model.
        let prompt = "Please be helpful and answer nicely. Thank you!";
        let profile = model_profile("qwen3.5-0.8b");
        let suggestions = lint_prompt(prompt, None, &profile);
        assert!(
            suggestions.len() >= 3,
            "expected ≥3 suggestions, got {:?}",
            suggestions
        );
        // Every suggestion cites a rule and offers a fix.
        for s in &suggestions {
            assert!(!s.rule.is_empty());
            assert!(!s.fix.is_empty());
        }
        // Specifically: system-prompt-missing, no-task-verb, politeness-padding.
        let rules: Vec<&str> = suggestions.iter().map(|s| s.rule.as_str()).collect();
        assert!(rules.contains(&"system-prompt-missing"));
        assert!(rules.contains(&"politeness-padding"));
    }

    #[test]
    fn token_budget_error_over_context() {
        let profile = ModelProfile {
            model_id: "tiny".into(),
            ctx_window: 10,
            supports_system_prompt: true,
            chat_template: "chatml".into(),
        };
        let prompt = "x".repeat(1000); // ≈250 tokens ≫ 10
        let s = lint_prompt(&prompt, None, &profile);
        let budget = s.iter().find(|s| s.rule == "token-budget").unwrap();
        assert_eq!(budget.level, SuggestionLevel::Error);
    }

    #[test]
    fn system_unsupported_flagged_on_gemma() {
        let profile = model_profile("gemma-4-e2b");
        let s = lint_prompt(
            "Summarize this in one sentence.",
            Some("You are a bot."),
            &profile,
        );
        assert!(s.iter().any(|s| s.rule == "system-prompt-unsupported"));
    }

    #[test]
    fn well_formed_prompt_has_few_or_no_warnings() {
        let profile = model_profile("qwen3.5-0.8b");
        let prompt = "Classify the message into one of: refund, cancel, question. Reply with only the label.";
        let s = lint_prompt(prompt, Some("You are an intent classifier."), &profile);
        // No Error/Warn-level issues for a clear, well-scoped prompt.
        assert!(
            !s.iter().any(|s| s.level != SuggestionLevel::Hint),
            "unexpected warnings: {s:?}"
        );
    }

    #[test]
    fn empty_prompt_is_an_error() {
        let s = lint_prompt("   ", None, &ModelProfile::generic("m"));
        assert_eq!(s[0].rule, "empty-prompt");
        assert_eq!(s[0].level, SuggestionLevel::Error);
    }

    #[test]
    fn suggestions_round_trip_through_serde() {
        let s = lint_prompt("please answer", None, &ModelProfile::generic("m"));
        let json = serde_json::to_string(&s).unwrap();
        let back: Vec<PromptSuggestion> = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn prompt_library_versions_and_write_back() {
        let dir = tempfile::TempDir::new().unwrap();
        let lib = PromptLibrary::with_dir(dir.path());
        assert!(lib.versions("intent").unwrap().is_empty());
        assert!(lib.latest("intent").unwrap().is_none());

        // write-back v1, then a revised v2.
        let v1 = lib
            .write(
                "intent",
                "Answer the question.",
                None,
                Some("initial"),
                None,
            )
            .unwrap();
        assert_eq!(v1.version, 1);
        let v2 = lib
            .write(
                "intent",
                "Classify into: refund, cancel, other. Reply with only the label.",
                Some("You are an intent classifier."),
                Some("applied prompt-opt: added task verb + output format"),
                None,
            )
            .unwrap();
        assert_eq!(v2.version, 2);

        assert_eq!(lib.versions("intent").unwrap(), vec![1, 2]);
        assert_eq!(lib.latest("intent").unwrap().unwrap().version, 2);
        let loaded = lib.load("intent", 1).unwrap();
        assert_eq!(loaded.text, "Answer the question.");
        assert_eq!(
            lib.load("intent", 2).unwrap().system.as_deref(),
            Some("You are an intent classifier.")
        );
    }

    #[test]
    fn prompt_library_rejects_traversal_ids() {
        let dir = tempfile::TempDir::new().unwrap();
        let lib = PromptLibrary::with_dir(dir.path());
        // Traversal, separators, empty, and a control char (NUL) — the last must
        // be rejected by the validator, not deep in the OS.
        for bad in ["../escape", "/etc/x", "a/b", "..", "", "a\0b"] {
            assert!(lib.write(bad, "x", None, None, None).is_err(), "{bad:?}");
            assert!(lib.versions(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn prompt_library_version_overflow_is_a_clean_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let lib = PromptLibrary::with_dir(dir.path());
        // Plant a maxed-out version slot; the next bump must error, not panic
        // (debug) or wrap to 0 and silently clobber (release).
        let pdir = dir.path().join("ovf");
        std::fs::create_dir_all(&pdir).unwrap();
        std::fs::write(pdir.join(format!("v{}.json", u32::MAX)), "{}").unwrap();
        let err = lib.write("ovf", "x", None, None, None).unwrap_err();
        assert!(format!("{err}").contains("exhausted"), "{err}");
    }

    #[test]
    fn prompt_library_skips_non_file_version_entries() {
        let dir = tempfile::TempDir::new().unwrap();
        let lib = PromptLibrary::with_dir(dir.path());
        // A *directory* named like a version file must not be counted (else
        // `load` would stat a dir / a FIFO would block).
        std::fs::create_dir_all(dir.path().join("nf").join("v5.json")).unwrap();
        let v = lib.write("nf", "hello", None, None, None).unwrap();
        assert_eq!(v.version, 1, "the bogus v5 dir must be ignored");
        assert_eq!(lib.versions("nf").unwrap(), vec![1]);
    }
}
