//! `xybrid run --decision`: a SystemOne decision request, answered by a
//! decision model.
//!
//! The request is read, parsed ([`xybrid_cli::systemone`]) and checked
//! ([`validate_structure`]) before any model is opened, so an invalid
//! request never loads one. Running it goes through one function,
//! [`run_decision_request`]: the seam where the runtime's decision support
//! plugs in. This build has none yet, so the seam refuses every request, and
//! `--dry-run` checks the request alone.

use anyhow::{Context, Result};
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::Path;
use xybrid_cli::systemone::{
    parse_request, render_response, RefusalKind, RequestError, RequestLimits,
    DEFAULT_MAX_BODY_BYTES,
};
use xybrid_core::ir::decision::{validate_structure, StructuralLimits};
use xybrid_core::ir::{DecisionQuestionKind, DecisionRequest, Decisions};

use crate::ui;

/// The model a decision request runs on, as the `run` flags name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelRef<'a> {
    /// `--model`: a registry id, with `--target` as the platform.
    Registry {
        id: &'a str,
        platform: Option<&'a str>,
    },
    /// `--model-file`: a GGUF file.
    File(&'a Path),
    /// `--directory`: a model directory with `model_metadata.json`.
    Directory(&'a Path),
    /// `--huggingface`: a Hugging Face repo, `REPO[:VARIANT]`.
    HuggingFace(&'a str),
    /// `--bundle`: a `.xyb` bundle.
    Bundle(&'a Path),
}

impl<'a> ModelRef<'a> {
    /// The model the `run` flags name; `None` when they name none.
    pub(crate) fn from_flags(
        model: Option<&'a str>,
        platform: Option<&'a str>,
        model_file: Option<&'a Path>,
        directory: Option<&'a Path>,
        huggingface: Option<&'a str>,
        bundle: Option<&'a Path>,
    ) -> Option<Self> {
        // Clap lets at most one of them through.
        model
            .map(|id| Self::Registry { id, platform })
            .or(model_file.map(Self::File))
            .or(directory.map(Self::Directory))
            .or(huggingface.map(Self::HuggingFace))
            .or(bundle.map(Self::Bundle))
    }

    /// The name the response gives the model: the registry id, the
    /// Hugging Face repo, or the file or directory name.
    pub(crate) fn name(&self) -> String {
        let file_name = |path: &Path| {
            path.file_name()
                .unwrap_or(path.as_os_str())
                .to_string_lossy()
                .into_owned()
        };
        match self {
            Self::Registry { id, .. } => (*id).to_string(),
            Self::HuggingFace(repo) => (*repo).to_string(),
            Self::File(path) | Self::Directory(path) | Self::Bundle(path) => file_name(path),
        }
    }
}

impl fmt::Display for ModelRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registry { id, .. } => write!(f, "{id} (registry)"),
            Self::File(path) => write!(f, "{} (GGUF file)", path.display()),
            Self::Directory(path) => write!(f, "{} (directory)", path.display()),
            Self::HuggingFace(repo) => write!(f, "{repo} (Hugging Face)"),
            Self::Bundle(path) => write!(f, "{} (bundle)", path.display()),
        }
    }
}

/// What `xybrid run --decision` was asked to do.
pub(crate) struct DecisionRun<'a> {
    /// The request file; `-` reads standard input.
    pub(crate) request: &'a Path,
    /// The model to run it on.
    pub(crate) model: ModelRef<'a>,
    /// Where to save the response, besides printing it.
    pub(crate) output: Option<&'a Path>,
    /// Check the request without running it.
    pub(crate) dry_run: bool,
}

/// Runs `xybrid run --decision`: reads and checks the request, answers it
/// through [`run_decision_request`], and prints the SystemOne response (and
/// saves it to `--output`).
pub(crate) fn handle_decision_run(run: DecisionRun<'_>) -> Result<()> {
    ui::header("Run · Decisions");
    ui::kv("Request", &request_name(run.request));
    ui::kv("Model", &run.model.to_string());

    let body = read_request(run.request, DEFAULT_MAX_BODY_BYTES)?;
    let request = parse_request(&body, &RequestLimits::default()).map_err(refusal)?;
    validate_structure(&request, &StructuralLimits::default())
        .context("decision request refused (positions count from 0)")?;
    ui::kv("Questions", &question_summary(&request));

    if run.dry_run {
        ui::section("Dry Run");
        println!();
        ui::ok("Decision request is valid");
        ui::warning("This build cannot run decision models yet, so the model was not checked");
        println!();
        return Ok(());
    }

    let decisions = run_decision_request(&run.model, &request)?;
    let response = render_response(&run.model.name(), &request, &decisions)
        .context("the decision runtime's answers do not answer the request")?;

    ui::section("Results");
    println!();
    println!("{response}");
    if let Some(path) = run.output {
        fs::write(path, &response)
            .with_context(|| format!("Failed to write the response to {}", path.display()))?;
        println!();
        ui::ok(&format!("Response saved to {}", path.display()));
    }
    println!();
    Ok(())
}

/// Answers a parsed, structurally valid decision request on `model`, all
/// questions or none.
///
/// The one seam between the CLI and the runtime's decision support: the
/// runtime integration replaces this body. Until then it refuses every
/// request before opening the model.
pub(crate) fn run_decision_request(
    model: &ModelRef<'_>,
    request: &DecisionRequest,
) -> Result<Decisions> {
    // The runtime will load `model` and answer `request`.
    let _ = (model, request);
    anyhow::bail!("decision models are not supported by this build yet")
}

/// The request's source as the header shows it.
fn request_name(path: &Path) -> String {
    if is_stdin(path) {
        "standard input".to_string()
    } else {
        path.display().to_string()
    }
}

fn is_stdin(path: &Path) -> bool {
    path.as_os_str() == "-"
}

/// Reads the request body, refusing one longer than `max` bytes before
/// reading past it.
fn read_request(path: &Path, max: usize) -> Result<Vec<u8>> {
    let too_large = || {
        anyhow::anyhow!("the request is larger than {max} bytes, the most that is read").context(
            format!(
                "decision request refused ({})",
                RefusalKind::RequestTooLarge
            ),
        )
    };
    let limit = u64::try_from(max).unwrap_or(u64::MAX).saturating_add(1);
    let mut body = Vec::new();
    if is_stdin(path) {
        std::io::stdin()
            .lock()
            .take(limit)
            .read_to_end(&mut body)
            .context("Failed to read the decision request from standard input")?;
    } else {
        let context = || format!("Failed to read the decision request {}", path.display());
        let file = fs::File::open(path).with_context(context)?;
        // Refuse a large file before reading any of it.
        if file.metadata().with_context(context)?.len() > limit - 1 {
            return Err(too_large());
        }
        file.take(limit)
            .read_to_end(&mut body)
            .with_context(context)?;
    }
    if body.len() > max {
        return Err(too_large());
    }
    Ok(body)
}

/// A parse refusal, named by its kind. A ceiling names no position.
fn refusal(error: RequestError) -> anyhow::Error {
    let context = match error.kind() {
        RefusalKind::RequestTooLarge => format!("decision request refused ({})", error.kind()),
        kind => format!("decision request refused ({kind}; positions count from 0)"),
    };
    anyhow::Error::new(error).context(context)
}

/// How many questions of each type: counts only, never content.
fn question_summary(request: &DecisionRequest) -> String {
    let questions = request.questions();
    let count = |is: fn(&DecisionQuestionKind) -> bool| {
        questions
            .iter()
            .filter(|question| is(question.kind()))
            .count()
    };
    let kinds = [
        ("choice", count(|kind| kind.as_choice().is_some())),
        ("noul", count(|kind| kind.as_yes_no().is_some())),
        ("score", count(|kind| kind.as_score().is_some())),
    ];
    let kinds: Vec<String> = kinds
        .iter()
        .filter(|(_, count)| *count > 0)
        .map(|(name, count)| format!("{count} {name}"))
        .collect();
    format!("{} ({})", questions.len(), kinds.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn the_model_comes_from_the_one_flag_given() {
        let path = PathBuf::from("models/d1-2B-Q4_K_M.gguf");
        assert_eq!(
            ModelRef::from_flags(None, None, None, None, None, None),
            None
        );
        let file = ModelRef::from_flags(None, None, Some(&path), None, None, None).unwrap();
        assert_eq!(file, ModelRef::File(&path));
        assert_eq!(file.name(), "d1-2B-Q4_K_M.gguf");
        let registry = ModelRef::from_flags(Some("d1-2b"), Some("linux"), None, None, None, None);
        assert_eq!(
            registry,
            Some(ModelRef::Registry {
                id: "d1-2b",
                platform: Some("linux")
            })
        );
        assert_eq!(registry.unwrap().name(), "d1-2b");
        let repo = ModelRef::from_flags(None, None, None, None, Some("LiquidAI/d1-2B-GGUF"), None);
        assert_eq!(repo.unwrap().name(), "LiquidAI/d1-2B-GGUF");
        let dir = PathBuf::from("/tmp/d1/");
        assert_eq!(ModelRef::Directory(&dir).name(), "d1");
    }

    #[test]
    fn the_seam_refuses_until_the_runtime_lands() {
        let request = parse_request(
            br#"{"state": null, "questions": {"q": {"type": "noul", "instructions": "x"}}}"#,
            &RequestLimits::default(),
        )
        .unwrap();
        let path = PathBuf::from("m.gguf");
        let error = run_decision_request(&ModelRef::File(&path), &request).unwrap_err();
        assert_eq!(
            error.to_string(),
            "decision models are not supported by this build yet"
        );
    }

    #[test]
    fn requests_are_read_up_to_the_ceiling() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("request.json");
        fs::write(&path, b"0123456789").unwrap();
        assert_eq!(read_request(&path, 10).unwrap(), b"0123456789");
        let error = read_request(&path, 9).unwrap_err();
        assert_eq!(
            format!("{error:#}"),
            "decision request refused (request_too_large): the request is larger than 9 bytes, \
             the most that is read"
        );
        let missing = read_request(&dir.path().join("missing.json"), 10).unwrap_err();
        assert!(
            missing
                .to_string()
                .contains("Failed to read the decision request"),
            "{missing:#}"
        );
    }

    #[test]
    fn refusals_name_their_kind() {
        let error = refusal(RequestError::MissingState);
        assert_eq!(
            format!("{error:#}"),
            "decision request refused (missing_state; positions count from 0): the request has \
             no \"state\" (use null for no state)"
        );
        let error = refusal(RequestError::TooManyValues { max: 4 });
        assert_eq!(
            format!("{error:#}"),
            "decision request refused (request_too_large): the request holds more than 4 JSON \
             values"
        );
    }

    #[test]
    fn the_summary_counts_questions_by_type() {
        let request = parse_request(
            br#"{"state": null, "questions": {
                "a": {"type": "noul", "instructions": "x"},
                "b": {"type": "score", "instructions": "x", "criteria": [null, null]},
                "c": {"type": "noul", "instructions": "x"}}}"#,
            &RequestLimits::default(),
        )
        .unwrap();
        assert_eq!(question_summary(&request), "3 (2 noul, 1 score)");
    }
}
