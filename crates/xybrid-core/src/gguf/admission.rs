//! Admission at the load boundary: a GGUF must be what its model's metadata
//! says it is.
//!
//! Metadata can be stale or wrong: a chat sidecar written next to a file
//! that is really a decision model, a bundle cached before its file was
//! classified, a hand-written `GgufDecision` scorer on a chat model. A
//! decision GGUF loaded as a chat model generates text from prompts it was
//! never trained on, and nothing in llama.cpp stops it (LiquidAI's d1 GGUFs
//! carry an ordinary chat template too). So every load route checks the file
//! itself against the metadata before it allocates a context: [`admit_gguf`]
//! reads the header, classifies it ([`classify`]) and refuses
//!
//! - a decision model declared as anything but a `ChoiceScorer` with a
//!   [`GgufDecision`](ChoiceScorerSpec::GgufDecision) scorer: a stale chat
//!   sidecar, a cached bundle;
//! - a `GgufDecision` scorer on a GGUF that is not a decision model, or one
//!   whose protocol no profile in this build admits;
//! - a `GgufDecision` scorer whose `protocol` differs from the GGUF's
//!   decision type: the protocol is an assertion, never an override;
//! - a header that does not read or classify completely, which is an error
//!   and never taken for an ordinary model.
//!
//! # Which files
//!
//! Admission classifies the GGUF llama.cpp builds a model from: the
//! `model_file` of a `Gguf`, `VisionLanguage` or `GgufDecision` template.
//! Other templates open no such file and are admitted as
//! [`GgufAdmission::NotDecision`] without reading anything: ONNX, Candle and
//! Core ML files are not GGUFs, whisper.cpp reads GGML, and the zzz speech
//! engine and a vision encoder's projector (llama.cpp's clip loader) read
//! their GGUFs with loaders of their own, which build no llama.cpp model.
//!
//! # What it guarantees
//!
//! A consistency check between metadata and file, not a cryptographic
//! identity: registry downloads are sha256-verified when they are fetched.
//! Admission records the file's length and modification time when it reads
//! the header; after the native load, the caller checks the file is still
//! the one admitted ([`AdmittedDecisionModel::verify_unchanged`]) and that
//! llama.cpp loaded the metadata the header holds (`verify_native_metadata`,
//! feature `llm-llamacpp`). The latter compares llama.cpp's string view of
//! every scalar key with the header: a scalar consistency check, not a file
//! identity (arrays, such as the vocabulary, and tensor data are not
//! compared).
//!
//! It checks decision identity only: whether the declared runtime can load
//! the file is still for that runtime to find out.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::classify::{classify, DecisionClassification, DecisionProfileId};
use super::error::GgufError;
use super::header::GgufHeader;
use crate::execution::choice::validate_scorer_spec;
use crate::execution::template::{
    ChoiceScorerSpec, DecisionProtocol, ExecutionTemplate, GgufDecisionSpec, ModelMetadata,
};
use crate::runtime_adapter::AdapterError;

/// Characters of a string from the model file an error quotes: enough to
/// name a decision type, too few for a damaged file to fill the message.
const MAX_QUOTED_CHARS: usize = 64;

/// Characters of a classification detail (which names keys from the file)
/// an error includes.
const MAX_DETAIL_CHARS: usize = 256;

/// What [`admit_gguf`] admitted.
#[derive(Debug)]
#[non_exhaustive]
pub enum GgufAdmission {
    /// An ordinary model: the template's GGUF classified as no decision
    /// model, or the template opens no GGUF through llama.cpp. Load it as
    /// before.
    NotDecision,
    /// A decision model, admitted by a profile of this build for the
    /// metadata's `GgufDecision` scorer.
    Decision(AdmittedDecisionModel),
}

/// A decision GGUF admitted for a `GgufDecision` scorer: everything a load
/// needs from the admission, as it was when the header was read.
pub struct AdmittedDecisionModel {
    spec: GgufDecisionSpec,
    profile: DecisionProfileId,
    path: PathBuf,
    header: GgufHeader,
    file_len: u64,
    modified: SystemTime,
}

impl AdmittedDecisionModel {
    /// The scorer spec, validated
    /// ([`validate_scorer_spec`]).
    pub fn spec(&self) -> &GgufDecisionSpec {
        &self.spec
    }

    /// The profile that admitted the GGUF.
    pub fn profile(&self) -> DecisionProfileId {
        self.profile
    }

    /// The model file: the spec's `model_file` resolved against the model
    /// directory.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The header admission read and classified: the single source of the
    /// model's template and temperatures (read through
    /// [`effective_f32`](super::effective_f32)), and what the loaded model
    /// is verified against.
    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// The file's length in bytes when its header was read.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// The file's modification time when its header was read.
    pub fn modified(&self) -> SystemTime {
        self.modified
    }

    /// Checks the file at [`Self::path`] still has the length and
    /// modification time it had when it was admitted. Call it after the
    /// native load: a file replaced in between fails here (unless the
    /// replacement keeps both, which `verify_native_metadata` then has to
    /// catch).
    ///
    /// # Errors
    ///
    /// [`AdapterError::IOError`] when the file cannot be read or has
    /// changed.
    pub fn verify_unchanged(&self) -> Result<(), AdapterError> {
        let now = file_identity(&self.path).map_err(|error| {
            AdapterError::IOError(std::io::Error::new(
                error.kind(),
                format!("cannot read the admitted GGUF: {error}"),
            ))
        })?;
        if now != (self.file_len, self.modified) {
            return Err(AdapterError::IOError(std::io::Error::other(
                "the admitted GGUF has changed since it was admitted",
            )));
        }
        Ok(())
    }
}

/// Summarized: a header holds the whole vocabulary.
impl fmt::Debug for AdmittedDecisionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdmittedDecisionModel")
            .field("spec", &self.spec)
            .field("profile", &self.profile)
            .field("path", &self.path)
            .field("metadata_keys", &self.header.metadata().len())
            .field("tensors", &self.header.tensors().len())
            .field("header_bytes", &self.header.header_bytes())
            .field("file_len", &self.file_len)
            .finish_non_exhaustive()
    }
}

/// Admits the GGUF a model's metadata names in `model_dir` (see the
/// [module documentation](self)): a decision model only as the
/// `GgufDecision` scorer of its protocol, an ordinary model only as
/// anything else.
///
/// Every load route calls it (sidecars, bundles and cached models,
/// standalone files, Hugging Face downloads with generated metadata) before
/// any context is allocated. A `GgufDecision` spec is validated first,
/// without reading anything. Reading the header reads only the header (8 MB
/// for d1-2B's 128,000-token vocabulary), within
/// [`GgufLimits`](super::GgufLimits)' default ceiling.
///
/// # Errors
///
/// - [`AdapterError::IOError`] when the file cannot be read, or changes
///   while its header is read;
/// - [`AdapterError::InvalidInput`] for an invalid `GgufDecision` spec, a
///   header that does not read or classify completely, and a `GgufDecision`
///   scorer whose protocol differs from the GGUF's decision type;
/// - [`AdapterError::UnsupportedModelCapability`] for a decision GGUF
///   declared as anything but a `GgufDecision` scorer, and for a
///   `GgufDecision` scorer on a GGUF that is not a decision model or whose
///   protocol no profile in this build admits.
///
/// Messages name the model id, the declared template and the file's
/// decision facts (types, keys), quoted and cut short; never request
/// content.
pub fn admit_gguf(
    metadata: &ModelMetadata,
    model_dir: &Path,
) -> Result<GgufAdmission, AdapterError> {
    let model_id = metadata.model_id.as_str();
    let (model_file, declared) = match &metadata.execution_template {
        ExecutionTemplate::ChoiceScorer { scorer } => match scorer {
            ChoiceScorerSpec::GgufDecision(spec) => {
                validate_scorer_spec(scorer)?;
                (spec.model_file.as_str(), Some(spec))
            }
            // An ONNX scorer reads no GGUF.
            _ => return Ok(GgufAdmission::NotDecision),
        },
        ExecutionTemplate::Gguf { model_file, .. }
        | ExecutionTemplate::VisionLanguage { model_file, .. } => (model_file.as_str(), None),
        _ => return Ok(GgufAdmission::NotDecision),
    };

    let path = model_dir.join(model_file);
    let (file_len, modified) = file_identity(&path).map_err(|error| unreadable(model_id, error))?;
    let header = GgufHeader::read_file(&path).map_err(|error| unclassifiable(model_id, error))?;
    let after = file_identity(&path).map_err(|error| unreadable(model_id, error))?;
    if after != (file_len, modified) {
        return Err(AdapterError::IOError(std::io::Error::other(format!(
            "model '{model_id}': its GGUF changed while its header was read"
        ))));
    }

    let classification = classify(&header).map_err(|error| unclassifiable(model_id, error))?;
    let profile = check_declaration(metadata, declared, &classification)?;
    Ok(match (declared, profile) {
        (Some(spec), Some(profile)) => GgufAdmission::Decision(AdmittedDecisionModel {
            spec: spec.clone(),
            profile,
            path,
            header,
            file_len,
            modified,
        }),
        _ => GgufAdmission::NotDecision,
    })
}

/// The profile that admits the classified GGUF for its declaration: `Some`
/// for a `GgufDecision` scorer, `None` for any other template, or the
/// refusal.
fn check_declaration(
    metadata: &ModelMetadata,
    declared: Option<&GgufDecisionSpec>,
    classification: &DecisionClassification,
) -> Result<Option<DecisionProfileId>, AdapterError> {
    let model_id = &metadata.model_id;

    // The GGUF's decision type, and the profile that admits it or why none
    // does.
    let (decision_type, admitted) = match classification {
        DecisionClassification::NotDecision => {
            return match declared {
                None => Ok(None),
                Some(spec) => Err(AdapterError::UnsupportedModelCapability {
                    model_id: model_id.clone(),
                    capability: format!("{} decisions", spec.protocol),
                    hint: "its GGUF has no decision type, so it is not a decision model: \
                           declare it with the template it was built for"
                        .to_string(),
                }),
            }
        }
        DecisionClassification::Supported(profile) => {
            (DecisionProtocol::from(*profile).as_str(), Ok(*profile))
        }
        DecisionClassification::Unsupported {
            decision_type,
            reason,
        } => (decision_type.as_str(), Err(reason)),
    };

    let Some(spec) = declared else {
        let hint = match admitted {
            Ok(profile) => format!(
                "its GGUF is a {profile} decision model (decision type {}), which loads only \
                 as a ChoiceScorer with a GgufDecision scorer: regenerate model_metadata.json, \
                 or remove a stale one",
                quoted(decision_type)
            ),
            Err(reason) => format!(
                "its GGUF is a decision model (decision type {}) that this build does not run \
                 ({}), so it does not load as any other model either",
                quoted(decision_type),
                bounded(&reason.to_string())
            ),
        };
        return Err(AdapterError::UnsupportedModelCapability {
            model_id: model_id.clone(),
            capability: format!("{} execution", template_label(&metadata.execution_template)),
            hint,
        });
    };

    if spec.protocol.as_str() != decision_type {
        return Err(AdapterError::InvalidInput(format!(
            "model '{model_id}' declares a GgufDecision scorer for the {} protocol, but its \
             GGUF's decision type is {}; the protocol is checked against the GGUF, never \
             applied to it",
            spec.protocol,
            quoted(decision_type)
        )));
    }
    match admitted {
        Ok(profile) => Ok(Some(profile)),
        Err(reason) => Err(AdapterError::UnsupportedModelCapability {
            model_id: model_id.clone(),
            capability: format!("{} decisions", spec.protocol),
            hint: format!(
                "its GGUF has that decision type, but no profile in this build admits it: {}",
                bounded(&reason.to_string())
            ),
        }),
    }
}

/// The file's length and modification time.
fn file_identity(path: &Path) -> std::io::Result<(u64, SystemTime)> {
    let metadata = std::fs::metadata(path)?;
    Ok((metadata.len(), metadata.modified()?))
}

/// An I/O failure on the model's GGUF, with its kind.
fn unreadable(model_id: &str, error: std::io::Error) -> AdapterError {
    AdapterError::IOError(std::io::Error::new(
        error.kind(),
        format!("model '{model_id}': cannot read its GGUF: {error}"),
    ))
}

/// A header that does not read or classify completely.
fn unclassifiable(model_id: &str, error: GgufError) -> AdapterError {
    match error {
        GgufError::Io(error) => unreadable(model_id, error),
        other => AdapterError::InvalidInput(format!(
            "model '{model_id}': its GGUF cannot be classified, so it does not load: {}",
            bounded(&other.to_string())
        )),
    }
}

/// How an error names a declared template: its `"type"` tag. Only the
/// templates admission reads a GGUF for reach a refusal.
fn template_label(template: &ExecutionTemplate) -> &'static str {
    match template {
        ExecutionTemplate::Gguf { .. } => "Gguf",
        ExecutionTemplate::VisionLanguage { .. } => "VisionLanguage",
        _ => "this template's",
    }
}

/// A string from the model file, quoted and cut to [`MAX_QUOTED_CHARS`].
fn quoted(text: &str) -> String {
    match text.char_indices().nth(MAX_QUOTED_CHARS) {
        None => format!("{text:?}"),
        Some((end, _)) => format!("{:?}…", &text[..end]),
    }
}

/// `text` cut to [`MAX_DETAIL_CHARS`].
fn bounded(text: &str) -> String {
    match text.char_indices().nth(MAX_DETAIL_CHARS) {
        None => text.to_string(),
        Some((end, _)) => format!("{}…", &text[..end]),
    }
}

#[cfg(test)]
mod tests;
