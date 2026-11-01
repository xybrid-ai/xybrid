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
//! - a `GgufDecision` scorer with a media companion (a `vision_encoder`
//!   block): the decision profiles read no media, and one that does comes
//!   with its own protocol;
//! - a header that does not read or classify completely, which is an error
//!   and never taken for an ordinary model.
//!
//! A header with more metadata keys than this build reads (4,096 by
//! default) is refused too, as a limit of this build rather than a fault of
//! the file: valid GGUFs can carry more.
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
//! So a decision GGUF declared as a projector or as a zzz model is not
//! detected here; those loaders fail on it.
//!
//! # What it guarantees
//!
//! A consistency check between metadata and file, not a cryptographic
//! identity: registry downloads are sha256-verified when they are fetched.
//! The header admission reads records the file it came from
//! ([`GgufHeader::file_stamp`]: size, modification time, device and inode
//! on Unix, and the SHA-256 of the header bytes), and a decision model's
//! load goes:
//!
//! 1. `admit_gguf(&metadata, model_dir)`, before any context exists;
//! 2. `xybrid_llama::LlamaModel::load` of [`AdmittedDecisionModel::path`];
//! 3. `verify_native_load(admitted.path(), admitted.header(), &model)`
//!    (feature `llm-llamacpp`), before the model is used: the file is still
//!    the one admitted, and llama.cpp holds the scalar metadata its header
//!    predicts.
//!
//! It checks decision identity only: whether the declared runtime can load
//! the file is still for that runtime to find out.

use std::fmt;
use std::path::{Path, PathBuf};

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
/// needs from the admission. The file's identity when it was read is its
/// header's [`file_stamp`](GgufHeader::file_stamp).
pub struct AdmittedDecisionModel {
    spec: GgufDecisionSpec,
    profile: DecisionProfileId,
    path: PathBuf,
    // Boxed: with its file stamp a header is larger than the rest of the
    // admission, and `GgufAdmission`'s other variant is empty.
    header: Box<GgufHeader>,
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

    /// The header admission read and classified, with its file stamp: the
    /// single source of the model's template and temperatures (read through
    /// [`effective_f32`](super::effective_f32)), and what `verify_native_load`
    /// checks the file and the loaded model against.
    pub fn header(&self) -> &GgufHeader {
        &self.header
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
/// any context is allocated. A `GgufDecision` spec and its metadata
/// ([`ModelMetadata::validate`]) are validated first, without reading
/// anything. Reading the header reads and hashes only the header (8 MB for
/// d1-2B's 128,000-token vocabulary), within [`GgufLimits`](super::GgufLimits)'
/// defaults: 128 MiB and 4,096 metadata keys, for every GGUF it reads.
///
/// # Errors
///
/// - [`AdapterError::IOError`] when the file cannot be read;
/// - [`AdapterError::InvalidInput`] for an invalid `GgufDecision` spec or
///   metadata (a `vision_encoder` block included), a header that does not
///   read or classify completely, and a `GgufDecision` scorer whose protocol
///   differs from the GGUF's decision type;
/// - [`AdapterError::UnsupportedModelCapability`] for a decision GGUF
///   declared as anything but a `GgufDecision` scorer, for a `GgufDecision`
///   scorer on a GGUF that is not a decision model or whose protocol no
///   profile in this build admits, and for a GGUF that declares more
///   metadata keys than this build's limit (a resource limit, reported with
///   both counts).
///
/// Messages name the model id, the declared template and the file's
/// decision facts (types, keys, counts), quoted and cut short; never
/// request content.
pub fn admit_gguf(
    metadata: &ModelMetadata,
    model_dir: &Path,
) -> Result<GgufAdmission, AdapterError> {
    let model_id = metadata.model_id.as_str();
    let (model_file, declared) = match &metadata.execution_template {
        ExecutionTemplate::ChoiceScorer { scorer } => match scorer {
            ChoiceScorerSpec::GgufDecision(spec) => {
                validate_scorer_spec(scorer)?;
                // Cross-field rules, a media companion's refusal among them.
                metadata.validate().map_err(|reason| {
                    AdapterError::InvalidInput(format!("model '{model_id}': {reason}"))
                })?;
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
    let header = GgufHeader::read_file(&path).map_err(|error| unclassifiable(model_id, error))?;
    let classification = classify(&header).map_err(|error| unclassifiable(model_id, error))?;
    let profile = check_declaration(metadata, declared, &classification)?;
    Ok(match (declared, profile) {
        (Some(spec), Some(profile)) => GgufAdmission::Decision(AdmittedDecisionModel {
            spec: spec.clone(),
            profile,
            path,
            header: Box::new(header),
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

/// A header that does not read or classify completely. An I/O failure stays
/// one, with its kind, and the key limit is reported as a limit.
fn unclassifiable(model_id: &str, error: GgufError) -> AdapterError {
    match error {
        GgufError::Io(error) => AdapterError::IOError(std::io::Error::new(
            error.kind(),
            format!("model '{model_id}': cannot read its GGUF: {error}"),
        )),
        // A resource limit of this build, not a fault of the file: valid
        // GGUFs can carry more keys (custom metadata, merge provenance).
        GgufError::TooManyEntries { count, limit } => AdapterError::UnsupportedModelCapability {
            model_id: model_id.to_string(),
            capability: "loading in this build".to_string(),
            hint: format!(
                "the GGUF declares {count} metadata keys, above this build's limit of {limit}"
            ),
        },
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
