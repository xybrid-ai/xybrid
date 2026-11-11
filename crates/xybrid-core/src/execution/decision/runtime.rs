//! One loaded model, one renderer and one exclusively owned decision context.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use xybrid_llama::{DecisionContext, LlamaError, LlamaModel};

use super::liquid::{DecisionProtocolError, LiquidProtocol};
use crate::execution::template::DecisionTemplate;
use crate::ir::{DecisionRequest, DecisionUsage, Decisions, Envelope};
use crate::runtime_adapter::AdapterError;

/// Includes file identity as well as size/time, so an atomic replacement with
/// preserved timestamps cannot reuse the previous model's vocabulary/context.
#[derive(PartialEq, Eq)]
struct ArtifactIdentity {
    length: u64,
    modified: std::time::SystemTime,
    created: Option<std::time::SystemTime>,
    #[cfg(unix)]
    unix: (u64, u64, i64, i64),
}

fn file_identity(path: &Path) -> Option<ArtifactIdentity> {
    let metadata = std::fs::metadata(path).ok()?;
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    Some(ArtifactIdentity {
        length: metadata.len(),
        modified: metadata.modified().ok()?,
        created: metadata.created().ok(),
        #[cfg(unix)]
        unix: (
            metadata.dev(),
            metadata.ino(),
            metadata.ctime(),
            metadata.ctime_nsec(),
        ),
    })
}

/// A cache entry includes all declared settings and the observed artifact identity.
pub(crate) struct DecisionRuntime {
    path: PathBuf,
    identity: Option<ArtifactIdentity>,
    spec: DecisionTemplate,
    model: LlamaModel,
    protocol: LiquidProtocol,
    // A failed pass poisons the context until it has been recreated.
    context: Option<DecisionContext>,
}

impl DecisionRuntime {
    pub(crate) fn load(
        model_id: &str,
        path: &Path,
        spec: &DecisionTemplate,
    ) -> Result<Self, AdapterError> {
        spec.validate()?;
        if !path.is_file() {
            // Name the file: the native loader only reports a content-free failure.
            return Err(AdapterError::ModelNotFound(path.display().to_string()));
        }
        let identity = file_identity(path);
        let model = LlamaModel::load_decision_cpu(
            path.to_str()
                .ok_or(DecisionProtocolError::InvalidMetadata)?,
        )
        .map_err(native_error)?;
        // A GGUF without the decision template is some other model, not a
        // damaged d1 artifact.
        if matches!(model.named_chat_template("systemone"), Ok(None)) {
            return Err(unadmitted_artifact(model_id));
        }
        let protocol =
            LiquidProtocol::load(&model).map_err(|error| admission_error(model_id, error))?;
        let context = DecisionContext::new(&model).map_err(native_error)?;
        if identity != file_identity(path) {
            return Err(DecisionProtocolError::InvalidMetadata.into());
        }
        Ok(Self {
            path: path.into(),
            identity,
            spec: spec.clone(),
            model,
            protocol,
            context: Some(context),
        })
    }

    pub(crate) fn matches(&self, path: &Path, spec: &DecisionTemplate) -> bool {
        self.path == path
            && self.spec == *spec
            && self.identity.is_some()
            && self.identity == file_identity(path)
    }

    pub(crate) fn run(
        &mut self,
        request: &DecisionRequest,
        abort: Option<&AtomicBool>,
    ) -> Result<Envelope, AdapterError> {
        cancelled(abort)?;
        // Preflight every question before the first native evaluation. Prompt
        // strings are released by preparation; only bounded tokens/plans remain.
        // Preparation reads the vocabulary and renders with a fresh interpreter:
        // a refusal here never touched the context, which stays usable.
        let prepared = self
            .protocol
            .prepare_controlled(&self.model, request, abort)?;
        cancelled(abort)?;
        let result = self.evaluate(&prepared, abort);
        if result.is_err() {
            // Once passes begin, any failure drops the context: cancellation
            // between passes and finalization failure, not only failures
            // reported by the native evaluator.
            self.context = None;
        }
        result
    }

    fn evaluate(
        &mut self,
        prepared: &[super::prepare::PreparedQuestion],
        abort: Option<&AtomicBool>,
    ) -> Result<Envelope, AdapterError> {
        if self.context.is_none() {
            self.context = Some(DecisionContext::new(&self.model).map_err(native_error)?);
        }
        let mut answers = Vec::with_capacity(prepared.len());
        let mut tokens = 0u64;
        for question in prepared {
            cancelled(abort)?;
            let result = self
                .context
                .as_mut()
                .expect("decision context initialized before evaluating a pass")
                .evaluate(&question.tokens, &question.select, abort);
            let logits = match result {
                Ok(logits) => logits,
                Err(error) => {
                    // Drop even when native recovery failed. A later request
                    // must recreate state and can never reuse a failed context.
                    self.context = None;
                    return Err(native_error(error));
                }
            };
            cancelled(abort)?;
            answers.push(super::math::finalize(question, &logits)?);
            tokens += question.tokens.len() as u64;
        }
        cancelled(abort)?;
        let usage = DecisionUsage::new(prepared.len() as u32, tokens);
        let mut output = Envelope::decisions(Decisions::new(answers, usage));
        output.set_metadata("tokens_in".into(), tokens.to_string());
        output.set_metadata("tokens_out".into(), "0".into());
        Ok(output)
    }
}

fn cancelled(abort: Option<&AtomicBool>) -> Result<(), AdapterError> {
    if abort.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(DecisionProtocolError::Cancelled.into());
    }
    Ok(())
}

/// The content-free refusal for an artifact that loads but is not the
/// admitted profile, so a host can tell "not a d1 text model" from a damaged
/// or missing file.
fn unadmitted_artifact(model_id: &str) -> AdapterError {
    AdapterError::UnsupportedModelCapability {
        model_id: model_id.into(),
        capability: format!("{} decisions", crate::execution::template::LIQUID_TEXT_V1),
        hint: "the artifact is not the admitted Liquid d1 text model: its architecture, \
               decision type, template or decision metadata differs"
            .into(),
    }
}

/// Maps a failed admission. Metadata the profile does not admit is a
/// capability the model lacks; metadata that cannot be read or parsed stays a
/// load failure.
fn admission_error(model_id: &str, error: DecisionProtocolError) -> AdapterError {
    match error {
        DecisionProtocolError::UnsupportedMetadata => unadmitted_artifact(model_id),
        other => other.into(),
    }
}

fn native_error(error: LlamaError) -> AdapterError {
    match error {
        LlamaError::DecisionCancelled => DecisionProtocolError::Cancelled.into(),
        // Never surface native diagnostics, template snippets or caller text.
        _ => DecisionProtocolError::NativeFailure.into(),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn replacing_same_size_artifact_with_preserved_mtime_invalidates_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"old").unwrap();
        let identity = file_identity(&path).unwrap();
        let replacement = dir.path().join("replacement.gguf");
        std::fs::write(&replacement, b"new").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&replacement)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(identity.modified))
            .unwrap();
        std::fs::rename(replacement, &path).unwrap();
        let replaced = file_identity(&path).unwrap();
        assert_eq!(identity.length, replaced.length);
        assert_eq!(identity.modified, replaced.modified);
        assert!(identity != replaced);
    }

    #[test]
    fn an_unadmitted_artifact_is_a_missing_capability_not_bad_input() {
        let error = admission_error("some-model", DecisionProtocolError::UnsupportedMetadata);
        let AdapterError::UnsupportedModelCapability {
            model_id,
            capability,
            ..
        } = error
        else {
            panic!("{error:?}");
        };
        assert_eq!(model_id, "some-model");
        assert_eq!(capability, "lfm3-d1/text/v1 decisions");

        // Metadata that cannot be read or parsed is a damaged artifact.
        assert!(matches!(
            admission_error("some-model", DecisionProtocolError::InvalidMetadata),
            AdapterError::InvalidInput(_)
        ));
    }
}
