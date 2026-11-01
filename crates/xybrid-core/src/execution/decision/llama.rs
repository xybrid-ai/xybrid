//! [`DecisionVocab`] over a loaded llama.cpp model.

use std::fmt;

use xybrid_llama::{LlamaModel, TokenAttr};

use super::{DecisionVocab, Token};
use crate::ir::decision::DecisionError;

/// A llama.cpp model's vocabulary, as the decision protocol reads it.
///
/// Construction walks the whole vocabulary once for its CONTROL spellings
/// (128,000 entries on d1-2B, a few milliseconds), so build it once per
/// loaded model.
pub struct LlamaVocab<'m> {
    model: &'m LlamaModel,
    control: Vec<Vec<u8>>,
}

impl<'m> LlamaVocab<'m> {
    /// The vocabulary of `model`: the texts of its CONTROL tokens of 2 bytes
    /// or more, in token-id order, as `init_quoted_spellings(CONTROL, 2)`
    /// collects them (`tools/server/server-decision.cpp:213-221`).
    ///
    /// # Errors
    ///
    /// [`DecisionError::VocabularyFailed`] when llama.cpp cannot report a
    /// token's attributes or text: a failure is never read as "not a
    /// CONTROL token".
    pub fn new(model: &'m LlamaModel) -> Result<Self, DecisionError> {
        let mut control = Vec::new();
        for id in 0..model.n_vocab() {
            let id = Token::try_from(id).map_err(|_| DecisionError::VocabularyFailed)?;
            let attr = model
                .token_attr(id)
                .ok()
                .flatten()
                .ok_or(DecisionError::VocabularyFailed)?;
            if !attr.intersects(TokenAttr::CONTROL) {
                continue;
            }
            let text = model
                .token_text(id)
                .ok()
                .flatten()
                .ok_or(DecisionError::VocabularyFailed)?;
            if text.len() >= 2 {
                control.push(text);
            }
        }
        Ok(Self { model, control })
    }
}

impl DecisionVocab for LlamaVocab<'_> {
    fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<Token>, DecisionError> {
        self.model
            .tokenize_with(text, add_special, parse_special)
            .map_err(|_| DecisionError::TokenizationFailed)
    }

    fn control_spellings(&self) -> &[Vec<u8>] {
        &self.control
    }
}

/// Counts only.
impl fmt::Debug for LlamaVocab<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlamaVocab")
            .field("control_spellings", &self.control.len())
            .finish()
    }
}
