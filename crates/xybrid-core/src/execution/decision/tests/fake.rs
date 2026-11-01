//! Test doubles: a vocabulary with chosen single-token forms, and synthetic
//! GGUF headers.

use std::collections::HashMap;

use crate::execution::decision::{
    DecisionLimits, DecisionModelFacts, DecisionVocab, Token, SYSTEMONE_TEMPLATE,
};
use crate::gguf::{DecisionProfileId, GgufHeader, GgufLimits};
use crate::ir::decision::DecisionError;

/// A vocabulary that tokenizes byte by byte (token `1000 + byte`), except
/// the forms it is told are single tokens.
///
/// With `parse_special`, a control spelling in the text is one token
/// (`2000 + its index`), as llama.cpp parses special tokens.
#[derive(Debug, Default, Clone)]
pub(super) struct FakeVocab {
    singles: HashMap<String, Token>,
    /// Forms that tokenize to two tokens even if they are one byte.
    pairs: Vec<String>,
    control: Vec<Vec<u8>>,
}

impl FakeVocab {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// `form` (tokenized without special tokens) is the single token `token`.
    pub(super) fn single(mut self, form: &str, token: Token) -> Self {
        self.singles.insert(form.to_string(), token);
        self
    }

    /// `form` (tokenized without special tokens) is two tokens.
    pub(super) fn pair(mut self, form: &str) -> Self {
        self.pairs.push(form.to_string());
        self
    }

    /// Each of `forms` is a single token, numbered from `first`.
    pub(super) fn singles<'a>(
        mut self,
        forms: impl IntoIterator<Item = &'a str>,
        first: Token,
    ) -> Self {
        for (offset, form) in forms.into_iter().enumerate() {
            self.singles
                .insert(form.to_string(), first + offset as Token);
        }
        self
    }

    /// These are the vocabulary's CONTROL spellings, in token-id order.
    pub(super) fn control<'a>(mut self, spellings: impl IntoIterator<Item = &'a str>) -> Self {
        self.control = spellings
            .into_iter()
            .map(|spelling| spelling.as_bytes().to_vec())
            .collect();
        self
    }

    /// Raw CONTROL spellings, for vocabularies the protocol must refuse.
    pub(super) fn control_bytes(mut self, spellings: Vec<Vec<u8>>) -> Self {
        self.control = spellings;
        self
    }

    /// d1-2B's label tokens as far as labels go: single letters and
    /// `" " + letter`, `00`..`99`, `yes`/`Yes`/`no`/`No`/`NO`, the digits,
    /// and the two-letter codes except the 89 that are not one token
    /// (`llamacpp-decision-reference.md` §6.5).
    pub(super) fn d1_labels(self) -> Self {
        let mut vocab = self;
        for c in (b'A'..=b'Z').chain(b'a'..=b'z') {
            let letter = char::from(c);
            vocab = vocab
                .single(&letter.to_string(), 100 + Token::from(c))
                .single(&format!(" {letter}"), 300 + Token::from(c));
        }
        for i in 0..100 {
            vocab = vocab.single(&format!("{i:02}"), 1500 + i);
        }
        for i in 0..10 {
            vocab = vocab.single(&i.to_string(), 24 + i);
        }
        let not_single: Vec<&str> = D1_NOT_SINGLE_TWO_LETTER.split_whitespace().collect();
        let mut next = 6000;
        for a in b'A'..=b'Z' {
            for b in b'A'..=b'Z' {
                let code: String = [char::from(a), char::from(b)].iter().collect();
                if !not_single.contains(&code.as_str()) {
                    vocab = vocab.single(&code, next);
                    next += 1;
                }
            }
        }
        // Last: `NO` is a two-letter code too, with d1-2B's token.
        for (form, token) in [
            ("yes", 11683),
            ("Yes", 12447),
            ("no", 2243),
            ("No", 4547),
            ("NO", 19598),
        ] {
            vocab = vocab.single(form, token);
        }
        vocab
    }
}

/// The two-letter codes that are not one token on d1-2B's vocabulary.
pub(super) const D1_NOT_SINGLE_TWO_LETTER: &str = "AJ BH CJ DZ EH EI EJ EO FG FN FU GJ HG JB JG KF KL KU LB LF LH LJ LN MJ PJ QD QE QF QG QI QJ QK QM QN QO QP QS QT QW QZ RJ SJ TG TN UH UO VB VJ VK VN WI WU XA XC XD XF XH XI XK XO XP XS XU XV YA YB YD YE YF YG YH YI YJ YK YL YO YU YV YZ ZB ZC ZH ZJ ZL ZM ZN ZP ZR ZV";

impl DecisionVocab for FakeVocab {
    fn tokenize(
        &self,
        text: &str,
        _add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<Token>, DecisionError> {
        if !parse_special {
            if let Some(&token) = self.singles.get(text) {
                return Ok(vec![token]);
            }
            if self.pairs.iter().any(|pair| pair == text) {
                return Ok(vec![999, 999]);
            }
        }
        let bytes = text.as_bytes();
        let mut tokens = Vec::with_capacity(bytes.len());
        let mut i = 0;
        'scan: while i < bytes.len() {
            if parse_special {
                for (index, spelling) in self.control.iter().enumerate() {
                    if bytes[i..].starts_with(spelling) {
                        tokens.push(2000 + index as Token);
                        i += spelling.len();
                        continue 'scan;
                    }
                }
            }
            tokens.push(1000 + Token::from(bytes[i]));
            i += 1;
        }
        Ok(tokens)
    }

    fn control_spellings(&self) -> &[Vec<u8>] {
        &self.control
    }

    /// One byte, a single-token form, or a control spelling: the most one
    /// of its tokens covers.
    fn max_token_bytes(&self) -> usize {
        self.singles
            .keys()
            .map(String::len)
            .chain(self.control.iter().map(Vec::len))
            .fold(1, usize::max)
    }
}

/// A GGUF header with exactly the given metadata, written and read back.
pub(super) struct GgufWriter {
    kv: Vec<(String, u32, Vec<u8>)>,
}

fn gguf_string(bytes: &[u8]) -> Vec<u8> {
    let mut out = (bytes.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(bytes);
    out
}

impl GgufWriter {
    pub(super) fn new() -> Self {
        Self { kv: Vec::new() }
    }

    pub(super) fn str(mut self, key: &str, value: &str) -> Self {
        self.kv
            .push((key.to_string(), 8, gguf_string(value.as_bytes())));
        self
    }

    pub(super) fn f32_bits(mut self, key: &str, bits: u32) -> Self {
        self.kv
            .push((key.to_string(), 6, bits.to_le_bytes().to_vec()));
        self
    }

    pub(super) fn bool(mut self, key: &str, value: bool) -> Self {
        self.kv.push((key.to_string(), 7, vec![u8::from(value)]));
        self
    }

    pub(super) fn without(mut self, key: &str) -> Self {
        self.kv.retain(|(k, _, _)| k != key);
        self
    }

    /// Sets `key` to an `f32`, replacing any earlier value in place.
    pub(super) fn set_f32(mut self, key: &str, value: f32) -> Self {
        let payload = value.to_bits().to_le_bytes().to_vec();
        if let Some(entry) = self.kv.iter_mut().find(|(k, _, _)| k == key) {
            *entry = (key.to_string(), 6, payload);
        } else {
            self.kv.push((key.to_string(), 6, payload));
        }
        self
    }

    pub(super) fn header(&self) -> GgufHeader {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensors
        bytes.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
        for (key, type_code, payload) in &self.kv {
            bytes.extend_from_slice(&gguf_string(key.as_bytes()));
            bytes.extend_from_slice(&type_code.to_le_bytes());
            bytes.extend_from_slice(payload);
        }
        GgufHeader::read(&bytes[..], &GgufLimits::default()).expect("a valid synthetic header")
    }
}

/// d1-2B's stored decision temperature (`f32` bits): 1.3373984, which reads
/// as `"1.337398"`.
pub(super) const D1_TEMPERATURE_BITS: u32 = 0x3fab2fdf;

/// The decision metadata of d1-2B, as a synthetic header.
pub(super) fn d1_header() -> GgufWriter {
    GgufWriter::new()
        .str("general.architecture", "lfm3")
        .str("lfm3.decision.type", "lfm3-d1")
        .f32_bits("lfm3.decision.temperature.noul", D1_TEMPERATURE_BITS)
        .f32_bits("lfm3.decision.temperature.choice", D1_TEMPERATURE_BITS)
        .f32_bits("lfm3.decision.temperature.score", D1_TEMPERATURE_BITS)
        .bool("lfm3.attention.causal", false)
        .str("tokenizer.chat_template.systemone", SYSTEMONE_TEMPLATE)
}

/// d1-2B's facts with `limits`.
pub(super) fn d1_facts(limits: DecisionLimits) -> DecisionModelFacts {
    DecisionModelFacts::new(DecisionProfileId::Lfm3D1Text, &d1_header().header(), limits)
        .expect("d1-2B's metadata")
}
