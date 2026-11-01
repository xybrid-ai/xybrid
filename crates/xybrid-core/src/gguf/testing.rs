//! A GGUF writer for tests: synthetic headers with exactly the keys, types
//! and tensor entries a test needs, malformed ones included.

use super::header::{GgufArray, GgufHeader, GgufValue};

/// Builds the bytes of a GGUF header.
#[derive(Clone)]
pub(super) struct GgufBuilder {
    version: u32,
    /// Each key with its type code and encoded payload, in file order.
    kv: Vec<(Vec<u8>, u32, Vec<u8>)>,
    tensors: Vec<u8>,
    n_tensors: u64,
    declared: Option<(u64, u64)>,
}

/// A GGUF string: its u64 length, then its bytes.
pub(super) fn gguf_string(bytes: &[u8]) -> Vec<u8> {
    let mut out = (bytes.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(bytes);
    out
}

impl GgufBuilder {
    pub(super) fn new() -> Self {
        Self {
            version: 3,
            kv: Vec::new(),
            tensors: Vec::new(),
            n_tensors: 0,
            declared: None,
        }
    }

    pub(super) fn version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Header counts that differ from what was added: (tensors, keys).
    pub(super) fn declare(mut self, n_tensors: u64, n_kv: u64) -> Self {
        self.declared = Some((n_tensors, n_kv));
        self
    }

    /// Drops every entry for `key`.
    pub(super) fn without(mut self, key: &str) -> Self {
        self.kv.retain(|(k, _, _)| k != key.as_bytes());
        self
    }

    /// A key with a raw type code and payload.
    pub(super) fn raw(mut self, key: &[u8], type_code: u32, payload: &[u8]) -> Self {
        self.kv.push((key.to_vec(), type_code, payload.to_vec()));
        self
    }

    pub(super) fn str(self, key: &str, value: &str) -> Self {
        self.raw(key.as_bytes(), 8, &gguf_string(value.as_bytes()))
    }

    pub(super) fn bytes_str(self, key: &str, value: &[u8]) -> Self {
        self.raw(key.as_bytes(), 8, &gguf_string(value))
    }

    pub(super) fn u8(self, key: &str, value: u8) -> Self {
        self.raw(key.as_bytes(), 0, &[value])
    }

    pub(super) fn i8(self, key: &str, value: i8) -> Self {
        self.raw(key.as_bytes(), 1, &value.to_le_bytes())
    }

    pub(super) fn u16(self, key: &str, value: u16) -> Self {
        self.raw(key.as_bytes(), 2, &value.to_le_bytes())
    }

    pub(super) fn i16(self, key: &str, value: i16) -> Self {
        self.raw(key.as_bytes(), 3, &value.to_le_bytes())
    }

    pub(super) fn u32(self, key: &str, value: u32) -> Self {
        self.raw(key.as_bytes(), 4, &value.to_le_bytes())
    }

    pub(super) fn i32(self, key: &str, value: i32) -> Self {
        self.raw(key.as_bytes(), 5, &value.to_le_bytes())
    }

    pub(super) fn f32(self, key: &str, value: f32) -> Self {
        self.raw(key.as_bytes(), 6, &value.to_le_bytes())
    }

    pub(super) fn bool(self, key: &str, value: bool) -> Self {
        self.raw(key.as_bytes(), 7, &[u8::from(value)])
    }

    pub(super) fn u64(self, key: &str, value: u64) -> Self {
        self.raw(key.as_bytes(), 10, &value.to_le_bytes())
    }

    pub(super) fn i64(self, key: &str, value: i64) -> Self {
        self.raw(key.as_bytes(), 11, &value.to_le_bytes())
    }

    pub(super) fn f64(self, key: &str, value: f64) -> Self {
        self.raw(key.as_bytes(), 12, &value.to_le_bytes())
    }

    /// An array of `elem_code` elements, `n` of them, encoded in `payload`.
    pub(super) fn array(self, key: &str, elem_code: u32, n: u64, payload: &[u8]) -> Self {
        let mut bytes = elem_code.to_le_bytes().to_vec();
        bytes.extend(n.to_le_bytes());
        bytes.extend_from_slice(payload);
        self.raw(key.as_bytes(), 9, &bytes)
    }

    pub(super) fn str_array(self, key: &str, items: &[&str]) -> Self {
        let payload: Vec<u8> = items
            .iter()
            .flat_map(|s| gguf_string(s.as_bytes()))
            .collect();
        self.array(key, 8, items.len() as u64, &payload)
    }

    pub(super) fn i32_array(self, key: &str, items: &[i32]) -> Self {
        let payload: Vec<u8> = items.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.array(key, 5, items.len() as u64, &payload)
    }

    pub(super) fn f32_array(self, key: &str, items: &[f32]) -> Self {
        let payload: Vec<u8> = items.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.array(key, 6, items.len() as u64, &payload)
    }

    /// A tensor entry with raw fields: `dims` are written as stored.
    pub(super) fn tensor(mut self, name: &str, dims: &[u64], ggml_type: u32, offset: u64) -> Self {
        self.tensors.extend(gguf_string(name.as_bytes()));
        self.tensors.extend((dims.len() as u32).to_le_bytes());
        for dim in dims {
            self.tensors.extend(dim.to_le_bytes());
        }
        self.tensors.extend(ggml_type.to_le_bytes());
        self.tensors.extend(offset.to_le_bytes());
        self.n_tensors += 1;
        self
    }

    pub(super) fn build(&self) -> Vec<u8> {
        let (n_tensors, n_kv) = self
            .declared
            .unwrap_or((self.n_tensors, self.kv.len() as u64));
        let mut out = b"GGUF".to_vec();
        out.extend(self.version.to_le_bytes());
        out.extend(n_tensors.to_le_bytes());
        out.extend(n_kv.to_le_bytes());
        for (key, type_code, payload) in &self.kv {
            out.extend(gguf_string(key));
            out.extend(type_code.to_le_bytes());
            out.extend_from_slice(payload);
        }
        out.extend_from_slice(&self.tensors);
        out
    }
}

/// `value`'s type code and payload, as a GGUF file stores them.
pub(super) fn encode_value(value: &GgufValue) -> (u32, Vec<u8>) {
    fn each<T>(items: &[T], encode: impl Fn(&T) -> Vec<u8>) -> Vec<u8> {
        items.iter().flat_map(encode).collect()
    }
    let code = value.gguf_type().code();
    let payload = match value {
        GgufValue::U8(v) => v.to_le_bytes().to_vec(),
        GgufValue::I8(v) => v.to_le_bytes().to_vec(),
        GgufValue::U16(v) => v.to_le_bytes().to_vec(),
        GgufValue::I16(v) => v.to_le_bytes().to_vec(),
        GgufValue::U32(v) => v.to_le_bytes().to_vec(),
        GgufValue::I32(v) => v.to_le_bytes().to_vec(),
        GgufValue::F32(v) => v.to_le_bytes().to_vec(),
        GgufValue::Bool(v) => vec![u8::from(*v)],
        GgufValue::String(bytes) => gguf_string(bytes),
        GgufValue::U64(v) => v.to_le_bytes().to_vec(),
        GgufValue::I64(v) => v.to_le_bytes().to_vec(),
        GgufValue::F64(v) => v.to_le_bytes().to_vec(),
        GgufValue::Array(array) => {
            let (elem, items) = match array {
                GgufArray::U8(v) => (0, each(v, |x| x.to_le_bytes().to_vec())),
                GgufArray::I8(v) => (1, each(v, |x| x.to_le_bytes().to_vec())),
                GgufArray::U16(v) => (2, each(v, |x| x.to_le_bytes().to_vec())),
                GgufArray::I16(v) => (3, each(v, |x| x.to_le_bytes().to_vec())),
                GgufArray::U32(v) => (4, each(v, |x| x.to_le_bytes().to_vec())),
                GgufArray::I32(v) => (5, each(v, |x| x.to_le_bytes().to_vec())),
                GgufArray::F32(v) => (6, each(v, |x| x.to_le_bytes().to_vec())),
                GgufArray::Bool(v) => (7, each(v, |x| vec![u8::from(*x)])),
                GgufArray::String(v) => (8, each(v, |x| gguf_string(x))),
                GgufArray::U64(v) => (10, each(v, |x| x.to_le_bytes().to_vec())),
                GgufArray::I64(v) => (11, each(v, |x| x.to_le_bytes().to_vec())),
                GgufArray::F64(v) => (12, each(v, |x| x.to_le_bytes().to_vec())),
            };
            let mut payload = (elem as u32).to_le_bytes().to_vec();
            payload.extend((array.len() as u64).to_le_bytes());
            payload.extend(items);
            payload
        }
    };
    (code, payload)
}

impl GgufBuilder {
    /// A builder holding `header`'s version, metadata and tensor entries,
    /// re-encoded, so a test can add keys to a real model's header.
    pub(super) fn from_header(header: &GgufHeader) -> Self {
        let mut builder = Self::new().version(header.version());
        for (key, value) in header.metadata() {
            let (code, payload) = encode_value(value);
            builder = builder.raw(key.as_bytes(), code, &payload);
        }
        for tensor in header.tensors() {
            builder = builder.tensor(
                tensor.name(),
                tensor.dims(),
                tensor.ggml_type(),
                tensor.offset(),
            );
        }
        builder
    }
}

/// d1-2B's decision temperature: 1.3373984098 (`f32` bits 0x3fab2fdf).
pub(super) const D1_TEMPERATURE: u32 = 0x3fab2fdf;

/// The metadata of an LFM3 d1 GGUF as LiquidAI's d1-2B carries it: the
/// keys classification reads, a few others, a token list and one tensor.
pub(super) fn d1_builder() -> GgufBuilder {
    let temperature = f32::from_bits(D1_TEMPERATURE);
    GgufBuilder::new()
        .str("general.architecture", "lfm3")
        .u32("lfm3.block_count", 24)
        .bool("lfm3.attention.causal", false)
        .str("lfm3.decision.type", "lfm3-d1")
        .f32("lfm3.decision.temperature.noul", temperature)
        .f32("lfm3.decision.temperature.choice", temperature)
        .f32("lfm3.decision.temperature.score", temperature)
        .str_array(
            "tokenizer.ggml.tokens",
            &["<|pad|>", "<|startoftext|>", "A"],
        )
        .str("tokenizer.chat_template.systemone", "{{ state }}")
        .str_array("tokenizer.chat_templates", &["systemone"])
        .tensor("token_embd.weight", &[2048, 3], 12, 0)
}
