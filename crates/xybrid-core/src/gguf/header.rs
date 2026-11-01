//! The typed GGUF header and its reader.

use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use super::error::GgufError;

/// The 4-byte magic every GGUF file starts with.
const MAGIC: [u8; 4] = *b"GGUF";
/// The newest GGUF version llama.cpp reads (`GGUF_VERSION`).
const MAX_VERSION: u32 = 3;
/// llama.cpp's limit on a tensor name, terminating NUL included
/// (`GGML_MAX_NAME`).
const MAX_TENSOR_NAME: usize = 64;
/// llama.cpp's limit on tensor dimensions (`GGML_MAX_DIMS`).
const MAX_DIMS: u32 = 4;
/// llama.cpp's limits on one string and on one array
/// (`GGUF_MAX_STRING_LENGTH`, `GGUF_MAX_ARRAY_ELEMENTS`).
const MAX_STRING_BYTES: u64 = 1 << 30;
const MAX_ARRAY_ELEMENTS: u64 = 1 << 30;
/// Elements up front for a vector whose length a file declares: a declared
/// count is not trusted with an allocation, the bytes read are.
const MAX_PREALLOCATED: u64 = 4096;

/// A GGUF value type, as the file encodes it (`enum gguf_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufType {
    /// `u8`
    U8,
    /// `i8`
    I8,
    /// `u16`
    U16,
    /// `i16`
    I16,
    /// `u32`
    U32,
    /// `i32`
    I32,
    /// `f32`
    F32,
    /// `bool`, one byte; any non-zero byte reads `true`, as in llama.cpp.
    Bool,
    /// A length-prefixed byte string.
    String,
    /// An array of one element type.
    Array,
    /// `u64`
    U64,
    /// `i64`
    I64,
    /// `f64`
    F64,
}

impl GgufType {
    /// The type with GGUF code `code`, if there is one.
    pub fn from_code(code: u32) -> Option<Self> {
        Some(match code {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }

    /// The type's GGUF code.
    pub fn code(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }

    /// llama.cpp's name for the type (`gguf_type_name`): `u8`, `str`, `arr`, ….
    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
            Self::String => "str",
            Self::Array => "arr",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F64 => "f64",
        }
    }

    /// Bytes one value of a fixed-size type takes; `None` for strings and
    /// arrays.
    fn fixed_size(self) -> Option<u64> {
        match self {
            Self::U8 | Self::I8 | Self::Bool => Some(1),
            Self::U16 | Self::I16 => Some(2),
            Self::U32 | Self::I32 | Self::F32 => Some(4),
            Self::U64 | Self::I64 | Self::F64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

impl fmt::Display for GgufType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One metadata value.
///
/// Strings are kept as the file stores them: GGUF does not promise UTF-8,
/// and llama.cpp does not check it. [`GgufValue::as_str`] reads one as text.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    /// `u8`
    U8(u8),
    /// `i8`
    I8(i8),
    /// `u16`
    U16(u16),
    /// `i16`
    I16(i16),
    /// `u32`
    U32(u32),
    /// `i32`
    I32(i32),
    /// `f32`
    F32(f32),
    /// `bool`
    Bool(bool),
    /// A string's bytes.
    String(Vec<u8>),
    /// An array.
    Array(GgufArray),
    /// `u64`
    U64(u64),
    /// `i64`
    I64(i64),
    /// `f64`
    F64(f64),
}

impl GgufValue {
    /// The value's GGUF type.
    pub fn gguf_type(&self) -> GgufType {
        match self {
            Self::U8(_) => GgufType::U8,
            Self::I8(_) => GgufType::I8,
            Self::U16(_) => GgufType::U16,
            Self::I16(_) => GgufType::I16,
            Self::U32(_) => GgufType::U32,
            Self::I32(_) => GgufType::I32,
            Self::F32(_) => GgufType::F32,
            Self::Bool(_) => GgufType::Bool,
            Self::String(_) => GgufType::String,
            Self::Array(_) => GgufType::Array,
            Self::U64(_) => GgufType::U64,
            Self::I64(_) => GgufType::I64,
            Self::F64(_) => GgufType::F64,
        }
    }

    /// The type as an error message names it: `str`, or `arr[str]` for an
    /// array of strings.
    pub fn type_name(&self) -> String {
        match self {
            Self::Array(array) => format!("arr[{}]", array.elem_type()),
            other => other.gguf_type().name().to_string(),
        }
    }

    /// A string value as text; `None` for another type or bytes that are
    /// not UTF-8.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(bytes) => std::str::from_utf8(bytes).ok(),
            _ => None,
        }
    }

    /// A string value's bytes; `None` for another type.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::String(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// A non-negative integer of any integer type; `None` for another type
    /// or a negative value.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(u64::from(v)),
            Self::U16(v) => Some(u64::from(v)),
            Self::U32(v) => Some(u64::from(v)),
            Self::U64(v) => Some(v),
            Self::I8(v) => u64::try_from(v).ok(),
            Self::I16(v) => u64::try_from(v).ok(),
            Self::I32(v) => u64::try_from(v).ok(),
            Self::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// A boolean value; `None` for another type.
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// An `f32` value; `None` for another type.
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    /// An array value; `None` for another type.
    pub fn as_array(&self) -> Option<&GgufArray> {
        match self {
            Self::Array(array) => Some(array),
            _ => None,
        }
    }
}

/// A typed array: every element has the array's element type. GGUF has no
/// arrays of arrays.
#[derive(Clone, PartialEq)]
pub enum GgufArray {
    /// `u8` elements.
    U8(Vec<u8>),
    /// `i8` elements.
    I8(Vec<i8>),
    /// `u16` elements.
    U16(Vec<u16>),
    /// `i16` elements.
    I16(Vec<i16>),
    /// `u32` elements.
    U32(Vec<u32>),
    /// `i32` elements.
    I32(Vec<i32>),
    /// `f32` elements.
    F32(Vec<f32>),
    /// `bool` elements.
    Bool(Vec<bool>),
    /// String elements, each the bytes the file stores.
    String(Vec<Vec<u8>>),
    /// `u64` elements.
    U64(Vec<u64>),
    /// `i64` elements.
    I64(Vec<i64>),
    /// `f64` elements.
    F64(Vec<f64>),
}

impl GgufArray {
    /// The element type.
    pub fn elem_type(&self) -> GgufType {
        match self {
            Self::U8(_) => GgufType::U8,
            Self::I8(_) => GgufType::I8,
            Self::U16(_) => GgufType::U16,
            Self::I16(_) => GgufType::I16,
            Self::U32(_) => GgufType::U32,
            Self::I32(_) => GgufType::I32,
            Self::F32(_) => GgufType::F32,
            Self::Bool(_) => GgufType::Bool,
            Self::String(_) => GgufType::String,
            Self::U64(_) => GgufType::U64,
            Self::I64(_) => GgufType::I64,
            Self::F64(_) => GgufType::F64,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    /// Whether the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements of a string array, as text where they are UTF-8;
    /// `None` for another element type.
    pub fn strings(&self) -> Option<impl Iterator<Item = Option<&str>>> {
        match self {
            Self::String(items) => Some(items.iter().map(|bytes| std::str::from_utf8(bytes).ok())),
            _ => None,
        }
    }
}

/// Summarized: vocabularies hold hundreds of thousands of elements.
impl fmt::Debug for GgufArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GgufArray")
            .field("elem_type", &self.elem_type())
            .field("len", &self.len())
            .finish()
    }
}

/// One tensor's entry in the header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    offset: u64,
}

impl GgufTensorInfo {
    /// The tensor's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Its dimensions, innermost first (`ne[0]` first), at most four.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Its `ggml_type` code, as stored. Not checked against the ggml types
    /// this build knows: llama.cpp checks it when it loads the tensors.
    pub fn ggml_type(&self) -> u32 {
        self.ggml_type
    }

    /// Its data offset, relative to the start of the data section.
    pub fn offset(&self) -> u64 {
        self.offset
    }
}

/// How much a header scan may read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufLimits {
    max_header_bytes: u64,
    max_metadata_entries: u64,
}

impl GgufLimits {
    /// The default ceiling: 128 MiB. Real headers are far smaller (a
    /// 262,144-token vocabulary with scores and merges is under 20 MiB);
    /// the ceiling bounds what a damaged or hostile file can make a scan
    /// read and allocate.
    pub const DEFAULT_MAX_HEADER_BYTES: u64 = 128 << 20;

    /// The default ceiling on metadata keys: 4,096. Real GGUFs carry tens
    /// (d1-2B has 45; a model card's tags, languages and a few merged base
    /// models stay near a hundred), so this is about ninety times d1-2B.
    /// Every key is later looked up in llama.cpp's own copy of the
    /// metadata, so the ceiling bounds that work too, and it is checked
    /// before any key is read.
    pub const DEFAULT_MAX_METADATA_ENTRIES: u64 = 4096;

    /// A scan that reads at most `max_header_bytes` bytes of header and the
    /// default number of metadata keys.
    pub const fn new(max_header_bytes: u64) -> Self {
        Self {
            max_header_bytes,
            max_metadata_entries: Self::DEFAULT_MAX_METADATA_ENTRIES,
        }
    }

    /// These limits with at most `max_metadata_entries` metadata keys.
    pub const fn with_max_metadata_entries(mut self, max_metadata_entries: u64) -> Self {
        self.max_metadata_entries = max_metadata_entries;
        self
    }

    /// The most header bytes a scan reads.
    pub const fn max_header_bytes(&self) -> u64 {
        self.max_header_bytes
    }

    /// The most metadata keys a header may declare.
    pub const fn max_metadata_entries(&self) -> u64 {
        self.max_metadata_entries
    }
}

impl Default for GgufLimits {
    fn default() -> Self {
        Self::new(Self::DEFAULT_MAX_HEADER_BYTES)
    }
}

/// A complete, typed read of a GGUF header: the version, every metadata key
/// in file order with its type and value (arrays typed, element by
/// element), and every tensor's entry.
///
/// A header only exists for a scan that read all of it: a truncated,
/// malformed or oversized header is an error, never a partial result.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufHeader {
    version: u32,
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<GgufTensorInfo>,
    header_bytes: u64,
}

impl GgufHeader {
    /// Read the header of the GGUF file at `path`, with the default
    /// [`GgufLimits`].
    ///
    /// # Errors
    ///
    /// See [`Self::read`]; also [`GgufError::Io`] when the file cannot be
    /// opened.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        Self::read_file_with_limits(path, &GgufLimits::default())
    }

    /// [`Self::read_file`] with explicit limits. The file's size bounds
    /// the scan too: a length that runs past the end of the file is
    /// malformed, as it is for llama.cpp.
    ///
    /// # Errors
    ///
    /// See [`Self::read`].
    pub fn read_file_with_limits(
        path: impl AsRef<Path>,
        limits: &GgufLimits,
    ) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        Self::read_bounded(BufReader::new(file), limits, Some(file_len))
    }

    /// Read a GGUF header from `reader`, which must be at the start of the
    /// file. Reads exactly the header (up to the end of the tensor entries)
    /// and nothing of the tensor data.
    ///
    /// # Errors
    ///
    /// [`GgufError::Io`] for a read failure, [`GgufError::Truncated`] when
    /// the input ends inside the header, [`GgufError::LimitExceeded`] when
    /// the header is larger than `limits` allow,
    /// [`GgufError::TooManyEntries`] when it declares more metadata keys than
    /// they allow, and [`GgufError::NotGguf`],
    /// [`GgufError::UnsupportedVersion`] or [`GgufError::Malformed`] for a
    /// header llama.cpp would refuse: a bad magic or version, an empty or
    /// duplicate key, an unknown type or an array of arrays, a misplaced
    /// `general.alignment`, or a bad tensor entry. Keys and tensor names
    /// must also be UTF-8.
    pub fn read(reader: impl Read, limits: &GgufLimits) -> Result<Self, GgufError> {
        Self::read_bounded(reader, limits, None)
    }

    fn read_bounded(
        reader: impl Read,
        limits: &GgufLimits,
        file_len: Option<u64>,
    ) -> Result<Self, GgufError> {
        let mut r = Reader {
            inner: reader,
            consumed: 0,
            limit: limits.max_header_bytes,
            file_len,
        };

        if r.array::<4>("the magic")? != MAGIC {
            return Err(GgufError::NotGguf);
        }
        let version = r.u32("the version")?;
        // llama.cpp's checks, in its order: 0 is invalid, a version with
        // zero low 16 bits is a big-endian file read on a little-endian host,
        // v1 is no longer supported, and v4+ is not known yet.
        if version == 0 || version & 0xFFFF == 0 || version == 1 || version > MAX_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let n_tensors = r.count("the tensor count")?;
        let n_kv = r.count("the metadata count")?;
        if n_kv > limits.max_metadata_entries {
            return Err(GgufError::TooManyEntries {
                count: n_kv,
                limit: limits.max_metadata_entries,
            });
        }

        let mut metadata = Vec::with_capacity(preallocation(n_kv));
        let mut seen = HashSet::new();
        for index in 0..n_kv {
            let key = r.string("a metadata key")?;
            let key = String::from_utf8(key)
                .map_err(|_| GgufError::Malformed(format!("metadata key {index} is not UTF-8")))?;
            if key.is_empty() {
                return Err(GgufError::Malformed(format!(
                    "metadata key {index} is empty"
                )));
            }
            if !seen.insert(key.clone()) {
                return Err(GgufError::Malformed(format!(
                    "duplicate metadata key {key:?}"
                )));
            }
            let value = r.value(&key)?;
            metadata.push((key, value));
        }

        let alignment = metadata.iter().find(|(key, _)| key == "general.alignment");
        if let Some((_, value)) = alignment {
            match value {
                GgufValue::U32(n) if n.is_power_of_two() => {}
                GgufValue::U32(n) => {
                    return Err(GgufError::Malformed(format!(
                        "general.alignment {n} is not a power of 2"
                    )))
                }
                other => {
                    return Err(GgufError::WrongType {
                        key: "general.alignment".to_string(),
                        expected: "u32",
                        found: other.type_name(),
                    })
                }
            }
        }

        let mut tensors = Vec::with_capacity(preallocation(n_tensors));
        let mut names = HashSet::new();
        for index in 0..n_tensors {
            let info = r.tensor_info(index)?;
            if !names.insert(info.name.clone()) {
                return Err(GgufError::Malformed(format!(
                    "duplicate tensor name {:?}",
                    info.name
                )));
            }
            tensors.push(info);
        }

        Ok(Self {
            version,
            metadata,
            tensors,
            header_bytes: r.consumed,
        })
    }

    /// The GGUF version (2 or 3).
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Every metadata key with its value, in file order.
    pub fn metadata(&self) -> impl ExactSizeIterator<Item = (&str, &GgufValue)> {
        self.metadata
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }

    /// The value of `key`, if the header has it.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, value)| value)
    }

    /// Every tensor's entry, in file order.
    pub fn tensors(&self) -> &[GgufTensorInfo] {
        &self.tensors
    }

    /// Bytes the header takes, from the magic to the end of the tensor
    /// entries (the data section starts at the next alignment boundary).
    pub fn header_bytes(&self) -> u64 {
        self.header_bytes
    }
}

/// Initial capacity for `declared` elements: grown as they are read.
fn preallocation(declared: u64) -> usize {
    usize::try_from(declared.min(MAX_PREALLOCATED)).unwrap_or(0)
}

/// A reader that counts what it consumes against the scan limit and, for a
/// file, the bytes the file has left.
struct Reader<R> {
    inner: R,
    consumed: u64,
    limit: u64,
    file_len: Option<u64>,
}

impl<R: Read> Reader<R> {
    /// Refuses to read `len` more bytes past the file's end or the limit.
    fn reserve(&self, len: u64, what: &str) -> Result<(), GgufError> {
        let end = self.consumed.checked_add(len);
        if let Some(file_len) = self.file_len {
            if end.is_none_or(|end| end > file_len) {
                return Err(GgufError::Truncated {
                    at: what.to_string(),
                });
            }
        }
        if end.is_none_or(|end| end > self.limit) {
            return Err(GgufError::LimitExceeded { limit: self.limit });
        }
        Ok(())
    }

    fn array<const N: usize>(&mut self, what: &str) -> Result<[u8; N], GgufError> {
        self.reserve(N as u64, what)?;
        let mut buf = [0u8; N];
        self.inner
            .read_exact(&mut buf)
            .map_err(|e| eof_as_truncated(e, what))?;
        self.consumed += N as u64;
        Ok(buf)
    }

    /// `len` bytes, allocated as they arrive rather than up front.
    fn bytes(&mut self, len: u64, what: &str) -> Result<Vec<u8>, GgufError> {
        self.reserve(len, what)?;
        let mut buf = Vec::with_capacity(preallocation(len));
        let read = (&mut self.inner).take(len).read_to_end(&mut buf)?;
        if read as u64 != len {
            return Err(GgufError::Truncated {
                at: what.to_string(),
            });
        }
        self.consumed += len;
        Ok(buf)
    }

    fn u32(&mut self, what: &str) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.array(what)?))
    }

    fn u64(&mut self, what: &str) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.array(what)?))
    }

    /// A count llama.cpp reads as `int64_t` and refuses when negative.
    fn count(&mut self, what: &str) -> Result<u64, GgufError> {
        let n = self.u64(what)?;
        if i64::try_from(n).is_err() {
            return Err(GgufError::Malformed(format!("{what} {n} is negative")));
        }
        Ok(n)
    }

    fn string(&mut self, what: &str) -> Result<Vec<u8>, GgufError> {
        let len = self.u64(what)?;
        if len > MAX_STRING_BYTES {
            return Err(GgufError::Malformed(format!(
                "{what} is {len} bytes; llama.cpp allows {MAX_STRING_BYTES}"
            )));
        }
        self.bytes(len, what)
    }

    fn type_code(&mut self, key: &str) -> Result<GgufType, GgufError> {
        let code = self.u32("a metadata type")?;
        GgufType::from_code(code).ok_or_else(|| {
            GgufError::Malformed(format!("metadata key {key:?} has unknown GGUF type {code}"))
        })
    }

    fn value(&mut self, key: &str) -> Result<GgufValue, GgufError> {
        let ty = self.type_code(key)?;
        if ty != GgufType::Array {
            return self.scalar(ty, key);
        }
        let elem = self.type_code(key)?;
        let n = self.u64("an array length")?;
        if n > MAX_ARRAY_ELEMENTS {
            return Err(GgufError::Malformed(format!(
                "array {key:?} has {n} elements; llama.cpp allows {MAX_ARRAY_ELEMENTS}"
            )));
        }
        let array = match elem.fixed_size() {
            Some(size) => {
                let len = n
                    .checked_mul(size)
                    .ok_or_else(|| GgufError::Malformed(format!("array {key:?} is too long")))?;
                fixed_array(elem, &self.bytes(len, "an array")?)
            }
            None if elem == GgufType::String => {
                // Each element takes at least its 8-byte length.
                let min = n
                    .checked_mul(8)
                    .ok_or_else(|| GgufError::Malformed(format!("array {key:?} is too long")))?;
                self.reserve(min, "an array")?;
                let mut items = Vec::with_capacity(preallocation(n));
                for _ in 0..n {
                    items.push(self.string("an array element")?);
                }
                GgufArray::String(items)
            }
            None => {
                return Err(GgufError::Malformed(format!(
                    "metadata key {key:?} is an array of arrays"
                )))
            }
        };
        Ok(GgufValue::Array(array))
    }

    fn scalar(&mut self, ty: GgufType, what: &str) -> Result<GgufValue, GgufError> {
        Ok(match ty {
            GgufType::U8 => GgufValue::U8(u8::from_le_bytes(self.array(what)?)),
            GgufType::I8 => GgufValue::I8(i8::from_le_bytes(self.array(what)?)),
            GgufType::U16 => GgufValue::U16(u16::from_le_bytes(self.array(what)?)),
            GgufType::I16 => GgufValue::I16(i16::from_le_bytes(self.array(what)?)),
            GgufType::U32 => GgufValue::U32(u32::from_le_bytes(self.array(what)?)),
            GgufType::I32 => GgufValue::I32(i32::from_le_bytes(self.array(what)?)),
            GgufType::F32 => GgufValue::F32(f32::from_le_bytes(self.array(what)?)),
            GgufType::Bool => GgufValue::Bool(self.array::<1>(what)?[0] != 0),
            GgufType::String => GgufValue::String(self.string(what)?),
            GgufType::U64 => GgufValue::U64(u64::from_le_bytes(self.array(what)?)),
            GgufType::I64 => GgufValue::I64(i64::from_le_bytes(self.array(what)?)),
            GgufType::F64 => GgufValue::F64(f64::from_le_bytes(self.array(what)?)),
            GgufType::Array => unreachable!("arrays are read by `value`"),
        })
    }

    fn tensor_info(&mut self, index: u64) -> Result<GgufTensorInfo, GgufError> {
        let name = self.string("a tensor name")?;
        let name = String::from_utf8(name)
            .map_err(|_| GgufError::Malformed(format!("tensor name {index} is not UTF-8")))?;
        if name.len() >= MAX_TENSOR_NAME {
            return Err(GgufError::Malformed(format!(
                "tensor name {index} is {} bytes; llama.cpp allows {}",
                name.len(),
                MAX_TENSOR_NAME - 1
            )));
        }
        let n_dims = self.u32("a tensor's dimension count")?;
        if n_dims > MAX_DIMS {
            return Err(GgufError::Malformed(format!(
                "tensor {name:?} has {n_dims} dimensions; llama.cpp allows {MAX_DIMS}"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        let mut elements: i64 = 1;
        for _ in 0..n_dims {
            let dim = self.u64("a tensor dimension")?;
            // llama.cpp reads dimensions as int64 and refuses negative ones.
            let signed = i64::try_from(dim).map_err(|_| {
                GgufError::Malformed(format!("tensor {name:?} has a negative dimension"))
            })?;
            elements = elements.checked_mul(signed).ok_or_else(|| {
                GgufError::Malformed(format!("tensor {name:?} has too many elements"))
            })?;
            dims.push(dim);
        }
        let ggml_type = self.u32("a tensor type")?;
        let offset = self.u64("a tensor offset")?;
        Ok(GgufTensorInfo {
            name,
            dims,
            ggml_type,
            offset,
        })
    }
}

/// An array of a fixed-size element type from its little-endian bytes.
fn fixed_array(elem: GgufType, bytes: &[u8]) -> GgufArray {
    fn each<const N: usize, T>(bytes: &[u8], from: fn([u8; N]) -> T) -> Vec<T> {
        // The reader read exactly `len * N` bytes, so nothing is left over.
        bytes
            .as_chunks::<N>()
            .0
            .iter()
            .map(|&chunk| from(chunk))
            .collect()
    }
    match elem {
        GgufType::U8 => GgufArray::U8(bytes.to_vec()),
        GgufType::I8 => GgufArray::I8(each(bytes, i8::from_le_bytes)),
        GgufType::U16 => GgufArray::U16(each(bytes, u16::from_le_bytes)),
        GgufType::I16 => GgufArray::I16(each(bytes, i16::from_le_bytes)),
        GgufType::U32 => GgufArray::U32(each(bytes, u32::from_le_bytes)),
        GgufType::I32 => GgufArray::I32(each(bytes, i32::from_le_bytes)),
        GgufType::F32 => GgufArray::F32(each(bytes, f32::from_le_bytes)),
        GgufType::Bool => GgufArray::Bool(bytes.iter().map(|&b| b != 0).collect()),
        GgufType::U64 => GgufArray::U64(each(bytes, u64::from_le_bytes)),
        GgufType::I64 => GgufArray::I64(each(bytes, i64::from_le_bytes)),
        GgufType::F64 => GgufArray::F64(each(bytes, f64::from_le_bytes)),
        GgufType::String | GgufType::Array => unreachable!("not a fixed-size type"),
    }
}

/// An end of input inside the header is a truncation, not an I/O failure.
fn eof_as_truncated(error: std::io::Error, what: &str) -> GgufError {
    if error.kind() == std::io::ErrorKind::UnexpectedEof {
        GgufError::Truncated {
            at: what.to_string(),
        }
    } else {
        GgufError::Io(error)
    }
}
