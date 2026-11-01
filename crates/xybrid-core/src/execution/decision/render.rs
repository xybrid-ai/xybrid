//! Rendering the pinned `systemone` template with minijinja, as llama.cpp's
//! own Jinja engine renders it.
//!
//! The environment is local to the decision protocol and built per
//! request: no other template shares it, and no crate-wide minijinja
//! feature changes for it (`preserve_order` would reorder maps that chat
//! templates build themselves). Order and number identity come from the
//! value adapter instead: every JSON array and object of the template input
//! reaches the template as an object over [`Escaped`] ([`JsonSeq`],
//! [`JsonMap`]) that enumerates in input order, and the `tojson` filter
//! writes those with llama.cpp's serializer ([`super::json`]). Scalars are
//! minijinja's own values, so `is string`, `is none` and truthiness behave
//! as llama.cpp's engine does (`common/jinja/value.h`: an empty string,
//! `0`, `0.0`, `-0.0`, `false`, `[]` and `{}` are false).
//!
//! The prompt never grows past a cap: the template writes into a sink that
//! refuses to (and `tojson` writes through one), so a prompt too long for its
//! pass is refused while it renders, not after.
//!
//! A failure renders nothing and says nothing: minijinja's messages can
//! quote values, so a failure is [`RenderError`], which carries none.

use std::fmt;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use minijinja::value::{Enumerator, Kwargs, Object, ObjectRepr, Value};
use minijinja::{AutoEscape, Environment, Error, ErrorKind};

use super::escape::Escaped;
use super::json::{to_json, JsonStyle};
use super::SYSTEMONE_TEMPLATE;

const TEMPLATE_NAME: &str = "systemone";

/// The most spaces per level `tojson(indent=...)` accepts; the template
/// uses 2.
const MAX_INDENT: i64 = 16;

/// Why a render stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RenderError {
    /// The prompt would be longer than the renderer's cap.
    TooLong,
    /// The template failed.
    Failed,
}

/// The pinned template, compiled, rendering prompts of at most `cap` bytes.
pub(super) struct Renderer {
    env: Environment<'static>,
    cap: usize,
    /// Set by `tojson` when its text would pass the cap.
    filter_too_long: Arc<AtomicBool>,
}

impl Renderer {
    /// A decision-local environment holding the pinned template: no
    /// auto-escaping, and llama.cpp's `tojson`; prompts of at most `cap`
    /// bytes.
    pub(super) fn new(cap: usize) -> Result<Self, RenderError> {
        let filter_too_long = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&filter_too_long);
        let mut env = Environment::new();
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.add_filter("tojson", move |value: &Value, kwargs: Kwargs| {
            tojson(value, kwargs, cap)?.ok_or_else(|| {
                flag.store(true, Ordering::Relaxed);
                Error::new(
                    ErrorKind::InvalidOperation,
                    "tojson: longer than the prompt's cap",
                )
            })
        });
        env.add_template(TEMPLATE_NAME, SYSTEMONE_TEMPLATE)
            .map_err(|_| RenderError::Failed)?;
        Ok(Self {
            env,
            cap,
            filter_too_long,
        })
    }

    /// The prompt for one template input, an [`Escaped::Object`] whose
    /// members are the template's globals: written into a buffer that never
    /// passes the cap.
    pub(super) fn render(&self, input: Arc<Escaped>) -> Result<String, RenderError> {
        if !matches!(*input, Escaped::Object(_)) {
            return Err(RenderError::Failed);
        }
        let template = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|_| RenderError::Failed)?;
        self.filter_too_long.store(false, Ordering::Relaxed);
        let mut out = Capped {
            bytes: Vec::new(),
            cap: self.cap,
            full: false,
        };
        match template.render_captured_to(to_value(&input), &mut out) {
            Ok(_) => String::from_utf8(out.bytes).map_err(|_| RenderError::Failed),
            Err(_) if out.full || self.filter_too_long.load(Ordering::Relaxed) => {
                Err(RenderError::TooLong)
            }
            Err(_) => Err(RenderError::Failed),
        }
    }
}

/// The rendered bytes, refusing any write that would pass `cap`.
struct Capped {
    bytes: Vec<u8>,
    cap: usize,
    full: bool,
}

impl io::Write for Capped {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.full || self.bytes.len() + buf.len() > self.cap {
            self.full = true;
            return Err(io::Error::other("the prompt is longer than its cap"));
        }
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A value of the template input as the template sees it.
pub(super) fn to_value(value: &Arc<Escaped>) -> Value {
    match &**value {
        Escaped::Null => Value::from(()),
        Escaped::Bool(value) => Value::from(*value),
        Escaped::Int(value) => Value::from(*value),
        Escaped::Float(value) => Value::from(*value),
        Escaped::Str(text) => Value::from(text.as_str()),
        Escaped::Array(_) => Value::from_object(JsonSeq(Arc::clone(value))),
        Escaped::Object(_) => Value::from_object(JsonMap(Arc::clone(value))),
    }
}

/// A JSON array of the template input.
struct JsonSeq(Arc<Escaped>);

impl JsonSeq {
    fn items(&self) -> &[Arc<Escaped>] {
        match &*self.0 {
            Escaped::Array(items) => items,
            _ => &[],
        }
    }
}

impl Object for JsonSeq {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Seq
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        self.items().get(key.as_usize()?).map(to_value)
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        Enumerator::Seq(self.items().len())
    }

    fn is_true(self: &Arc<Self>) -> bool {
        !self.items().is_empty()
    }
}

/// A JSON object of the template input: members in input order.
struct JsonMap(Arc<Escaped>);

impl JsonMap {
    fn members(&self) -> &[(String, Arc<Escaped>)] {
        match &*self.0 {
            Escaped::Object(members) => members,
            _ => &[],
        }
    }
}

impl Object for JsonMap {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Map
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        let key = key.as_str()?;
        self.members()
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| to_value(value))
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        Enumerator::Values(
            self.members()
                .iter()
                .map(|(key, _)| Value::from(key.as_str()))
                .collect(),
        )
    }

    fn is_true(self: &Arc<Self>) -> bool {
        !self.members().is_empty()
    }
}

/// Kind and size only: the values are the caller's content.
impl fmt::Debug for JsonSeq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

/// Kind and size only: the values are the caller's content.
impl fmt::Debug for JsonMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

/// llama.cpp's `tojson` (`common/jinja/value.cpp:250-277`), for the
/// arguments the template uses: none, or `indent`, a non-negative integer
/// (or `none`). `ensure_ascii` and `sort_keys` are accepted as `false`, their
/// default; anything else is refused, rather than rendered differently
/// (`separators`, `ensure_ascii=true`, `sort_keys=true` — which llama.cpp
/// refuses too —, positional arguments, unknown names). `None` when the text
/// would be longer than `cap` bytes.
pub(super) fn tojson(value: &Value, kwargs: Kwargs, cap: usize) -> Result<Option<String>, Error> {
    let indent = match kwargs.get::<Option<Value>>("indent")? {
        None => None,
        Some(indent) if indent.is_none() => None,
        Some(indent) => {
            let spaces = indent
                .is_integer()
                .then(|| indent.as_i64())
                .flatten()
                .filter(|spaces| (0..=MAX_INDENT).contains(spaces))
                .ok_or_else(|| unsupported("indent"))?;
            Some(usize::try_from(spaces).map_err(|_| unsupported("indent"))?)
        }
    };
    for flag in ["ensure_ascii", "sort_keys"] {
        if kwargs.get::<Option<bool>>(flag)? == Some(true) {
            return Err(unsupported(flag));
        }
    }
    kwargs.assert_all_used()?;

    let style = JsonStyle { indent };
    if let Some(seq) = value.downcast_object_ref::<JsonSeq>() {
        return Ok(to_json(&seq.0, style, cap));
    }
    if let Some(map) = value.downcast_object_ref::<JsonMap>() {
        return Ok(to_json(&map.0, style, cap));
    }
    let scalar = if value.is_none() || value.is_undefined() {
        Escaped::Null
    } else if let Some(text) = value.as_str() {
        Escaped::Str(text.to_string())
    } else if value.is_integer() {
        Escaped::Int(value.as_i64().ok_or_else(|| unsupported("value"))?)
    } else if value.is_number() {
        Escaped::Float(f64::try_from(value.clone()).map_err(|_| unsupported("value"))?)
    } else if let Ok(flag) = bool::try_from(value.clone()) {
        Escaped::Bool(flag)
    } else {
        return Err(unsupported("value"));
    };
    Ok(to_json(&scalar, style, cap))
}

fn unsupported(argument: &str) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!("tojson: unsupported {argument}"),
    )
}
