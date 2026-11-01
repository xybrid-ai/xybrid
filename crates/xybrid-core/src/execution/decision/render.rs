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
//! A failure renders nothing and says nothing: minijinja's messages can
//! quote values, so every error is [`DecisionError::RenderFailed`].

use std::fmt;
use std::sync::Arc;

use minijinja::value::{Enumerator, Kwargs, Object, ObjectRepr, Value};
use minijinja::{AutoEscape, Environment, Error, ErrorKind};

use super::escape::Escaped;
use super::facts::SYSTEMONE_TEMPLATE;
use super::json::{to_json, JsonStyle};
use crate::ir::decision::DecisionError;

const TEMPLATE_NAME: &str = "systemone";

/// The most spaces per level `tojson(indent=...)` accepts; the template
/// uses 2.
const MAX_INDENT: i64 = 16;

/// The pinned template, compiled.
pub(super) struct Renderer {
    env: Environment<'static>,
}

impl Renderer {
    /// A decision-local environment holding the pinned template: no
    /// auto-escaping, and llama.cpp's `tojson`.
    pub(super) fn new() -> Result<Self, DecisionError> {
        let mut env = Environment::new();
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.add_filter("tojson", tojson);
        env.add_template(TEMPLATE_NAME, SYSTEMONE_TEMPLATE)
            .map_err(|_| DecisionError::RenderFailed)?;
        Ok(Self { env })
    }

    /// The prompt for one template input: an [`Escaped::Object`] whose
    /// members are the template's globals.
    pub(super) fn render(&self, input: Arc<Escaped>) -> Result<String, DecisionError> {
        if !matches!(*input, Escaped::Object(_)) {
            return Err(DecisionError::RenderFailed);
        }
        self.env
            .get_template(TEMPLATE_NAME)
            .and_then(|template| template.render(to_value(&input)))
            .map_err(|_| DecisionError::RenderFailed)
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
/// refuses too —, positional arguments, unknown names).
pub(super) fn tojson(value: &Value, kwargs: Kwargs) -> Result<String, Error> {
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
        return Ok(to_json(&seq.0, style));
    }
    if let Some(map) = value.downcast_object_ref::<JsonMap>() {
        return Ok(to_json(&map.0, style));
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
    Ok(to_json(&scalar, style))
}

fn unsupported(argument: &str) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!("tojson: unsupported {argument}"),
    )
}
