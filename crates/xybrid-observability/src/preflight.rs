//! Bounded preflight of content values (CONTRACT.md "Content pipeline").
//!
//! [`measure`] walks any `Serialize` value without copying it and computes the
//! canonical JSON form's nesting depth and conservative preflight cost:
//!
//! ```text
//! cost(null | boolean | number) = 16
//! cost(string)                  = 16 + utf8_len(string)
//! cost(array)                   = 16 + Σ cost(element)
//! cost(object)                  = 16 + Σ (16 + utf8_len(key) + cost(member value))
//! ```
//!
//! The walk stops as soon as the depth exceeds [`MAX_JSON_DEPTH`] or the cost
//! exceeds the budget, so it never recurses deeper than 129 levels and its work
//! is bounded by the budget.

use std::fmt;

use serde::ser::{self, Serialize};

use crate::semconv::{MAX_JSON_DEPTH, PREFLIGHT_NODE_BYTES};

const NODE: u64 = PREFLIGHT_NODE_BYTES as u64;

/// Why a value failed preflight. Carries no content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreflightError {
    /// Nesting deeper than [`MAX_JSON_DEPTH`].
    Depth,
    /// Cost above the traversal budget.
    Budget,
    /// The value cannot be represented as contract JSON.
    Unrepresentable,
}

impl fmt::Display for PreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Depth => "nesting too deep",
            Self::Budget => "traversal budget exceeded",
            Self::Unrepresentable => "value not representable as JSON",
        })
    }
}

impl std::error::Error for PreflightError {}

impl ser::Error for PreflightError {
    fn custom<T: fmt::Display>(_msg: T) -> Self {
        Self::Unrepresentable
    }
}

/// Measures `value`; returns its preflight cost when it fits `budget` and the
/// depth guard.
pub(crate) fn measure<T: Serialize + ?Sized>(
    value: &T,
    budget: u64,
) -> Result<u64, PreflightError> {
    let mut meter = Meter { cost: 0, budget };
    value.serialize(Walker {
        meter: &mut meter,
        depth: 0,
    })?;
    Ok(meter.cost)
}

/// Unbounded cost of an already validated (depth-bounded) JSON value.
pub(crate) fn cost_of(value: &serde_json::Value) -> u64 {
    measure(value, u64::MAX).unwrap_or(u64::MAX)
}

struct Meter {
    cost: u64,
    budget: u64,
}

impl Meter {
    fn add(&mut self, bytes: u64) -> Result<(), PreflightError> {
        self.cost = self.cost.saturating_add(bytes);
        if self.cost > self.budget {
            return Err(PreflightError::Budget);
        }
        Ok(())
    }
}

fn len(s: &str) -> u64 {
    s.len() as u64
}

/// Serializer for one value at `depth` (number of enclosing containers).
struct Walker<'a> {
    meter: &'a mut Meter,
    depth: usize,
}

impl<'a> Walker<'a> {
    fn scalar(self, extra: u64) -> Result<(), PreflightError> {
        self.meter.add(NODE + extra)
    }

    fn open(self) -> Result<Container<'a>, PreflightError> {
        let depth = self.depth + 1;
        if depth > MAX_JSON_DEPTH {
            return Err(PreflightError::Depth);
        }
        self.meter.add(NODE)?;
        Ok(Container {
            meter: self.meter,
            depth,
        })
    }

    /// `{variant: value}` (externally tagged newtype/tuple/struct variants).
    fn open_variant(self, variant: &'static str) -> Result<Container<'a>, PreflightError> {
        let outer = self.open()?;
        outer.meter.add(NODE + len(variant))?;
        let depth = outer.depth;
        let meter = outer.meter;
        let inner = Walker { meter, depth }.open()?;
        Ok(inner)
    }
}

/// An open array or object; children are measured at `depth`.
struct Container<'a> {
    meter: &'a mut Meter,
    depth: usize,
}

impl Container<'_> {
    fn child<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), PreflightError> {
        value.serialize(Walker {
            meter: self.meter,
            depth: self.depth,
        })
    }

    fn member<T: Serialize + ?Sized>(
        &mut self,
        key: &str,
        value: &T,
    ) -> Result<(), PreflightError> {
        self.meter.add(NODE + len(key))?;
        self.child(value)
    }
}

impl<'a> ser::Serializer for Walker<'a> {
    type Ok = ();
    type Error = PreflightError;
    type SerializeSeq = Container<'a>;
    type SerializeTuple = Container<'a>;
    type SerializeTupleStruct = Container<'a>;
    type SerializeTupleVariant = Container<'a>;
    type SerializeMap = Container<'a>;
    type SerializeStruct = Container<'a>;
    type SerializeStructVariant = Container<'a>;

    fn serialize_bool(self, _: bool) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_i8(self, _: i8) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_i16(self, _: i16) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_i32(self, _: i32) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_i64(self, _: i64) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_i128(self, _: i128) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_u8(self, _: u8) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_u16(self, _: u16) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_u32(self, _: u32) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_u64(self, _: u64) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_u128(self, _: u128) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_f32(self, _: f32) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_f64(self, _: f64) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_char(self, v: char) -> Result<(), PreflightError> {
        self.scalar(v.len_utf8() as u64)
    }
    fn serialize_str(self, v: &str) -> Result<(), PreflightError> {
        self.scalar(len(v))
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<(), PreflightError> {
        // JSON serializes bytes as an array of numbers.
        let depth = self.depth + 1;
        if depth > MAX_JSON_DEPTH {
            return Err(PreflightError::Depth);
        }
        self.meter
            .add(NODE.saturating_add(NODE.saturating_mul(v.len() as u64)))
    }
    fn serialize_none(self) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), PreflightError> {
        value.serialize(self)
    }
    fn serialize_unit(self) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), PreflightError> {
        self.scalar(0)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<(), PreflightError> {
        self.scalar(len(variant))
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), PreflightError> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), PreflightError> {
        let mut outer = self.open()?;
        outer.member(variant, value)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Container<'a>, PreflightError> {
        self.open()
    }
    fn serialize_tuple(self, _: usize) -> Result<Container<'a>, PreflightError> {
        self.open()
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Container<'a>, PreflightError> {
        self.open()
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Container<'a>, PreflightError> {
        self.open_variant(variant)
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Container<'a>, PreflightError> {
        self.open()
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Container<'a>, PreflightError> {
        self.open()
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Container<'a>, PreflightError> {
        self.open_variant(variant)
    }
}

impl ser::SerializeSeq for Container<'_> {
    type Ok = ();
    type Error = PreflightError;
    fn serialize_element<T: Serialize + ?Sized>(
        &mut self,
        value: &T,
    ) -> Result<(), PreflightError> {
        self.child(value)
    }
    fn end(self) -> Result<(), PreflightError> {
        Ok(())
    }
}

impl ser::SerializeTuple for Container<'_> {
    type Ok = ();
    type Error = PreflightError;
    fn serialize_element<T: Serialize + ?Sized>(
        &mut self,
        value: &T,
    ) -> Result<(), PreflightError> {
        self.child(value)
    }
    fn end(self) -> Result<(), PreflightError> {
        Ok(())
    }
}

impl ser::SerializeTupleStruct for Container<'_> {
    type Ok = ();
    type Error = PreflightError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), PreflightError> {
        self.child(value)
    }
    fn end(self) -> Result<(), PreflightError> {
        Ok(())
    }
}

impl ser::SerializeTupleVariant for Container<'_> {
    type Ok = ();
    type Error = PreflightError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), PreflightError> {
        self.child(value)
    }
    fn end(self) -> Result<(), PreflightError> {
        Ok(())
    }
}

impl ser::SerializeMap for Container<'_> {
    type Ok = ();
    type Error = PreflightError;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), PreflightError> {
        let key_len = key.serialize(KeyLen)?;
        self.meter.add(NODE + key_len)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), PreflightError> {
        self.child(value)
    }
    fn end(self) -> Result<(), PreflightError> {
        Ok(())
    }
}

impl ser::SerializeStruct for Container<'_> {
    type Ok = ();
    type Error = PreflightError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), PreflightError> {
        self.member(key, value)
    }
    fn end(self) -> Result<(), PreflightError> {
        Ok(())
    }
}

impl ser::SerializeStructVariant for Container<'_> {
    type Ok = ();
    type Error = PreflightError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), PreflightError> {
        self.member(key, value)
    }
    fn end(self) -> Result<(), PreflightError> {
        Ok(())
    }
}

/// Measures the UTF-8 length of a JSON object key (serde_json accepts string,
/// char, integer and boolean keys; anything else is unrepresentable).
struct KeyLen;

fn digits<T: fmt::Display>(v: T) -> u64 {
    struct Count(u64);
    impl fmt::Write for Count {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            self.0 += s.len() as u64;
            Ok(())
        }
    }
    let mut c = Count(0);
    let _ = fmt::write(&mut c, format_args!("{v}"));
    c.0
}

macro_rules! key_ints {
    ($($m:ident: $t:ty),*) => {
        $(fn $m(self, v: $t) -> Result<u64, PreflightError> { Ok(digits(v)) })*
    };
}

impl ser::Serializer for KeyLen {
    type Ok = u64;
    type Error = PreflightError;
    type SerializeSeq = ser::Impossible<u64, PreflightError>;
    type SerializeTuple = ser::Impossible<u64, PreflightError>;
    type SerializeTupleStruct = ser::Impossible<u64, PreflightError>;
    type SerializeTupleVariant = ser::Impossible<u64, PreflightError>;
    type SerializeMap = ser::Impossible<u64, PreflightError>;
    type SerializeStruct = ser::Impossible<u64, PreflightError>;
    type SerializeStructVariant = ser::Impossible<u64, PreflightError>;

    key_ints!(
        serialize_i8: i8, serialize_i16: i16, serialize_i32: i32, serialize_i64: i64,
        serialize_i128: i128, serialize_u8: u8, serialize_u16: u16, serialize_u32: u32,
        serialize_u64: u64, serialize_u128: u128
    );

    fn serialize_bool(self, v: bool) -> Result<u64, PreflightError> {
        Ok(if v { 4 } else { 5 })
    }
    fn serialize_f32(self, _: f32) -> Result<u64, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_f64(self, _: f64) -> Result<u64, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_char(self, v: char) -> Result<u64, PreflightError> {
        Ok(v.len_utf8() as u64)
    }
    fn serialize_str(self, v: &str) -> Result<u64, PreflightError> {
        Ok(len(v))
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<u64, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_none(self) -> Result<u64, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, _: &T) -> Result<u64, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_unit(self) -> Result<u64, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<u64, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<u64, PreflightError> {
        Ok(len(variant))
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<u64, PreflightError> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: &T,
    ) -> Result<u64, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStruct, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, PreflightError> {
        Err(PreflightError::Unrepresentable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::{InputMessage, Part};
    use serde_json::{json, Value};

    /// The contract's reference cost over a parsed JSON value.
    fn reference_cost(v: &Value) -> u64 {
        match v {
            Value::Null | Value::Bool(_) | Value::Number(_) => 16,
            Value::String(s) => 16 + s.len() as u64,
            Value::Array(items) => 16 + items.iter().map(reference_cost).sum::<u64>(),
            Value::Object(map) => {
                16 + map
                    .iter()
                    .map(|(k, v)| 16 + k.len() as u64 + reference_cost(v))
                    .sum::<u64>()
            }
        }
    }

    fn nested(depth: usize) -> Value {
        let mut v = json!("x");
        for _ in 0..depth {
            v = Value::Array(vec![v]);
        }
        v
    }

    #[test]
    fn typed_values_cost_the_same_as_their_canonical_json() {
        let messages = vec![
            InputMessage::text("user", "héllo"),
            InputMessage {
                role: "tool".into(),
                parts: vec![Part::ToolCallResponse {
                    id: Some("c1".into()),
                    response: json!({"a": [1, null, "x"], "b": {"c": true}}),
                }],
                name: None,
            },
        ];
        let canonical = serde_json::to_value(&messages).unwrap();
        assert_eq!(
            measure(&messages, u64::MAX).unwrap(),
            reference_cost(&canonical)
        );
        assert_eq!(cost_of(&canonical), reference_cost(&canonical));
    }

    #[test]
    fn depth_guard_allows_128_and_rejects_129() {
        assert!(measure(&nested(128), u64::MAX).is_ok());
        assert_eq!(measure(&nested(129), u64::MAX), Err(PreflightError::Depth));
        // Far deeper values stop at the guard instead of recursing through.
        assert_eq!(measure(&nested(5000), u64::MAX), Err(PreflightError::Depth));
        assert!(measure(&json!([]), u64::MAX).is_ok());
    }

    #[test]
    fn budget_stops_the_walk() {
        let v = json!({"k": "x".repeat(100)});
        let cost = reference_cost(&v);
        assert_eq!(measure(&v, cost).unwrap(), cost);
        assert_eq!(measure(&v, cost - 1), Err(PreflightError::Budget));
    }
}
