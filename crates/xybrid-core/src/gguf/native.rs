//! llama.cpp's string view of GGUF metadata, and the check that a loaded
//! model matches the header read from its file.
//!
//! llama.cpp keeps a model's scalar metadata as strings (`gguf_kv_to_str`)
//! and its accessors (`llama_model_meta_val_str` and friends) return those.
//! Integers are written in decimal, booleans as `true`/`false`, strings up
//! to their first NUL byte, and floats through `std::to_string`, which is
//! C's `%f`: six decimals, correctly rounded. llama.cpp's decision server
//! reads a temperature back with `strtof`, so the temperature it uses is
//! the six-decimal string's value, not the stored `f32`: d1-2B stores
//! 1.3373984098 and runs at 1.3373980522 (`"1.337398"`).

use super::error::GgufError;
use super::header::{GgufHeader, GgufValue};

/// The bytes llama.cpp's metadata accessors return for `value`, or `None`
/// for an array, which llama.cpp does not keep.
pub fn native_value(value: &GgufValue) -> Option<Vec<u8>> {
    let text = match *value {
        GgufValue::U8(v) => v.to_string(),
        GgufValue::I8(v) => v.to_string(),
        GgufValue::U16(v) => v.to_string(),
        GgufValue::I16(v) => v.to_string(),
        GgufValue::U32(v) => v.to_string(),
        GgufValue::I32(v) => v.to_string(),
        GgufValue::U64(v) => v.to_string(),
        GgufValue::I64(v) => v.to_string(),
        GgufValue::F32(v) => native_f32(v),
        GgufValue::F64(v) => c_fixed6(v),
        GgufValue::Bool(v) => (if v { "true" } else { "false" }).to_string(),
        GgufValue::String(ref bytes) => {
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            return Some(bytes[..end].to_vec());
        }
        GgufValue::Array(_) => return None,
    };
    Some(text.into_bytes())
}

/// An `f32` as llama.cpp's metadata accessors write it:
/// `std::to_string(float)`, six decimals (`1.337398`).
pub fn native_f32(value: f32) -> String {
    c_fixed6(f64::from(value))
}

/// The `f32` llama.cpp ends up with when it reads `value` back from its
/// metadata string with `strtof`: the stored value rounded to six
/// decimals. This is the effective decision temperature.
pub fn effective_f32(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    native_f32(value)
        .parse()
        .expect("C's %f output of a finite value is a decimal number")
}

/// C's `printf("%f", x)`: the exact binary value rounded to six decimals,
/// ties to even, which Rust's fixed-precision formatting also does. C
/// spells the non-finite values `nan`, `-nan`, `inf` and `-inf`.
fn c_fixed6(x: f64) -> String {
    if x.is_nan() {
        return if x.is_sign_negative() { "-nan" } else { "nan" }.to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    format!("{x:.6}")
}

/// Checks that llama.cpp loaded the model `header` describes: the same
/// scalar keys, each with the string [`native_value`] predicts. A
/// disagreement between this scanner and llama.cpp's parser fails here.
///
/// This compares llama.cpp's string projection of the scalars, so it cannot
/// see a type change that keeps the string (`f32` 1.3373984 and the string
/// `"1.337398"`) or an array-valued key: on its own it does not prove the
/// loaded file is the scanned one.
///
/// llama.cpp's metadata is read with one bulk lookup of the header's keys
/// and a count, never enumerated: its only enumeration goes by index and
/// walks its map from the start each time.
///
/// # Errors
///
/// [`GgufError::NativeMismatch`] naming the first key (in file order) whose
/// presence or value differs, [`GgufError::NativeKeyCount`] when llama.cpp
/// holds keys the header does not, and [`GgufError::Native`] when llama.cpp
/// cannot report the model's metadata.
#[cfg(feature = "llm-llamacpp")]
pub fn verify_native_metadata(
    header: &GgufHeader,
    model: &xybrid_llama::LlamaModel,
) -> Result<(), GgufError> {
    let native = |error: xybrid_llama::LlamaError| GgufError::Native(error.to_string());
    let scalars = scalar_projection(header);
    let keys: Vec<&str> = scalars.iter().map(|(key, _)| *key).collect();
    let count = model.meta_count().map_err(native)?;
    let values = model.meta_values(&keys).map_err(native)?;
    compare_native(&scalars, count, &values)
}

/// Every scalar key of `header` with the string llama.cpp keeps for it, in
/// file order.
pub(crate) fn scalar_projection(header: &GgufHeader) -> Vec<(&str, Vec<u8>)> {
    header
        .metadata()
        .filter_map(|(key, value)| native_value(value).map(|bytes| (key, bytes)))
        .collect()
}

/// The comparison behind [`verify_native_metadata`]: `values[i]` is what
/// llama.cpp holds for `scalars[i]`'s key, and `native_count` the number of
/// keys it holds. All of the header's keys found with equal values, and no
/// more keys than that, means the two sets are equal.
pub(crate) fn compare_native(
    scalars: &[(&str, Vec<u8>)],
    native_count: usize,
    values: &[Option<Vec<u8>>],
) -> Result<(), GgufError> {
    debug_assert_eq!(scalars.len(), values.len(), "one value per key");
    for ((key, expected), value) in scalars.iter().zip(values) {
        if value.as_deref() != Some(expected.as_slice()) {
            return Err(GgufError::NativeMismatch {
                key: (*key).to_string(),
            });
        }
    }
    if native_count != scalars.len() {
        return Err(GgufError::NativeKeyCount {
            native: native_count,
            header: scalars.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `std::to_string` of these `f32` bit patterns, from glibc (g++ 11,
    /// x86_64): ties round to even, huge values print every digit, and the
    /// non-finite ones have C's spellings.
    const F32_VECTORS: &[(u32, &str)] = &[
        (0x3fab2fdf, "1.337398"), // d1-2B's decision temperatures
        (0x405882cc, "3.382983"), // d1-omni-2B's
        (0x3c000000, "0.007812"), // 1/128, a tie: down to even
        (0x3cc00000, "0.023438"), // 3/128, a tie: up to even
        (0x3d200000, "0.039062"), // 5/128, a tie: down to even
        (0x3f000000, "0.500000"),
        (0x3f800000, "1.000000"),
        (0x80000000, "-0.000000"),
        (0x00000000, "0.000000"),
        (0x33d6bf95, "0.000000"), // 1e-7
        (0x350637bd, "0.000000"), // 5e-7, just below the tie
        (0x35000000, "0.000000"), // 2^-21
        (0x35800000, "0.000001"), // 2^-20
        (0x7149f2ca, "1000000015047466219876688855040.000000"),
        (0x7f7fffff, "340282346638528859811704183484516925440.000000"),
        (0x00800000, "0.000000"), // smallest normal
        (0x00000001, "0.000000"), // smallest subnormal
        (0xc0200000, "-2.500000"),
        (0x47f12065, "123456.789062"),
        (0x3dcccccd, "0.100000"),
        (0xb089705f, "-0.000000"), // -1e-9 keeps its sign
        (0x4b800000, "16777216.000000"),
        (0x40000002, "2.000000"),
        (0x3f7ffff8, "1.000000"),
        (0x3f7fffff, "1.000000"),
        (0x7f800000, "inf"),
        (0xff800000, "-inf"),
        (0x7fc00000, "nan"),
        (0xffc00000, "-nan"),
    ];

    /// The same for `f64` values (`std::to_string(double)`).
    const F64_VECTORS: &[(u64, &str)] = &[
        (0x3ff565fbebd8160c, "1.337398"),
        (0x3ea0c6f7a0b5ed8d, "0.000000"),
        (0x8000000000000000, "-0.000000"),
        (0x3e90c6f7a0b5ed8d, "0.000000"),
        (0x3fbf9ad85dfa871a, "0.123456"),
        (0x4341c37937e08000, "10000000000000000.000000"),
    ];

    #[test]
    fn floats_read_as_cs_six_decimals() {
        for &(bits, expected) in F32_VECTORS {
            assert_eq!(
                native_f32(f32::from_bits(bits)),
                expected,
                "f32 {bits:#010x}"
            );
        }
        for &(bits, expected) in F64_VECTORS {
            assert_eq!(c_fixed6(f64::from_bits(bits)), expected, "f64 {bits:#018x}");
        }
    }

    #[test]
    fn the_effective_temperature_is_the_six_decimal_string_read_back() {
        // The decision server's own number: strtof("1.337398") = 0x3fab2fdc.
        let stored = f32::from_bits(0x3fab2fdf);
        assert_eq!(effective_f32(stored).to_bits(), 0x3fab2fdc);
        assert_eq!(
            effective_f32(1e-7),
            0.0,
            "rounds to zero, which llama.cpp refuses"
        );
        assert_eq!(effective_f32(f32::from_bits(0x35800000)), 1e-6);
        assert!(effective_f32(f32::NAN).is_nan());
        assert_eq!(effective_f32(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn scalars_read_as_llama_cpp_writes_them() {
        let native = |value: GgufValue| String::from_utf8(native_value(&value).unwrap()).unwrap();
        assert_eq!(
            native(GgufValue::U8(200)),
            "200",
            "a number, not a character"
        );
        assert_eq!(native(GgufValue::I8(-5)), "-5");
        assert_eq!(native(GgufValue::U16(65535)), "65535");
        assert_eq!(native(GgufValue::I16(-32768)), "-32768");
        assert_eq!(native(GgufValue::U32(4096)), "4096");
        assert_eq!(native(GgufValue::I32(-1)), "-1");
        assert_eq!(native(GgufValue::U64(u64::MAX)), "18446744073709551615");
        assert_eq!(native(GgufValue::I64(i64::MIN)), "-9223372036854775808");
        assert_eq!(native(GgufValue::Bool(true)), "true");
        assert_eq!(native(GgufValue::Bool(false)), "false");
        assert_eq!(native(GgufValue::F64(0.5)), "0.500000");
        assert_eq!(native(GgufValue::String(b"lfm3".to_vec())), "lfm3");
        assert_eq!(
            native_value(&GgufValue::String(b"cut\0here".to_vec())).unwrap(),
            b"cut",
            "a C string ends at its first NUL"
        );
        assert_eq!(
            native_value(&GgufValue::String(vec![0xff, 0x00])).unwrap(),
            [0xff],
            "bytes, UTF-8 or not"
        );
        let array = GgufValue::Array(super::super::header::GgufArray::U32(vec![1]));
        assert_eq!(native_value(&array), None, "llama.cpp keeps no arrays");
    }
}
