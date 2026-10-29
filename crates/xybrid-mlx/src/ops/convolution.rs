//! Convolution tensor ops.
//!
//! MLX uses channel-last input layout for 1D convolution: `(N, L, C_in)`.
//! We keep that layout explicit here because LFM's short-convolution blocks
//! are trained against the MLX-LM convention, not PyTorch's `NCL` layout.

use crate::array::MlxArray;
use crate::error::{MlxError, MlxResult};
use crate::ffi;
use crate::stream::MlxStream;

/// Borrow the caller-provided stream or, if `None`, fetch a clone of the
/// thread-local default CPU stream.
enum StreamRef<'a> {
    Borrowed(&'a MlxStream),
    Owned(MlxStream),
}

impl StreamRef<'_> {
    fn as_stream(&self) -> &MlxStream {
        match self {
            StreamRef::Borrowed(s) => s,
            StreamRef::Owned(s) => s,
        }
    }
}

fn resolve_stream(opt: Option<&MlxStream>) -> MlxResult<StreamRef<'_>> {
    match opt {
        Some(s) => Ok(StreamRef::Borrowed(s)),
        None => Ok(StreamRef::Owned(MlxStream::default_cpu()?)),
    }
}

/// 1D convolution over channel-last input.
///
/// - `input`: shape `(N, L, C_in)`.
/// - `weight`: shape `(C_out, K, C_in)`.
/// - `stride`, `dilation`, and `groups` must be positive.
/// - `padding` must be non-negative and is symmetric.
///
/// The output shape is `(N, L_out, C_out)`. MLX performs cross-correlation,
/// matching common neural-network convolution layers.
pub fn conv1d(
    input: &MlxArray,
    weight: &MlxArray,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    stream: Option<&MlxStream>,
) -> MlxResult<MlxArray> {
    if input.ndim() != 3 {
        return Err(MlxError::InvalidShape {
            expected: vec![0, 0, 0],
            got: input.shape(),
        });
    }
    if weight.ndim() != 3 {
        return Err(MlxError::InvalidShape {
            expected: vec![0, 0, 0],
            got: weight.shape(),
        });
    }
    if stride <= 0 || dilation <= 0 || groups <= 0 || padding < 0 {
        return Err(MlxError::Internal(format!(
            "conv1d expects stride/dilation/groups > 0 and padding >= 0 (got stride={stride}, padding={padding}, dilation={dilation}, groups={groups})"
        )));
    }

    let s = resolve_stream(stream)?;
    // SAFETY: `input`, `weight`, and the stream are live for the duration
    // of the call; scalar parameters were range-checked above.
    let raw = unsafe {
        ffi::op_conv1d(
            input.as_raw(),
            weight.as_raw(),
            stride,
            padding,
            dilation,
            groups,
            s.as_stream().as_raw(),
        )?
    };
    Ok(MlxArray::from_raw(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: &[f32], expected: &[f32], tol: f32, ctx: &str) {
        assert_eq!(
            got.len(),
            expected.len(),
            "{ctx}: length mismatch (got {}, expected {})",
            got.len(),
            expected.len()
        );
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff <= tol,
                "{ctx}: element {i} differs: got {g}, expected {e}, diff {diff}"
            );
        }
    }

    #[test]
    fn conv1d_matches_channel_last_reference() {
        // input: N=1, L=4, C=1
        // weight: C_out=1, K=2, C_in=1
        // output[t] = input[t] * 10 + input[t + 1] * 1
        let input = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1]).unwrap();
        let weight = MlxArray::from_slice_f32(&[10.0, 1.0], &[1, 2, 1]).unwrap();
        let out = conv1d(&input, &weight, 1, 0, 1, 1, None).unwrap();

        assert_eq!(out.shape(), vec![1, 3, 1]);
        assert_close(
            &out.to_vec_f32().unwrap(),
            &[12.0, 23.0, 34.0],
            2.0e-6,
            "conv1d channel-last",
        );
    }

    #[test]
    fn conv1d_rejects_non_positive_parameters() {
        let input = MlxArray::from_slice_f32(&[1.0, 2.0], &[1, 2, 1]).unwrap();
        let weight = MlxArray::from_slice_f32(&[1.0], &[1, 1, 1]).unwrap();
        let err = conv1d(&input, &weight, 0, 0, 1, 1, None).unwrap_err();
        assert!(
            err.to_string().contains("stride/dilation/groups > 0"),
            "got {err}"
        );
    }
}
