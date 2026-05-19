//! Shared MLX linear weight handling.
//!
//! Decoder architectures use the same projection shape at runtime:
//! `input @ weight.T`. Dense SafeTensors can run through normal `matmul`;
//! MLX 4-bit affine bundles store packed U32 weights with sibling
//! `scales`/`biases` tensors and must use `mlx_quantized_matmul`.

use safetensors::Dtype as StDtype;
use std::ffi::CString;
use xybrid_mlx::ops::{dequantize, matmul, quantized_matmul_prechecked, slice, transpose};
use xybrid_mlx::{MlxArray, MlxDtype, MlxStream};

use super::super::model::{MlxLlmError, MlxLlmResult};
use super::super::weights::SafeTensorBundle;

/// Quantization metadata from an MLX model config.
#[derive(Debug, Clone)]
pub struct LinearQuantization {
    pub group_size: i32,
    pub bits: i32,
    pub mode: String,
}

impl LinearQuantization {
    pub fn new(bits: u32, group_size: u32, mode: impl AsRef<str>) -> MlxLlmResult<Self> {
        let bits = i32::try_from(bits).map_err(|_| {
            MlxLlmError::ConfigInvalid(format!("quantization bits={bits} exceeds i32"))
        })?;
        let group_size = i32::try_from(group_size).map_err(|_| {
            MlxLlmError::ConfigInvalid(format!("quantization group_size={group_size} exceeds i32"))
        })?;
        let mode = mode.as_ref().trim();
        if mode.is_empty() {
            return Err(MlxLlmError::ConfigInvalid(
                "quantization mode must not be empty".into(),
            ));
        }
        Ok(Self {
            group_size,
            bits,
            mode: mode.to_string(),
        })
    }
}

/// Runtime representation of a projection weight.
#[derive(Debug)]
pub enum LinearWeight {
    Dense(DenseLinear),
    Quantized(QuantizedLinear),
}

/// Dense projection with both original and matmul-ready transposed weights.
#[derive(Debug)]
pub struct DenseLinear {
    pub weight: MlxArray,
    /// `None` only for non-rank-2 placeholder tensors used by load/manifest
    /// tests; real forward calls reject those shapes with a config error.
    transposed: Option<MlxArray>,
}

/// Packed affine quantized projection weight plus its scale/bias metadata.
#[derive(Debug)]
pub struct QuantizedLinear {
    pub weight: MlxArray,
    pub scales: MlxArray,
    pub biases: MlxArray,
    pub group_size: i32,
    pub bits: i32,
    pub mode: String,
    mode_cstr: CString,
}

impl LinearWeight {
    pub fn dense(weight: MlxArray) -> MlxLlmResult<Self> {
        let transposed = if weight.ndim() == 2 {
            Some(transpose(&weight, &[1, 0], None)?.contiguous_on_stream(None)?)
        } else {
            None
        };
        Ok(Self::Dense(DenseLinear { weight, transposed }))
    }

    pub fn load(
        weights: &SafeTensorBundle,
        base_weight_name: &str,
        quant: Option<&LinearQuantization>,
    ) -> MlxLlmResult<Self> {
        let info = weights.tensor_info(base_weight_name)?;
        match info.dtype {
            StDtype::U32 => {
                let quant = quant.ok_or_else(|| MlxLlmError::WeightLoad {
                    path: weights.path_for_error(),
                    reason: format!(
                        "tensor `{base_weight_name}` is quantized (U32) but config has no quantization block"
                    ),
                })?;
                Ok(Self::Quantized(load_quantized_linear(
                    weights,
                    base_weight_name,
                    quant,
                )?))
            }
            _ => Self::dense(weights.read_array(base_weight_name)?),
        }
    }

    pub fn forward(&self, input: &MlxArray, stream: Option<&MlxStream>) -> MlxLlmResult<MlxArray> {
        match self {
            Self::Dense(weight) => {
                let transposed = weight.transposed.as_ref().ok_or_else(|| {
                    MlxLlmError::ConfigInvalid(format!(
                        "dense linear weight must be rank-2 for forward, got shape {:?}",
                        weight.weight.shape()
                    ))
                })?;
                matmul(input, transposed, stream).map_err(Into::into)
            }
            Self::Quantized(weight) => quantized_matmul_prechecked(
                input,
                &weight.weight,
                &weight.scales,
                Some(&weight.biases),
                true,
                weight.group_size,
                weight.bits,
                weight.mode_cstr.as_c_str(),
                stream,
            )
            .map_err(Into::into),
        }
    }

    pub fn slice_output_rows(
        &self,
        start: usize,
        rows: usize,
        stream: Option<&MlxStream>,
    ) -> MlxLlmResult<Self> {
        match self {
            Self::Dense(weight) => {
                let sliced = slice_axis0_rows(&weight.weight, start, rows, "dense linear", stream)?;
                let sliced = sliced.contiguous_on_stream(stream)?;
                sliced.eval()?;
                Self::dense(sliced)
            }
            Self::Quantized(weight) => {
                let mode_cstr = CString::new(weight.mode.as_str()).map_err(|_| {
                    MlxLlmError::ConfigInvalid("MLX quantization mode contains NUL".into())
                })?;
                let sliced_weight =
                    slice_axis0_rows(&weight.weight, start, rows, "quantized weight", stream)?
                        .contiguous_on_stream(stream)?;
                let sliced_scales =
                    slice_axis0_rows(&weight.scales, start, rows, "quantized scales", stream)?
                        .contiguous_on_stream(stream)?;
                let sliced_biases =
                    slice_axis0_rows(&weight.biases, start, rows, "quantized biases", stream)?
                        .contiguous_on_stream(stream)?;
                sliced_weight.eval()?;
                sliced_scales.eval()?;
                sliced_biases.eval()?;
                Ok(Self::Quantized(QuantizedLinear {
                    weight: sliced_weight,
                    scales: sliced_scales,
                    biases: sliced_biases,
                    group_size: weight.group_size,
                    bits: weight.bits,
                    mode: weight.mode.clone(),
                    mode_cstr,
                }))
            }
        }
    }

    pub fn dense_ref(&self) -> Option<&MlxArray> {
        match self {
            Self::Dense(weight) => Some(&weight.weight),
            Self::Quantized(_) => None,
        }
    }
}

fn slice_axis0_rows(
    array: &MlxArray,
    start: usize,
    rows: usize,
    label: &str,
    stream: Option<&MlxStream>,
) -> MlxLlmResult<MlxArray> {
    let shape = array.shape();
    if shape.is_empty() {
        return Err(MlxLlmError::ConfigInvalid(format!(
            "{label} slice requires rank >= 1, got scalar"
        )));
    }
    let axis0 = usize::try_from(shape[0]).map_err(|_| {
        MlxLlmError::ConfigInvalid(format!("{label} axis-0 dimension {} is invalid", shape[0]))
    })?;
    let stop = start.checked_add(rows).ok_or_else(|| {
        MlxLlmError::ConfigInvalid(format!("{label} output-row slice overflows usize"))
    })?;
    if rows == 0 || stop > axis0 {
        return Err(MlxLlmError::ConfigInvalid(format!(
            "{label} output-row slice [{start}, {stop}) exceeds axis-0 length {axis0}"
        )));
    }
    let start_i32 = i32::try_from(start).map_err(|_| {
        MlxLlmError::ConfigInvalid(format!("{label} slice start {start} exceeds i32"))
    })?;
    let stop_i32 = i32::try_from(stop).map_err(|_| {
        MlxLlmError::ConfigInvalid(format!("{label} slice stop {stop} exceeds i32"))
    })?;
    let mut starts = vec![0; shape.len()];
    let mut stops = shape;
    let strides = vec![1; starts.len()];
    starts[0] = start_i32;
    stops[0] = stop_i32;
    slice(array, &starts, &stops, &strides, stream).map_err(Into::into)
}

/// Load a tensor that must be consumed by dense-only ops.
///
/// Embedding lookup and tied logits do not have a row-wise quantized MLX-C
/// operation today, so quantized tensors are dequantized once at load time.
/// This trades resident memory for correctness until a quantized gather path
/// exists.
pub fn load_dense_or_dequantized(
    weights: &SafeTensorBundle,
    weight_name: &str,
    quant: Option<&LinearQuantization>,
) -> MlxLlmResult<MlxArray> {
    let info = weights.tensor_info(weight_name)?;
    if info.dtype == StDtype::U32 {
        let quant = quant.ok_or_else(|| MlxLlmError::WeightLoad {
            path: weights.path_for_error(),
            reason: format!(
                "tensor `{weight_name}` is quantized (U32) but config has no quantization block"
            ),
        })?;
        dequantize_weight(weights, weight_name, quant)
    } else {
        weights.read_array(weight_name)
    }
}

fn load_quantized_linear(
    weights: &SafeTensorBundle,
    base_weight_name: &str,
    quant: &LinearQuantization,
) -> MlxLlmResult<QuantizedLinear> {
    let (scales_name, biases_name) = quantized_sibling_names(base_weight_name)?;
    require_tensor(weights, &scales_name)?;
    require_tensor(weights, &biases_name)?;
    let mode_cstr = CString::new(quant.mode.as_str())
        .map_err(|_| MlxLlmError::ConfigInvalid("MLX quantization mode contains NUL".into()))?;
    Ok(QuantizedLinear {
        weight: weights.read_array(base_weight_name)?,
        scales: weights.read_array(&scales_name)?,
        biases: weights.read_array(&biases_name)?,
        group_size: quant.group_size,
        bits: quant.bits,
        mode: quant.mode.clone(),
        mode_cstr,
    })
}

fn dequantize_weight(
    weights: &SafeTensorBundle,
    weight_name: &str,
    quant: &LinearQuantization,
) -> MlxLlmResult<MlxArray> {
    let (scales_name, biases_name) = quantized_sibling_names(weight_name)?;
    require_tensor(weights, &scales_name)?;
    require_tensor(weights, &biases_name)?;
    let weight = weights.read_array(weight_name)?;
    let scales = weights.read_array(&scales_name)?;
    let biases = weights.read_array(&biases_name)?;
    dequantize(
        &weight,
        &scales,
        Some(&biases),
        quant.group_size,
        quant.bits,
        &quant.mode,
        Some(MlxDtype::F32),
        None,
    )
    .map_err(Into::into)
}

fn quantized_sibling_names(base_weight_name: &str) -> MlxLlmResult<(String, String)> {
    let base = base_weight_name
        .strip_suffix(".weight")
        .ok_or_else(|| MlxLlmError::WeightLoad {
            path: std::path::PathBuf::from(base_weight_name),
            reason: format!("quantized linear tensor `{base_weight_name}` must end in `.weight`"),
        })?;
    Ok((format!("{base}.scales"), format!("{base}.biases")))
}

fn require_tensor(weights: &SafeTensorBundle, name: &str) -> MlxLlmResult<()> {
    if weights.has_tensor(name)? {
        return Ok(());
    }
    Err(MlxLlmError::WeightLoad {
        path: weights.path_for_error(),
        reason: format!("missing quantized tensor sibling `{name}`"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    #[test]
    fn quantized_linear_loads_from_safetensors_and_runs_forward() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;
        use xybrid_mlx::MlxArray;

        let tmp = tempfile::TempDir::new().unwrap();
        let packed_weight = le_u32_bytes(&[0; 16]);
        let scales = le_f32_bytes(&[1.0, 1.0]);
        let biases = le_f32_bytes(&[0.0, 0.0]);
        let tensors = vec![
            (
                "proj.weight".to_string(),
                TensorView::new(Dtype::U32, vec![2, 8], &packed_weight).unwrap(),
            ),
            (
                "proj.scales".to_string(),
                TensorView::new(Dtype::F32, vec![2, 1], &scales).unwrap(),
            ),
            (
                "proj.biases".to_string(),
                TensorView::new(Dtype::F32, vec![2, 1], &biases).unwrap(),
            ),
        ];
        let path = tmp.path().join("model.safetensors");
        safetensors::serialize_to_file(tensors, &None, &path).unwrap();

        let bundle = SafeTensorBundle::from_single_file(path);
        let quant = LinearQuantization::new(4, 64, "affine").unwrap();
        let weight = LinearWeight::load(&bundle, "proj.weight", Some(&quant)).unwrap();

        assert!(matches!(weight, LinearWeight::Quantized(_)));
        let input = MlxArray::from_slice_f32(&[0.25; 64], &[1, 64]).unwrap();
        let output = weight.forward(&input, None).unwrap();
        assert_eq!(output.shape(), vec![1, 2]);
        let values = output.to_vec_f32().unwrap();
        assert_eq!(values.len(), 2);
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    #[test]
    fn dense_linear_constructor_runs_forward() {
        use xybrid_mlx::MlxArray;

        let weight = MlxArray::from_slice_f32(
            &[
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0,
            ],
            &[2, 3],
        )
        .unwrap();
        let linear = LinearWeight::dense(weight).unwrap();
        let input = MlxArray::from_slice_f32(&[1.0, 10.0, 100.0], &[1, 3]).unwrap();

        let output = linear.forward(&input, None).unwrap();

        assert_eq!(output.shape(), vec![1, 2]);
        let values = output.to_vec_f32().unwrap();
        assert_eq!(values, vec![321.0, 654.0]);
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    #[test]
    fn dense_linear_slice_output_rows_keeps_forward_semantics() {
        use xybrid_mlx::MlxArray;

        let weight = MlxArray::from_slice_f32(
            &[
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0,
            ],
            &[3, 3],
        )
        .unwrap();
        let linear = LinearWeight::dense(weight).unwrap();
        let sliced = linear.slice_output_rows(1, 2, None).unwrap();
        let input = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();

        let got = sliced.forward(&input, None).unwrap();

        assert_eq!(got.shape(), vec![1, 2]);
        assert_eq!(got.to_vec_f32().unwrap(), vec![32.0, 50.0]);
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    fn le_u32_bytes(values: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * std::mem::size_of::<u32>());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    fn le_f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }
}
