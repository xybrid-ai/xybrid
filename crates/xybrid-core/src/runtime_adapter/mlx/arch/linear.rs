//! Shared MLX linear weight handling.
//!
//! Decoder architectures use the same projection shape at runtime:
//! `input @ weight.T`. Dense SafeTensors can run through normal `matmul`;
//! MLX 4-bit affine bundles store packed U32 weights with sibling
//! `scales`/`biases` tensors and must use `mlx_quantized_matmul`.

use safetensors::Dtype as StDtype;
use xybrid_mlx::ops::{dequantize, matmul, quantized_matmul, transpose};
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
    Dense(MlxArray),
    Quantized(QuantizedLinear),
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
}

impl LinearWeight {
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
            _ => Ok(Self::Dense(weights.read_array(base_weight_name)?)),
        }
    }

    pub fn forward(&self, input: &MlxArray, stream: Option<&MlxStream>) -> MlxLlmResult<MlxArray> {
        match self {
            Self::Dense(weight) => {
                matmul(input, &transpose(weight, &[1, 0], stream)?, stream).map_err(Into::into)
            }
            Self::Quantized(weight) => quantized_matmul(
                input,
                &weight.weight,
                &weight.scales,
                Some(&weight.biases),
                true,
                weight.group_size,
                weight.bits,
                &weight.mode,
                stream,
            )
            .map_err(Into::into),
        }
    }

    pub fn dense_ref(&self) -> Option<&MlxArray> {
        match self {
            Self::Dense(weight) => Some(weight),
            Self::Quantized(_) => None,
        }
    }
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
    Ok(QuantizedLinear {
        weight: weights.read_array(base_weight_name)?,
        scales: weights.read_array(&scales_name)?,
        biases: weights.read_array(&biases_name)?,
        group_size: quant.group_size,
        bits: quant.bits,
        mode: quant.mode.clone(),
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
