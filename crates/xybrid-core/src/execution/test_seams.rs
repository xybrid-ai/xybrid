//! Test and benchmark seams for execution internals.
//!
//! These helpers are visible to in-tree tests and downstream consumers that
//! enable `dev-tools`. They keep benchmarks on the real executor path without
//! making internal preprocessing modules part of the stable SDK surface.

use ndarray::ArrayD;

use super::{preprocessing, types::PreprocessedData, ImageTensorLayout};
use crate::ir::Envelope;
use crate::runtime_adapter::AdapterError;

/// Run the real `ImageIngress` preprocessing step and return its tensor.
pub fn image_ingress_tensor(
    input: &Envelope,
    channels: usize,
    layout: ImageTensorLayout,
) -> Result<ArrayD<f32>, AdapterError> {
    let data = PreprocessedData::from_envelope(input)?;
    match preprocessing::image::image_ingress_step(data, channels, layout)? {
        PreprocessedData::Tensor(tensor) => Ok(tensor),
        other => Err(AdapterError::InvalidInput(format!(
            "ImageIngress did not produce a tensor: {other:?}"
        ))),
    }
}
