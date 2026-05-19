//! Safe wrappers around MLX tensor operations.
//!
//! Each submodule groups related ops by rough topic, mirroring the PRD's
//! breakdown:
//!
//! - [`basic`] — matmul, add, mul, softmax, norms, cast, activations (US-006)
//! - [`attention`] — rope, scaled_dot_product_attention, gather, concat,
//!   reshape, transpose (US-007)
//! - [`convolution`] — channel-last conv1d for hybrid architectures
//!
//! All ops follow a common shape:
//!
//! ```text
//! fn op(args, stream: Option<&MlxStream>) -> MlxResult<MlxArray>
//! ```
//!
//! Passing `None` for the stream falls through to
//! [`crate::MlxStream::default_cpu`]. Ops that need GPU dispatch must
//! pass an explicit GPU stream; we do not default to GPU because the
//! CPU path works in every runtime context (including CI / headless
//! builds) whereas the GPU path requires a reachable Metal device.

pub mod attention;
pub mod basic;
pub mod convolution;

pub use attention::{
    concat, gather, reshape, rope, scaled_dot_product_attention, slice, slice_update, transpose,
};
pub use basic::{
    add, argmax, cast, dequantize, div, erf, exp, gelu, gelu_tanh, layer_norm, matmul, mul, neg,
    quantized_matmul, quantized_matmul_prechecked, reciprocal, rms_norm, sigmoid, silu, softmax,
    sqrt, square, sub, tanh,
};
pub use convolution::conv1d;
