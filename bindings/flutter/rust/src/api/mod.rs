// Flutter FFI API module
// Thin wrappers with #[frb] attributes that delegate to xybrid-sdk

pub mod envelope;
pub mod model;
pub mod result;

// Re-export all public types for convenient access
pub use envelope::FfiEnvelope;
pub use model::{FfiModel, FfiModelLoader};
pub use result::FfiResult;
