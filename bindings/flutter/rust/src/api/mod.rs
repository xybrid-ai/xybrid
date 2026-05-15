// Flutter FFI API module
// Thin wrappers with #[frb] attributes that delegate to xybrid-sdk

/// Binding identifier reported in the `X-Xybrid-Client` registry header
/// for Flutter apps. Routed through `xybrid_sdk::set_binding` at every
/// FFI entry so registry calls are attributed correctly even on
/// platforms that skip `init_sdk_cache_dir` (iOS/macOS), and on entry
/// points the host hits before `init_sdk_cache_dir` is called (the
/// push-state setters in [`device`]).
pub(crate) const FLUTTER_BINDING: &str = "flutter";

pub mod context;
pub mod device;
pub mod envelope;
pub mod model;
pub mod pipeline;
pub mod result;
pub mod sdk_client;

// Re-export all public types for convenient access
pub use context::{FfiConversationContext, FfiMessageRole};
pub use device::{FfiThermalState, XybridDevice};
pub use envelope::FfiEnvelope;
pub use model::{
    FfiGenerationConfig, FfiModel, FfiModelLoader, FfiRunOptions, FfiStreamEvent, FfiStreamToken,
};
pub use pipeline::FfiPipeline;
pub use result::FfiResult;
