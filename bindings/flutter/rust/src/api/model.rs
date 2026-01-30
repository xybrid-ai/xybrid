//! Model loading FFI wrappers for Flutter.
use flutter_rust_bridge::frb;
use std::sync::Arc;
use xybrid_sdk::{ModelLoader, XybridModel};

use super::result::FfiResult;

/// FFI wrapper for ModelLoader (preparatory step before loading).
#[frb(opaque)]
pub struct FfiModelLoader(ModelLoader);

/// FFI wrapper for a loaded XybridModel ready for inference.
#[frb(opaque)]
pub struct FfiModel(Arc<XybridModel>);

impl FfiModelLoader {
    #[frb(sync)]
    pub fn from_registry(model_id: String) -> FfiModelLoader {
        FfiModelLoader(ModelLoader::from_registry(&model_id))
    }
    #[frb(sync)]
    pub fn from_bundle(path: String) -> Result<FfiModelLoader, String> {
        ModelLoader::from_bundle(&path).map(FfiModelLoader).map_err(|e| e.to_string())
    }
    pub async fn load(&self) -> Result<FfiModel, String> {
        self.0.load_async().await.map(|m| FfiModel(Arc::new(m))).map_err(|e| e.to_string())
    }
}

impl FfiModel {
    pub fn run(&self, envelope: super::envelope::FfiEnvelope) -> Result<FfiResult, String> {
        let result = self.0.run(&envelope.into_envelope()).map_err(|e| e.to_string())?;
        Ok(FfiResult::from_inference_result(&result))
    }
}

