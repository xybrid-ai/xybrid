//! Native Core ML implementation for Apple platforms.

use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::{
    AdapterError, AdapterResult, ModelMetadata, RuntimeAdapter, RuntimeAdapterExt,
};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureType, MLFeatureValue,
    MLModel, MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

/// An Objective-C model confined to the Core ML worker thread.
struct WorkerModel {
    model: Retained<MLModel>,
    input_name: String,
    input_shape: Vec<u64>,
    input_data_type: MLMultiArrayDataType,
    output_name: String,
}

enum Command {
    Load {
        path: PathBuf,
        model_id: String,
        reply: Sender<AdapterResult<ModelMetadata>>,
    },
    Infer {
        model_id: String,
        values: Vec<f32>,
        reply: Sender<AdapterResult<Vec<f32>>>,
    },
    Unload {
        model_id: String,
        reply: Sender<AdapterResult<()>>,
    },
    Shutdown,
}

/// Native Core ML adapter for single-input, single-output tensor models.
///
/// The first implementation deliberately supports one narrow, composable
/// model class: an [`EnvelopeKind::Embedding`] is mapped to a model's sole
/// `MLMultiArray` input, and its sole `MLMultiArray` output is returned as an
/// embedding. Native `.mlmodel` and `.mlpackage` assets are compiled for the
/// current device before loading; precompiled `.mlmodelc` bundles are loaded
/// directly.
///
/// Core ML objects are confined to a dedicated worker thread because the
/// generated Objective-C bindings do not claim `Send` or `Sync`. Only owned
/// Rust paths, vectors, metadata, and errors cross the channel boundary.
pub struct CoreMLRuntimeAdapter {
    commands: Sender<Command>,
    worker: Option<JoinHandle<()>>,
    worker_start_error: Option<String>,
    models: HashMap<String, ModelMetadata>,
    current_model: Option<String>,
    metal_available: bool,
}

impl CoreMLRuntimeAdapter {
    /// Creates an empty native Core ML adapter.
    pub fn new() -> Self {
        let (commands, receiver) = mpsc::channel();
        let (worker, worker_start_error) = match std::thread::Builder::new()
            .name("xybrid-coreml".to_string())
            .spawn(move || worker_loop(receiver))
        {
            Ok(worker) => (Some(worker), None),
            Err(error) => (None, Some(error.to_string())),
        };
        Self {
            commands,
            worker,
            worker_start_error,
            models: HashMap::new(),
            current_model: None,
            metal_available: Self::detect_metal_availability(),
        }
    }

    fn detect_metal_availability() -> bool {
        // `MTLCreateSystemDefaultDevice` has no preconditions and returns an
        // owned optional device. Dropping it releases this one-shot probe.
        objc2_metal::MTLCreateSystemDefaultDevice().is_some()
    }

    /// Returns whether this device exposes a default Metal device.
    pub fn has_metal(&self) -> bool {
        self.metal_available
    }

    fn validate_model_file(model_path: &str) -> AdapterResult<PathBuf> {
        let path = Path::new(model_path);
        if !path.exists() {
            return Err(AdapterError::ModelNotFound(format!(
                "Model file or directory not found: {model_path}"
            )));
        }

        let extension = path.extension().and_then(|value| value.to_str());
        if !matches!(extension, Some("mlmodel" | "mlpackage" | "mlmodelc")) {
            return Err(AdapterError::InvalidInput(format!(
                "Unsupported Core ML model path '{model_path}'; expected .mlmodel, .mlpackage, or .mlmodelc"
            )));
        }
        Ok(path.to_path_buf())
    }

    fn extract_model_id(path: &Path) -> String {
        path.file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("unknown")
            .to_string()
    }

    fn disconnected(&self, context: &str) -> AdapterError {
        let detail = self
            .worker_start_error
            .as_deref()
            .map(|error| format!(" ({error})"))
            .unwrap_or_default();
        AdapterError::RuntimeError(format!(
            "Core ML worker disconnected while {context}{detail}"
        ))
    }

    fn infer_values(&self, model_id: &str, input: &Envelope) -> AdapterResult<Envelope> {
        if !self.models.contains_key(model_id) {
            return Err(AdapterError::ModelNotLoaded(format!(
                "Model '{model_id}' is not loaded"
            )));
        }
        let values = match &input.kind {
            EnvelopeKind::Embedding(values) => values.clone(),
            other => {
                return Err(AdapterError::InvalidInput(format!(
                    "Native Core ML MVP accepts Embedding input, received {}",
                    other.as_str()
                )))
            }
        };

        let (reply, response) = mpsc::channel();
        self.commands
            .send(Command::Infer {
                model_id: model_id.to_string(),
                values,
                reply,
            })
            .map_err(|_| self.disconnected("starting inference"))?;
        let values = response
            .recv()
            .map_err(|_| self.disconnected("waiting for inference"))??;
        Ok(Envelope::new(EnvelopeKind::Embedding(values)))
    }
}

impl Default for CoreMLRuntimeAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CoreMLRuntimeAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CoreMLRuntimeAdapter")
            .field("models", &self.models.keys().collect::<Vec<_>>())
            .field("current_model", &self.current_model)
            .field("metal_available", &self.metal_available)
            .field("worker_start_error", &self.worker_start_error)
            .finish_non_exhaustive()
    }
}

impl Drop for CoreMLRuntimeAdapter {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl RuntimeAdapter for CoreMLRuntimeAdapter {
    fn name(&self) -> &str {
        "coreml"
    }

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["mlpackage", "mlmodel", "mlmodelc"]
    }

    fn load_model(&mut self, path: &str) -> AdapterResult<()> {
        let path = Self::validate_model_file(path)?;
        let model_id = Self::extract_model_id(&path);
        if self.models.contains_key(&model_id) {
            log::warn!("Model '{model_id}' is already loaded, skipping reload");
            self.current_model = Some(model_id);
            return Ok(());
        }

        let (reply, response) = mpsc::channel();
        self.commands
            .send(Command::Load {
                path,
                model_id: model_id.clone(),
                reply,
            })
            .map_err(|_| self.disconnected("loading a model"))?;
        let metadata = response
            .recv()
            .map_err(|_| self.disconnected("waiting for model loading"))??;
        self.models.insert(model_id.clone(), metadata);
        self.current_model = Some(model_id);
        Ok(())
    }

    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope> {
        let model_id = self.current_model.as_ref().ok_or_else(|| {
            AdapterError::ModelNotLoaded("No model loaded. Call load_model() first.".to_string())
        })?;
        self.infer_values(model_id, input)
    }
}

impl RuntimeAdapterExt for CoreMLRuntimeAdapter {
    fn is_loaded(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    fn get_metadata(&self, model_id: &str) -> AdapterResult<&ModelMetadata> {
        self.models.get(model_id).ok_or_else(|| {
            AdapterError::ModelNotLoaded(format!("Model '{model_id}' is not loaded"))
        })
    }

    fn infer(&self, model_id: &str, input: &Envelope) -> AdapterResult<Envelope> {
        self.infer_values(model_id, input)
    }

    fn unload_model(&mut self, model_id: &str) -> AdapterResult<()> {
        if !self.models.contains_key(model_id) {
            return Err(AdapterError::ModelNotLoaded(format!(
                "Model '{model_id}' is not loaded"
            )));
        }
        let (reply, response) = mpsc::channel();
        self.commands
            .send(Command::Unload {
                model_id: model_id.to_string(),
                reply,
            })
            .map_err(|_| self.disconnected("unloading a model"))?;
        response
            .recv()
            .map_err(|_| self.disconnected("waiting for model unloading"))??;
        self.models.remove(model_id);
        if self.current_model.as_deref() == Some(model_id) {
            self.current_model = None;
        }
        Ok(())
    }

    fn list_loaded_models(&self) -> Vec<String> {
        let mut model_ids = self.models.keys().cloned().collect::<Vec<_>>();
        model_ids.sort();
        model_ids
    }
}

fn worker_loop(receiver: Receiver<Command>) {
    let mut models = HashMap::<String, WorkerModel>::new();
    while let Ok(command) = receiver.recv() {
        match command {
            Command::Load {
                path,
                model_id,
                reply,
            } => {
                let result = autoreleasepool(|_| load_native_model(&path, &model_id));
                match result {
                    Ok((model, metadata)) => {
                        models.insert(model_id, model);
                        let _ = reply.send(Ok(metadata));
                    }
                    Err(error) => {
                        let _ = reply.send(Err(error));
                    }
                }
            }
            Command::Infer {
                model_id,
                values,
                reply,
            } => {
                let result = autoreleasepool(|_| {
                    let model = models.get(&model_id).ok_or_else(|| {
                        AdapterError::ModelNotLoaded(format!("Model '{model_id}' is not loaded"))
                    })?;
                    infer_native(model, &values)
                });
                let _ = reply.send(result);
            }
            Command::Unload { model_id, reply } => {
                let result = if models.remove(&model_id).is_some() {
                    Ok(())
                } else {
                    Err(AdapterError::ModelNotLoaded(format!(
                        "Model '{model_id}' is not loaded"
                    )))
                };
                let _ = reply.send(result);
            }
            Command::Shutdown => break,
        }
    }
}

fn ns_error(context: &str, error: &NSError) -> String {
    format!("{context}: {}", error.localizedDescription())
}

#[expect(
    deprecated,
    reason = "RuntimeAdapter loading is synchronous and Core ML offers no nondeprecated synchronous compiler"
)]
fn model_url(path: &Path) -> AdapterResult<Retained<NSURL>> {
    let path_string = path.to_string_lossy();
    let source_url = NSURL::fileURLWithPath(&NSString::from_str(&path_string));
    if path.extension().and_then(|value| value.to_str()) == Some("mlmodelc") {
        return Ok(source_url);
    }

    // Core ML's synchronous compiler is deprecated in favour of its async
    // counterpart, but `RuntimeAdapter::load_model` is synchronous. This call
    // preserves that trait contract and runs only on the worker during loading.
    unsafe {
        MLModel::compileModelAtURL_error(&source_url).map_err(|error| {
            AdapterError::RuntimeError(ns_error("Failed to compile Core ML model", &error))
        })
    }
}

fn single_multi_array_feature(
    model: &MLModel,
    input: bool,
) -> AdapterResult<(String, Vec<u64>, MLMultiArrayDataType)> {
    // SAFETY: All returned Core ML description objects are retained by the
    // generated bindings. We do not mutate the model or dictionaries.
    let descriptions = unsafe {
        let description = model.modelDescription();
        if input {
            description.inputDescriptionsByName()
        } else {
            description.outputDescriptionsByName()
        }
    };
    let keys = descriptions.allKeys();
    if keys.len() != 1 {
        let direction = if input { "input" } else { "output" };
        return Err(AdapterError::InvalidInput(format!(
            "Native Core ML MVP requires exactly one model {direction}; found {}",
            keys.len()
        )));
    }

    let key = keys.objectAtIndex(0);
    let feature = descriptions.objectForKey(&key).ok_or_else(|| {
        AdapterError::RuntimeError("Core ML model description lost its feature".to_string())
    })?;
    // SAFETY: These are immutable accessors on a retained description.
    if unsafe { feature.r#type() } != MLFeatureType::MultiArray {
        let direction = if input { "input" } else { "output" };
        return Err(AdapterError::InvalidInput(format!(
            "Native Core ML MVP requires an MLMultiArray {direction}; '{}' has a different feature type",
            key
        )));
    }
    // SAFETY: The type check above guarantees a multi-array constraint.
    let constraint = unsafe { feature.multiArrayConstraint() }.ok_or_else(|| {
        AdapterError::RuntimeError(format!(
            "Core ML feature '{}' has no multi-array constraint",
            key
        ))
    })?;
    let shape = unsafe { constraint.shape() }
        .to_vec()
        .into_iter()
        .map(|dimension| dimension.as_u64())
        .collect::<Vec<_>>();
    let data_type = unsafe { constraint.dataType() };
    Ok((key.to_string(), shape, data_type))
}

fn load_native_model(path: &Path, model_id: &str) -> AdapterResult<(WorkerModel, ModelMetadata)> {
    let url = model_url(path)?;
    // SAFETY: `new` allocates and initializes an Objective-C configuration.
    let configuration = unsafe { MLModelConfiguration::new() };
    // `All` lets Core ML choose between CPU, GPU, and Neural Engine based on
    // model compatibility and the current Apple device.
    unsafe { configuration.setComputeUnits(MLComputeUnits::All) };
    // SAFETY: URL and configuration are valid retained Foundation objects.
    let model =
        unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &configuration) }
            .map_err(|error| {
                AdapterError::RuntimeError(ns_error(
                    "Failed to load compiled Core ML model",
                    &error,
                ))
            })?;

    let (input_name, input_shape, input_data_type) = single_multi_array_feature(&model, true)?;
    let (output_name, output_shape, _) = single_multi_array_feature(&model, false)?;
    let metadata = ModelMetadata {
        model_id: model_id.to_string(),
        version: "1.0.0".to_string(),
        runtime_type: "coreml".to_string(),
        model_path: path.to_string_lossy().into_owned(),
        input_schema: HashMap::from([(input_name.clone(), input_shape.clone())]),
        output_schema: HashMap::from([(output_name.clone(), output_shape)]),
    };
    Ok((
        WorkerModel {
            model,
            input_name,
            input_shape,
            input_data_type,
            output_name,
        },
        metadata,
    ))
}

fn expected_element_count(shape: &[u64]) -> AdapterResult<usize> {
    let count = shape.iter().try_fold(1_u64, |total, dimension| {
        total.checked_mul(*dimension).ok_or_else(|| {
            AdapterError::InvalidInput("Core ML input shape overflows usize".to_string())
        })
    })?;
    usize::try_from(count).map_err(|_| {
        AdapterError::InvalidInput("Core ML input is too large for this platform".to_string())
    })
}

fn make_multi_array(
    values: &[f32],
    shape: &[u64],
    data_type: MLMultiArrayDataType,
) -> AdapterResult<Retained<MLMultiArray>> {
    let expected = expected_element_count(shape)?;
    if values.len() != expected {
        return Err(AdapterError::InvalidInput(format!(
            "Core ML model expects {expected} input values for shape {shape:?}, received {}",
            values.len()
        )));
    }
    let dimensions = shape
        .iter()
        .map(|dimension| NSNumber::new_u64(*dimension))
        .collect::<Vec<_>>();
    let ns_shape = NSArray::from_retained_slice(&dimensions);
    // SAFETY: Shape values and the Core ML-provided data type are valid.
    let array = unsafe {
        MLMultiArray::initWithShape_dataType_error(MLMultiArray::alloc(), &ns_shape, data_type)
            .map_err(|error| {
                AdapterError::InvalidInput(ns_error("Failed to allocate Core ML input", &error))
            })?
    };
    for (index, value) in values.iter().enumerate() {
        let index = isize::try_from(index)
            .map_err(|_| AdapterError::InvalidInput("Core ML input index overflow".to_string()))?;
        // SAFETY: `index` is within the allocated array's element count.
        unsafe { array.setObject_atIndexedSubscript(&NSNumber::new_f32(*value), index) };
    }
    Ok(array)
}

fn infer_native(model: &WorkerModel, values: &[f32]) -> AdapterResult<Vec<f32>> {
    let array = make_multi_array(values, &model.input_shape, model.input_data_type)?;
    // SAFETY: The feature value retains the valid multi-array.
    let feature_value = unsafe { MLFeatureValue::featureValueWithMultiArray(&array) };
    let value_object: Retained<AnyObject> = feature_value.into_super().into_super();
    let input_name = NSString::from_str(&model.input_name);
    let dictionary = NSDictionary::from_retained_objects(&[&*input_name], &[value_object]);
    // SAFETY: The dictionary contains a supported MLFeatureValue object.
    let provider = unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &dictionary,
        )
        .map_err(|error| {
            AdapterError::InvalidInput(ns_error(
                "Failed to create Core ML feature provider",
                &error,
            ))
        })?
    };
    let provider: &ProtocolObject<dyn MLFeatureProvider> = ProtocolObject::from_ref(&*provider);

    // SAFETY: The provider matches the schema captured at model loading.
    let output = unsafe {
        model
            .model
            .predictionFromFeatures_error(provider)
            .map_err(|error| {
                AdapterError::InferenceFailed(ns_error("Core ML prediction failed", &error))
            })?
    };
    let output_name = NSString::from_str(&model.output_name);
    // SAFETY: The model schema was checked for exactly one named output.
    let output_value = unsafe { output.featureValueForName(&output_name) }.ok_or_else(|| {
        AdapterError::InferenceFailed(format!(
            "Core ML prediction omitted output '{}'",
            model.output_name
        ))
    })?;
    // SAFETY: The output's feature type was checked at model loading.
    let output_array = unsafe { output_value.multiArrayValue() }.ok_or_else(|| {
        AdapterError::InferenceFailed(format!(
            "Core ML output '{}' is not an MLMultiArray",
            model.output_name
        ))
    })?;
    let count = usize::try_from(unsafe { output_array.count() }).map_err(|_| {
        AdapterError::InferenceFailed("Core ML returned an invalid output size".to_string())
    })?;
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let index = isize::try_from(index).map_err(|_| {
            AdapterError::InferenceFailed("Core ML output index overflow".to_string())
        })?;
        // SAFETY: `index` is less than `output_array.count()`.
        result.push(unsafe { output_array.objectAtIndexedSubscript(index) }.as_f32());
    }
    Ok(result)
}
