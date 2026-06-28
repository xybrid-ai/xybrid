//! Godot 4.5+ GDExtension bindings for xybrid.
//!
//! The Godot-facing API uses `Dictionary` records so GDScript can branch on
//! `{ ok = true }` / `{ ok = false }` without exceptions crossing the extension
//! boundary. All SDK translation routes through `xybrid-ffi-facade`.

use std::sync::Arc;

use godot::builtin::{Array, GString, PackedByteArray, PackedFloat32Array, VarDictionary, Variant};
use godot::classes::{IRefCounted, RefCounted};
use godot::prelude::*;
use xybrid_ffi_facade as facade;

type Dictionary = VarDictionary;

struct XybridGodotExtension;

// SAFETY: godot-rust owns the generated GDExtension entry point for this
// marker type; the type carries no state and only registers Godot classes.
#[gdextension]
unsafe impl ExtensionLibrary for XybridGodotExtension {}

#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct XybridRuntime {
    base: Base<RefCounted>,
}

#[godot_api]
impl IRefCounted for XybridRuntime {
    fn init(base: Base<Self::Base>) -> Self {
        Self { base }
    }
}

#[godot_api]
impl XybridRuntime {
    #[func]
    pub fn init(
        &self,
        #[opt(default = "")] api_key: GString,
        #[opt(default = "")] gateway_url: GString,
        #[opt(default = "")] ingest_url: GString,
        #[opt(default = "")] cache_dir: GString,
    ) -> Dictionary {
        let cache_dir = cache_dir.to_string();
        if !cache_dir.trim().is_empty() {
            facade::init_sdk_cache_dir(cache_dir);
        }

        facade::set_binding("godot".to_string());
        facade::configure_runtime(
            optional_string(api_key),
            optional_string(gateway_url),
            optional_string(ingest_url),
        );
        ok_nil()
    }

    #[func]
    pub fn set_cache_dir(&self, path: GString) -> Dictionary {
        let path = path.to_string();
        if path.trim().is_empty() {
            return err_message("cache_dir cannot be blank");
        }

        facade::init_sdk_cache_dir(path);
        ok_nil()
    }

    #[func]
    pub fn model_from_registry(
        &self,
        id: GString,
        #[opt(default = "")] platform: GString,
    ) -> Dictionary {
        let id = id.to_string();
        if id.trim().is_empty() {
            return err_message("model id cannot be blank");
        }

        let platform = platform.to_string();
        let loader = if platform.trim().is_empty() {
            facade::ModelLoader::from_registry(id)
        } else {
            facade::ModelLoader::from_registry_with_platform(id, platform)
        };
        ok_variant(loader_to_gd(loader).to_variant())
    }

    #[func]
    pub fn model_from_directory(&self, path: GString) -> Dictionary {
        match facade::ModelLoader::from_directory(path.to_string()) {
            Ok(loader) => ok_variant(loader_to_gd(loader).to_variant()),
            Err(error) => err(error),
        }
    }

    #[func]
    pub fn model_from_bundle(&self, path: GString) -> Dictionary {
        match facade::ModelLoader::from_bundle(path.to_string()) {
            Ok(loader) => ok_variant(loader_to_gd(loader).to_variant()),
            Err(error) => err(error),
        }
    }

    #[func]
    pub fn model_from_huggingface(&self, repo: GString) -> Dictionary {
        let repo = repo.to_string();
        if repo.trim().is_empty() {
            return err_message("HuggingFace repo cannot be blank");
        }

        ok_variant(loader_to_gd(facade::ModelLoader::from_huggingface(repo)).to_variant())
    }
}

#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct XybridModelLoader {
    base: Base<RefCounted>,
    inner: Option<Arc<facade::ModelLoader>>,
}

#[godot_api]
impl IRefCounted for XybridModelLoader {
    fn init(base: Base<Self::Base>) -> Self {
        Self { base, inner: None }
    }
}

#[godot_api]
impl XybridModelLoader {
    #[func]
    pub fn load(&self) -> Dictionary {
        let Some(loader) = &self.inner else {
            return err_message("XybridModelLoader was not created by XybridRuntime");
        };

        match loader.load() {
            Ok(model) => ok_variant(model_to_gd(model).to_variant()),
            Err(error) => err(error),
        }
    }
}

#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct XybridModel {
    base: Base<RefCounted>,
    inner: Option<Arc<facade::XybridModel>>,
}

#[godot_api]
impl IRefCounted for XybridModel {
    fn init(base: Base<Self::Base>) -> Self {
        Self { base, inner: None }
    }
}

#[godot_api]
impl XybridModel {
    #[func]
    pub fn run(
        &self,
        envelope: Dictionary,
        generation_config: Dictionary,
        run_options: Dictionary,
    ) -> Dictionary {
        let Some(model) = &self.inner else {
            return err_message("XybridModel was not created by XybridModelLoader");
        };

        let envelope = match dictionary_to_envelope(&envelope) {
            Ok(envelope) => envelope,
            Err(message) => return err_message(message),
        };
        let generation_config = dictionary_to_generation_config(&generation_config);
        let mut options = dictionary_to_run_options(&run_options);
        if let Some(config) = generation_config {
            options.generation_config = Some(config);
        }

        match model.run_with_options(envelope, options, None) {
            Ok(result) => ok_variant(result_to_dictionary(result).to_variant()),
            Err(error) => err(error),
        }
    }

    #[func]
    pub fn run_with_context(
        &self,
        envelope: Dictionary,
        context: Gd<XybridConversationContext>,
        generation_config: Dictionary,
    ) -> Dictionary {
        let Some(model) = &self.inner else {
            return err_message("XybridModel was not created by XybridModelLoader");
        };
        let Some(context) = context.bind().inner.clone() else {
            return err_message("XybridConversationContext is not initialized");
        };
        let envelope = match dictionary_to_envelope(&envelope) {
            Ok(envelope) => envelope,
            Err(message) => return err_message(message),
        };
        let generation_config = dictionary_to_generation_config(&generation_config);

        match model.run_with_context(envelope, context, generation_config) {
            Ok(result) => ok_variant(result_to_dictionary(result).to_variant()),
            Err(error) => err(error),
        }
    }

    #[func]
    pub fn warmup(&self) -> Dictionary {
        let Some(model) = &self.inner else {
            return err_message("XybridModel was not created by XybridModelLoader");
        };

        match model.warmup() {
            Ok(()) => ok_nil(),
            Err(error) => err(error),
        }
    }

    #[func]
    pub fn unload(&self) -> Dictionary {
        let Some(model) = &self.inner else {
            return err_message("XybridModel was not created by XybridModelLoader");
        };

        match model.unload() {
            Ok(()) => ok_nil(),
            Err(error) => err(error),
        }
    }
}

#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct XybridConversationContext {
    base: Base<RefCounted>,
    inner: Option<Arc<facade::ConversationContextHandle>>,
}

#[godot_api]
impl IRefCounted for XybridConversationContext {
    fn init(base: Base<Self::Base>) -> Self {
        Self {
            base,
            inner: Some(facade::ConversationContextHandle::new()),
        }
    }
}

#[godot_api]
impl XybridConversationContext {
    #[func]
    pub fn push(&self, envelope: Dictionary) -> Dictionary {
        let Some(context) = &self.inner else {
            return err_message("XybridConversationContext is not initialized");
        };
        let envelope = match dictionary_to_envelope(&envelope) {
            Ok(envelope) => envelope,
            Err(message) => return err_message(message),
        };

        match context.push(envelope) {
            Ok(()) => ok_nil(),
            Err(error) => err(error),
        }
    }

    #[func]
    pub fn set_system(&self, envelope: Dictionary) -> Dictionary {
        let Some(context) = &self.inner else {
            return err_message("XybridConversationContext is not initialized");
        };
        let envelope = match dictionary_to_envelope(&envelope) {
            Ok(envelope) => envelope,
            Err(message) => return err_message(message),
        };

        match context.set_system(envelope) {
            Ok(()) => ok_nil(),
            Err(error) => err(error),
        }
    }

    #[func]
    pub fn clear(&self) {
        if let Some(context) = &self.inner {
            context.clear();
        }
    }
}

fn loader_to_gd(loader: Arc<facade::ModelLoader>) -> Gd<XybridModelLoader> {
    Gd::from_init_fn(|base| XybridModelLoader {
        base,
        inner: Some(loader),
    })
}

fn model_to_gd(model: Arc<facade::XybridModel>) -> Gd<XybridModel> {
    Gd::from_init_fn(|base| XybridModel {
        base,
        inner: Some(model),
    })
}

fn optional_string(value: GString) -> Option<String> {
    let value = value.to_string();
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

fn ok_nil() -> Dictionary {
    let mut dict = Dictionary::new();
    dict.set("ok", true);
    dict.set("value", &Variant::nil());
    dict
}

fn ok_variant(value: Variant) -> Dictionary {
    let mut dict = Dictionary::new();
    dict.set("ok", true);
    dict.set("value", &value);
    dict
}

fn err(error: facade::Error) -> Dictionary {
    let mut dict = Dictionary::new();
    dict.set("ok", false);
    dict.set("code", i64::from(error.code()));
    dict.set("retryable", error.is_retryable());
    dict.set("message", error.to_string());
    dict
}

fn err_message(message: impl Into<String>) -> Dictionary {
    let mut dict = Dictionary::new();
    dict.set("ok", false);
    dict.set("code", 0_i64);
    dict.set("retryable", false);
    dict.set("message", message.into());
    dict
}

fn dictionary_to_envelope(dict: &Dictionary) -> Result<facade::Envelope, String> {
    let kind = get_string(dict, "kind").ok_or_else(|| "envelope.kind is required".to_string())?;
    let mut envelope = match kind.as_str() {
        "text" => facade::Envelope::text(get_string(dict, "text").unwrap_or_default()),
        "audio" => facade::Envelope::audio(get_bytes(dict, "bytes")?),
        "embedding" => facade::Envelope::embedding(get_f32s(dict, "values")?),
        "image" => facade::Envelope::image(
            get_bytes(dict, "bytes")?,
            get_string(dict, "format").unwrap_or_else(|| "png".to_string()),
        ),
        "multipart" => {
            let parts = get_dictionary_array(dict, "parts")?
                .into_iter()
                .map(|part| dictionary_to_envelope(&part))
                .collect::<Result<Vec<_>, _>>()?;
            facade::Envelope::multipart(parts)
        }
        _ => return Err(format!("unsupported envelope.kind `{kind}`")),
    };

    if let Some(role) = get_string(dict, "role") {
        let role = match role.as_str() {
            "system" => facade::MessageRole::System,
            "user" => facade::MessageRole::User,
            "assistant" => facade::MessageRole::Assistant,
            _ => return Err(format!("unsupported envelope.role `{role}`")),
        };
        envelope = envelope.with_role(role);
    }

    Ok(envelope)
}

fn dictionary_to_generation_config(dict: &Dictionary) -> Option<facade::GenerationConfig> {
    if dict.is_empty() {
        return None;
    }

    Some(facade::GenerationConfig {
        max_tokens: get_i64(dict, "max_tokens").and_then(|v| u32::try_from(v).ok()),
        temperature: get_f64(dict, "temperature").map(|v| v as f32),
        top_p: get_f64(dict, "top_p").map(|v| v as f32),
        min_p: get_f64(dict, "min_p").map(|v| v as f32),
        top_k: get_i64(dict, "top_k").and_then(|v| u32::try_from(v).ok()),
        repetition_penalty: get_f64(dict, "repetition_penalty").map(|v| v as f32),
        stop_sequences: get_string_array(dict, "stop_sequences").unwrap_or_default(),
    })
}

fn dictionary_to_run_options(dict: &Dictionary) -> facade::RunOptions {
    let abort_on = get_string_array(dict, "abort_on")
        .unwrap_or_default()
        .into_iter()
        .filter_map(|signal| match signal.as_str() {
            "memory_pressure_warn" => Some(facade::AbortSignal::MemoryPressureWarn),
            "memory_pressure_critical" => Some(facade::AbortSignal::MemoryPressureCritical),
            "thermal_hot" => Some(facade::AbortSignal::ThermalHot),
            "thermal_critical" => Some(facade::AbortSignal::ThermalCritical),
            _ => None,
        })
        .collect();

    facade::RunOptions {
        fallback_to_cloud: get_bool(dict, "fallback_to_cloud").unwrap_or(false),
        max_grace_tokens: get_i64(dict, "max_grace_tokens")
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0),
        correlation_id: get_string(dict, "correlation_id").filter(|v| !v.trim().is_empty()),
        abort_on,
        ..Default::default()
    }
}

fn result_to_dictionary(result: facade::InferenceResult) -> Dictionary {
    let text = result.text().map(str::to_string);
    let audio_bytes = result.audio_bytes().map(<[u8]>::to_vec);
    let embedding = result.embedding().map(<[f32]>::to_vec);

    let mut dict = Dictionary::new();
    dict.set("output_type", output_type_name(result.output_type));
    dict.set("model_id", result.model_id);
    dict.set("latency_ms", i64::from(result.latency_ms));
    dict.set("metrics", &metrics_to_dictionary(result.metrics));
    dict.set("envelope", &envelope_to_dictionary(&result.envelope));
    if let Some(text) = text {
        dict.set("text", text);
    }
    if let Some(bytes) = audio_bytes {
        dict.set("audio_bytes", &PackedByteArray::from(bytes.as_slice()));
    }
    if let Some(values) = embedding {
        dict.set("embedding", &PackedFloat32Array::from(values.as_slice()));
    }
    dict
}

fn metrics_to_dictionary(metrics: facade::InferenceMetrics) -> Dictionary {
    let mut dict = Dictionary::new();
    dict.set("total_ms", i64::from(metrics.total_ms));
    set_optional_u32(&mut dict, "ttft_ms", metrics.ttft_ms);
    set_optional_f32(&mut dict, "tokens_per_second", metrics.tokens_per_second);
    set_optional_f32(&mut dict, "prefill_tps", metrics.prefill_tps);
    set_optional_f32(&mut dict, "decode_tps", metrics.decode_tps);
    set_optional_u32(&mut dict, "tokens_out", metrics.tokens_out);

    let mut stages = Array::<Dictionary>::new();
    for stage in metrics.stage_latencies_ms {
        let mut stage_dict = Dictionary::new();
        stage_dict.set("stage_id", stage.stage_id);
        stage_dict.set("latency_ms", i64::from(stage.latency_ms));
        stages.push(&stage_dict);
    }
    dict.set("stage_latencies_ms", &stages);
    dict
}

fn envelope_to_dictionary(envelope: &facade::Envelope) -> Dictionary {
    let mut dict = Dictionary::new();
    match &envelope.kind {
        facade::EnvelopeKind::Text { text } => {
            dict.set("kind", "text");
            dict.set("text", text.as_str());
        }
        facade::EnvelopeKind::Audio { bytes } => {
            dict.set("kind", "audio");
            dict.set("bytes", &PackedByteArray::from(bytes.as_slice()));
        }
        facade::EnvelopeKind::Embedding { values } => {
            dict.set("kind", "embedding");
            dict.set("values", &PackedFloat32Array::from(values.as_slice()));
        }
        facade::EnvelopeKind::Image { bytes, format } => {
            dict.set("kind", "image");
            dict.set("bytes", &PackedByteArray::from(bytes.as_slice()));
            dict.set("format", format.as_str());
        }
        facade::EnvelopeKind::MultiPart { parts } => {
            dict.set("kind", "multipart");
            let mut arr = Array::<Dictionary>::new();
            for part in parts {
                arr.push(&envelope_to_dictionary(part));
            }
            dict.set("parts", &arr);
        }
    }

    if let Some(role) = envelope.role() {
        dict.set("role", role.as_str());
    }
    dict
}

fn output_type_name(output_type: facade::OutputType) -> &'static str {
    match output_type {
        facade::OutputType::Text => "text",
        facade::OutputType::Audio => "audio",
        facade::OutputType::Embedding => "embedding",
        facade::OutputType::Unknown => "unknown",
    }
}

fn set_optional_u32(dict: &mut Dictionary, key: &str, value: Option<u32>) {
    if let Some(value) = value {
        dict.set(key, i64::from(value));
    }
}

fn set_optional_f32(dict: &mut Dictionary, key: &str, value: Option<f32>) {
    if let Some(value) = value {
        dict.set(key, f64::from(value));
    }
}

fn get_variant(dict: &Dictionary, key: &str) -> Option<Variant> {
    dict.get(key)
}

fn get_string(dict: &Dictionary, key: &str) -> Option<String> {
    get_variant(dict, key).and_then(|v| v.try_to::<GString>().ok().map(|s| s.to_string()))
}

fn get_bool(dict: &Dictionary, key: &str) -> Option<bool> {
    get_variant(dict, key).and_then(|v| v.try_to::<bool>().ok())
}

fn get_i64(dict: &Dictionary, key: &str) -> Option<i64> {
    get_variant(dict, key).and_then(|v| v.try_to::<i64>().ok())
}

fn get_f64(dict: &Dictionary, key: &str) -> Option<f64> {
    get_variant(dict, key).and_then(|v| v.try_to::<f64>().ok())
}

fn get_bytes(dict: &Dictionary, key: &str) -> Result<Vec<u8>, String> {
    let Some(value) = get_variant(dict, key) else {
        return Ok(Vec::new());
    };

    if let Ok(bytes) = value.try_to::<PackedByteArray>() {
        return Ok(bytes.as_slice().to_vec());
    }
    if let Ok(values) = value.try_to::<Array<Variant>>() {
        return values
            .iter_shared()
            .map(|v| {
                let value = v
                    .try_to::<i64>()
                    .map_err(|_| format!("{key} contains non-integer value"))?;
                byte_from_i64(value, key)
            })
            .collect();
    }
    if let Ok(values) = value.try_to::<Array<i64>>() {
        return values
            .iter_shared()
            .map(|v| byte_from_i64(v, key))
            .collect();
    }

    Err(format!("{key} must be a PackedByteArray or Array[int]"))
}

fn byte_from_i64(value: i64, key: &str) -> Result<u8, String> {
    u8::try_from(value).map_err(|_| format!("{key} contains byte outside 0..=255"))
}

fn get_f32s(dict: &Dictionary, key: &str) -> Result<Vec<f32>, String> {
    let Some(value) = get_variant(dict, key) else {
        return Ok(Vec::new());
    };

    if let Ok(values) = value.try_to::<PackedFloat32Array>() {
        return Ok(values.as_slice().to_vec());
    }
    if let Ok(values) = value.try_to::<Array<Variant>>() {
        return values
            .iter_shared()
            .map(|v| {
                v.try_to::<f64>()
                    .map(|value| value as f32)
                    .map_err(|_| format!("{key} contains non-float value"))
            })
            .collect();
    }
    if let Ok(values) = value.try_to::<Array<f64>>() {
        return Ok(values.iter_shared().map(|v| v as f32).collect());
    }

    Err(format!(
        "{key} must be a PackedFloat32Array or Array[float]"
    ))
}

fn get_string_array(dict: &Dictionary, key: &str) -> Option<Vec<String>> {
    get_variant(dict, key).and_then(|value| {
        if let Ok(values) = value.try_to::<Array<Variant>>() {
            Some(
                values
                    .iter_shared()
                    .filter_map(|v| v.try_to::<GString>().ok().map(|s| s.to_string()))
                    .collect(),
            )
        } else {
            value
                .try_to::<Array<GString>>()
                .ok()
                .map(|values| values.iter_shared().map(|s| s.to_string()).collect())
        }
    })
}

fn get_dictionary_array(dict: &Dictionary, key: &str) -> Result<Vec<Dictionary>, String> {
    let Some(value) = get_variant(dict, key) else {
        return Ok(Vec::new());
    };

    if let Ok(values) = value.try_to::<Array<Dictionary>>() {
        return Ok(values.iter_shared().collect());
    }
    if let Ok(values) = value.try_to::<Array<Variant>>() {
        return values
            .iter_shared()
            .map(|v| {
                v.try_to::<Dictionary>()
                    .map_err(|_| format!("{key} elements must be Dictionaries"))
            })
            .collect();
    }

    Err(format!("{key} must be an Array of Dictionaries"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Godot Dictionary requires an initialized Godot engine"]
    fn text_envelope_round_trips_through_dictionary() {
        let mut dict = Dictionary::new();
        dict.set("kind", "text");
        dict.set("text", "hello");
        dict.set("role", "user");

        let envelope = dictionary_to_envelope(&dict).unwrap();
        assert_eq!(envelope.role(), Some(facade::MessageRole::User));

        let back = envelope_to_dictionary(&envelope);
        assert_eq!(get_string(&back, "kind").as_deref(), Some("text"));
        assert_eq!(get_string(&back, "text").as_deref(), Some("hello"));
    }

    #[test]
    #[ignore = "Godot Dictionary requires an initialized Godot engine"]
    fn generation_config_reads_optional_overrides() {
        let mut dict = Dictionary::new();
        dict.set("max_tokens", 32_i64);
        dict.set("temperature", 0.25_f64);

        let config = dictionary_to_generation_config(&dict).unwrap();
        assert_eq!(config.max_tokens, Some(32));
        assert_eq!(config.temperature, Some(0.25));
    }

    #[test]
    #[ignore = "Godot Dictionary requires an initialized Godot engine"]
    fn error_dictionary_contains_stable_fields() {
        let dict = err(facade::Error::NotLoaded);
        assert_eq!(get_bool(&dict, "ok"), Some(false));
        assert_eq!(get_i64(&dict, "code"), Some(9));
        assert_eq!(get_bool(&dict, "retryable"), Some(false));
    }

    #[test]
    fn byte_from_i64_accepts_only_byte_range() {
        assert_eq!(byte_from_i64(0, "bytes"), Ok(0));
        assert_eq!(byte_from_i64(255, "bytes"), Ok(255));
        assert_eq!(
            byte_from_i64(-1, "bytes"),
            Err("bytes contains byte outside 0..=255".to_string())
        );
        assert_eq!(
            byte_from_i64(256, "bytes"),
            Err("bytes contains byte outside 0..=255".to_string())
        );
    }
}
