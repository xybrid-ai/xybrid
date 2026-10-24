//! Model-free coverage of the existing run/stream surface with the inert client.
#![cfg(all(feature = "tts-zzz-android-companion", not(feature = "tts-zzz")))]

use serde_json::json;
use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};

#[test]
fn unavailable_companion_suppresses_cached_capabilities() {
    let names = xybrid_core::features::enabled();
    assert!(!names.contains(&"tts-zzz"));
    assert!(!names.contains(&"zzz-phonon-31m"));
    assert!(names.contains(&"vision"));
    assert_eq!(names.as_ptr(), xybrid_core::features::enabled().as_ptr());
}

#[test]
fn both_templates_reject_batch_and_stream_before_assets_or_audio() {
    let directory = tempfile::tempdir().unwrap();
    let mut executor = TemplateExecutor::new(directory.path().to_str().unwrap());
    let registered = TemplateExecutor::default_runtimes();
    assert!(registered.contains_key("zzz"));
    assert!(registered.contains_key("zzz-phonon"));
    let expected = xybrid_core::runtime_adapter::zzz::assert_engine_abi()
        .expect_err("the production companion identity is unconfigured")
        .to_string();
    assert!(expected.contains("zzz engine unavailable:"));
    for template in [
        json!({"type":"ZzzEmbed", "model_file":"missing.gguf"}),
        json!({"type":"ZzzPhonon", "model_file":"missing.gguf"}),
    ] {
        let metadata: ModelMetadata = serde_json::from_value(json!({
            "model_id":"unavailable-companion", "version":"1", "files":[],
            "execution_template":template,
        }))
        .unwrap();
        let input = Envelope::new(EnvelopeKind::Text("hello".into()));
        assert!(!executor.is_model_loaded(&metadata));
        let error = executor.execute(&metadata, &input, None).unwrap_err();
        assert!(error.to_string().contains(&expected), "{error}");
        let mut packets = 0;
        let error = executor
            .execute_tts_streaming_controlled(&metadata, &input, &|| false, &mut |_| {
                packets += 1;
                true
            })
            .unwrap_err();
        assert!(error.to_string().contains(&expected), "{error}");
        let error = executor
            .execute_tts_streaming(&metadata, &input, &mut |_, _| {
                packets += 1;
                true
            })
            .unwrap_err();
        assert!(error.to_string().contains(&expected), "{error}");
        assert_eq!(packets, 0);
        assert!(!executor.is_model_loaded(&metadata));
    }
    assert_eq!(directory.path().read_dir().unwrap().count(), 0);
}
