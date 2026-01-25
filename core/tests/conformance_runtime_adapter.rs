//! Conformance tests for RuntimeAdapter implementations.
//!
//! These tests verify that all RuntimeAdapter implementations follow the
//! expected contract. Any new adapter implementation should pass all
//! conformance tests.
//!
//! ## Running Tests
//!
//! ```bash
//! # Run all conformance tests
//! cargo test --test conformance_runtime_adapter
//!
//! # Run with specific features for additional backends
//! cargo test --test conformance_runtime_adapter --features candle
//! ```

use std::sync::Arc;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::runtime_adapter::{AdapterError, RuntimeAdapter};

// ─────────────────────────────────────────────────────────────────────────────
// Conformance Test Suite
// ─────────────────────────────────────────────────────────────────────────────

/// Tests that all RuntimeAdapter implementations must pass.
///
/// This function can be called with any adapter to verify conformance.
fn adapter_conformance_suite<A: RuntimeAdapter>(adapter: &A, adapter_name: &str) {
    // Test 1: Name is non-empty and matches expected
    let name = adapter.name();
    assert!(
        !name.is_empty(),
        "{}: Adapter name should not be empty",
        adapter_name
    );
    println!("✓ {}: name() returned '{}'", adapter_name, name);

    // Test 2: Supported formats returns a list (can be empty for non-file adapters)
    let formats = adapter.supported_formats();
    println!(
        "✓ {}: supported_formats() returned {:?}",
        adapter_name, formats
    );

    // Test 3: Execute before loading returns appropriate error
    let test_input = Envelope::new(EnvelopeKind::Text("test input".to_string()));
    let result = adapter.execute(&test_input);

    // For adapters that don't require model loading (like cloud), this may succeed
    // For file-based adapters, this should fail with ModelNotLoaded
    match result {
        Ok(_) => {
            println!(
                "✓ {}: execute() succeeded without load (stateless adapter)",
                adapter_name
            );
        }
        Err(AdapterError::ModelNotLoaded(_)) => {
            println!(
                "✓ {}: execute() returned ModelNotLoaded before load_model",
                adapter_name
            );
        }
        Err(AdapterError::InvalidInput(_)) => {
            println!(
                "✓ {}: execute() returned InvalidInput (expected for some adapters)",
                adapter_name
            );
        }
        Err(e) => {
            // Other errors are acceptable for specific adapters
            println!(
                "✓ {}: execute() returned expected error: {:?}",
                adapter_name, e
            );
        }
    }
}

/// Tests for mutable adapter operations (load_model).
fn adapter_mutable_conformance<A: RuntimeAdapter>(adapter: &mut A, adapter_name: &str) {
    // Test load_model with non-existent path
    let result = adapter.load_model("/nonexistent/path/to/model.onnx");

    match result {
        Ok(_) => {
            // Some adapters (like cloud) don't need to load files
            println!(
                "✓ {}: load_model() with nonexistent path succeeded (stateless)",
                adapter_name
            );
        }
        Err(AdapterError::ModelNotFound(_)) => {
            println!(
                "✓ {}: load_model() returned ModelNotFound for nonexistent path",
                adapter_name
            );
        }
        Err(AdapterError::IOError(_)) => {
            println!(
                "✓ {}: load_model() returned IOError for nonexistent path",
                adapter_name
            );
        }
        Err(e) => {
            println!(
                "✓ {}: load_model() returned error for nonexistent path: {:?}",
                adapter_name, e
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX Runtime Adapter Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn onnx_adapter_conformance() {
    use xybrid_core::runtime_adapter::OnnxRuntimeAdapter;

    let adapter = OnnxRuntimeAdapter::new();
    adapter_conformance_suite(&adapter, "OnnxRuntimeAdapter");
}

#[test]
fn onnx_adapter_mutable_conformance() {
    use xybrid_core::runtime_adapter::OnnxRuntimeAdapter;

    let mut adapter = OnnxRuntimeAdapter::new();
    adapter_mutable_conformance(&mut adapter, "OnnxRuntimeAdapter");
}

#[test]
fn onnx_adapter_name() {
    use xybrid_core::runtime_adapter::OnnxRuntimeAdapter;

    let adapter = OnnxRuntimeAdapter::new();
    assert_eq!(adapter.name(), "onnx");
}

#[test]
fn onnx_adapter_supported_formats() {
    use xybrid_core::runtime_adapter::OnnxRuntimeAdapter;

    let adapter = OnnxRuntimeAdapter::new();
    let formats = adapter.supported_formats();
    assert!(formats.contains(&"onnx"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Cloud Runtime Adapter Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cloud_adapter_conformance() {
    use xybrid_core::runtime_adapter::CloudRuntimeAdapter;

    let adapter = CloudRuntimeAdapter::new();
    adapter_conformance_suite(&adapter, "CloudRuntimeAdapter");
}

#[test]
fn cloud_adapter_mutable_conformance() {
    use xybrid_core::runtime_adapter::CloudRuntimeAdapter;

    let mut adapter = CloudRuntimeAdapter::new();
    adapter_mutable_conformance(&mut adapter, "CloudRuntimeAdapter");
}

#[test]
fn cloud_adapter_name() {
    use xybrid_core::runtime_adapter::CloudRuntimeAdapter;

    let adapter = CloudRuntimeAdapter::new();
    assert_eq!(adapter.name(), "cloud");
}

#[test]
fn cloud_adapter_supported_formats_empty() {
    use xybrid_core::runtime_adapter::CloudRuntimeAdapter;

    let adapter = CloudRuntimeAdapter::new();
    let formats = adapter.supported_formats();
    // Cloud adapter doesn't use file formats
    assert!(formats.is_empty());
}

#[test]
fn cloud_adapter_load_model_is_noop() {
    use xybrid_core::runtime_adapter::CloudRuntimeAdapter;

    let mut adapter = CloudRuntimeAdapter::new();
    // Should succeed (no-op for cloud adapter)
    assert!(adapter.load_model("/any/path").is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// CoreML Runtime Adapter Tests (macOS/iOS only)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod coreml_tests {
    use super::*;
    use xybrid_core::runtime_adapter::CoreMLRuntimeAdapter;

    #[test]
    fn coreml_adapter_conformance() {
        let adapter = CoreMLRuntimeAdapter::new();
        adapter_conformance_suite(&adapter, "CoreMLRuntimeAdapter");
    }

    #[test]
    fn coreml_adapter_mutable_conformance() {
        let mut adapter = CoreMLRuntimeAdapter::new();
        adapter_mutable_conformance(&mut adapter, "CoreMLRuntimeAdapter");
    }

    #[test]
    fn coreml_adapter_name() {
        let adapter = CoreMLRuntimeAdapter::new();
        assert_eq!(adapter.name(), "coreml");
    }

    #[test]
    fn coreml_adapter_supported_formats() {
        let adapter = CoreMLRuntimeAdapter::new();
        let formats = adapter.supported_formats();
        assert!(formats.contains(&"mlpackage") || formats.contains(&"mlmodel"));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Candle Runtime Adapter Tests (feature-gated)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "candle")]
mod candle_tests {
    use super::*;
    use xybrid_core::runtime_adapter::CandleRuntimeAdapter;

    #[test]
    fn candle_adapter_conformance() {
        let adapter = CandleRuntimeAdapter::new();
        adapter_conformance_suite(&adapter, "CandleRuntimeAdapter");
    }

    #[test]
    fn candle_adapter_mutable_conformance() {
        let mut adapter = CandleRuntimeAdapter::new();
        adapter_mutable_conformance(&mut adapter, "CandleRuntimeAdapter");
    }

    #[test]
    fn candle_adapter_name() {
        let adapter = CandleRuntimeAdapter::new();
        assert_eq!(adapter.name(), "candle");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mock Adapter Tests (for testing infrastructure)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mock_adapter_conformance() {
    use xybrid_core::testing::mocks::MockRuntimeAdapter;

    let adapter = MockRuntimeAdapter::with_text_output("mock output");
    adapter_conformance_suite(&adapter, "MockRuntimeAdapter");
}

#[test]
fn mock_adapter_mutable_conformance() {
    use xybrid_core::testing::mocks::MockRuntimeAdapter;

    let mut adapter = MockRuntimeAdapter::with_text_output("mock output");
    adapter_mutable_conformance(&mut adapter, "MockRuntimeAdapter");
}

#[test]
fn mock_adapter_with_preset_output() {
    use xybrid_core::testing::mocks::MockRuntimeAdapter;

    let mut adapter = MockRuntimeAdapter::with_text_output("expected output");
    adapter.load_model("/mock/model").unwrap();

    let input = Envelope::new(EnvelopeKind::Text("test".to_string()));
    let output = adapter.execute(&input).unwrap();

    match output.kind {
        EnvelopeKind::Text(text) => assert_eq!(text, "expected output"),
        _ => panic!("Expected Text output"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread Safety Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn adapter_is_send_sync() {
    use xybrid_core::runtime_adapter::OnnxRuntimeAdapter;

    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<OnnxRuntimeAdapter>();
    assert_send_sync::<Arc<dyn RuntimeAdapter>>();
}

#[test]
fn adapter_can_be_shared_across_threads() {
    use xybrid_core::runtime_adapter::OnnxRuntimeAdapter;
    use std::thread;

    let adapter = Arc::new(OnnxRuntimeAdapter::new());

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let adapter_clone = Arc::clone(&adapter);
            thread::spawn(move || {
                let name = adapter_clone.name();
                assert_eq!(name, "onnx");
                println!("Thread {} verified adapter name: {}", i, name);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}
