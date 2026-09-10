//! Run the native Core ML adapter against a single-input tensor model.

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use xybrid_core::ir::{Envelope, EnvelopeKind};
    use xybrid_core::runtime_adapter::{CoreMLRuntimeAdapter, RuntimeAdapter};

    let model_path = std::env::args().nth(1).ok_or(
        "usage: cargo run -p xybrid-core --example coreml_linear --features coreml -- <model.mlmodel>",
    )?;
    let mut adapter = CoreMLRuntimeAdapter::new();
    adapter.load_model(&model_path)?;

    let input = Envelope::new(EnvelopeKind::Embedding(vec![1.0, 2.0, 3.0, 4.0]));
    let output = adapter.execute(&input)?;
    println!("{output:?}");
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn main() {
    eprintln!("The native Core ML example requires macOS or iOS.");
}
