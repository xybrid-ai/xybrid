//! Example demonstrating the #[hybrid::route] macro usage.
//!
//! This example shows how to use the macro to annotate inference functions
//! that will be routed by the orchestrator.

use xybrid_sdk::hybrid;

/// Example ASR (Automatic Speech Recognition) stage function.
///
/// This function would typically process audio input and return text.
/// The `#[hybrid::route]` macro marks it for orchestrator routing.
#[hybrid::route]
fn asr_stage(input: String) -> String {
    // TODO: In future versions, this will be wrapped with orchestrator calls
    // For now, it's a placeholder that compiles but doesn't transform
    format!("asr_output: {}", input)
}

/// Example TTS (Text-to-Speech) stage function.
#[hybrid::route]
fn tts_stage(input: String) -> String {
    format!("tts_output: {}", input)
}

/// Example LLM inference stage.
#[hybrid::route]
fn llm_stage(input: String) -> String {
    format!("llm_output: {}", input)
}

fn main() {
    println!("🎤 Xybrid SDK Macro Demo");
    println!("{}", "=".repeat(60));
    println!();

    println!("📝 Annotated functions:");
    println!("  - #[hybrid::route] fn asr_stage()");
    println!("  - #[hybrid::route] fn tts_stage()");
    println!("  - #[hybrid::route] fn llm_stage()");
    println!();

    println!("ℹ️  Note: The macro is currently a placeholder.");
    println!("   Future versions will transform these functions to:");
    println!("   1. Create StageDescriptor from function metadata");
    println!("   2. Wrap with orchestrator.execute_stage() calls");
    println!("   3. Handle input/output envelope conversion");
    println!("   4. Inject DeviceMetrics and LocalAvailability handling");
    println!();

    // Demonstrate that the functions still work (unchanged for now)
    let result = asr_stage("test input".to_string());
    println!("✅ Function call works: {}", result);
}
