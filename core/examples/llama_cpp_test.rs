//! LlamaCpp Backend Test
//!
//! Tests the llama.cpp backend directly (without going through TemplateExecutor).
//! Useful for verifying the generation loop works correctly.
//!
//! Run with:
//!   cargo run --example llama_cpp_test -p xybrid-core --features local-llm-llamacpp
//!
//! Requires a GGUF model. Download a small one:
//!   mkdir -p test_models/tinyllama
//!   curl -L -o test_models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
//!     https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

#[cfg(feature = "local-llm-llamacpp")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use xybrid_core::runtime_adapter::llama_cpp::LlamaCppBackend;
    use xybrid_core::runtime_adapter::llm::{ChatMessage, GenerationConfig, LlmBackend, LlmConfig};

    println!("═══════════════════════════════════════════════════════");
    println!("  LlamaCpp Backend Test");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Find a GGUF model
    let model_paths = [
        "test_models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "test_models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "test_models/smollm2-135m-instruct-q8_0.gguf",
    ];

    let model_path = model_paths
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .ok_or("No GGUF model found. Download one first (see file header).")?;

    println!("📁 Using model: {}", model_path);
    println!();

    // Create backend
    println!("🔧 Creating LlamaCppBackend...");
    let mut backend = LlamaCppBackend::new()?;
    println!("✅ Backend created");

    // Load model
    println!("📦 Loading model...");
    let config = LlmConfig::new(*model_path)
        .with_context_length(2048)
        .with_gpu_layers(0); // CPU only for testing

    backend.load(&config)?;
    println!("✅ Model loaded");
    println!();

    // Test generation
    let messages = vec![
        ChatMessage::system("You are a helpful assistant. Be concise."),
        ChatMessage::user("What is 2 + 2?"),
    ];

    let gen_config = GenerationConfig {
        max_tokens: 64,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        ..Default::default()
    };

    println!("💬 Prompt: What is 2 + 2?");
    println!("🔄 Generating...");
    println!();

    let start = std::time::Instant::now();
    let output = backend.generate(&messages, &gen_config)?;
    let elapsed = start.elapsed();

    println!("═══════════════════════════════════════════════════════");
    println!("📤 Response:");
    println!("═══════════════════════════════════════════════════════");
    println!("{}", output.text);
    println!("═══════════════════════════════════════════════════════");
    println!();

    println!("📊 Stats:");
    println!("   Tokens generated: {}", output.tokens_generated);
    println!("   Time: {:.2}s ({:.2} tokens/sec)",
        elapsed.as_secs_f32(),
        output.tokens_per_second);
    println!("   Finish reason: {}", output.finish_reason);
    println!();

    println!("✅ LlamaCpp backend test PASSED!");

    Ok(())
}

#[cfg(not(feature = "local-llm-llamacpp"))]
fn main() {
    eprintln!("This example requires the `local-llm-llamacpp` feature.");
    eprintln!("Run with: cargo run --example llama_cpp_test -p xybrid-core --features local-llm-llamacpp");
}
