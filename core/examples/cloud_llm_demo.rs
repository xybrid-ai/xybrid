//! Cloud LLM API Demo
//!
//! This example demonstrates using the LLM client to call
//! OpenAI and Anthropic APIs.
//!
//! Prerequisites:
//! - Set OPENAI_API_KEY environment variable for OpenAI
//! - Set ANTHROPIC_API_KEY environment variable for Anthropic
//!
//! Usage:
//!   cargo run --example cloud_llm_demo [openai|anthropic] [prompt]

use std::time::Instant;
use xybrid_core::llm::{Llm, LlmConfig, CompletionRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();

    println!("═══════════════════════════════════════════════════════");
    println!("  Cloud LLM API Demo");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let provider_arg = args.get(1).map(|s| s.as_str()).unwrap_or("openai");
    let prompt = args.get(2).cloned().unwrap_or_else(|| {
        "What is the capital of France? Answer in one sentence.".to_string()
    });

    // Determine provider
    let provider = provider_arg;

    println!("🤖 Provider: {}", provider);
    println!("📝 Prompt: \"{}\"", prompt);
    println!();

    // Check for API key based on provider
    let env_var = match provider {
        "anthropic" => "ANTHROPIC_API_KEY",
        _ => "OPENAI_API_KEY",
    };

    if std::env::var(env_var).is_err() {
        eprintln!("❌ {} not set", env_var);
        eprintln!();
        eprintln!("Please set the environment variable:");
        eprintln!("  export {}=your-api-key", env_var);
        return Ok(());
    }

    // Create client with direct provider access
    let client_start = Instant::now();
    let config = LlmConfig::direct(provider);
    let client = Llm::with_config(config)?;
    let client_latency = client_start.elapsed();
    println!("✅ Client created successfully");
    println!("   ⏱️  Client init: {:.2}ms", client_latency.as_secs_f64() * 1000.0);
    println!();

    // Build request
    let request = CompletionRequest::new(&prompt)
        .with_system("You are a helpful assistant. Be concise.")
        .with_max_tokens(100)
        .with_temperature(0.7);

    println!("🔄 Sending request...");

    // Send request with timing
    let request_start = Instant::now();
    let response = client.complete(request)?;
    let request_latency = request_start.elapsed();

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Response");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("📤 Model: {}", response.model);
    println!();
    println!("💬 Response:");
    println!("   {}", response.text);
    println!();

    if let Some(usage) = &response.usage {
        println!("📊 Token Usage:");
        println!("   Prompt:     {} tokens", usage.prompt_tokens);
        println!("   Completion: {} tokens", usage.completion_tokens);
        println!("   Total:      {} tokens", usage.total_tokens);
        println!();
    }

    if let Some(finish_reason) = &response.finish_reason {
        println!("🏁 Finish Reason: {}", finish_reason);
        if finish_reason == "length" {
            println!("   ⚠️  Response was truncated (hit max_tokens)");
        }
    }

    let total_latency = total_start.elapsed();

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Latency Summary");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("⏱️  Client initialization: {:>8.2}ms", client_latency.as_secs_f64() * 1000.0);
    println!("⏱️  API request:           {:>8.2}ms", request_latency.as_secs_f64() * 1000.0);
    println!("───────────────────────────────────────────────────────");
    println!("⏱️  Total:                 {:>8.2}ms", total_latency.as_secs_f64() * 1000.0);
    println!();

    println!("═══════════════════════════════════════════════════════");
    println!("  Demo Complete!");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("🎯 Features demonstrated:");
    println!("   • Provider abstraction (OpenAI/Anthropic)");
    println!("   • System prompts");
    println!("   • Temperature control");
    println!("   • Token usage tracking");
    println!("   • Latency measurement");
    println!();

    Ok(())
}
