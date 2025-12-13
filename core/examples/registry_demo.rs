//! Example demonstrating the Registry for storing and retrieving bundles.
//!
//! This example shows how to use LocalRegistry to store policy bundles
//! and retrieve them for use with the orchestrator.
//!
//! Run with: `cargo run --example registry_demo`

use xybrid_core::registry::{LocalRegistry, Registry};

fn main() {
    println!("📦 Registry Demo: Bundle Storage and Retrieval");
    println!("{}", "=".repeat(60));
    println!();

    // Create a registry in a temporary location (for demo)
    // In production, use LocalRegistry::default() for user's home directory
    let mut registry =
        LocalRegistry::new("./target/registry_demo").expect("Failed to create registry");

    // Store a policy bundle
    println!("💾 Storing policy bundle...");
    let policy_bundle = r#"
version: "1.0.0"
rules:
  - id: "audio_rule"
    expression: "input.kind == 'AudioRaw'"
    action: "deny"
"#;

    let metadata = registry
        .store_bundle("policy-hiiipe", "1.0.0", policy_bundle.as_bytes().to_vec())
        .expect("Failed to store bundle");

    println!(
        "   Stored: {} v{} ({} bytes)",
        metadata.id, metadata.version, metadata.size_bytes
    );
    println!();

    // Retrieve the bundle
    println!("📥 Retrieving bundle...");
    let retrieved = registry
        .get_bundle("policy-hiiipe", Some("1.0.0"))
        .expect("Failed to retrieve bundle");

    let retrieved_str = String::from_utf8_lossy(&retrieved);
    println!("   Retrieved {} bytes", retrieved.len());
    println!(
        "   Content preview: {}",
        retrieved_str.lines().next().unwrap_or("")
    );
    println!();

    // List all bundles
    println!("📋 Listing all bundles...");
    let bundles = registry.list_bundles().expect("Failed to list bundles");
    println!("   Found {} bundle(s):", bundles.len());
    for bundle in &bundles {
        println!(
            "   - {} v{} ({} bytes)",
            bundle.id, bundle.version, bundle.size_bytes
        );
    }
    println!();

    // Get metadata without loading content
    println!("📊 Getting bundle metadata...");
    let meta = registry
        .get_metadata("policy-hiiipe", Some("1.0.0"))
        .expect("Failed to get metadata");
    println!("   ID: {}", meta.id);
    println!("   Version: {}", meta.version);
    println!("   Path: {}", meta.path);
    println!("   Size: {} bytes", meta.size_bytes);
    println!();

    // Try to get latest version (when version is None)
    println!("🔍 Getting latest version...");
    let latest = registry
        .get_bundle("policy-hiiipe", None)
        .expect("Failed to get latest bundle");
    println!("   Retrieved latest version ({} bytes)", latest.len());
    println!();

    println!("✅ Registry demo completed successfully!");
}
