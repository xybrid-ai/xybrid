//! OrchestrationAuthority Demo
//!
//! Demonstrates how the authority makes different routing decisions based on:
//! - Device capabilities (battery, network RTT, temperature)
//! - Model requirements
//! - Explicit pipeline targets
//!
//! Run with:
//!   cargo run -p xybrid-core --example authority_demo

use xybrid_core::context::DeviceMetrics;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::orchestrator::{
    LocalAuthority, ModelConstraints, ModelRequest, OrchestrationAuthority, PolicyOutcome,
    PolicyRequest, ResolvedTarget, StageContext,
};
use xybrid_core::pipeline::ExecutionTarget;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         OrchestrationAuthority Decision Demo                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Create a LocalAuthority (default, offline)
    let authority = LocalAuthority::new();
    println!("Authority: {} (fully offline, no phone-home)\n", authority.name());

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO 1: Single Model Execution - ASR (Speech Recognition)
    // ═══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════════");
    println!("SCENARIO 1: Single Model Execution (ASR - whisper-tiny)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // --- LOW-END DEVICE ---
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ LOW-END DEVICE: Old phone, low battery, poor network           │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let low_end_metrics = DeviceMetrics {
        network_rtt: 350, // High latency (poor connection)
        battery: 12,      // Low battery
        temperature: 42.0, // Running hot
    };

    demo_single_model(&authority, "whisper-tiny", &low_end_metrics);

    // --- HIGH-END DEVICE ---
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ HIGH-END DEVICE: Flagship phone, good battery, fast network    │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let high_end_metrics = DeviceMetrics {
        network_rtt: 45,  // Low latency (fast connection)
        battery: 85,      // Good battery
        temperature: 32.0, // Cool
    };

    demo_single_model(&authority, "whisper-tiny", &high_end_metrics);

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO 2: Pipeline Execution - ASR → LLM → TTS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n\n═══════════════════════════════════════════════════════════════════");
    println!("SCENARIO 2: Pipeline Execution (Voice Assistant)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Pipeline: Audio → ASR → LLM → TTS → Audio\n");

    // --- LOW-END DEVICE ---
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ LOW-END DEVICE: Budget phone, weak connectivity                │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let low_end_metrics = DeviceMetrics {
        network_rtt: 280, // Moderate-high latency
        battery: 25,      // Medium-low battery
        temperature: 38.0,
    };

    demo_pipeline(&authority, &low_end_metrics);

    // --- HIGH-END DEVICE ---
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ HIGH-END DEVICE: Latest flagship, excellent connectivity       │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let high_end_metrics = DeviceMetrics {
        network_rtt: 30,  // Excellent latency
        battery: 92,      // Full battery
        temperature: 28.0, // Cool
    };

    demo_pipeline(&authority, &high_end_metrics);

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO 3: Explicit Target Override
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n\n═══════════════════════════════════════════════════════════════════");
    println!("SCENARIO 3: Explicit Target Override (Privacy-Sensitive)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    demo_explicit_target(&authority, &high_end_metrics);

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO 4: Model Selection
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("SCENARIO 4: Model Selection (with constraints)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    demo_model_selection(&authority);

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                        Demo Complete                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

/// Demo: Single model execution with policy and target resolution
fn demo_single_model(authority: &LocalAuthority, model_id: &str, metrics: &DeviceMetrics) {
    println!("\nDevice Metrics:");
    println!("  • Network RTT: {}ms", metrics.network_rtt);
    println!("  • Battery: {}%", metrics.battery);
    println!("  • Temperature: {}°C", metrics.temperature);

    // Step 1: Apply Policy
    println!("\n[1] Policy Evaluation:");
    let policy_request = PolicyRequest {
        stage_id: "asr".to_string(),
        envelope: Envelope::new(EnvelopeKind::Audio(vec![0u8; 1024])),
        metrics: metrics.clone(),
    };

    let policy_decision = authority.apply_policy(&policy_request);
    print_decision("Policy", &policy_decision.result, &policy_decision);

    // Step 2: Resolve Target
    println!("\n[2] Target Resolution:");
    let stage_context = StageContext {
        stage_id: "asr".to_string(),
        model_id: model_id.to_string(),
        input_kind: EnvelopeKind::Audio(vec![]),
        metrics: metrics.clone(),
        explicit_target: None, // Let authority decide
    };

    let target_decision = authority.resolve_target(&stage_context);
    print_decision("Target", &target_decision.result, &target_decision);

    // Summary
    println!("\n  📊 Summary:");
    let target_str = match &target_decision.result {
        ResolvedTarget::Device => "🏠 ON-DEVICE (local inference)",
        ResolvedTarget::Cloud { .. } => "☁️  CLOUD (remote inference)",
        ResolvedTarget::Server { .. } => "🖥️  SERVER (custom endpoint)",
    };
    println!("     → Execute {} {}", model_id, target_str);
}

/// Demo: Multi-stage pipeline execution
fn demo_pipeline(authority: &LocalAuthority, metrics: &DeviceMetrics) {
    println!("\nDevice Metrics:");
    println!("  • Network RTT: {}ms", metrics.network_rtt);
    println!("  • Battery: {}%", metrics.battery);
    println!("  • Temperature: {}°C", metrics.temperature);

    let stages = vec![
        ("asr", "whisper-tiny", EnvelopeKind::Audio(vec![])),
        ("llm", "qwen2.5-0.5b", EnvelopeKind::Text("transcribed text".to_string())),
        ("tts", "kokoro-82m", EnvelopeKind::Text("response text".to_string())),
    ];

    println!("\n  Stage Routing Decisions:\n");

    for (stage_id, model_id, input_kind) in stages {
        // Policy check
        let policy_request = PolicyRequest {
            stage_id: stage_id.to_string(),
            envelope: Envelope::new(input_kind.clone()),
            metrics: metrics.clone(),
        };
        let policy = authority.apply_policy(&policy_request);

        // Target resolution
        let context = StageContext {
            stage_id: stage_id.to_string(),
            model_id: model_id.to_string(),
            input_kind,
            metrics: metrics.clone(),
            explicit_target: None,
        };
        let target = authority.resolve_target(&context);

        let icon = match &target.result {
            ResolvedTarget::Device => "🏠",
            ResolvedTarget::Cloud { .. } => "☁️",
            ResolvedTarget::Server { .. } => "🖥️",
        };

        let policy_icon = match &policy.result {
            PolicyOutcome::Allow => "✅",
            PolicyOutcome::Deny { .. } => "❌",
            PolicyOutcome::Transform { .. } => "🔄",
        };

        println!(
            "  {} {} ({}) → {} {} [confidence: {:.0}%]",
            policy_icon,
            stage_id.to_uppercase(),
            model_id,
            icon,
            target.result,
            target.confidence * 100.0
        );
        println!("      └─ {}", target.reason);
        println!();
    }
}

/// Demo: Explicit target override for privacy-sensitive data
fn demo_explicit_target(authority: &LocalAuthority, metrics: &DeviceMetrics) {
    println!("Even with excellent network conditions, privacy-sensitive pipelines");
    println!("can force on-device execution via explicit target.\n");

    // Without explicit target (would route to cloud)
    let context_auto = StageContext {
        stage_id: "medical-asr".to_string(),
        model_id: "whisper-tiny".to_string(),
        input_kind: EnvelopeKind::Audio(vec![]),
        metrics: metrics.clone(),
        explicit_target: None, // Auto-routing
    };

    let decision_auto = authority.resolve_target(&context_auto);
    println!("  [AUTO] Medical ASR without explicit target:");
    println!("    → {} ({})", decision_auto.result, decision_auto.reason);

    // With explicit device target (forces local)
    let context_forced = StageContext {
        stage_id: "medical-asr".to_string(),
        model_id: "whisper-tiny".to_string(),
        input_kind: EnvelopeKind::Audio(vec![]),
        metrics: metrics.clone(),
        explicit_target: Some(ExecutionTarget::Device), // Force on-device
    };

    let decision_forced = authority.resolve_target(&context_forced);
    println!("\n  [FORCED] Medical ASR with target=device:");
    println!("    → {} ({})", decision_forced.result, decision_forced.reason);
    println!("\n  📋 HIPAA Compliance: Data never leaves the device!");
}

/// Demo: Model selection with constraints
fn demo_model_selection(authority: &LocalAuthority) {
    // Request without constraints
    let request_unconstrained = ModelRequest {
        model_id: "whisper-tiny".to_string(),
        task: "asr".to_string(),
        constraints: ModelConstraints::default(),
    };

    let selection = authority.select_model(&request_unconstrained);
    println!("  Model Selection (unconstrained):");
    println!("    Model: {}", selection.result.model_id);
    println!("    Source: {:?}", selection.result.source);
    println!("    Reason: {}", selection.reason);

    // Request with size constraint
    let request_constrained = ModelRequest {
        model_id: "qwen2.5-0.5b".to_string(),
        task: "llm".to_string(),
        constraints: ModelConstraints {
            max_size_mb: Some(500),
            required_accuracy: None,
            prefer_quantized: true,
        },
    };

    let selection = authority.select_model(&request_constrained);
    println!("\n  Model Selection (with constraints: max 500MB, prefer quantized):");
    println!("    Model: {}", selection.result.model_id);
    println!("    Source: {:?}", selection.result.source);
    println!("    Reason: {}", selection.reason);
}

/// Helper to print decision details
fn print_decision<T: std::fmt::Debug>(
    _name: &str,
    result: &T,
    decision: &xybrid_core::orchestrator::AuthorityDecision<T>,
) {
    println!("    Result: {:?}", result);
    println!("    Reason: {}", decision.reason);
    println!(
        "    Source: {} | Confidence: {:.0}%",
        decision.source,
        decision.confidence * 100.0
    );
}
