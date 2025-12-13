//! Hiiipe Demo Example
//!
//! This example demonstrates the complete Hiiipe workflow:
//! Mic Input (AudioRaw) → Local ASR → Cloud Motivator → Local TTS
//!
//! The Hiiipe pipeline is a motivational voice companion — a "hype-man" for your workout.
//! It demonstrates hybrid routing, privacy-aware inference, and low latency edge performance.
//!
//! Run with: `cargo run --example hiiipe`
//!
//! This example simulates:
//! - ASR (Speech-to-Text): Local if available (low latency)
//! - Motivator (LLM): Cloud (heavy reasoning, emotional tone generation)
//! - TTS (Text-to-Speech): Local (low latency playback)
//!
//! Policy: Sensitive audio never leaves device. Text may be routed to cloud LLM.
//! Latency optimization: prefer local where possible.

use xybrid_core::context::{DeviceMetrics, Envelope, EnvelopeKind, StageDescriptor};
use xybrid_core::event_bus::OrchestratorEvent;
use xybrid_core::orchestrator::Orchestrator;
use xybrid_core::routing_engine::LocalAvailability;

fn describe_kind(kind: &EnvelopeKind) -> String {
    match kind {
        EnvelopeKind::Audio(_) => "Audio<bytes>".to_string(),
        EnvelopeKind::Text(text) => format!("Text<{}>", text),
        EnvelopeKind::Embedding(vec) => format!("Embedding<len={}>", vec.len()),
    }
}

fn main() {
    println!("🎤 Hiiipe Demo: Hybrid AI Inference Pipeline");
    println!("A motivational voice companion for your workout\n");
    println!("{}", "=".repeat(70));
    println!();

    // Create orchestrator with default components
    let mut orchestrator = Orchestrator::new();

    // Subscribe to events for user-friendly output
    let event_bus = orchestrator.event_bus();
    let subscription = event_bus.subscribe();

    // Start event listener in background for formatted output
    let event_handle = std::thread::spawn(move || {
        loop {
            match subscription.recv() {
                Ok(event) => {
                    match event {
                        OrchestratorEvent::PipelineStart { stages } => {
                            println!("📋 Pipeline Starting");
                            println!("   Stages: {}", stages.join(" → "));
                            println!();
                        }
                        OrchestratorEvent::StageStart { stage_name } => {
                            println!("▶️  Processing: {}", stage_name);
                        }
                        OrchestratorEvent::PolicyEvaluated {
                            stage_name: _,
                            allowed,
                            reason,
                        } => {
                            let status = if allowed { "✓ ALLOWED" } else { "✗ DENIED" };
                            if let Some(ref reason_str) = reason {
                                println!("   📜 Policy: {} ({})", status, reason_str);
                            } else {
                                println!("   📜 Policy: {}", status);
                            }
                        }
                        OrchestratorEvent::RoutingDecided {
                            stage_name,
                            target,
                            reason,
                        } => {
                            println!("   🎯 Routing: {} → {} ({})", stage_name, target, reason);
                        }
                        OrchestratorEvent::ExecutionStarted {
                            stage_name: _,
                            target,
                        } => {
                            println!("   ⚙️  Executing on: {}", target);
                        }
                        OrchestratorEvent::ExecutionCompleted {
                            stage_name: _,
                            target,
                            execution_time_ms,
                        } => {
                            println!(
                                "   ✓ Execution complete on {} ({}ms)",
                                target, execution_time_ms
                            );
                        }
                        OrchestratorEvent::StageComplete {
                            stage_name,
                            target,
                            latency_ms,
                        } => {
                            println!(
                                "   ✅ Stage complete: {} on {} (total: {}ms)",
                                stage_name, target, latency_ms
                            );
                            println!();
                        }
                        OrchestratorEvent::PipelineComplete { total_latency_ms } => {
                            println!("{}", "=".repeat(70));
                            println!(
                                "🎉 Pipeline Complete! Total latency: {}ms",
                                total_latency_ms
                            );
                            println!();
                            break; // exit event loop gracefully
                        }
                        _ => {}
                    }
                }
                Err(_) => break,
            }
        }
    });

    // Define the Hiiipe pipeline stages (simplified names as per spec)
    let stages = vec![
        StageDescriptor::new("asr"),       // ASR (Speech-to-Text)
        StageDescriptor::new("motivator"), // Motivator (LLM)
        StageDescriptor::new("tts"),       // TTS (Text-to-Speech)
    ];

    // Simulate mic input (raw audio)
    println!("📥 Input: Raw audio from microphone (AudioRaw)");
    let mic_input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 1600]));

    // Simulate device metrics
    println!("📊 Device Metrics:");
    let metrics = DeviceMetrics {
        network_rtt: 120,  // Moderate network latency
        battery: 70,       // Good battery
        temperature: 25.0, // Normal temperature
    };
    println!("   Network RTT: {}ms", metrics.network_rtt);
    println!("   Battery: {}%", metrics.battery);
    println!("   Temperature: {}°C", metrics.temperature);
    println!();

    // Define model availability (matching Hiiipe demo spec)
    // ASR: local preferred, TTS: local preferred, Motivator: cloud only
    println!("📦 Model Availability:");
    let availability_fn = |stage: &str| -> LocalAvailability {
        match stage {
            "asr" => {
                println!("   ✅ ASR available locally");
                LocalAvailability::new(true)
            }
            "tts" => {
                println!("   ✅ TTS available locally");
                LocalAvailability::new(true)
            }
            "motivator" => {
                println!("   ❌ Motivator only available in cloud");
                LocalAvailability::new(false)
            }
            _ => {
                println!("   ❌ Unknown stage: {}", stage);
                LocalAvailability::new(false)
            }
        }
    };
    println!();

    // Execute the pipeline
    println!("🚀 Executing Pipeline...");
    println!("{}", "=".repeat(70));
    println!();
    println!("📊 Telemetry Logs (JSON format):");
    println!();

    match orchestrator.execute_pipeline(&stages, &mic_input, &metrics, &availability_fn) {
        Ok(results) => {
            // Print summary results
            println!();
            println!("{}", "=".repeat(70));
            println!("📊 Pipeline Results Summary:");
            println!("{}", "=".repeat(70));

            for (i, result) in results.iter().enumerate() {
                println!("\nStage {}: {}", i + 1, result.stage);
                let input_desc = if i == 0 {
                    "AudioRaw".to_string()
                } else {
                    describe_kind(&results[i - 1].output.kind)
                };
                println!("   Input:  {}", input_desc);
                println!("   Output: {}", describe_kind(&result.output.kind));
                println!("   Routing: {}", result.routing_decision.target);
                println!("   Reason: {}", result.routing_decision.reason);
                println!("   Latency: {}ms", result.latency_ms);
            }

            println!();
            println!("{}", "=".repeat(70));
            println!("✨ Demo completed successfully!");
            println!();
            println!("💡 This demo validates:");
            println!("   • Orchestrator pipeline sequencing");
            println!("   • Policy + routing integration");
            println!("   • Local/cloud hybrid execution");
            println!("   • EventBus + Telemetry output");
        }
        Err(e) => {
            eprintln!("❌ Pipeline execution failed: {}", e);
            std::process::exit(1);
        }
    }

    // Wait for event thread to complete (it will exit gracefully on PipelineComplete)
    let _ = event_handle.join();
}
