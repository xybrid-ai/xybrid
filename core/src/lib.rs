//! Xybrid Core - The Rust orchestrator runtime for hybrid cloud-edge AI inference.

pub mod audio;
pub mod bundler;
pub mod context;
pub mod control_sync;
pub mod device;
pub mod device_adapter;
pub mod event_bus;
pub mod execution_template;
pub mod executor;
pub mod ir;
pub mod orchestrator;
pub mod phonemizer;
pub mod pipeline;
pub mod policy_engine;
pub mod preprocessing;
pub mod registry;
pub mod registry_config;
pub mod registry_index;
pub mod registry_remote;
pub mod registry_resolver;
pub mod routing_engine;
pub mod runtime_adapter;
pub mod stage_resolver;
pub mod stream_manager;
pub mod streaming;
pub mod target;
pub mod telemetry;
pub mod template_executor;

// Universal Architecture System - configuration structures for ML models
pub mod universal;

// High-level client APIs (abstract over execution source)
pub mod llm;
pub mod tts;

// Gateway API types (OpenAI-compatible)
pub mod gateway;

// Internal provider implementations (not for direct use)
pub(crate) mod cloud_llm;
