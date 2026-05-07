//! Routing Engine module - Decides where to execute each model stage.
//!
//! The Routing Engine merges information from the Policy Engine, device metrics, and model
//! availability to choose execution targets (local, cloud, or fallback).

use crate::context::DeviceMetrics;
use crate::device::MemoryPressure;
use crate::orchestrator::policy_engine::PolicyResult;
use crate::telemetry::{should_log, Severity};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

const CPU_HISTORY_MAX_STAGES: usize = 256;

/// Target location for model execution.
#[derive(Debug, Clone, PartialEq)]
pub enum RouteTarget {
    Local,
    Cloud,
    Fallback(String),
}

impl RouteTarget {
    /// Convert RouteTarget to a string representation for logging.
    pub fn as_str(&self) -> &str {
        match self {
            RouteTarget::Local => "local",
            RouteTarget::Cloud => "cloud",
            RouteTarget::Fallback(_) => "fallback",
        }
    }

    /// Convert RouteTarget to JSON-compatible string for telemetry.
    pub fn to_json_string(&self) -> String {
        match self {
            RouteTarget::Local => "local".to_string(),
            RouteTarget::Cloud => "cloud".to_string(),
            RouteTarget::Fallback(id) => format!("fallback:{}", id),
        }
    }
}

impl fmt::Display for RouteTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RouteTarget::Local => write!(f, "local"),
            RouteTarget::Cloud => write!(f, "cloud"),
            RouteTarget::Fallback(id) => write!(f, "fallback:{}", id),
        }
    }
}

/// Summary of recent local reliability under similar routing conditions.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LocalReliabilityHint {
    pub recent_abort_rate: f32,
    pub sample_size: u32,
}

impl LocalReliabilityHint {
    pub const EMPTY: Self = Self {
        recent_abort_rate: 0.0,
        sample_size: 0,
    };
}

/// Routing decision for a stage execution.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub stage: String,
    pub target: RouteTarget,
    pub reason: String,
    pub timestamp_ms: u64,
    pub local_reliability_hint: LocalReliabilityHint,
}

impl RoutingDecision {
    /// Convert RoutingDecision to JSON format for telemetry logging.
    ///
    /// Uses serde so that caller-supplied strings (model_id interpolated
    /// into hysteresis/history_bias `reason` fields) are properly escaped.
    /// A raw `format!()` template would corrupt the log line whenever
    /// `reason` or `stage` contained `"`, `\`, or a control character.
    pub fn to_json(&self) -> String {
        #[derive(serde::Serialize)]
        struct LocalReliabilityHintWire {
            recent_abort_rate: f32,
            sample_size: u32,
        }

        #[derive(serde::Serialize)]
        struct RoutingDecisionWire<'a> {
            stage: &'a str,
            target: String,
            reason: &'a str,
            timestamp_ms: u64,
            local_reliability_hint: LocalReliabilityHintWire,
        }

        serde_json::to_string(&RoutingDecisionWire {
            stage: &self.stage,
            target: self.target.to_json_string(),
            reason: &self.reason,
            timestamp_ms: self.timestamp_ms,
            local_reliability_hint: LocalReliabilityHintWire {
                recent_abort_rate: self.local_reliability_hint.recent_abort_rate,
                sample_size: self.local_reliability_hint.sample_size,
            },
        })
        .unwrap_or_else(|_| String::from("{}"))
    }
}

/// Local model availability information.
#[derive(Debug, Clone)]
pub struct LocalAvailability {
    pub local_model_exists: bool,
    // TODO: Add more fields (model size, version, etc.)
}

impl LocalAvailability {
    /// Create a new LocalAvailability instance.
    pub fn new(exists: bool) -> Self {
        Self {
            local_model_exists: exists,
        }
    }
}

/// Routing Engine trait for making routing decisions.
pub trait RoutingEngine {
    /// Decide the execution target for a stage.
    fn decide(
        &mut self,
        stage: &str,
        metrics: &DeviceMetrics,
        policy: &PolicyResult,
        availability: &LocalAvailability,
    ) -> RoutingDecision;

    /// Record feedback about a routing decision's performance.
    fn record_feedback(&mut self, decision: &RoutingDecision, latency_ms: u32);
}

/// Default implementation of RoutingEngine using heuristic-based routing.
///
/// The ladder is *evidence-only cloud, default-local*: we route to cloud only
/// when (a) policy forbids local, (b) the local model isn't available, or (c)
/// a current observation of device stress (memory, sustained CPU, throttle)
/// argues that local will fail. Anything else stays local.
///
/// Order, first match wins:
/// 1. `policy.allowed == false` → Local
/// 2. `!availability.local_model_exists` → Cloud
/// 3. `capabilities.should_throttle()` (battery low or thermal Hot/Critical) → Cloud
/// 4. `resource.memory_pressure == Critical` → Cloud
/// 5. Sustained CPU ≥ 95 % for N samples → Cloud
/// 6. Default → Local
pub struct DefaultRoutingEngine {
    cpu_sustain_samples: usize,
    cpu_sustain_threshold_pct: f32,
    cpu_history: HashMap<String, VecDeque<bool>>,
}

impl DefaultRoutingEngine {
    /// Create a new DefaultRoutingEngine instance.
    pub fn new() -> Self {
        Self {
            cpu_sustain_samples: 2,
            cpu_sustain_threshold_pct: 95.0,
            cpu_history: HashMap::new(),
        }
    }

    /// Create a routing engine with a custom sustained-CPU sample window.
    pub fn with_cpu_sustain_samples(mut self, samples: usize) -> Self {
        self.cpu_sustain_samples = samples.max(1);
        self
    }

    /// Get current timestamp in milliseconds.
    fn current_timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Log routing decision to telemetry (MVP: stdout JSON).
    fn log_decision(&self, decision: &RoutingDecision) {
        // Only log if verbosity is high enough (Info level for routing decisions)
        if should_log(Severity::Info) {
            println!("{}", decision.to_json());
        }
    }

    fn cpu_is_sustained(&mut self, stage: &str, cpu_pct: Option<f32>) -> bool {
        let is_hot = cpu_pct
            .map(|pct| pct >= self.cpu_sustain_threshold_pct)
            .unwrap_or(false);
        if !self.cpu_history.contains_key(stage) && self.cpu_history.len() >= CPU_HISTORY_MAX_STAGES
        {
            if let Some(victim) = self.cpu_history.keys().next().cloned() {
                self.cpu_history.remove(&victim);
            }
        }
        let history = self.cpu_history.entry(stage.to_string()).or_default();
        history.push_back(is_hot);
        while history.len() > self.cpu_sustain_samples {
            history.pop_front();
        }
        history.len() == self.cpu_sustain_samples && history.iter().all(|hot| *hot)
    }

    fn decision(
        stage: &str,
        target: RouteTarget,
        reason: impl Into<String>,
        timestamp_ms: u64,
    ) -> RoutingDecision {
        RoutingDecision {
            stage: stage.to_string(),
            target,
            reason: reason.into(),
            timestamp_ms,
            local_reliability_hint: LocalReliabilityHint::EMPTY,
        }
    }
}

impl Default for DefaultRoutingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingEngine for DefaultRoutingEngine {
    fn decide(
        &mut self,
        stage: &str,
        metrics: &DeviceMetrics,
        policy: &PolicyResult,
        availability: &LocalAvailability,
    ) -> RoutingDecision {
        let timestamp_ms = Self::current_timestamp_ms();

        // Step 1: policy deny always keeps execution local.
        if !policy.allowed {
            let reason = format!(
                "policy_deny: {}",
                policy
                    .reason
                    .as_deref()
                    .unwrap_or("policy denied cloud execution")
            );
            let decision = Self::decision(stage, RouteTarget::Local, reason, timestamp_ms);
            self.log_decision(&decision);
            return decision;
        }

        // Step 2: if the local model is absent, cloud is the only usable target.
        if !availability.local_model_exists {
            let decision = Self::decision(
                stage,
                RouteTarget::Cloud,
                "model_unavailable: local model not found",
                timestamp_ms,
            );
            self.log_decision(&decision);
            return decision;
        }

        // Step 3: route stressed devices to cloud when policy allows it.
        if metrics.capabilities.should_throttle() {
            let reason = format!(
                "stress_throttle: battery {}%, thermal {:?}",
                metrics.capabilities.battery_level(),
                metrics.capabilities.thermal_state()
            );
            let decision = Self::decision(stage, RouteTarget::Cloud, reason, timestamp_ms);
            self.log_decision(&decision);
            return decision;
        }

        if metrics.resource.memory_pressure == MemoryPressure::Critical {
            let reason = "stress_memory: memory pressure critical".to_string();
            let decision = Self::decision(stage, RouteTarget::Cloud, reason, timestamp_ms);
            self.log_decision(&decision);
            return decision;
        }

        if self.cpu_is_sustained(stage, metrics.resource.cpu_pct) {
            let reason = format!(
                "stress_cpu_sustained: CPU >= {:.0}% for {} samples",
                self.cpu_sustain_threshold_pct, self.cpu_sustain_samples
            );
            let decision = Self::decision(stage, RouteTarget::Cloud, reason, timestamp_ms);
            self.log_decision(&decision);
            return decision;
        }

        // Default: prefer local. Cloud is opt-in via the rules above; we no
        // longer speculate about "optimal conditions" or accelerator presence.
        let decision = Self::decision(stage, RouteTarget::Local, "default_local", timestamp_ms);
        self.log_decision(&decision);
        decision
    }

    fn record_feedback(&mut self, _decision: &RoutingDecision, _latency_ms: u32) {
        // MVP: Feedback tracking is a no-op
        // TODO: Implement feedback tracking for adaptive routing
    }
}

#[cfg(test)]
mod tests {
    use super::super::policy_engine::PolicyResult;
    use super::*;
    use crate::device::{HardwareCapabilities, MemoryPressure, ResourceSnapshot, ThermalState};

    fn metrics_with_live_state(
        battery: u8,
        thermal_state: ThermalState,
        memory_pressure: MemoryPressure,
        cpu_pct: Option<f32>,
    ) -> DeviceMetrics {
        let capabilities = HardwareCapabilities {
            battery_level: battery,
            thermal_state,
            ..Default::default()
        };

        let mut resource = ResourceSnapshot::unknown();
        resource.memory_pressure = memory_pressure;
        resource.cpu_pct = cpu_pct;
        resource.thermal_state = thermal_state;
        resource.battery_pct = Some(battery);

        DeviceMetrics {
            capabilities,
            resource,
        }
    }

    #[test]
    fn test_policy_deny_routes_local() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics = DeviceMetrics::default();
        let policy = PolicyResult::deny("test policy denial".to_string());
        let availability = LocalAvailability::new(true);

        let decision = engine.decide("test_stage", &metrics, &policy, &availability);

        assert_eq!(decision.target, RouteTarget::Local);
        assert!(decision.reason.contains("policy_deny"));
        assert_eq!(decision.stage, "test_stage");
    }

    #[test]
    fn stress_throttle_routes_cloud() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics =
            metrics_with_live_state(10, ThermalState::Normal, MemoryPressure::Normal, None);
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        let decision = engine.decide("test_stage", &metrics, &policy, &availability);

        assert_eq!(decision.target, RouteTarget::Cloud);
        assert!(decision.reason.contains("stress_throttle"));
    }

    #[test]
    fn critical_memory_routes_cloud() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics =
            metrics_with_live_state(50, ThermalState::Normal, MemoryPressure::Critical, None);
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        let decision = engine.decide("test_stage", &metrics, &policy, &availability);

        assert_eq!(decision.target, RouteTarget::Cloud);
        assert!(decision.reason.contains("stress_memory"));
    }

    #[test]
    fn warm_but_not_stressed_conditions_do_not_fire_stress_branch() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics = metrics_with_live_state(25, ThermalState::Normal, MemoryPressure::Warn, None);
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        let decision = engine.decide("test_stage", &metrics, &policy, &availability);

        assert!(!decision.reason.contains("stress_"));
    }

    #[test]
    fn sustained_cpu_routes_cloud_on_second_hot_sample() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics =
            metrics_with_live_state(50, ThermalState::Normal, MemoryPressure::Normal, Some(96.0));
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        let first = engine.decide("test_stage", &metrics, &policy, &availability);
        let second = engine.decide("test_stage", &metrics, &policy, &availability);

        assert!(!first.reason.contains("stress_cpu_sustained"));
        assert_eq!(second.target, RouteTarget::Cloud);
        assert!(second.reason.contains("stress_cpu_sustained"));
    }

    #[test]
    fn model_unavailable_overrides_stress() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics =
            metrics_with_live_state(10, ThermalState::Hot, MemoryPressure::Critical, Some(99.0));
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(false);

        let decision = engine.decide("test_stage", &metrics, &policy, &availability);

        assert_eq!(decision.target, RouteTarget::Cloud);
        assert!(decision.reason.contains("model_unavailable"));
    }

    #[test]
    fn sustained_cpu_history_is_stage_scoped() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics =
            metrics_with_live_state(50, ThermalState::Normal, MemoryPressure::Normal, Some(96.0));
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        let stage_a_first = engine.decide("stage_a", &metrics, &policy, &availability);
        let stage_b_first = engine.decide("stage_b", &metrics, &policy, &availability);
        let stage_a_second = engine.decide("stage_a", &metrics, &policy, &availability);

        assert!(!stage_a_first.reason.contains("stress_cpu_sustained"));
        assert!(!stage_b_first.reason.contains("stress_cpu_sustained"));
        assert!(stage_a_second.reason.contains("stress_cpu_sustained"));
    }

    #[test]
    fn sustained_cpu_history_map_stays_bounded() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics =
            metrics_with_live_state(50, ThermalState::Normal, MemoryPressure::Normal, Some(96.0));
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        for idx in 0..(CPU_HISTORY_MAX_STAGES + 32) {
            let _ = engine.decide(&format!("stage-{idx}"), &metrics, &policy, &availability);
        }

        assert!(
            engine.cpu_history.len() <= CPU_HISTORY_MAX_STAGES,
            "CPU history should stay bounded"
        );
    }

    #[test]
    fn sustained_cpu_resets_when_sample_drops_below_threshold() {
        let mut engine = DefaultRoutingEngine::new();
        let hot =
            metrics_with_live_state(50, ThermalState::Normal, MemoryPressure::Normal, Some(96.0));
        let cool =
            metrics_with_live_state(50, ThermalState::Normal, MemoryPressure::Normal, Some(20.0));
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        let _ = engine.decide("test_stage", &hot, &policy, &availability);
        let _ = engine.decide("test_stage", &cool, &policy, &availability);
        let decision = engine.decide("test_stage", &hot, &policy, &availability);

        assert!(!decision.reason.contains("stress_cpu_sustained"));
    }

    #[test]
    fn test_missing_model_routes_cloud() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics = DeviceMetrics::default();
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(false);

        let decision = engine.decide("test_stage", &metrics, &policy, &availability);

        assert_eq!(decision.target, RouteTarget::Cloud);
        assert!(decision.reason.contains("model_unavailable"));
    }

    #[test]
    fn default_unstressed_conditions_route_local() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics =
            metrics_with_live_state(80, ThermalState::Normal, MemoryPressure::Normal, Some(20.0));
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        let decision = engine.decide("test_stage", &metrics, &policy, &availability);

        assert_eq!(decision.target, RouteTarget::Local);
        assert!(decision.reason.contains("default_local"));
    }

    #[test]
    fn test_routing_decision_json_format() {
        let decision = RoutingDecision {
            stage: "motivator".to_string(),
            target: RouteTarget::Cloud,
            reason: "low network latency (110ms)".to_string(),
            timestamp_ms: 1730559412312,
            local_reliability_hint: LocalReliabilityHint::EMPTY,
        };

        let json = decision.to_json();
        assert!(json.contains("\"stage\":\"motivator\""));
        assert!(json.contains("\"target\":\"cloud\""));
        assert!(json.contains("\"reason\":\"low network latency (110ms)\""));
        assert!(json.contains("\"timestamp_ms\":1730559412312"));
    }

    #[test]
    fn routing_decision_json_escapes_special_characters_in_reason() {
        // Pre-fix, the hand-rolled format!() template would emit a quote
        // verbatim, producing a malformed JSON line whenever a model_id
        // interpolated into the reason carried `"`, `\`, or a control char.
        let decision = RoutingDecision {
            stage: "stage-1".to_string(),
            target: RouteTarget::Cloud,
            reason: r#"hysteresis: recent local abort for model 'weird-"model' (stress_memory)"#
                .to_string(),
            timestamp_ms: 1730559412312,
            local_reliability_hint: LocalReliabilityHint::EMPTY,
        };

        let json = decision.to_json();
        // Must be parseable JSON post-fix.
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("to_json output must be valid JSON");
        assert_eq!(parsed["stage"], "stage-1");
        assert_eq!(
            parsed["reason"],
            r#"hysteresis: recent local abort for model 'weird-"model' (stress_memory)"#
        );
    }

    #[test]
    fn test_route_target_to_json_string() {
        assert_eq!(RouteTarget::Local.to_json_string(), "local");
        assert_eq!(RouteTarget::Cloud.to_json_string(), "cloud");
        assert_eq!(
            RouteTarget::Fallback("model_v2".to_string()).to_json_string(),
            "fallback:model_v2"
        );
    }

    #[test]
    fn test_route_target_display() {
        assert_eq!(format!("{}", RouteTarget::Local), "local");
        assert_eq!(format!("{}", RouteTarget::Cloud), "cloud");
        assert_eq!(
            format!("{}", RouteTarget::Fallback("model_v2".to_string())),
            "fallback:model_v2"
        );

        // Verify Display works with to_string()
        assert_eq!(RouteTarget::Local.to_string(), "local");
        assert_eq!(RouteTarget::Cloud.to_string(), "cloud");
        assert_eq!(
            RouteTarget::Fallback("backup".to_string()).to_string(),
            "fallback:backup"
        );
    }

    #[test]
    fn boundary_battery_throttle_threshold() {
        let mut engine = DefaultRoutingEngine::new();
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        // Battery 19% with normal thermal: should_throttle() returns true
        // (HardwareCapabilities throttles on battery < 20).
        let metrics =
            metrics_with_live_state(19, ThermalState::Normal, MemoryPressure::Normal, None);
        let decision = engine.decide("test_stage", &metrics, &policy, &availability);
        assert_eq!(decision.target, RouteTarget::Cloud);
        assert!(decision.reason.contains("stress_throttle"));

        // Battery 20% with normal thermal: should_throttle() returns false,
        // and with no other stress signals we land on default_local.
        let metrics =
            metrics_with_live_state(20, ThermalState::Normal, MemoryPressure::Normal, None);
        let decision = engine.decide("test_stage", &metrics, &policy, &availability);
        assert_eq!(decision.target, RouteTarget::Local);
    }
}
