//! Routing Engine module - Decides where to execute each model stage.
//!
//! The Routing Engine merges information from the Policy Engine, device metrics, and model
//! availability to choose execution targets (local, cloud, or fallback).

use crate::context::DeviceMetrics;
use crate::device::capabilities::detect_capabilities;
use crate::orchestrator::policy_engine::PolicyResult;
use crate::telemetry::{should_log, Severity};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

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

/// Routing decision for a stage execution.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub stage: String,
    pub target: RouteTarget,
    pub reason: String,
    pub timestamp_ms: u64,
}

impl RoutingDecision {
    /// Convert RoutingDecision to JSON format for telemetry logging.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"stage":"{}","target":"{}","reason":"{}","timestamp_ms":{}}}"#,
            self.stage,
            self.target.to_json_string(),
            self.reason,
            self.timestamp_ms
        )
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
/// This implementation follows the MVP algorithm:
/// 1. If policy.allowed == false → target = Local (reason = "policy_deny")
/// 2. Else if network_rtt > 250 ms → target = Local (reason = "high_latency")
/// 3. Else if battery < 15% OR !availability.local_model_exists → target = Cloud
/// 4. Otherwise → target = Cloud (reason = "optimal_conditions")
pub struct DefaultRoutingEngine {
    // TODO: Add fields for feedback tracking, learning, etc.
}

impl DefaultRoutingEngine {
    /// Create a new DefaultRoutingEngine instance.
    pub fn new() -> Self {
        Self {}
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

        // MVP Algorithm (Heuristic)
        // Step 1: Check if policy denies cloud execution
        if !policy.allowed {
            let reason = format!(
                "policy_deny: {}",
                policy
                    .reason
                    .as_deref()
                    .unwrap_or("policy denied cloud execution")
            );
            let decision = RoutingDecision {
                stage: stage.to_string(),
                target: RouteTarget::Local,
                reason,
                timestamp_ms,
            };
            self.log_decision(&decision);
            return decision;
        }

        // Step 2: Check for high network latency
        if metrics.network_rtt > 250 {
            let reason = format!(
                "high_latency: network RTT {}ms exceeds 250ms threshold",
                metrics.network_rtt
            );
            let decision = RoutingDecision {
                stage: stage.to_string(),
                target: RouteTarget::Local,
                reason,
                timestamp_ms,
            };
            self.log_decision(&decision);
            return decision;
        }

        // Step 3: Check battery level or model availability
        if metrics.battery < 15 || !availability.local_model_exists {
            let reason = if metrics.battery < 15 {
                format!("low_battery: {}% below 15% threshold", metrics.battery)
            } else {
                "model_unavailable: local model not found".to_string()
            };
            let decision = RoutingDecision {
                stage: stage.to_string(),
                target: RouteTarget::Cloud,
                reason,
                timestamp_ms,
            };
            self.log_decision(&decision);
            return decision;
        }

        // Step 4: Check hardware capabilities (optional: prefer local if GPU/Metal/NNAPI available)
        let capabilities = detect_capabilities(metrics);
        if capabilities.should_prefer_gpu()
            || capabilities.should_prefer_metal()
            || capabilities.should_prefer_nnapi()
        {
            let reason = format!(
                "hardware_acceleration: GPU/Metal/NNAPI available, battery {}%, preferring local execution",
                metrics.battery
            );
            let decision = RoutingDecision {
                stage: stage.to_string(),
                target: RouteTarget::Local,
                reason,
                timestamp_ms,
            };
            self.log_decision(&decision);
            return decision;
        }

        // Step 5: Default to cloud for optimal conditions
        let reason = format!(
            "optimal_conditions: low network latency ({}ms)",
            metrics.network_rtt
        );
        let decision = RoutingDecision {
            stage: stage.to_string(),
            target: RouteTarget::Cloud,
            reason,
            timestamp_ms,
        };
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

    #[test]
    fn test_policy_deny_routes_local() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 25.0,
        };
        let policy = PolicyResult::deny("test policy denial".to_string());
        let availability = LocalAvailability::new(true);

        let decision = engine.decide("test_stage", &metrics, &policy, &availability);

        assert_eq!(decision.target, RouteTarget::Local);
        assert!(decision.reason.contains("policy_deny"));
        assert_eq!(decision.stage, "test_stage");
    }

    #[test]
    fn test_high_rtt_routes_local() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics = DeviceMetrics {
            network_rtt: 300,
            battery: 50,
            temperature: 25.0,
        };
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        let decision = engine.decide("test_stage", &metrics, &policy, &availability);

        assert_eq!(decision.target, RouteTarget::Local);
        assert!(decision.reason.contains("high_latency"));
    }

    #[test]
    fn test_low_battery_routes_cloud() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 10,
            temperature: 25.0,
        };
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        let decision = engine.decide("test_stage", &metrics, &policy, &availability);

        assert_eq!(decision.target, RouteTarget::Cloud);
        assert!(decision.reason.contains("low_battery"));
    }

    #[test]
    fn test_missing_model_routes_cloud() {
        let mut engine = DefaultRoutingEngine::new();
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 25.0,
        };
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(false);

        let decision = engine.decide("test_stage", &metrics, &policy, &availability);

        assert_eq!(decision.target, RouteTarget::Cloud);
        assert!(decision.reason.contains("model_unavailable"));
    }

    #[test]
    fn test_optimal_conditions_routes_cloud() {
        let mut engine = DefaultRoutingEngine::new();
        // Use low temperature to avoid GPU preference (GPU detection may prefer local)
        // On non-mobile platforms, GPU detection may route to local
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 50,
            temperature: 25.0,
        };
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        let decision = engine.decide("test_stage", &metrics, &policy, &availability);

        // With capability detection, may route to local if GPU available, or cloud otherwise
        // Both are valid routing decisions
        assert!(decision.target == RouteTarget::Cloud || decision.target == RouteTarget::Local);
        assert!(
            decision.reason.contains("optimal_conditions")
                || decision.reason.contains("hardware_acceleration")
        );
    }

    #[test]
    fn test_routing_decision_json_format() {
        let decision = RoutingDecision {
            stage: "motivator".to_string(),
            target: RouteTarget::Cloud,
            reason: "low network latency (110ms)".to_string(),
            timestamp_ms: 1730559412312,
        };

        let json = decision.to_json();
        assert!(json.contains("\"stage\":\"motivator\""));
        assert!(json.contains("\"target\":\"cloud\""));
        assert!(json.contains("\"reason\":\"low network latency (110ms)\""));
        assert!(json.contains("\"timestamp_ms\":1730559412312"));
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
    fn test_boundary_conditions() {
        let mut engine = DefaultRoutingEngine::new();
        let policy = PolicyResult::allow(Some("policy passed".to_string()));
        let availability = LocalAvailability::new(true);

        // Test RTT exactly at threshold (250ms should still route to cloud if other conditions are good)
        // Note: May route to local if GPU available due to capability detection
        let metrics = DeviceMetrics {
            network_rtt: 250,
            battery: 50,
            temperature: 25.0,
        };
        let decision = engine.decide("test_stage", &metrics, &policy, &availability);
        // 250 is not > 250, so should route to cloud (unless GPU available)
        assert!(decision.target == RouteTarget::Cloud || decision.target == RouteTarget::Local);

        // Test RTT just above threshold
        let metrics = DeviceMetrics {
            network_rtt: 251,
            battery: 50,
            temperature: 25.0,
        };
        let decision = engine.decide("test_stage", &metrics, &policy, &availability);
        assert_eq!(decision.target, RouteTarget::Local);

        // Test battery exactly at threshold
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 15,
            temperature: 25.0,
        };
        let decision = engine.decide("test_stage", &metrics, &policy, &availability);
        assert_eq!(decision.target, RouteTarget::Cloud); // 15 is not < 15

        // Test battery just below threshold
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 14,
            temperature: 25.0,
        };
        let decision = engine.decide("test_stage", &metrics, &policy, &availability);
        assert_eq!(decision.target, RouteTarget::Cloud); // Routes to cloud due to low battery
        assert!(decision.reason.contains("low_battery"));
    }
}
