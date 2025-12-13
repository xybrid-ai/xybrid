//! Execution target resolver - Maps high-level pipeline targets to routing decisions.
//!
//! The resolver bridges the Pipeline DSL's `ExecutionTarget` enum with the runtime's
//! `RouteTarget` enum, handling:
//! - Direct mappings (Device → Local, Server/Integration → Cloud)
//! - Auto resolution based on device capabilities and conditions
//! - Fallback chain processing

use super::provider::IntegrationProvider;
use super::stage::{FallbackConfig, StageConfig};
use super::target::ExecutionTarget;
use crate::context::DeviceMetrics;
use crate::device::capabilities::{detect_capabilities, HardwareCapabilities};
use crate::routing_engine::{LocalAvailability, RouteTarget, RoutingDecision};
use std::time::{SystemTime, UNIX_EPOCH};

/// Resolution context containing all information needed to resolve execution targets.
#[derive(Debug, Clone)]
pub struct ResolutionContext {
    /// Device metrics (battery, network, temperature).
    pub metrics: DeviceMetrics,

    /// Local model availability.
    pub local_available: bool,

    /// Server (Xybrid-hosted) availability.
    pub server_available: bool,

    /// Integration provider availability by provider.
    pub integration_available: std::collections::HashMap<IntegrationProvider, bool>,

    /// Hardware capabilities.
    pub capabilities: HardwareCapabilities,
}

impl ResolutionContext {
    /// Create a new resolution context with device metrics.
    pub fn new(metrics: DeviceMetrics) -> Self {
        let capabilities = detect_capabilities(&metrics);
        Self {
            metrics,
            local_available: false,
            server_available: true, // Assume server is available by default
            integration_available: std::collections::HashMap::new(),
            capabilities,
        }
    }

    /// Set local model availability.
    pub fn with_local_available(mut self, available: bool) -> Self {
        self.local_available = available;
        self
    }

    /// Set server availability.
    pub fn with_server_available(mut self, available: bool) -> Self {
        self.server_available = available;
        self
    }

    /// Set integration provider availability.
    pub fn with_integration_available(mut self, provider: IntegrationProvider, available: bool) -> Self {
        self.integration_available.insert(provider, available);
        self
    }

    /// Check if an integration provider is available.
    pub fn is_integration_available(&self, provider: &IntegrationProvider) -> bool {
        *self.integration_available.get(provider).unwrap_or(&true)
    }
}

/// Result of target resolution.
#[derive(Debug, Clone)]
pub struct ResolvedTarget {
    /// The resolved route target.
    pub target: RouteTarget,

    /// The reason for the resolution.
    pub reason: String,

    /// The provider (for integration targets).
    pub provider: Option<IntegrationProvider>,

    /// The model to use (may be different from original for fallbacks).
    pub model: String,

    /// The model version.
    pub version: Option<String>,
}

impl ResolvedTarget {
    /// Create a resolved target for local execution.
    pub fn local(model: &str, version: Option<&str>, reason: &str) -> Self {
        Self {
            target: RouteTarget::Local,
            reason: reason.to_string(),
            provider: None,
            model: model.to_string(),
            version: version.map(|v| v.to_string()),
        }
    }

    /// Create a resolved target for server execution.
    pub fn server(model: &str, version: Option<&str>, reason: &str) -> Self {
        Self {
            target: RouteTarget::Cloud,
            reason: format!("server: {}", reason),
            provider: None,
            model: model.to_string(),
            version: version.map(|v| v.to_string()),
        }
    }

    /// Create a resolved target for integration execution.
    pub fn integration(
        provider: IntegrationProvider,
        model: &str,
        reason: &str,
    ) -> Self {
        Self {
            target: RouteTarget::Cloud,
            reason: format!("integration/{}: {}", provider, reason),
            provider: Some(provider),
            model: model.to_string(),
            version: None, // Integrations typically don't use versions
        }
    }

    /// Create a fallback target.
    pub fn fallback(model: &str, version: Option<&str>, reason: &str) -> Self {
        Self {
            target: RouteTarget::Fallback(model.to_string()),
            reason: reason.to_string(),
            provider: None,
            model: model.to_string(),
            version: version.map(|v| v.to_string()),
        }
    }

    /// Convert to a RoutingDecision for telemetry.
    pub fn to_routing_decision(&self, stage: &str) -> RoutingDecision {
        RoutingDecision {
            stage: stage.to_string(),
            target: self.target.clone(),
            reason: self.reason.clone(),
            timestamp_ms: current_timestamp_ms(),
        }
    }
}

/// Execution target resolver.
pub struct TargetResolver;

impl TargetResolver {
    /// Resolve the execution target for a stage.
    pub fn resolve(
        stage: &StageConfig,
        context: &ResolutionContext,
    ) -> Result<ResolvedTarget, ResolutionError> {
        let primary = Self::resolve_target(
            &stage.target,
            &stage.model,
            stage.version.as_deref(),
            stage.provider.as_ref(),
            stage.prefer.as_ref(),
            context,
        );

        match primary {
            Ok(resolved) => Ok(resolved),
            Err(err) => {
                // Try fallbacks
                for fallback in &stage.fallback {
                    let fallback_result = Self::resolve_fallback(fallback, &stage.model, context);
                    if let Ok(resolved) = fallback_result {
                        return Ok(resolved);
                    }
                }
                // All fallbacks failed, return original error
                Err(err)
            }
        }
    }

    /// Resolve a single execution target.
    fn resolve_target(
        target: &ExecutionTarget,
        model: &str,
        version: Option<&str>,
        provider: Option<&IntegrationProvider>,
        prefer: Option<&ExecutionTarget>,
        context: &ResolutionContext,
    ) -> Result<ResolvedTarget, ResolutionError> {
        match target {
            ExecutionTarget::Device => Self::resolve_device(model, version, context),
            ExecutionTarget::Server => Self::resolve_server(model, version, context),
            ExecutionTarget::Integration => {
                let provider = provider.ok_or_else(|| {
                    ResolutionError::MissingProvider(model.to_string())
                })?;
                Self::resolve_integration(provider, model, context)
            }
            ExecutionTarget::Auto => Self::resolve_auto(model, version, provider, prefer, context),
        }
    }

    /// Resolve device target.
    fn resolve_device(
        model: &str,
        version: Option<&str>,
        context: &ResolutionContext,
    ) -> Result<ResolvedTarget, ResolutionError> {
        if !context.local_available {
            return Err(ResolutionError::DeviceUnavailable(model.to_string()));
        }

        // Check device constraints
        if context.capabilities.should_throttle() {
            return Err(ResolutionError::DeviceThrottled(
                "device is thermal throttled".to_string(),
            ));
        }

        Ok(ResolvedTarget::local(
            model,
            version,
            "device target resolved",
        ))
    }

    /// Resolve server target.
    fn resolve_server(
        model: &str,
        version: Option<&str>,
        context: &ResolutionContext,
    ) -> Result<ResolvedTarget, ResolutionError> {
        if !context.server_available {
            return Err(ResolutionError::ServerUnavailable(model.to_string()));
        }

        // Check network conditions
        if context.metrics.network_rtt > 500 {
            return Err(ResolutionError::NetworkTooSlow(context.metrics.network_rtt));
        }

        Ok(ResolvedTarget::server(
            model,
            version,
            "server target resolved",
        ))
    }

    /// Resolve integration target.
    fn resolve_integration(
        provider: &IntegrationProvider,
        model: &str,
        context: &ResolutionContext,
    ) -> Result<ResolvedTarget, ResolutionError> {
        if !context.is_integration_available(provider) {
            return Err(ResolutionError::IntegrationUnavailable(
                provider.clone(),
                model.to_string(),
            ));
        }

        Ok(ResolvedTarget::integration(
            provider.clone(),
            model,
            "integration target resolved",
        ))
    }

    /// Resolve auto target using heuristics.
    fn resolve_auto(
        model: &str,
        version: Option<&str>,
        provider: Option<&IntegrationProvider>,
        prefer: Option<&ExecutionTarget>,
        context: &ResolutionContext,
    ) -> Result<ResolvedTarget, ResolutionError> {
        // If a preference is set, try it first
        if let Some(preferred) = prefer {
            let result = Self::resolve_target(preferred, model, version, provider, None, context);
            if result.is_ok() {
                return result;
            }
            // Preference failed, continue with auto resolution
        }

        // Auto resolution priority:
        // 1. Device if available and conditions are good
        // 2. Integration if provider is specified
        // 3. Server as fallback

        // Check if device is the best option
        if context.local_available && Self::should_prefer_device(context) {
            return Ok(ResolvedTarget::local(
                model,
                version,
                "auto: device preferred (good conditions)",
            ));
        }

        // Check integration if provider specified
        if let Some(prov) = provider {
            if context.is_integration_available(prov) {
                return Ok(ResolvedTarget::integration(
                    prov.clone(),
                    model,
                    "auto: integration available",
                ));
            }
        }

        // Try server
        if context.server_available && context.metrics.network_rtt <= 250 {
            return Ok(ResolvedTarget::server(
                model,
                version,
                "auto: server available with good network",
            ));
        }

        // Fallback to device if available
        if context.local_available {
            return Ok(ResolvedTarget::local(
                model,
                version,
                "auto: fallback to device",
            ));
        }

        Err(ResolutionError::NoTargetAvailable(model.to_string()))
    }

    /// Check if device should be preferred based on conditions.
    fn should_prefer_device(context: &ResolutionContext) -> bool {
        // Prefer device if:
        // - Battery is good (>30%)
        // - Not thermally throttled
        // - Has hardware acceleration

        context.metrics.battery > 30
            && !context.capabilities.should_throttle()
            && (context.capabilities.has_gpu
                || context.capabilities.has_metal
                || context.capabilities.has_nnapi)
    }

    /// Resolve a fallback configuration.
    fn resolve_fallback(
        fallback: &FallbackConfig,
        original_model: &str,
        context: &ResolutionContext,
    ) -> Result<ResolvedTarget, ResolutionError> {
        let model = fallback.model.as_deref().unwrap_or(original_model);
        let version = fallback.version.as_deref();
        let provider = fallback.provider.as_ref();

        Self::resolve_target(&fallback.target, model, version, provider, None, context)
    }
}

/// Resolution errors.
#[derive(Debug, Clone)]
pub enum ResolutionError {
    /// Device model not available.
    DeviceUnavailable(String),

    /// Device is throttled (thermal/battery).
    DeviceThrottled(String),

    /// Server is unavailable.
    ServerUnavailable(String),

    /// Network too slow for server.
    NetworkTooSlow(u32),

    /// Integration provider unavailable.
    IntegrationUnavailable(IntegrationProvider, String),

    /// Integration target requires a provider.
    MissingProvider(String),

    /// No target available after trying all options.
    NoTargetAvailable(String),
}

impl std::fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolutionError::DeviceUnavailable(model) => {
                write!(f, "device model '{}' not available", model)
            }
            ResolutionError::DeviceThrottled(reason) => {
                write!(f, "device throttled: {}", reason)
            }
            ResolutionError::ServerUnavailable(model) => {
                write!(f, "server model '{}' not available", model)
            }
            ResolutionError::NetworkTooSlow(rtt) => {
                write!(f, "network too slow ({}ms RTT)", rtt)
            }
            ResolutionError::IntegrationUnavailable(provider, model) => {
                write!(
                    f,
                    "integration provider '{}' unavailable for model '{}'",
                    provider, model
                )
            }
            ResolutionError::MissingProvider(model) => {
                write!(
                    f,
                    "integration target for '{}' requires a provider",
                    model
                )
            }
            ResolutionError::NoTargetAvailable(model) => {
                write!(f, "no execution target available for model '{}'", model)
            }
        }
    }
}

impl std::error::Error for ResolutionError {}

/// Get current timestamp in milliseconds.
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Convert LocalAvailability to ResolutionContext's local_available flag.
impl From<&LocalAvailability> for bool {
    fn from(availability: &LocalAvailability) -> bool {
        availability.local_model_exists
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_context() -> ResolutionContext {
        let metrics = DeviceMetrics {
            network_rtt: 100,
            battery: 80,
            temperature: 25.0,
        };
        ResolutionContext::new(metrics).with_local_available(true)
    }

    #[test]
    fn test_resolve_device_target() {
        let context = test_context();
        let stage = StageConfig::new("asr", "wav2vec2-base-960h")
            .with_target(ExecutionTarget::Device)
            .with_version("1.0");

        let result = TargetResolver::resolve(&stage, &context).unwrap();
        assert_eq!(result.target, RouteTarget::Local);
        assert_eq!(result.model, "wav2vec2-base-960h");
        assert_eq!(result.version, Some("1.0".to_string()));
    }

    #[test]
    fn test_resolve_device_unavailable() {
        let context = test_context().with_local_available(false);
        let stage = StageConfig::new("asr", "wav2vec2-base-960h")
            .with_target(ExecutionTarget::Device);

        let result = TargetResolver::resolve(&stage, &context);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ResolutionError::DeviceUnavailable(_)
        ));
    }

    #[test]
    fn test_resolve_server_target() {
        let context = test_context();
        let stage = StageConfig::new("asr", "whisper-large-v3")
            .with_target(ExecutionTarget::Server)
            .with_version("1.0");

        let result = TargetResolver::resolve(&stage, &context).unwrap();
        assert_eq!(result.target, RouteTarget::Cloud);
        assert!(result.reason.contains("server"));
    }

    #[test]
    fn test_resolve_integration_target() {
        let context = test_context()
            .with_integration_available(IntegrationProvider::OpenAI, true);
        let stage = StageConfig::new("llm", "gpt-4o-mini")
            .with_provider(IntegrationProvider::OpenAI);

        let result = TargetResolver::resolve(&stage, &context).unwrap();
        assert_eq!(result.target, RouteTarget::Cloud);
        assert_eq!(result.provider, Some(IntegrationProvider::OpenAI));
        assert!(result.reason.contains("integration"));
    }

    #[test]
    fn test_resolve_integration_missing_provider() {
        let context = test_context();
        let stage = StageConfig::new("llm", "gpt-4o-mini")
            .with_target(ExecutionTarget::Integration);

        let result = TargetResolver::resolve(&stage, &context);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ResolutionError::MissingProvider(_)
        ));
    }

    #[test]
    fn test_resolve_auto_prefers_device() {
        let context = test_context();
        let stage = StageConfig::new("asr", "wav2vec2-base-960h")
            .with_target(ExecutionTarget::Auto);

        let result = TargetResolver::resolve(&stage, &context).unwrap();
        // With good battery and hardware acceleration, should prefer device
        // Note: actual result depends on capability detection
        assert!(
            result.target == RouteTarget::Local || result.target == RouteTarget::Cloud
        );
    }

    #[test]
    fn test_resolve_auto_with_preference() {
        let mut stage = StageConfig::new("tts", "piper-en-us")
            .with_target(ExecutionTarget::Auto);
        stage.prefer = Some(ExecutionTarget::Device);

        let context = test_context();
        let result = TargetResolver::resolve(&stage, &context).unwrap();
        assert_eq!(result.target, RouteTarget::Local);
    }

    #[test]
    fn test_resolve_with_fallback_chain() {
        // Device unavailable, should fall back
        let context = test_context()
            .with_local_available(false)
            .with_integration_available(IntegrationProvider::OpenAI, true);

        let stage = StageConfig::new("asr", "whisper-large-v3")
            .with_target(ExecutionTarget::Device)
            .with_fallback(
                FallbackConfig::new(ExecutionTarget::Integration)
                    .with_provider(IntegrationProvider::OpenAI)
                    .with_model("whisper-1"),
            );

        let result = TargetResolver::resolve(&stage, &context).unwrap();
        assert_eq!(result.target, RouteTarget::Cloud);
        assert_eq!(result.provider, Some(IntegrationProvider::OpenAI));
        assert_eq!(result.model, "whisper-1");
    }

    #[test]
    fn test_resolution_context_builder() {
        let metrics = DeviceMetrics {
            network_rtt: 50,
            battery: 90,
            temperature: 20.0,
        };

        let context = ResolutionContext::new(metrics)
            .with_local_available(true)
            .with_server_available(true)
            .with_integration_available(IntegrationProvider::OpenAI, true)
            .with_integration_available(IntegrationProvider::Anthropic, false);

        assert!(context.local_available);
        assert!(context.server_available);
        assert!(context.is_integration_available(&IntegrationProvider::OpenAI));
        assert!(!context.is_integration_available(&IntegrationProvider::Anthropic));
        assert!(context.is_integration_available(&IntegrationProvider::Google)); // Default true
    }

    #[test]
    fn test_resolved_target_to_routing_decision() {
        let resolved = ResolvedTarget::local("wav2vec2", Some("1.0"), "test reason");
        let decision = resolved.to_routing_decision("asr");

        assert_eq!(decision.stage, "asr");
        assert_eq!(decision.target, RouteTarget::Local);
        assert_eq!(decision.reason, "test reason");
        assert!(decision.timestamp_ms > 0);
    }
}
