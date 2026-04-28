//! Typed cooperative abort reasons shared across routing and execution.

use serde::{Deserialize, Serialize};

/// Stable reason for a cooperative abort or local-to-cloud handoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbortReason {
    UserCancelled,
    StressThrottle,
    StressMemory,
    StressThermal,
    StressCpuSustained,
    BudgetExceeded,
}

impl AbortReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UserCancelled => "user_cancelled",
            Self::StressThrottle => "stress_throttle",
            Self::StressMemory => "stress_memory",
            Self::StressThermal => "stress_thermal",
            Self::StressCpuSustained => "stress_cpu_sustained",
            Self::BudgetExceeded => "budget_exceeded",
        }
    }
}

impl std::fmt::Display for AbortReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for AbortReason {}

/// Error marker used by streaming callbacks to preserve typed cloud-fallback
/// aborts through dynamic `Box<dyn Error>` callback boundaries.
#[derive(Debug, Clone)]
pub struct CloudFallbackAbort {
    reason: AbortReason,
}

impl CloudFallbackAbort {
    pub fn new(reason: AbortReason) -> Self {
        Self { reason }
    }

    pub fn reason(&self) -> AbortReason {
        self.reason
    }
}

impl std::fmt::Display for CloudFallbackAbort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "aborted for cloud fallback: {}", self.reason)
    }
}

impl std::error::Error for CloudFallbackAbort {}

pub fn cloud_fallback_reason_from_error(
    error: &(dyn std::error::Error + 'static),
) -> Option<AbortReason> {
    error
        .downcast_ref::<CloudFallbackAbort>()
        .map(CloudFallbackAbort::reason)
}
