//! Deployment promotion records — the **provenanced canary record** of the
//! gating contract.
//!
//! `eval ship` promotes a candidate that passes its evalset gate. The promotion
//! is recorded with full provenance — candidate SHA, `evalset@version`, scorer
//! version, judge identity, and the device/profile constraints that allowed the
//! ramp — *so any shipped deployment can be reconstructed and any later
//! regression attributed*.
//!
//! This module owns the record + local store **and the device-side delivery
//! resolution**: [`DeploymentStore::active_for_skill`] resolves which deployment
//! a device cohort should run (canary-aware, rollback-aware) against the local
//! registry, and [`PromotionRecord::ramp_to`] / [`PromotionRecord::roll_back`]
//! drive the lifecycle. The only piece that stays server-side is the
//! *remote fleet transport* — pushing the resolved deployment to real devices;
//! locally the registry is the authority (a `RemoteAuthority` swaps the
//! store backing without changing this interface). The store mirrors
//! [`super::run::EvalRunStore`]: dependency-injected base dir, validated
//! single-component ids.
//!
//! **Trust model.** Locally the store is single-user and unauthenticated: the
//! gate is re-evaluated from the referenced run's *recorded* scores (it does
//! not re-execute the model), so a record is as trustworthy as the developer's
//! own `~/.xybrid` — the right bar for a local loop. The real integrity boundary
//! (verifying `candidate.model_sha256` against fetched bytes, rejecting tampered
//! run records) belongs to the `RemoteAuthority` seam, where a promotion record
//! crosses into a device fleet.

use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::eval::format::EvalError;
use crate::eval::run::{CandidateRef, JudgeIdentity};
use crate::eval::stats::{ConfidenceInterval, GateVerdict};

/// Promotion-record filename inside a deployment directory.
pub const PROMOTION_FILE: &str = "promotion.json";

/// Max promotion-record size accepted on load (DoS guard; mirrors the run cap).
const MAX_PROMOTION_FILE_BYTES: u64 = 4 * 1024 * 1024;

/// Lifecycle of a deployment. Transitions past `Pending` are the remote
/// authority's to make (canary ramp → active, or auto-rollback); this local
/// store only mints `Pending` records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentStatus {
    /// Gated and recorded, awaiting canary ramp.
    #[default]
    Pending,
    /// Fully ramped (remote authority).
    Active,
    /// Auto-rolled-back (remote authority).
    RolledBack,
}

/// A fully-provenanced promotion record — everything needed to reconstruct a
/// shipped deployment and attribute a later regression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionRecord {
    /// Unique deployment id.
    pub deployment_id: String,
    /// The named deployment target this promotes (absent when running purely
    /// against the local store).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill: Option<String>,
    /// Evalset that gated the promotion.
    pub evalset: String,
    /// Exact evalset version the gate ran against.
    pub evalset_version: u32,
    /// The promoted candidate (model id + SHA + config + prompt).
    pub candidate: CandidateRef,
    /// The gate verdict (always `Pass` for a real promotion).
    pub gate_verdict: GateVerdict,
    /// Aggregate quality at promotion time.
    pub quality: f64,
    /// Confidence interval on quality.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ci: Option<ConfidenceInterval>,
    /// Scorer schema version (a bump invalidates comparisons).
    pub scorer_version: String,
    /// Judge identity, when a judge backed the gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<JudgeIdentity>,
    /// Initial canary ramp percentage.
    pub canary_pct: u8,
    /// Device/profile constraints that allowed the ramp (e.g. `os=ios>=16`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub device_constraints: Vec<String>,
    /// The run id that produced the gating verdict.
    pub run_id: String,
    /// ISO-8601 creation timestamp (injected — never stamped implicitly).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
    /// Deployment lifecycle status.
    #[serde(default)]
    pub status: DeploymentStatus,
    /// Why the deployment was rolled back, if it was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback_reason: Option<String>,
}

impl PromotionRecord {
    /// Ramp the canary to `pct` (clamped to 100). At 100% the deployment is
    /// fully `Active`. A no-op once rolled back.
    pub fn ramp_to(&mut self, pct: u8) {
        if self.status == DeploymentStatus::RolledBack {
            return;
        }
        self.canary_pct = pct.min(100);
        if self.canary_pct >= 100 {
            self.status = DeploymentStatus::Active;
        }
    }

    /// Record a rollback with its trigger.
    pub fn roll_back(&mut self, trigger: &RollbackTrigger) {
        self.status = DeploymentStatus::RolledBack;
        self.rollback_reason = Some(trigger.describe());
    }
}

/// Why a deployment should roll back (the auto-rollback triggers of the gating
/// contract). The remote authority delivers the rollback; this decides
/// *whether*.
#[derive(Debug, Clone, PartialEq)]
pub enum RollbackTrigger {
    /// The eval gate no longer passes on the candidate's exact SHA.
    GateRegression,
    /// Live failure-rate (explicit `report()` + auto-flag signals) exceeds the
    /// evalset baseline by more than the margin.
    LiveFailureRate {
        /// Observed live failure rate (0..1).
        live: f64,
        /// The baseline blessed at ship time (0..1).
        baseline: f64,
    },
}

impl RollbackTrigger {
    /// A human-readable description for the deployment record.
    pub fn describe(&self) -> String {
        match self {
            RollbackTrigger::GateRegression => "eval gate no longer passes".to_string(),
            RollbackTrigger::LiveFailureRate { live, baseline } => {
                format!("live failure-rate {live:.3} exceeded baseline {baseline:.3}")
            }
        }
    }
}

/// Decide whether a live deployment should roll back (the same cases that chose
/// the winning deployment watch it in production). Returns the first triggered
/// reason, or `None` to keep ramping. `gate_passes` re-runs the evalset on the
/// candidate; `live_failure_rate` is the production signal (absent ⇒ not yet
/// observed).
pub fn rollback_decision(
    gate_passes: bool,
    live_failure_rate: Option<f64>,
    baseline_failure_rate: f64,
    margin: f64,
) -> Option<RollbackTrigger> {
    if !gate_passes {
        return Some(RollbackTrigger::GateRegression);
    }
    if let Some(live) = live_failure_rate {
        if live.is_finite() && live > baseline_failure_rate + margin.max(0.0) {
            return Some(RollbackTrigger::LiveFailureRate {
                live,
                baseline: baseline_failure_rate,
            });
        }
    }
    None
}

/// Validate a deployment id is a single safe path component (no traversal, no
/// control chars — the latter only fail deep in the OS otherwise).
fn validate_deployment_id(id: &str) -> Result<&str, EvalError> {
    let mut comps = Path::new(id).components();
    match (comps.next(), comps.next()) {
        (Some(Component::Normal(c)), None)
            if c == OsStr::new(id) && !id.chars().any(|ch| ch.is_control()) =>
        {
            Ok(id)
        }
        _ => Err(EvalError::Invalid(format!(
            "invalid deployment id {id:?}: must be a single path component"
        ))),
    }
}

/// On-disk store for promotion records. Dependency-injected base dir;
/// production resolves `~/.xybrid/deployments`.
#[derive(Debug, Clone)]
pub struct DeploymentStore {
    base: PathBuf,
}

impl DeploymentStore {
    /// A store rooted at an explicit base directory.
    pub fn with_dir(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    /// The default store at `~/.xybrid/deployments`.
    pub fn default_location() -> Result<Self, EvalError> {
        let home = dirs::home_dir()
            .ok_or_else(|| EvalError::Io("could not resolve home directory".into()))?;
        Ok(Self::with_dir(home.join(".xybrid").join("deployments")))
    }

    /// The base directory.
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// Persist a promotion record; returns the deployment directory. The write
    /// is atomic (temp + rename) so a crash mid-write cannot leave a torn record
    /// that breaks every later load.
    pub fn save(&self, record: &PromotionRecord) -> Result<PathBuf, EvalError> {
        let id = validate_deployment_id(&record.deployment_id)?;
        let dir = self.base.join(id);
        std::fs::create_dir_all(&dir)
            .map_err(|e| EvalError::Io(format!("{}: {e}", dir.display())))?;
        let path = dir.join(PROMOTION_FILE);
        let json = serde_json::to_string_pretty(record)
            .map_err(|e| EvalError::Io(format!("serialize promotion: {e}")))?;
        let tmp = dir.join(format!(".{PROMOTION_FILE}.{}.tmp", std::process::id()));
        std::fs::write(&tmp, json).map_err(|e| EvalError::Io(format!("{}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, &path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            EvalError::Io(format!("{}: {e}", path.display()))
        })?;
        Ok(dir)
    }

    /// Load a promotion record by deployment id (size-capped, regular-file only).
    pub fn load(&self, deployment_id: &str) -> Result<PromotionRecord, EvalError> {
        let id = validate_deployment_id(deployment_id)?;
        let path = self.base.join(id).join(PROMOTION_FILE);
        let meta = std::fs::metadata(&path)
            .map_err(|e| EvalError::Io(format!("{}: {e}", path.display())))?;
        if !meta.is_file() || meta.len() > MAX_PROMOTION_FILE_BYTES {
            return Err(EvalError::Invalid(format!(
                "promotion record {} is not a regular file or is too large",
                path.display()
            )));
        }
        let src = std::fs::read_to_string(&path)
            .map_err(|e| EvalError::Io(format!("{}: {e}", path.display())))?;
        serde_json::from_str(&src)
            .map_err(|e| EvalError::Invalid(format!("deployment {deployment_id}: {e}")))
    }

    /// List saved deployment ids.
    pub fn list(&self) -> Result<Vec<String>, EvalError> {
        if !self.base.exists() {
            return Ok(Vec::new());
        }
        let mut ids = Vec::new();
        for entry in std::fs::read_dir(&self.base)
            .map_err(|e| EvalError::Io(format!("{}: {e}", self.base.display())))?
        {
            let entry = entry.map_err(|e| EvalError::Io(e.to_string()))?;
            if entry.path().join(PROMOTION_FILE).exists() {
                if let Some(name) = entry.file_name().to_str() {
                    ids.push(name.to_string());
                }
            }
        }
        ids.sort();
        Ok(ids)
    }

    /// All promotion records for a skill (any status), newest first (by
    /// `created`). Unreadable records are skipped.
    pub fn deployments_for_skill(&self, skill: &str) -> Result<Vec<PromotionRecord>, EvalError> {
        let mut recs: Vec<PromotionRecord> = self
            .list()?
            .iter()
            .filter_map(|id| self.load(id).ok())
            .filter(|r| r.skill.as_deref() == Some(skill))
            .collect();
        // Newest first — `created` is an RFC3339 timestamp stamped at ship time;
        // records without one sort last. Tie-break on the id so resolution is
        // deterministic when two records share a timestamp (e.g. synced records).
        recs.sort_by(|a, b| {
            b.created
                .cmp(&a.created)
                .then_with(|| b.deployment_id.cmp(&a.deployment_id))
        });
        Ok(recs)
    }

    /// Resolve the deployment a device in `cohort` (0..100) should run for
    /// `skill` — the device-side of OTA delivery (the actual fleet transport is
    /// server-side; this resolves against the local registry).
    ///
    /// Canary semantics: the newest non-rolled-back deployment whose ramp covers
    /// the cohort (`cohort < canary_pct`) or is fully `Active` wins; otherwise
    /// the device stays on the previous active deployment. `None` if nothing
    /// applies (a rolled-back-only skill).
    pub fn active_for_skill(
        &self,
        skill: &str,
        cohort: u8,
    ) -> Result<Option<PromotionRecord>, EvalError> {
        for r in self.deployments_for_skill(skill)? {
            if r.status == DeploymentStatus::RolledBack {
                continue;
            }
            if r.status == DeploymentStatus::Active || (cohort as u16) < r.canary_pct as u16 {
                return Ok(Some(r));
            }
        }
        Ok(None)
    }
}

/// Current UTC time as an RFC3339 timestamp — stamped on promotion records so
/// deployments for a skill are orderable.
pub fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::run::CandidateRef;
    use tempfile::TempDir;

    fn record(id: &str) -> PromotionRecord {
        PromotionRecord {
            deployment_id: id.to_string(),
            skill: None,
            evalset: "intent".into(),
            evalset_version: 3,
            candidate: CandidateRef::new("qwen3.5-0.8b"),
            gate_verdict: GateVerdict::Pass,
            quality: 0.96,
            ci: None,
            scorer_version: "eval-scorer-v0".into(),
            judge: None,
            canary_pct: 5,
            device_constraints: vec!["os=ios>=16".into()],
            run_id: "run_abc".into(),
            created: Some("2026-06-14".into()),
            status: DeploymentStatus::Pending,
            rollback_reason: None,
        }
    }

    #[test]
    fn rollback_decision_triggers() {
        // Gate regression always rolls back.
        assert_eq!(
            rollback_decision(false, None, 0.05, 0.02),
            Some(RollbackTrigger::GateRegression)
        );
        // Gate passes + live within baseline+margin → keep ramping.
        assert_eq!(rollback_decision(true, Some(0.06), 0.05, 0.02), None);
        // Gate passes but live failure-rate spikes past baseline+margin → roll back.
        assert_eq!(
            rollback_decision(true, Some(0.20), 0.05, 0.02),
            Some(RollbackTrigger::LiveFailureRate {
                live: 0.20,
                baseline: 0.05
            })
        );
        // No live signal yet → no rollback.
        assert_eq!(rollback_decision(true, None, 0.05, 0.02), None);
        // NaN live signal is ignored (never a spurious rollback).
        assert_eq!(rollback_decision(true, Some(f64::NAN), 0.05, 0.02), None);
    }

    #[test]
    fn lifecycle_ramp_and_rollback() {
        let mut r = record("dep_l");
        r.ramp_to(50);
        assert_eq!(r.canary_pct, 50);
        assert_eq!(r.status, DeploymentStatus::Pending);
        r.ramp_to(100);
        assert_eq!(r.canary_pct, 100);
        assert_eq!(r.status, DeploymentStatus::Active);
        // ramp past 100 clamps.
        r.ramp_to(250);
        assert_eq!(r.canary_pct, 100);
        // rollback records the trigger and freezes ramps.
        r.roll_back(&RollbackTrigger::GateRegression);
        assert_eq!(r.status, DeploymentStatus::RolledBack);
        assert!(r.rollback_reason.as_deref().unwrap().contains("gate"));
        r.ramp_to(10); // no-op after rollback
        assert_eq!(r.status, DeploymentStatus::RolledBack);
    }

    #[test]
    fn promotion_record_round_trips() {
        let r = record("dep_1");
        let json = serde_json::to_string(&r).unwrap();
        let back: PromotionRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
        // provenance is all present
        assert!(json.contains("evalset_version"));
        assert!(json.contains("scorer_version"));
        assert!(json.contains("device_constraints"));
    }

    #[test]
    fn store_round_trips_in_temp_dir() {
        let dir = TempDir::new().unwrap();
        let store = DeploymentStore::with_dir(dir.path());
        assert!(store.list().unwrap().is_empty());
        let d = store.save(&record("dep_x")).unwrap();
        assert!(d.starts_with(dir.path()));
        assert_eq!(store.list().unwrap(), vec!["dep_x".to_string()]);
        assert_eq!(store.load("dep_x").unwrap().deployment_id, "dep_x");
    }

    #[test]
    fn store_rejects_traversal_deployment_ids() {
        let dir = TempDir::new().unwrap();
        let store = DeploymentStore::with_dir(dir.path());
        for bad in ["../escape", "/etc/passwd", "a/b", "..", ""] {
            assert!(store.load(bad).is_err(), "load should reject {bad:?}");
        }
        let mut r = record("../evil");
        assert!(store.save(&r).is_err());
        assert!(!dir.path().parent().unwrap().join("evil").exists());
        r.deployment_id = "dep_ok".into();
        assert!(store.save(&r).is_ok());
    }

    #[test]
    fn active_for_skill_resolves_canary_and_rollback() {
        let dir = TempDir::new().unwrap();
        let store = DeploymentStore::with_dir(dir.path());
        // older fully-active deployment for the skill
        let mut old = record("dep_old");
        old.skill = Some("intent".into());
        old.status = DeploymentStatus::Active;
        old.canary_pct = 100;
        old.created = Some("2026-06-10T00:00:00Z".into());
        store.save(&old).unwrap();
        // newer canary at 5%
        let mut new = record("dep_new");
        new.skill = Some("intent".into());
        new.status = DeploymentStatus::Pending;
        new.canary_pct = 5;
        new.created = Some("2026-06-14T00:00:00Z".into());
        store.save(&new).unwrap();

        // cohort 3 (< 5%) gets the new canary; cohort 50 stays on old active.
        assert_eq!(
            store
                .active_for_skill("intent", 3)
                .unwrap()
                .unwrap()
                .deployment_id,
            "dep_new"
        );
        assert_eq!(
            store
                .active_for_skill("intent", 50)
                .unwrap()
                .unwrap()
                .deployment_id,
            "dep_old"
        );
        // unknown skill → nothing to run.
        assert!(store.active_for_skill("other", 0).unwrap().is_none());

        // rolling the canary back → cohort 3 falls back to the old active.
        let mut rb = store.load("dep_new").unwrap();
        rb.roll_back(&RollbackTrigger::GateRegression);
        store.save(&rb).unwrap();
        assert_eq!(
            store
                .active_for_skill("intent", 3)
                .unwrap()
                .unwrap()
                .deployment_id,
            "dep_old"
        );
        assert_eq!(store.deployments_for_skill("intent").unwrap().len(), 2);
    }
}
