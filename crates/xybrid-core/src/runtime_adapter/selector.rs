//! Runtime backend selector: which local text backend runs a model.
//!
//! MLX is opt-in. The selector returns it only when the caller names it, or
//! when it is the only local text backend this build has:
//!
//! 1. An explicit `backend:` override (from `model_metadata.json`, a
//!    pipeline-stage YAML, the SDK's `with_backend` or the CLI's `--backend`)
//!    wins, when that backend is compiled into this build and, for MLX, the
//!    Metal runtime probe succeeds.
//! 2. Else [`BackendChoice::LlamaCpp`] if compiled in.
//! 3. Else [`BackendChoice::Mistral`] if compiled in.
//! 4. Else [`BackendChoice::Mlx`] if this build can run it.
//! 5. Else [`SelectorError::NoBackendAvailable`].
//!
//! So `backend: auto` picks the same backend on a build with MLX as on one
//! without: MLX refuses grammars and images, and a request that works on
//! llama.cpp must not start failing because the build gained a runtime.
//!
//! Explicit overrides that point at an unusable backend (not compiled for
//! this build, not supported by this target, or blocked by the MLX runtime
//! probe) produce [`SelectorError::ExplicitBackendUnavailable`] with the
//! current target and the list of backends that are usable, e.g. "MLX backend
//! requested but not available on this platform (target: linux-x86_64).
//! Available backends: [llamacpp]".

use std::sync::OnceLock;
use thiserror::Error;

// =============================================================================
// BackendChoice
// =============================================================================

/// The set of local text backends the runtime selector can return.
///
/// Cross-platform: this enum exists regardless of which backend features
/// are compiled in. The [`SelectorCfg`] struct records which backends are
/// actually available for the current build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendChoice {
    /// Apple Silicon macOS MLX backend (via `vendor/mlx-apple/mlx.xcframework`).
    Mlx,
    /// llama.cpp backend (Android + desktop fallback).
    LlamaCpp,
    /// mistral.rs backend (desktop only — SIGILL risk on Android).
    Mistral,
}

impl BackendChoice {
    /// Canonical lower-case identifier used in YAML, JSON, and log lines.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::LlamaCpp => "llamacpp",
            Self::Mistral => "mistral",
        }
    }

    /// Parse a case-insensitive backend identifier.
    ///
    /// Returns `Ok(None)` for the sentinel values `"auto"` and `""` (empty
    /// string) which mean "let the selector decide". Returns
    /// [`SelectorError::UnknownBackend`] for any other string.
    ///
    /// Common aliases: `"llama.cpp"`, `"llama_cpp"` → LlamaCpp;
    /// `"mistral.rs"`, `"mistralrs"` → Mistral.
    pub fn parse(s: &str) -> Result<Option<Self>, SelectorError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(None),
            "mlx" => Ok(Some(Self::Mlx)),
            "llamacpp" | "llama.cpp" | "llama_cpp" | "llama-cpp" => Ok(Some(Self::LlamaCpp)),
            "mistral" | "mistral.rs" | "mistralrs" => Ok(Some(Self::Mistral)),
            _ => Err(SelectorError::UnknownBackend(s.to_string())),
        }
    }
}

impl std::fmt::Display for BackendChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// =============================================================================
// SelectorError
// =============================================================================

/// Errors produced by the backend selector.
#[derive(Debug, Error)]
pub enum SelectorError {
    /// An explicit `backend:` override was requested but that backend is
    /// not available on this host — either the cargo feature isn't
    /// compiled in, or (for MLX) the Metal runtime probe failed.
    ///
    /// Error text matches the PRD's literal wording, e.g.
    /// `"MLX backend requested but not available on this platform
    /// (target: linux-x86_64). Available backends: [llamacpp, mistral]"`.
    #[error(
        "{} backend requested but not available on this platform (target: {target}). \
         Available backends: [{}]",
        fmt_backend_uppercase(requested),
        fmt_backend_list(available)
    )]
    ExplicitBackendUnavailable {
        /// The backend the caller asked for.
        requested: BackendChoice,
        /// Short `os-arch` string for the current target (e.g. `linux-x86_64`).
        target: String,
        /// Backends that ARE available on this build.
        available: Vec<BackendChoice>,
    },

    /// No local text backend is compiled into this build.
    #[error(
        "no local text backend available for model `{model_id}` (target: {target}, available: [{}])",
        fmt_backend_list(available)
    )]
    NoBackendAvailable {
        /// The model the caller asked the selector to route.
        model_id: String,
        /// Short `os-arch` string for the current target.
        target: String,
        /// Backends that ARE available on this build (expected to be empty
        /// when this variant is returned).
        available: Vec<BackendChoice>,
    },

    /// The explicit-backend string wasn't one of the recognised aliases.
    #[error("unknown backend identifier `{0}` (expected one of: auto, mlx, llamacpp, mistral)")]
    UnknownBackend(String),
}

fn fmt_backend_list(backends: &[BackendChoice]) -> String {
    backends
        .iter()
        .map(|b| b.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Uppercase variant used in the PRD-mandated
/// `"MLX backend requested..."` error prefix. Other backends still get
/// upper-cased display to keep the error prefix stylistically consistent
/// across backends.
fn fmt_backend_uppercase(b: &BackendChoice) -> &'static str {
    match b {
        BackendChoice::Mlx => "MLX",
        BackendChoice::LlamaCpp => "LlamaCpp",
        BackendChoice::Mistral => "Mistral",
    }
}

// =============================================================================
// SelectorCfg + capability probes
// =============================================================================

/// Snapshot of which backends are available on the current build/host.
///
/// Split from the selection logic so tests can construct arbitrary
/// configurations without being pinned to whatever the host machine
/// happens to compile.
#[derive(Debug, Clone)]
pub struct SelectorCfg {
    /// Short `os-arch` target string (e.g. `macos-aarch64`, `linux-x86_64`).
    pub target: String,
    /// True iff the host is `aarch64-apple-darwin` (required for the MLX path).
    pub host_is_apple_arm64: bool,
    /// True iff the `llm-mlx` cargo feature is compiled into this build.
    pub mlx_compiled: bool,
    /// True iff the `llm-llamacpp` cargo feature is compiled in.
    pub llamacpp_compiled: bool,
    /// True iff the `llm-mistral` cargo feature is compiled in.
    pub mistral_compiled: bool,
    /// True iff the MLX capability probe ([`mlx_runtime_available`])
    /// successfully created a Metal GPU stream. Always false when
    /// `llm-mlx-runtime` is disabled or the host is not Apple Silicon macOS.
    pub mlx_runtime_ok: bool,
}

impl SelectorCfg {
    /// Snapshot the current build + host.
    ///
    /// The probe is run via [`mlx_runtime_available`] which caches its
    /// result in a `OnceLock`, so repeated calls are cheap.
    #[must_use]
    pub fn current() -> Self {
        Self {
            target: current_target(),
            host_is_apple_arm64: cfg!(all(target_os = "macos", target_arch = "aarch64")),
            mlx_compiled: cfg!(feature = "llm-mlx"),
            llamacpp_compiled: cfg!(feature = "llm-llamacpp"),
            mistral_compiled: cfg!(feature = "llm-mistral"),
            mlx_runtime_ok: mlx_runtime_available(),
        }
    }

    /// Backends usable by this build on this host — the `available_backends`
    /// field in selector log lines and error messages.
    ///
    /// MLX is included only when the feature is compiled, the host is Apple
    /// Silicon macOS, and the runtime probe succeeded. A failed Metal probe
    /// means MLX cannot execute, so listing it as available would make
    /// explicit-backend errors point users at the backend that just failed.
    #[must_use]
    pub fn available_backends(&self) -> Vec<BackendChoice> {
        let mut v = Vec::new();
        if self.mlx_compiled && self.host_is_apple_arm64 && self.mlx_runtime_ok {
            v.push(BackendChoice::Mlx);
        }
        if self.llamacpp_compiled {
            v.push(BackendChoice::LlamaCpp);
        }
        if self.mistral_compiled {
            v.push(BackendChoice::Mistral);
        }
        v
    }
}

/// Build a short `os-arch` target string for error messages and telemetry.
#[must_use]
pub fn current_target() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Probe the MLX runtime. Result is cached in a `OnceLock` so subsequent
/// calls are cheap (the probe itself allocates a Metal GPU stream on the
/// first call, which can take ~5ms).
///
/// Always returns `false` when:
/// - The `llm-mlx-runtime` feature is not enabled (we have no way to
///   actually construct an `MlxStream` to probe with), or
/// - The host is not `aarch64-apple-darwin`.
///
/// On Apple Silicon macOS with `llm-mlx-runtime` on, returns true iff
/// `xybrid_mlx::MlxStream::default_gpu()` succeeded on the first attempt.
#[must_use]
pub fn mlx_runtime_available() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(probe_mlx_runtime)
}

#[cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]
fn probe_mlx_runtime() -> bool {
    // Building a GPU stream touches the Metal device enumeration path,
    // which is the earliest point MLX fails when Metal is unavailable
    // (e.g. headless CI with no GPU). That makes it the right probe: if
    // this succeeds, every subsequent op in the adapter will have a
    // working GPU handle.
    xybrid_mlx::MlxStream::default_gpu().is_ok()
}

#[cfg(not(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
)))]
const fn probe_mlx_runtime() -> bool {
    false
}

// =============================================================================
// Selection
// =============================================================================

/// Inputs to a single selection call.
#[derive(Debug, Clone)]
pub struct SelectionParams<'a> {
    /// The model ID being resolved (e.g. `qwen3.5-3b`).
    pub model_id: &'a str,
    /// Explicit backend override from `model_metadata.json`'s `backend:`
    /// field, a pipeline-stage YAML `backend:` field, the SDK or the CLI.
    /// `None` means "let the selector decide via the default priority".
    pub explicit: Option<BackendChoice>,
}

impl<'a> SelectionParams<'a> {
    /// Construct a selection request for the given model with no explicit
    /// override (equivalent to `backend: auto`).
    #[must_use]
    pub const fn new(model_id: &'a str) -> Self {
        Self {
            model_id,
            explicit: None,
        }
    }

    /// Add an explicit backend override.
    #[must_use]
    pub const fn with_explicit(mut self, explicit: Option<BackendChoice>) -> Self {
        self.explicit = explicit;
        self
    }
}

/// Pick the backend for `model_id` with no override, using the current
/// build's capabilities. Embedding models that can run on a text backend go
/// through the same selection.
///
/// Equivalent to
/// `select_with_cfg(&SelectionParams::new(model_id), &SelectorCfg::current())`.
///
/// # Errors
///
/// [`SelectorError::NoBackendAvailable`] when this build has no local text
/// backend.
pub fn select_llm_backend(model_id: &str) -> Result<BackendChoice, SelectorError> {
    select_with_cfg(&SelectionParams::new(model_id), &SelectorCfg::current())
}

/// Same as [`select_llm_backend`], with an optional explicit backend and a
/// pre-built [`SelectorCfg`] (so tests can simulate any host).
///
/// # Errors
///
/// [`SelectorError::ExplicitBackendUnavailable`] when the requested backend
/// cannot run on this build or host, and
/// [`SelectorError::NoBackendAvailable`] when no backend was requested and
/// none is compiled in.
pub fn select_with_cfg(
    params: &SelectionParams<'_>,
    cfg: &SelectorCfg,
) -> Result<BackendChoice, SelectorError> {
    let available = cfg.available_backends();

    // (1) Explicit override wins. Both the "not compiled in" case and
    //     the "MLX compiled but Metal probe failed" case fall through
    //     to ExplicitBackendUnavailable so callers get one error shape.
    if let Some(requested) = params.explicit {
        let backend_ok = available.contains(&requested);
        let runtime_ok = requested != BackendChoice::Mlx || cfg.mlx_runtime_ok;
        if backend_ok && runtime_ok {
            log_decision(params.model_id, requested, &available, "explicit_override");
            return Ok(requested);
        }
        log::info!(
            "backend_selector: chosen_backend=err \
             available_backends=[{}] reason=\"explicit backend `{}` unavailable on {}\" \
             model_id={}",
            fmt_backend_list(&available),
            requested,
            cfg.target,
            params.model_id,
        );
        return Err(SelectorError::ExplicitBackendUnavailable {
            requested,
            target: cfg.target.clone(),
            available,
        });
    }

    // (2) llama.cpp. MLX is never preferred over it: see the module docs.
    if cfg.llamacpp_compiled {
        log_decision(
            params.model_id,
            BackendChoice::LlamaCpp,
            &available,
            "fallback to llamacpp",
        );
        return Ok(BackendChoice::LlamaCpp);
    }

    // (3) mistral.rs.
    if cfg.mistral_compiled {
        log_decision(
            params.model_id,
            BackendChoice::Mistral,
            &available,
            "fallback to mistral",
        );
        return Ok(BackendChoice::Mistral);
    }

    // (4) MLX, on a build whose only local text backend it is. Whether the
    // model is an MLX SafeTensors bundle is decided at load; this branch
    // only refuses to report "no local text backend compiled in" while MLX
    // is compiled and probe-verified.
    if cfg.host_is_apple_arm64 && cfg.mlx_compiled && cfg.mlx_runtime_ok {
        log_decision(
            params.model_id,
            BackendChoice::Mlx,
            &available,
            "fallback to mlx (only compiled backend with runtime ok)",
        );
        return Ok(BackendChoice::Mlx);
    }

    // (5) Nothing available.
    log::info!(
        "backend_selector: chosen_backend=err \
         available_backends=[{}] reason=\"no local text backend compiled in\" model_id={}",
        fmt_backend_list(&available),
        params.model_id,
    );
    Err(SelectorError::NoBackendAvailable {
        model_id: params.model_id.to_string(),
        target: cfg.target.clone(),
        available,
    })
}

fn log_decision(model_id: &str, chosen: BackendChoice, available: &[BackendChoice], reason: &str) {
    log::info!(
        "backend_selector: chosen_backend={} available_backends=[{}] reason=\"{}\" model_id={}",
        chosen,
        fmt_backend_list(available),
        reason,
        model_id,
    );
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fully-permissive cfg (every backend compiled, MLX probe ok,
    /// Apple Silicon macOS host). Individual tests toggle fields off.
    fn apple_arm64_everything() -> SelectorCfg {
        SelectorCfg {
            target: "macos-aarch64".to_string(),
            host_is_apple_arm64: true,
            mlx_compiled: true,
            llamacpp_compiled: true,
            mistral_compiled: true,
            mlx_runtime_ok: true,
        }
    }

    fn linux_x86_64_llamacpp_only() -> SelectorCfg {
        SelectorCfg {
            target: "linux-x86_64".to_string(),
            host_is_apple_arm64: false,
            mlx_compiled: false,
            llamacpp_compiled: true,
            mistral_compiled: false,
            mlx_runtime_ok: false,
        }
    }

    // ------------------------------------------------------------------
    // BackendChoice parsing
    // ------------------------------------------------------------------

    #[test]
    fn backend_choice_parse_auto_is_none() {
        assert!(matches!(BackendChoice::parse("auto"), Ok(None)));
        assert!(matches!(BackendChoice::parse(""), Ok(None)));
        assert!(matches!(BackendChoice::parse("AUTO"), Ok(None)));
    }

    #[test]
    fn backend_choice_parse_canonical_identifiers() {
        assert_eq!(
            BackendChoice::parse("mlx").unwrap(),
            Some(BackendChoice::Mlx)
        );
        assert_eq!(
            BackendChoice::parse("llamacpp").unwrap(),
            Some(BackendChoice::LlamaCpp)
        );
        assert_eq!(
            BackendChoice::parse("mistral").unwrap(),
            Some(BackendChoice::Mistral)
        );
    }

    #[test]
    fn backend_choice_parse_aliases() {
        assert_eq!(
            BackendChoice::parse("LLAMA.CPP").unwrap(),
            Some(BackendChoice::LlamaCpp)
        );
        assert_eq!(
            BackendChoice::parse("llama_cpp").unwrap(),
            Some(BackendChoice::LlamaCpp)
        );
        assert_eq!(
            BackendChoice::parse("llama-cpp").unwrap(),
            Some(BackendChoice::LlamaCpp)
        );
        assert_eq!(
            BackendChoice::parse("mistral.rs").unwrap(),
            Some(BackendChoice::Mistral)
        );
        assert_eq!(
            BackendChoice::parse("MistralRS").unwrap(),
            Some(BackendChoice::Mistral)
        );
    }

    #[test]
    fn backend_choice_parse_unknown_returns_err() {
        assert!(matches!(
            BackendChoice::parse("onnx"),
            Err(SelectorError::UnknownBackend(s)) if s == "onnx"
        ));
    }

    #[test]
    fn backend_choice_as_str_round_trip() {
        for b in [
            BackendChoice::Mlx,
            BackendChoice::LlamaCpp,
            BackendChoice::Mistral,
        ] {
            assert_eq!(BackendChoice::parse(b.as_str()).unwrap(), Some(b));
        }
    }

    // ------------------------------------------------------------------
    // available_backends()
    // ------------------------------------------------------------------

    #[test]
    fn available_backends_excludes_mlx_when_not_apple_arm64() {
        let cfg = SelectorCfg {
            target: "linux-x86_64".to_string(),
            host_is_apple_arm64: false,
            mlx_compiled: true, // would-be compiled but host doesn't match
            llamacpp_compiled: true,
            mistral_compiled: false,
            mlx_runtime_ok: false,
        };
        let avail = cfg.available_backends();
        assert_eq!(avail, vec![BackendChoice::LlamaCpp]);
    }

    #[test]
    fn available_backends_includes_mlx_on_apple_arm64_when_compiled() {
        let cfg = SelectorCfg {
            target: "macos-aarch64".to_string(),
            host_is_apple_arm64: true,
            mlx_compiled: true,
            llamacpp_compiled: true,
            mistral_compiled: false,
            mlx_runtime_ok: true,
        };
        let avail = cfg.available_backends();
        assert_eq!(avail, vec![BackendChoice::Mlx, BackendChoice::LlamaCpp]);
    }

    #[test]
    fn available_backends_excludes_mlx_when_runtime_probe_fails() {
        let cfg = SelectorCfg {
            target: "macos-aarch64".to_string(),
            host_is_apple_arm64: true,
            mlx_compiled: true,
            llamacpp_compiled: true,
            mistral_compiled: true,
            mlx_runtime_ok: false,
        };
        let avail = cfg.available_backends();
        assert_eq!(avail, vec![BackendChoice::LlamaCpp, BackendChoice::Mistral]);
    }

    // ------------------------------------------------------------------
    // Branch (1): Explicit override
    // ------------------------------------------------------------------

    #[test]
    fn branch_1_explicit_override_wins_when_backend_available() {
        let cfg = apple_arm64_everything();
        let params =
            SelectionParams::new("qwen3.5-3b").with_explicit(Some(BackendChoice::LlamaCpp));
        assert_eq!(
            select_with_cfg(&params, &cfg).unwrap(),
            BackendChoice::LlamaCpp,
        );
    }

    #[test]
    fn branch_1_explicit_mlx_on_non_apple_returns_descriptive_error() {
        let cfg = linux_x86_64_llamacpp_only();
        let params = SelectionParams::new("qwen3.5-3b").with_explicit(Some(BackendChoice::Mlx));

        let err = select_with_cfg(&params, &cfg).unwrap_err();
        // Matches the PRD's required error shape (word-for-word):
        //   "MLX backend requested but not available on this platform
        //    (target: linux-x86_64). Available backends: [llamacpp, mistral]"
        let msg = err.to_string();
        assert_eq!(
            msg,
            "MLX backend requested but not available on this platform \
             (target: linux-x86_64). Available backends: [llamacpp]",
            "selector error text must match the PRD's wording verbatim (except \
             for the available-backends list, which varies by build)",
        );
        match err {
            SelectorError::ExplicitBackendUnavailable {
                requested,
                target,
                available,
            } => {
                assert_eq!(requested, BackendChoice::Mlx);
                assert_eq!(target, "linux-x86_64");
                assert_eq!(available, vec![BackendChoice::LlamaCpp]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn branch_1_explicit_mlx_when_runtime_probe_fails() {
        // MLX feature compiled, host is Apple Silicon, but Metal probe
        // failed (e.g. headless CI) — same error shape as "not compiled".
        let mut cfg = apple_arm64_everything();
        cfg.mlx_runtime_ok = false;
        let params = SelectionParams::new("qwen3.5-3b").with_explicit(Some(BackendChoice::Mlx));
        let err = select_with_cfg(&params, &cfg).unwrap_err();
        assert!(matches!(
            err,
            SelectorError::ExplicitBackendUnavailable {
                requested: BackendChoice::Mlx,
                available,
                ..
            } if available == vec![BackendChoice::LlamaCpp, BackendChoice::Mistral]
        ));
    }

    // ------------------------------------------------------------------
    // MLX is opt-in: no override never selects it over another backend
    // ------------------------------------------------------------------

    #[test]
    fn no_override_keeps_llamacpp_on_a_build_that_can_run_mlx() {
        let cfg = apple_arm64_everything();
        let params = SelectionParams::new("qwen3.5-3b");
        assert_eq!(
            select_with_cfg(&params, &cfg).unwrap(),
            BackendChoice::LlamaCpp,
        );
    }

    #[test]
    fn no_override_keeps_mistral_over_mlx_when_llamacpp_is_absent() {
        let mut cfg = apple_arm64_everything();
        cfg.llamacpp_compiled = false;
        let params = SelectionParams::new("qwen3.5-3b");
        assert_eq!(
            select_with_cfg(&params, &cfg).unwrap(),
            BackendChoice::Mistral,
        );
    }

    #[test]
    fn no_override_selects_the_same_backend_with_and_without_mlx() {
        let with_mlx = apple_arm64_everything();
        let mut without_mlx = apple_arm64_everything();
        without_mlx.mlx_compiled = false;
        without_mlx.mlx_runtime_ok = false;
        let params = SelectionParams::new("qwen3.5-3b");
        assert_eq!(
            select_with_cfg(&params, &with_mlx).unwrap(),
            select_with_cfg(&params, &without_mlx).unwrap(),
        );
    }

    #[test]
    fn an_explicit_mlx_override_selects_mlx() {
        let cfg = apple_arm64_everything();
        let params = SelectionParams::new("qwen3.5-3b").with_explicit(Some(BackendChoice::Mlx));
        assert_eq!(select_with_cfg(&params, &cfg).unwrap(), BackendChoice::Mlx);
    }

    #[test]
    fn an_mlx_only_build_selects_mlx_without_an_override() {
        let mut cfg = apple_arm64_everything();
        cfg.llamacpp_compiled = false;
        cfg.mistral_compiled = false;
        let params = SelectionParams::new("qwen3.5-3b");
        assert_eq!(select_with_cfg(&params, &cfg).unwrap(), BackendChoice::Mlx);
    }

    #[test]
    fn an_mlx_only_build_without_a_runtime_has_no_backend() {
        let mut cfg = apple_arm64_everything();
        cfg.llamacpp_compiled = false;
        cfg.mistral_compiled = false;
        cfg.mlx_runtime_ok = false;
        let params = SelectionParams::new("qwen3.5-3b");
        assert!(matches!(
            select_with_cfg(&params, &cfg),
            Err(SelectorError::NoBackendAvailable { .. })
        ));
    }

    // ------------------------------------------------------------------
    // Branch (2): LlamaCpp
    // ------------------------------------------------------------------

    #[test]
    fn branch_2_llamacpp_when_not_apple() {
        let cfg = linux_x86_64_llamacpp_only();
        let params = SelectionParams::new("qwen3.5-3b");
        assert_eq!(
            select_with_cfg(&params, &cfg).unwrap(),
            BackendChoice::LlamaCpp,
        );
    }

    // ------------------------------------------------------------------
    // Branch (3): Mistral fallback
    // ------------------------------------------------------------------

    #[test]
    fn branch_3_mistral_when_no_llamacpp() {
        let cfg = SelectorCfg {
            target: "linux-x86_64".to_string(),
            host_is_apple_arm64: false,
            mlx_compiled: false,
            llamacpp_compiled: false,
            mistral_compiled: true,
            mlx_runtime_ok: false,
        };
        let params = SelectionParams::new("qwen3.5-3b");
        assert_eq!(
            select_with_cfg(&params, &cfg).unwrap(),
            BackendChoice::Mistral,
        );
    }

    // ------------------------------------------------------------------
    // Branch (5): No backend available
    // ------------------------------------------------------------------

    #[test]
    fn branch_5_no_backend_available_returns_err() {
        let cfg = SelectorCfg {
            target: "freebsd-x86_64".to_string(),
            host_is_apple_arm64: false,
            mlx_compiled: false,
            llamacpp_compiled: false,
            mistral_compiled: false,
            mlx_runtime_ok: false,
        };
        let params = SelectionParams::new("qwen3.5-3b");
        let err = select_with_cfg(&params, &cfg).unwrap_err();
        match err {
            SelectorError::NoBackendAvailable {
                model_id,
                target,
                available,
            } => {
                assert_eq!(model_id, "qwen3.5-3b");
                assert_eq!(target, "freebsd-x86_64");
                assert!(available.is_empty());
            }
            _ => panic!("wrong variant"),
        }
    }

    // ------------------------------------------------------------------
    // Public entry-point smoke test: at least runs without panicking.
    // ------------------------------------------------------------------

    #[test]
    fn current_cfg_can_be_built_without_panicking() {
        let _ = SelectorCfg::current();
    }

    #[test]
    fn current_target_is_well_formed() {
        let t = current_target();
        assert!(t.contains('-'), "target must be `os-arch`: {t}");
    }

    #[test]
    fn mlx_runtime_available_is_stable() {
        // The OnceLock caches the first call, so subsequent calls must
        // agree. Also serves as a smoke test that we don't panic on
        // non-Apple hosts (where the probe is `const fn` returning false).
        let a = mlx_runtime_available();
        let b = mlx_runtime_available();
        assert_eq!(a, b);
    }
}
