//! `xybrid eval` — the eval-driven development loop (flag → collect → compare →
//! gate → ship).
//!
//! The local loop runs entirely on your machine: scaffold an evalset, validate
//! it, run a candidate, compare candidates, and gate in CI. Graders are implied
//! by the task type — Tier 1 never picks a metric. Runs and gates are statistical
//! (pass / fail / inconclusive), not raw-number compares.
//!
//! Commands that don't run a model (`inspect`, `init`, `show`, `diff`, and the
//! stored-run form of `gate`) work fully offline with no backend features.
//! `run` / `compare` (and `gate --model`) execute the candidate through the same
//! path as `xybrid run`, so they need a platform preset built in.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Subcommand;

use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::template_executor::TemplateExecutor;
use xybrid_sdk::eval::{
    gate_policy, run_evalset, CandidateRef, Case, CaseOutcome, Environment, EvalRunStore, Evalset,
    GateVerdict, GradeOutput, InboxClient, InboxQuery, LoadedEvalset, OverlapJudge, Run,
    RunOptions, TaskType, Verdict,
};
use xybrid_sdk::registry_client::RegistryClient;
use xybrid_sdk::ExecutionProviderInfo;

use crate::ui;

/// Maximum size of a single `file:` payload sidecar read during a run (DoS guard).
const MAX_PAYLOAD_BYTES: u64 = 256 * 1024 * 1024;

/// `xybrid eval` subcommands.
#[derive(Subcommand)]
pub enum EvalCommand {
    /// Validate an evalset and print a summary.
    Inspect {
        /// Path to the evalset directory (`evals/<name>/`).
        #[arg(value_name = "PATH")]
        path: PathBuf,
    },
    /// Scaffold a new evalset for a task.
    Init {
        /// Task type: classify | chat | summarize | extract | asr | tts | embedding | vlm.
        #[arg(value_name = "TASK")]
        task: String,
        /// Evalset name (defaults to the task name).
        #[arg(long)]
        name: Option<String>,
        /// Parent directory to scaffold under (an `evals/<name>/` dir is created).
        #[arg(long, default_value = "evals")]
        dir: PathBuf,
        /// Overwrite an existing evalset.
        #[arg(long)]
        force: bool,
    },
    /// Pull flagged cases from the inbox into the local evalset (review queue).
    Pull {
        /// Path to the evalset directory.
        #[arg(value_name = "EVALSET")]
        evalset: PathBuf,
        /// Inbox file (JSONL of pending cases). Defaults to
        /// `~/.xybrid/inbox/<evalset>.jsonl` (the platform sync target).
        #[arg(long, value_name = "PATH")]
        inbox: Option<PathBuf>,
        /// Accept all pending cases without prompting (CI / scripting).
        #[arg(long)]
        accept_all: bool,
        /// Show what would be pulled without writing anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// View the platform failure inbox (flagged results + monitor auto-flags).
    Inbox {
        /// Look-back window: 1d | 7d | 30d | all.
        #[arg(long, default_value = "7d")]
        period: String,
        /// Filter to one model id.
        #[arg(long, value_name = "ID")]
        model: Option<String>,
        /// Filter by source: report | signal | all.
        #[arg(long, default_value = "all")]
        source: String,
        /// Filter explicit rating: up | down.
        #[arg(long)]
        rating: Option<String>,
        /// Maximum number of items to show.
        #[arg(long, default_value_t = 50)]
        limit: u32,
    },
    /// Run a candidate against an evalset and score it.
    Run {
        /// Path to the evalset directory.
        #[arg(value_name = "EVALSET")]
        evalset: PathBuf,
        /// Candidate model id.
        #[arg(long, value_name = "ID")]
        model: String,
        /// Limit to the first N cases.
        #[arg(long)]
        limit: Option<usize>,
        /// Don't capture per-case outputs into the run record.
        #[arg(long)]
        no_capture: bool,
        /// Run store directory (defaults to `~/.xybrid/eval-runs`).
        #[arg(long, value_name = "DIR")]
        store: Option<PathBuf>,
    },
    /// Compare candidates on an evalset and print a leaderboard.
    Compare {
        /// Path to the evalset directory.
        #[arg(value_name = "EVALSET")]
        evalset: PathBuf,
        /// Candidate model id (repeatable). The first is the baseline.
        #[arg(long = "model", value_name = "ID")]
        models: Vec<String>,
        /// Suggest comparable candidates from the registry by task type.
        #[arg(long)]
        auto: bool,
        /// Run store directory.
        #[arg(long, value_name = "DIR")]
        store: Option<PathBuf>,
    },
    /// Evaluate the gate for a stored run (or run fresh with `--model`).
    Gate {
        /// Path to the evalset directory (its gate config is applied).
        #[arg(value_name = "EVALSET")]
        evalset: PathBuf,
        /// Evaluate a previously-stored run instead of running fresh.
        #[arg(long, value_name = "RUN_ID")]
        run: Option<String>,
        /// Run fresh against this model, then gate.
        #[arg(long, value_name = "ID", conflicts_with = "run")]
        model: Option<String>,
        /// Treat an inconclusive gate as a failure (exit non-zero).
        #[arg(long)]
        strict: bool,
        /// Run store directory.
        #[arg(long, value_name = "DIR")]
        store: Option<PathBuf>,
    },
    /// Ship a candidate that passes its gate: record a provenanced promotion
    /// (the artifact a remote backend ramps over-the-air).
    Ship {
        /// Path to the evalset directory (its gate must pass).
        #[arg(value_name = "EVALSET")]
        evalset: PathBuf,
        /// Promote a previously-stored run.
        #[arg(long, value_name = "RUN_ID")]
        run: Option<String>,
        /// Run fresh against this model, gate, then promote.
        #[arg(long, value_name = "ID", conflicts_with = "run")]
        model: Option<String>,
        /// Named deployment target this serves (its active deployment is
        /// resolved per device cohort via `eval resolve`).
        #[arg(long, value_name = "SLUG")]
        skill: Option<String>,
        /// Initial canary ramp percentage.
        #[arg(long, default_value = "5")]
        canary: u8,
        /// Device/profile constraint that allowed the ramp (repeatable).
        #[arg(long = "constraint", value_name = "EXPR")]
        constraints: Vec<String>,
        /// Run store directory.
        #[arg(long, value_name = "DIR")]
        store: Option<PathBuf>,
        /// Deployment store directory (defaults to `~/.xybrid/deployments`).
        #[arg(long, value_name = "DIR")]
        deployments: Option<PathBuf>,
    },
    /// Ramp a deployment's canary % — re-gates on the exact candidate first
    /// (promotion past canary requires a still-passing eval).
    Promote {
        /// Deployment id.
        #[arg(value_name = "DEPLOYMENT_ID")]
        deployment_id: String,
        /// Target canary percentage.
        #[arg(long, value_name = "PCT")]
        to: u8,
        /// Evalset directory (re-evaluated against the gating run).
        #[arg(long, value_name = "EVALSET")]
        evalset: PathBuf,
        /// Run store directory.
        #[arg(long, value_name = "DIR")]
        store: Option<PathBuf>,
        /// Deployment store directory.
        #[arg(long, value_name = "DIR")]
        deployments: Option<PathBuf>,
    },
    /// Roll a deployment back (records the trigger; a remote backend delivers).
    Rollback {
        /// Deployment id.
        #[arg(value_name = "DEPLOYMENT_ID")]
        deployment_id: String,
        /// Reason for the rollback.
        #[arg(long, value_name = "REASON")]
        reason: Option<String>,
        /// Deployment store directory.
        #[arg(long, value_name = "DIR")]
        deployments: Option<PathBuf>,
    },
    /// List recorded deployments (optionally for one skill).
    Deployments {
        /// Filter to a skill slug.
        #[arg(long, value_name = "SLUG")]
        skill: Option<String>,
        /// Deployment store directory.
        #[arg(long, value_name = "DIR")]
        deployments: Option<PathBuf>,
    },
    /// Resolve the active deployment a device cohort runs for a skill (the
    /// device-side of over-the-air delivery, against the local registry).
    Resolve {
        /// Deployment target slug.
        #[arg(value_name = "SLUG")]
        skill: String,
        /// Device cohort bucket (0–99); decides canary inclusion.
        #[arg(long, default_value = "0")]
        cohort: u8,
        /// Deployment store directory.
        #[arg(long, value_name = "DIR")]
        deployments: Option<PathBuf>,
    },
    /// Re-print a stored run.
    Show {
        /// Run id.
        #[arg(value_name = "RUN_ID")]
        run_id: String,
        /// Run store directory.
        #[arg(long, value_name = "DIR")]
        store: Option<PathBuf>,
    },
    /// Side-by-side delta between two stored runs.
    Diff {
        /// First run id.
        #[arg(value_name = "RUN_A")]
        run_a: String,
        /// Second run id.
        #[arg(value_name = "RUN_B")]
        run_b: String,
        /// Run store directory.
        #[arg(long, value_name = "DIR")]
        store: Option<PathBuf>,
    },
}

/// Dispatch `xybrid eval [subcommand]`. Zero-arg discovers `evals/` in the cwd.
///
/// `api_key` / `platform_url` carry the CLI's resolved global `--api-key` /
/// `--platform-url` flags so `eval inbox` honors them (clap doesn't write parsed
/// flags back to the environment, so the env-only path would silently ignore the
/// flag forms).
pub fn handle_eval_command(
    command: Option<EvalCommand>,
    api_key: Option<&str>,
    platform_url: &str,
) -> Result<()> {
    match command {
        None => discover_evalsets(),
        Some(EvalCommand::Inspect { path }) => inspect(&path),
        Some(EvalCommand::Init {
            task,
            name,
            dir,
            force,
        }) => init(&task, name.as_deref(), &dir, force),
        Some(EvalCommand::Pull {
            evalset,
            inbox,
            accept_all,
            dry_run,
        }) => pull(&evalset, inbox.as_deref(), accept_all, dry_run),
        Some(EvalCommand::Inbox {
            period,
            model,
            source,
            rating,
            limit,
        }) => inbox_view(
            &period,
            model.as_deref(),
            &source,
            rating.as_deref(),
            limit,
            api_key,
            platform_url,
        ),
        Some(EvalCommand::Run {
            evalset,
            model,
            limit,
            no_capture,
            store,
        }) => run(&evalset, &model, limit, !no_capture, store.as_deref()),
        Some(EvalCommand::Compare {
            evalset,
            models,
            auto,
            store,
        }) => compare(&evalset, &models, auto, store.as_deref()),
        Some(EvalCommand::Gate {
            evalset,
            run,
            model,
            strict,
            store,
        }) => gate(
            &evalset,
            run.as_deref(),
            model.as_deref(),
            strict,
            store.as_deref(),
        ),
        Some(EvalCommand::Ship {
            evalset,
            run,
            model,
            skill,
            canary,
            constraints,
            store,
            deployments,
        }) => ship(
            &evalset,
            run.as_deref(),
            model.as_deref(),
            skill.as_deref(),
            canary,
            constraints,
            store.as_deref(),
            deployments.as_deref(),
        ),
        Some(EvalCommand::Promote {
            deployment_id,
            to,
            evalset,
            store,
            deployments,
        }) => promote(
            &deployment_id,
            to,
            &evalset,
            store.as_deref(),
            deployments.as_deref(),
        ),
        Some(EvalCommand::Rollback {
            deployment_id,
            reason,
            deployments,
        }) => rollback(&deployment_id, reason.as_deref(), deployments.as_deref()),
        Some(EvalCommand::Deployments { skill, deployments }) => {
            list_deployments(skill.as_deref(), deployments.as_deref())
        }
        Some(EvalCommand::Resolve {
            skill,
            cohort,
            deployments,
        }) => resolve_skill(&skill, cohort, deployments.as_deref()),
        Some(EvalCommand::Show { run_id, store }) => show(&run_id, store.as_deref()),
        Some(EvalCommand::Diff {
            run_a,
            run_b,
            store,
        }) => diff(&run_a, &run_b, store.as_deref()),
    }
}

// ============================================================================
// Pure helpers (unit-tested)
// ============================================================================

/// Parse a task verb into a [`TaskType`], with a helpful error.
fn parse_task(task: &str) -> Result<TaskType> {
    serde_yaml::from_str::<TaskType>(task.trim()).map_err(|_| {
        anyhow::anyhow!(
            "unknown task '{task}'. Valid tasks: classify, chat, summarize, extract, asr, tts, embedding, vlm"
        )
    })
}

/// The human-readable default grader for a task (informational; Tier 1 never
/// picks one).
fn default_grader_label(task: TaskType) -> &'static str {
    match task {
        TaskType::Classify => "label match",
        TaskType::Asr => "word error rate",
        TaskType::Extract => "JSON field match",
        TaskType::Chat | TaskType::Summarize | TaskType::Vlm => "LLM judge",
        TaskType::Tts => "golden output",
        TaskType::Embedding => "recall@k",
    }
}

/// Exit code for a gate verdict. Pass/Inconclusive are CI-neutral (0) unless
/// `strict` treats inconclusive as a block; Fail always blocks (2 — distinct
/// from the anyhow error path so the offline/not-found handler doesn't misfire).
fn gate_exit_code(verdict: GateVerdict, strict: bool) -> i32 {
    match verdict {
        GateVerdict::Pass => 0,
        GateVerdict::Inconclusive => {
            if strict {
                2
            } else {
                0
            }
        }
        GateVerdict::Fail => 2,
    }
}

/// Whether a manifest declares an enforceable absolute gate (a quality or
/// latency threshold). `gate` short-circuits to a no-op when this is false —
/// the report-only `Pass` `GatePolicy::evaluate` returns is for display, not a
/// CI gate. (Non-inferiority is a `compare`-only concern, never reachable from
/// `gate`, so it is intentionally not part of this predicate.)
fn manifest_has_gate_thresholds(manifest: &Evalset) -> bool {
    manifest
        .gate
        .as_ref()
        .is_some_and(|g| g.min_quality.is_some() || g.max_p95_latency_ms.is_some())
}

/// HF-style task strings a verb maps to, for registry candidate suggestion.
/// Lossy by nature (three task vocabularies exist); ambiguous verbs return broad
/// sets and are surfaced as suggestions, never auto-run blindly.
fn verb_to_hf_tasks(task: TaskType) -> &'static [&'static str] {
    match task {
        TaskType::Classify => &["text-classification", "image-classification"],
        TaskType::Asr => &["automatic-speech-recognition", "speech-recognition"],
        TaskType::Tts => &["text-to-speech"],
        TaskType::Embedding => &["feature-extraction", "sentence-similarity"],
        // Broad/ambiguous — suggestions only.
        TaskType::Chat | TaskType::Summarize | TaskType::Extract | TaskType::Vlm => {
            &["text-generation"]
        }
    }
}

/// Recommend the winning run index per the winner-selection policy: only a
/// **Pass** verdict is eligible (Fail *and* Inconclusive are never a green),
/// flaky candidates and non-finite quality are dropped, then maximize quality,
/// breaking ties by lower p95 latency, then smaller bundle. Returns `None` if
/// nothing qualifies.
fn recommend_winner(runs: &[Run]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, r) in runs.iter().enumerate() {
        // Only a clean Pass is promotable: an inconclusive candidate is never a
        // green, and a non-finite quality can never be trusted/ranked.
        if r.scores.verdict != GateVerdict::Pass || r.scores.flaky || !r.scores.quality.is_finite()
        {
            continue;
        }
        best = Some(match best {
            None => i,
            Some(b) => {
                if better_run(&runs[i].scores, &runs[b].scores) {
                    i
                } else {
                    b
                }
            }
        });
    }
    best
}

/// Whether run `a`'s scores beat `b`'s under the deterministic tie-break order.
fn better_run(a: &xybrid_sdk::eval::Scores, b: &xybrid_sdk::eval::Scores) -> bool {
    const EPS: f64 = 1e-9;
    if (a.quality - b.quality).abs() > EPS {
        return a.quality > b.quality;
    }
    match (a.latency_p95_ms, b.latency_p95_ms) {
        (Some(la), Some(lb)) if (la - lb).abs() > EPS => return la < lb,
        _ => {}
    }
    match (a.bundle_mb, b.bundle_mb) {
        (Some(ba), Some(bb)) if (ba - bb).abs() > EPS => return ba < bb,
        _ => {}
    }
    false
}

/// Format a gate verdict for display.
fn verdict_label(v: GateVerdict) -> String {
    match v {
        GateVerdict::Pass => ui::success("pass").to_string(),
        GateVerdict::Fail => ui::error("fail").to_string(),
        GateVerdict::Inconclusive => ui::warn("inconclusive").to_string(),
    }
}

// ============================================================================
// Commands — offline (no model)
// ============================================================================

fn discover_evalsets() -> Result<()> {
    let root = Path::new("evals");
    if !root.is_dir() {
        ui::header("Eval");
        ui::hint("No `evals/` directory here. Scaffold one with:");
        ui::hint("  xybrid eval init <task>");
        return Ok(());
    }
    ui::header("Evalsets");
    let mut found = 0;
    for entry in std::fs::read_dir(root)? {
        let dir = entry?.path();
        if dir.join("evalset.yaml").exists() {
            match LoadedEvalset::load(&dir) {
                Ok(set) => {
                    found += 1;
                    ui::bullet(
                        &set.manifest.name,
                        &format!(
                            "{:?} · {} cases · v{}",
                            set.manifest.task,
                            set.cases.len(),
                            set.manifest.version
                        ),
                    );
                }
                Err(e) => ui::warning(&format!("{}: {e}", dir.display())),
            }
        }
    }
    if found == 0 {
        ui::hint("No evalsets found. Create one with `xybrid eval init <task>`.");
    } else {
        ui::footer(&format!(
            "{found} evalset(s) · run one with `xybrid eval run <path> --model <id>`"
        ));
    }
    Ok(())
}

fn inspect(path: &Path) -> Result<()> {
    let set = LoadedEvalset::load(path)
        .with_context(|| format!("failed to load evalset at {}", path.display()))?;
    let m = &set.manifest;
    ui::header(&format!("Evalset · {}", m.name));
    println!();
    ui::kv("Task", &format!("{:?}", m.task).to_lowercase());
    ui::kv("Kind", &format!("{:?}", m.kind).to_lowercase());
    ui::kv("Version", &m.version.to_string());
    ui::kv("Cases", &set.cases.len().to_string());
    ui::kv("Grader", default_grader_label(m.task));
    if !m.labels.is_empty() {
        ui::kv("Labels", &m.labels.join(", "));
    }
    if let Some(gate) = &m.gate {
        if let Some(q) = gate.min_quality {
            ui::kv("Gate quality", &format!("≥ {q:.2}"));
        }
        if let Some(l) = gate.max_p95_latency_ms {
            ui::kv("Gate p95", &format!("≤ {l:.0} ms"));
        }
    }
    // Provenance + lifecycle summary.
    let flagged = set
        .cases
        .iter()
        .filter(|c| matches!(c.source, xybrid_sdk::eval::CaseSource::Flagged))
        .count();
    let quarantined = set.cases.iter().filter(|c| c.is_quarantined()).count();
    if flagged > 0 || quarantined > 0 {
        ui::kv(
            "Provenance",
            &format!("{flagged} flagged · {quarantined} quarantined"),
        );
    }
    if let Some(sample) = set.cases.first() {
        ui::section("Sample case");
        println!();
        ui::kv("id", &sample.id);
        ui::kv("input", &format!("{:?}", sample.input));
        if let Some(exp) = &sample.expected {
            ui::kv("expected", &format!("{:?}", exp));
        } else {
            ui::kv("expected", "(golden mode — none)");
        }
    }
    println!();
    ui::ok(&format!("Evalset valid ({} cases)", set.cases.len()));
    Ok(())
}

fn init(task: &str, name: Option<&str>, dir: &Path, force: bool) -> Result<()> {
    let task = parse_task(task)?;
    let name = name.unwrap_or_else(|| task_default_name(task)).to_string();
    let target = dir.join(&name);
    let manifest_path = target.join("evalset.yaml");
    if manifest_path.exists() && !force {
        anyhow::bail!(
            "evalset already exists at {} (use --force to overwrite)",
            manifest_path.display()
        );
    }
    std::fs::create_dir_all(&target)
        .with_context(|| format!("failed to create {}", target.display()))?;

    let mut manifest = Evalset::new(&name, task);
    if task == TaskType::Classify {
        manifest.labels = vec!["label_a".into(), "label_b".into()];
    }
    let yaml = serde_yaml::to_string(&manifest).context("failed to serialize manifest")?;
    std::fs::write(&manifest_path, yaml)?;
    std::fs::write(target.join("cases.jsonl"), "")?;

    ui::header(&format!("Eval init · {name}"));
    println!();
    ui::ok(&format!("Scaffolded {}", target.display()));
    ui::kv("Task", &format!("{task:?}").to_lowercase());
    ui::kv("Grader", default_grader_label(task));
    println!();
    ui::hint("Add cases to cases.jsonl, then:");
    ui::hint(&format!(
        "  xybrid eval run {} --model <id>",
        target.display()
    ));
    Ok(())
}

/// `xybrid eval inbox` — view the platform failure inbox: the read side of the
/// collect loop (explicit `Feedback` flags + monitor `Signal` auto-flags),
/// the terminal twin of the console inbox. Read-only; reuses the SDK
/// [`InboxClient`] against `/v1/telemetry/feedback` (auth via `XYBRID_API_KEY`,
/// base via `XYBRID_API_URL`). Minting full cases from these items needs the
/// original inference input (payload capture, joined by `trace_id`) and stays
/// on `eval pull`'s local path for now.
#[allow(clippy::too_many_arguments)]
fn inbox_view(
    period: &str,
    model: Option<&str>,
    source: &str,
    rating: Option<&str>,
    limit: u32,
    api_key: Option<&str>,
    platform_url: &str,
) -> Result<()> {
    // Resolve auth: the global `--api-key` flag (already env-fellback by clap)
    // wins; fall back to a raw `XYBRID_API_KEY` read so a bare env var also
    // works. The base URL is the CLI's resolved `--platform-url` (flag / env /
    // default), so both flag and env forms target the same place.
    let key = api_key.map(str::to_string).or_else(|| {
        std::env::var("XYBRID_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
    });
    let Some(key) = key else {
        ui::header("Eval inbox");
        ui::err("No API key set — the failure inbox is a platform feature.");
        ui::hint("Pass --api-key or set XYBRID_API_KEY to view flagged results.");
        ui::hint("Get a free key at https://dashboard.xybrid.dev");
        return Ok(());
    };
    let client = InboxClient::new(platform_url, key);

    let query = InboxQuery {
        period: Some(period.to_string()),
        model_id: model.map(str::to_string),
        source: (source != "all").then(|| source.to_string()),
        rating: rating.map(str::to_string),
        limit: Some(limit),
    };

    let resp = match client.fetch(&query) {
        Ok(r) => r,
        Err(e) => {
            // A rejected key reads as an HTTP 401/403 — surface the actionable
            // hint the missing-key branch can't (the request did go out).
            let msg = e.to_string();
            if msg.contains("HTTP 401") || msg.contains("HTTP 403") {
                ui::header("Eval inbox");
                ui::err("Platform rejected the API key (HTTP 401/403).");
                ui::hint("Check --api-key / XYBRID_API_KEY and the workspace it belongs to.");
                return Ok(());
            }
            return Err(anyhow::Error::new(e).context("failed to fetch the failure inbox"));
        }
    };

    ui::header(&format!("Eval inbox · {} window", resp.period));
    println!();
    let s = &resp.summary;
    ui::kv("Total flags", &s.total.to_string());
    ui::kv("Reported down", &s.down_count.to_string());
    ui::kv("Auto-flagged", &s.signal_count.to_string());
    if let Some(rate) = s.negative_rate {
        ui::kv("Negative rate", &format!("{:.0}%", rate * 100.0));
    }
    if !s.by_model.is_empty() {
        let top = s
            .by_model
            .iter()
            .map(|r| format!("{} ({})", r.key, r.count))
            .collect::<Vec<_>>()
            .join(", ");
        ui::kv("By model", &top);
    }
    println!();

    if resp.items.is_empty() {
        ui::hint("Inbox is empty for these filters.");
        ui::hint(
            "As users flag results (result.report) and the monitor auto-flags \
             degraded outputs, cases land here.",
        );
        return Ok(());
    }

    let mut table = ui::Table::new(vec!["When", "Source", "Model", "Task", "Detail", "Trace"]);
    let rows: Vec<[String; 6]> = resp
        .items
        .iter()
        .map(|it| {
            let src = match it.source.as_str() {
                "report" => match it.rating.as_deref() {
                    Some("up") => "reported up".to_string(),
                    _ => "reported down".to_string(),
                },
                _ => "auto-flag".to_string(),
            };
            let detail = if let Some(e) = &it.expected {
                format!("expected: {e}")
            } else if let Some(n) = &it.note {
                n.clone()
            } else if let Some(sn) = &it.signal_name {
                match &it.signal_kind {
                    Some(k) => format!("{k}: {sn}"),
                    None => sn.clone(),
                }
            } else {
                "-".to_string()
            };
            [
                inbox_fmt_when(&it.created_at),
                src,
                it.model_id.clone().unwrap_or_else(|| "-".to_string()),
                it.task.clone().unwrap_or_else(|| "-".to_string()),
                inbox_trunc(&detail, 48),
                it.trace_id
                    .as_deref()
                    .map(inbox_short_trace)
                    .unwrap_or_else(|| "-".to_string()),
            ]
        })
        .collect();
    for r in &rows {
        table.row(r.iter().map(String::as_str).collect());
    }
    table.print();

    println!();
    ui::footer(&format!(
        "{} of {} flagged result(s) · source: {}",
        resp.items.len(),
        resp.total,
        resp.data_source
    ));
    Ok(())
}

/// Compact `MM-DD HH:MM` rendering of an ISO-8601 timestamp (`YYYY-MM-DDThh:mm…`
/// → `MM-DD hh:mm`). Char-safe: a short or non-ASCII string (a malformed server
/// response) falls back to the raw value instead of panicking on a byte slice.
fn inbox_fmt_when(iso: &str) -> String {
    let chars: Vec<char> = iso.chars().collect();
    if chars.len() >= 16 {
        chars[5..16]
            .iter()
            .map(|&c| if c == 'T' { ' ' } else { c })
            .collect()
    } else {
        iso.to_string()
    }
}

/// Shorten a trace id for the table's last column.
fn inbox_short_trace(id: &str) -> String {
    if id.chars().count() > 12 {
        let head: String = id.chars().take(10).collect();
        format!("{head}…")
    } else {
        id.to_string()
    }
}

/// Truncate `s` to at most `n` characters, appending an ellipsis when cut.
fn inbox_trunc(s: &str, n: usize) -> String {
    if s.chars().count() > n {
        let head: String = s.chars().take(n).collect();
        format!("{head}…")
    } else {
        s.to_string()
    }
}

/// Default inbox path for an evalset: `~/.xybrid/inbox/<name>.jsonl` (the
/// platform-sync target). A real platform `pull` fetches from the inbox API;
/// this local file is the offline predecessor.
fn default_inbox_path(name: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().context("could not resolve home directory")?;
    Ok(home
        .join(".xybrid")
        .join("inbox")
        .join(format!("{name}.jsonl")))
}

/// Read pending cases from a local inbox file (JSONL of `Case`). Missing inbox →
/// empty (nothing to pull). Size-capped against a hostile/huge inbox.
fn read_inbox(path: &Path) -> Result<Vec<Case>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let meta = std::fs::metadata(path).with_context(|| format!("stat inbox {}", path.display()))?;
    if !meta.is_file() {
        anyhow::bail!("inbox {} is not a regular file", path.display());
    }
    if meta.len() > 64 * 1024 * 1024 {
        anyhow::bail!(
            "inbox {} is too large ({} bytes)",
            path.display(),
            meta.len()
        );
    }
    let src = std::fs::read_to_string(path)
        .with_context(|| format!("reading inbox {}", path.display()))?;
    let mut cases = Vec::new();
    for (i, line) in src.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let c: Case = serde_json::from_str(t)
            .with_context(|| format!("invalid inbox case {}:{}", path.display(), i + 1))?;
        cases.push(c);
    }
    Ok(cases)
}

/// Select pulled cases that are new to the evalset (dedupe by id), preserving
/// inbox order. Pure + unit-tested.
fn select_new_cases(existing: &[Case], pulled: &[Case]) -> Vec<Case> {
    use std::collections::HashSet;
    let have: HashSet<&str> = existing.iter().map(|c| c.id.as_str()).collect();
    pulled
        .iter()
        .filter(|c| !have.contains(c.id.as_str()))
        .cloned()
        .collect()
}

/// Append accepted cases to `cases.jsonl` and bump the manifest `version`.
/// Returns the new version. Pure-ish (filesystem only); unit-tested via a temp
/// dir.
fn append_and_bump(dir: &Path, manifest: &Evalset, accepted: &[Case]) -> Result<u32> {
    let cases_path = dir.join("cases.jsonl");
    let mut body = std::fs::read_to_string(&cases_path).unwrap_or_default();
    if !body.is_empty() && !body.ends_with('\n') {
        body.push('\n');
    }
    for c in accepted {
        body.push_str(&serde_json::to_string(c).context("serialize pulled case")?);
        body.push('\n');
    }
    std::fs::write(&cases_path, body)
        .with_context(|| format!("writing {}", cases_path.display()))?;

    let mut bumped = manifest.clone();
    bumped.version += 1;
    let yaml = serde_yaml::to_string(&bumped).context("serialize manifest")?;
    std::fs::write(dir.join("evalset.yaml"), yaml).context("writing evalset.yaml")?;
    Ok(bumped.version)
}

/// Rewrite the inbox with the remaining (un-pulled) cases.
fn write_inbox(path: &Path, remaining: &[Case]) -> Result<()> {
    let mut body = String::new();
    for c in remaining {
        body.push_str(&serde_json::to_string(c)?);
        body.push('\n');
    }
    std::fs::write(path, body).with_context(|| format!("writing inbox {}", path.display()))?;
    Ok(())
}

/// `xybrid eval pull` — drain the inbox into the evalset through a review queue.
fn pull(evalset_dir: &Path, inbox: Option<&Path>, accept_all: bool, dry_run: bool) -> Result<()> {
    let set = LoadedEvalset::load(evalset_dir)
        .with_context(|| format!("failed to load evalset at {}", evalset_dir.display()))?;
    let inbox_path = match inbox {
        Some(p) => p.to_path_buf(),
        None => default_inbox_path(&set.manifest.name)?,
    };

    ui::header(&format!("Eval pull · {}", set.manifest.name));
    println!();
    let pending = read_inbox(&inbox_path)?;
    if pending.is_empty() {
        ui::hint(&format!("No pending cases in {}", inbox_path.display()));
        ui::hint("Flag results in your app (result.report) — the platform syncs them here.");
        return Ok(());
    }
    let fresh = select_new_cases(&set.cases, &pending);
    ui::kv(
        "Pending",
        &format!(
            "{} ({} new, {} already pulled)",
            pending.len(),
            fresh.len(),
            pending.len() - fresh.len()
        ),
    );
    if fresh.is_empty() {
        return Ok(());
    }

    // Decide which fresh cases to accept.
    let accepted: Vec<Case> = if dry_run || accept_all {
        fresh.clone()
    } else {
        review_queue(&fresh)
    };

    if dry_run {
        println!();
        for c in &accepted {
            ui::bullet(&c.id, &format!("{:?}", c.input));
        }
        ui::footer(&format!(
            "{} case(s) would be pulled (dry run)",
            accepted.len()
        ));
        return Ok(());
    }

    if accepted.is_empty() {
        ui::hint("Nothing accepted; inbox unchanged.");
        return Ok(());
    }

    let new_version = append_and_bump(evalset_dir, &set.manifest, &accepted)?;
    // Remove accepted from the inbox; skipped/discarded handling: accepted are
    // gone, everything else stays pending for next time.
    use std::collections::HashSet;
    let accepted_ids: HashSet<&str> = accepted.iter().map(|c| c.id.as_str()).collect();
    let remaining: Vec<Case> = pending
        .into_iter()
        .filter(|c| !accepted_ids.contains(c.id.as_str()))
        .collect();
    write_inbox(&inbox_path, &remaining)?;

    println!();
    ui::ok(&format!(
        "Pulled {} case(s) → {} now v{}",
        accepted.len(),
        set.manifest.name,
        new_version
    ));
    Ok(())
}

/// Interactive review queue: accept / skip / discard each fresh case. Skipped
/// and discarded both stay out of the evalset (discarded would also be dropped
/// server-side in the platform flow). Falls back to accepting on EOF (piped
/// input) so non-interactive use without `--accept-all` still progresses.
fn review_queue(fresh: &[Case]) -> Vec<Case> {
    use std::io::Write;
    let mut accepted = Vec::new();
    for c in fresh {
        print!(
            "  {} {}  [a]ccept / [s]kip / [d]iscard? ",
            ui::accent(&c.id),
            ui::dim(&format!("{:?}", c.input))
        );
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) => {
                // EOF — accept the rest by default.
                accepted.push(c.clone());
            }
            Ok(_) => match line.trim().chars().next() {
                Some('s') | Some('S') => {}
                Some('d') | Some('D') => {}
                _ => accepted.push(c.clone()),
            },
            Err(_) => accepted.push(c.clone()),
        }
    }
    accepted
}

fn show(run_id: &str, store: Option<&Path>) -> Result<()> {
    let store = open_store(store)?;
    let run = store
        .load(run_id)
        .with_context(|| format!("no run '{run_id}' in {}", store.base().display()))?;
    print_run(&run);
    Ok(())
}

fn diff(run_a: &str, run_b: &str, store: Option<&Path>) -> Result<()> {
    let store = open_store(store)?;
    let a = store
        .load(run_a)
        .with_context(|| format!("no run '{run_a}'"))?;
    let b = store
        .load(run_b)
        .with_context(|| format!("no run '{run_b}'"))?;

    ui::header("Eval diff");
    println!();
    let mut table = ui::Table::new(vec![
        "metric",
        &a.candidate.model_id,
        &b.candidate.model_id,
        "Δ",
    ]);
    let qa = a.scores.quality;
    let qb = b.scores.quality;
    table.row(vec![
        "quality",
        &format!("{qa:.3}"),
        &format!("{qb:.3}"),
        &format!("{:+.3}", qb - qa),
    ]);
    table.row(vec![
        "pass",
        &format!("{}/{}", a.scores.pass, a.scores.pass + a.scores.fail),
        &format!("{}/{}", b.scores.pass, b.scores.pass + b.scores.fail),
        "",
    ]);
    let pa = a.scores.latency_p95_ms.unwrap_or(0.0);
    let pb = b.scores.latency_p95_ms.unwrap_or(0.0);
    table.row(vec![
        "p95 ms",
        &format!("{pa:.0}"),
        &format!("{pb:.0}"),
        &format!("{:+.0}", pb - pa),
    ]);
    table.print();
    println!();
    Ok(())
}

fn gate(
    evalset: &Path,
    run_id: Option<&str>,
    model: Option<&str>,
    strict: bool,
    store: Option<&Path>,
) -> Result<()> {
    let set = LoadedEvalset::load(evalset)
        .with_context(|| format!("failed to load evalset at {}", evalset.display()))?;

    // A gate command must enforce a threshold. If the manifest declares no
    // enforceable gate criteria, don't claim Pass (the report-only Pass that
    // `GatePolicy::evaluate` returns is for display paths, not for `gate`): warn
    // and exit 0 as a no-op. Done before any model run so it can't waste work.
    if !manifest_has_gate_thresholds(&set.manifest) {
        ui::header(&format!("Gate · {}", set.manifest.name));
        println!();
        ui::warning("evalset has no gate thresholds; nothing to enforce");
        return Ok(());
    }

    let run = if let Some(id) = run_id {
        open_store(store)?
            .load(id)
            .with_context(|| format!("no run '{id}'"))?
    } else if let Some(model_id) = model {
        execute_run(&set, model_id, None, true)?
    } else {
        anyhow::bail!("specify --run <id> to gate a stored run or --model <id> to run fresh");
    };

    // Re-evaluate against the CURRENT manifest gate. A stored run's verdict was
    // frozen at run time; tightening `gate.min_quality` and re-gating must use
    // the new policy, not the stale verdict.
    let policy = gate_policy(&set.manifest);
    let scores: Vec<f64> = run
        .cases
        .iter()
        .filter(|c| c.verdict != Verdict::Unblessed)
        .map(|c| c.score)
        .collect();
    let decision = policy.evaluate(&scores, run.scores.latency_p95_ms, None);

    ui::header(&format!("Gate · {}", set.manifest.name));
    println!();
    ui::kv("Candidate", &run.candidate.model_id);
    ui::kv("Quality", &format!("{:.3}", decision.quality));
    if let Some(ci) = &decision.ci {
        ui::kv("95% CI", &format!("[{:.3}, {:.3}]", ci.low, ci.high));
    }
    ui::kv("Verdict", &verdict_label(decision.verdict));
    ui::kv("Reason", &decision.reason);
    println!();

    let code = gate_exit_code(decision.verdict, strict);
    if code == 0 {
        ui::ok("Gate passed (or inconclusive — not blocking)");
    } else {
        ui::err("Gate blocked");
    }
    // Exit explicitly (not via Err) so the main error-chain handler for offline /
    // model-not-found errors doesn't misrender a clean gate failure.
    std::process::exit(code);
}

/// `xybrid eval ship` — promote a candidate that passes its gate by writing a
/// provenanced promotion record. The over-the-air *delivery* (canary ramp,
/// rollback) is a remote backend's job; this produces the artifact it consumes.
#[allow(clippy::too_many_arguments)]
fn ship(
    evalset: &Path,
    run_id: Option<&str>,
    model: Option<&str>,
    skill: Option<&str>,
    canary: u8,
    constraints: Vec<String>,
    store: Option<&Path>,
    deployments: Option<&Path>,
) -> Result<()> {
    let set = LoadedEvalset::load(evalset)
        .with_context(|| format!("failed to load evalset at {}", evalset.display()))?;
    // Shipping requires an enforceable gate — you cannot promote on "no criteria".
    if !manifest_has_gate_thresholds(&set.manifest) {
        anyhow::bail!(
            "evalset '{}' has no gate thresholds — refusing to ship without an eval gate",
            set.manifest.name
        );
    }

    let run = if let Some(id) = run_id {
        open_store(store)?
            .load(id)
            .with_context(|| format!("no run '{id}'"))?
    } else if let Some(model_id) = model {
        execute_run(&set, model_id, None, true)?
    } else {
        anyhow::bail!("specify --run <id> to ship a stored run or --model <id> to run fresh");
    };

    // Re-evaluate against the current gate; only a Pass may ship.
    let policy = gate_policy(&set.manifest);
    let scores: Vec<f64> = run
        .cases
        .iter()
        .filter(|c| c.verdict != Verdict::Unblessed)
        .map(|c| c.score)
        .collect();
    let decision = policy.evaluate(&scores, run.scores.latency_p95_ms, None);

    ui::header(&format!(
        "Ship · {} → {}",
        set.manifest.name, run.candidate.model_id
    ));
    println!();
    ui::kv("Quality", &format!("{:.3}", decision.quality));
    ui::kv("Verdict", &verdict_label(decision.verdict));

    if decision.verdict != GateVerdict::Pass {
        println!();
        ui::err(&format!("Gate did not pass: {}", decision.reason));
        anyhow::bail!(
            "refusing to ship: eval gate is {}",
            format!("{:?}", decision.verdict).to_lowercase()
        );
    }

    let record = xybrid_sdk::eval::PromotionRecord {
        deployment_id: format!("dep_{}", uuid::Uuid::new_v4().simple()),
        skill: skill.map(String::from),
        evalset: run.evalset.clone(),
        evalset_version: run.evalset_version,
        candidate: run.candidate.clone(),
        gate_verdict: decision.verdict,
        quality: decision.quality,
        ci: decision.ci.clone(),
        scorer_version: run.scores.scorer_version.clone(),
        judge: run.scores.judge.clone(),
        canary_pct: canary,
        device_constraints: constraints,
        run_id: run.run_id.clone(),
        created: Some(xybrid_sdk::eval::now_rfc3339()),
        status: xybrid_sdk::eval::DeploymentStatus::Pending,
        rollback_reason: None,
    };

    let store = match deployments {
        Some(dir) => xybrid_sdk::eval::DeploymentStore::with_dir(dir),
        None => xybrid_sdk::eval::DeploymentStore::default_location()
            .map_err(|e| anyhow::anyhow!("{e}"))?,
    };
    let dir = store.save(&record).map_err(|e| anyhow::anyhow!("{e}"))?;

    println!();
    ui::ok(&format!("Promotion recorded: {}", record.deployment_id));
    ui::kv(
        "Candidate",
        &format!("{} (run {})", record.candidate.model_id, record.run_id),
    );
    ui::kv(
        "Evalset",
        &format!("{} v{}", record.evalset, record.evalset_version),
    );
    ui::kv("Canary", &format!("{}%", record.canary_pct));
    if !record.device_constraints.is_empty() {
        ui::kv("Constraints", &record.device_constraints.join(", "));
    }
    ui::kv("Record", &dir.join("promotion.json").display().to_string());
    println!();
    ui::hint("Over-the-air canary ramp + auto-rollback are delivered by a remote backend");
    ui::hint("(not yet available); this record is the gated, provenanced artifact it consumes.");
    Ok(())
}

fn open_deployment_store(deployments: Option<&Path>) -> Result<xybrid_sdk::eval::DeploymentStore> {
    match deployments {
        Some(dir) => Ok(xybrid_sdk::eval::DeploymentStore::with_dir(dir)),
        None => xybrid_sdk::eval::DeploymentStore::default_location()
            .map_err(|e| anyhow::anyhow!("{e}")),
    }
}

/// `xybrid eval promote` — ramp a deployment's canary, re-gating first: the
/// candidate's gating run is re-evaluated against the *current* evalset gate
/// (promotion past canary requires a still-passing eval). This re-checks the
/// recorded run's scores against the live policy — it catches a tightened gate
/// or a changed evalset; it does not re-execute the model or re-measure latency
/// (that is the runner's job, and the `RemoteAuthority` seam's where the
/// fleet-side metrics live).
fn promote(
    deployment_id: &str,
    to: u8,
    evalset: &Path,
    store: Option<&Path>,
    deployments: Option<&Path>,
) -> Result<()> {
    let dstore = open_deployment_store(deployments)?;
    let mut rec = dstore
        .load(deployment_id)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if rec.status == xybrid_sdk::eval::DeploymentStatus::RolledBack {
        anyhow::bail!("deployment {deployment_id} was rolled back; cannot ramp");
    }

    // Re-evaluate the candidate's recorded gating run against the current
    // evalset gate (does not re-run the model — see fn docs).
    let set = LoadedEvalset::load(evalset)
        .with_context(|| format!("failed to load evalset at {}", evalset.display()))?;
    let run = open_store(store)?
        .load(&rec.run_id)
        .with_context(|| format!("gating run '{}' not found", rec.run_id))?;
    let policy = gate_policy(&set.manifest);
    let scores: Vec<f64> = run
        .cases
        .iter()
        .filter(|c| c.verdict != Verdict::Unblessed)
        .map(|c| c.score)
        .collect();
    let decision = policy.evaluate(&scores, run.scores.latency_p95_ms, None);

    ui::header(&format!("Promote · {deployment_id}"));
    println!();
    ui::kv("Re-gate", &verdict_label(decision.verdict));
    if decision.verdict != GateVerdict::Pass {
        println!();
        ui::err(&format!("gate no longer passes: {}", decision.reason));
        ui::hint("Consider `xybrid eval rollback` instead.");
        anyhow::bail!("ramp blocked: gate is not passing");
    }

    rec.ramp_to(to);
    dstore.save(&rec).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!();
    ui::ok(&format!(
        "Ramped {deployment_id} → {}% ({})",
        rec.canary_pct,
        format!("{:?}", rec.status).to_lowercase()
    ));
    ui::hint("Over-the-air delivery of the ramp is a remote backend's job.");
    Ok(())
}

/// `xybrid eval rollback` — record a rollback (a remote backend delivers it).
fn rollback(deployment_id: &str, reason: Option<&str>, deployments: Option<&Path>) -> Result<()> {
    let dstore = open_deployment_store(deployments)?;
    let mut rec = dstore
        .load(deployment_id)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rec.roll_back(&xybrid_sdk::eval::RollbackTrigger::GateRegression);
    if let Some(r) = reason {
        rec.rollback_reason = Some(r.to_string());
    }
    dstore.save(&rec).map_err(|e| anyhow::anyhow!("{e}"))?;
    ui::header(&format!("Rollback · {deployment_id}"));
    println!();
    ui::ok(&format!(
        "Rolled back: {}",
        rec.rollback_reason.as_deref().unwrap_or("(no reason)")
    ));
    ui::hint("A remote backend delivers the rollback to the fleet.");
    Ok(())
}

/// `xybrid eval deployments` — list recorded deployments (optionally per skill).
fn list_deployments(skill: Option<&str>, deployments: Option<&Path>) -> Result<()> {
    let dstore = open_deployment_store(deployments)?;
    let records: Vec<xybrid_sdk::eval::PromotionRecord> = match skill {
        Some(s) => dstore
            .deployments_for_skill(s)
            .map_err(|e| anyhow::anyhow!("{e}"))?,
        None => dstore
            .list()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .iter()
            .filter_map(|id| dstore.load(id).ok())
            .collect(),
    };
    ui::header("Deployments");
    println!();
    if records.is_empty() {
        ui::hint("No deployments recorded. Ship one with `xybrid eval ship`.");
        return Ok(());
    }
    let mut table = ui::Table::new(vec!["deployment", "skill", "candidate", "canary", "status"]);
    for r in &records {
        table.row(vec![
            &r.deployment_id,
            r.skill.as_deref().unwrap_or("-"),
            &r.candidate.model_id,
            &format!("{}%", r.canary_pct),
            &format!("{:?}", r.status).to_lowercase(),
        ]);
    }
    table.print();
    println!();
    Ok(())
}

/// `xybrid eval resolve` — the active deployment a device cohort runs for a
/// skill (the device-side of over-the-air delivery, against the local registry).
fn resolve_skill(skill: &str, cohort: u8, deployments: Option<&Path>) -> Result<()> {
    let dstore = open_deployment_store(deployments)?;
    ui::header(&format!("Resolve · {skill} (cohort {cohort})"));
    println!();
    match dstore
        .active_for_skill(skill, cohort)
        .map_err(|e| anyhow::anyhow!("{e}"))?
    {
        Some(r) => {
            ui::ok(&format!("Active deployment: {}", r.deployment_id));
            ui::kv("Candidate", &r.candidate.model_id);
            ui::kv("Evalset", &format!("{} v{}", r.evalset, r.evalset_version));
            ui::kv("Canary", &format!("{}%", r.canary_pct));
            ui::kv("Status", &format!("{:?}", r.status).to_lowercase());
        }
        None => {
            ui::warning(&format!(
                "No active deployment for '{skill}' at cohort {cohort}"
            ));
            ui::hint("Ship one with `xybrid eval ship <evalset> --skill <slug>`.");
        }
    }
    Ok(())
}

// ============================================================================
// Commands — model execution
// ============================================================================

fn run(
    evalset: &Path,
    model: &str,
    limit: Option<usize>,
    capture: bool,
    store: Option<&Path>,
) -> Result<()> {
    let mut set = LoadedEvalset::load(evalset)
        .with_context(|| format!("failed to load evalset at {}", evalset.display()))?;
    if let Some(n) = limit {
        set.cases.truncate(n);
    }
    ui::header(&format!("Eval run · {} → {}", set.manifest.name, model));
    let run = execute_run(&set, model, None, capture)?;

    let store = open_store(store)?;
    let dir = store.save(&run)?;
    // Emit an EvalRun telemetry event so the run lands on the console compare
    // leaderboard. Opt-out gated + a no-op without a configured exporter, so a
    // local-only run (no API key) emits nothing.
    xybrid_sdk::telemetry::publish_eval_run_event(&run);
    print_run(&run);
    ui::footer(&format!("Saved {} · {}", run.run_id, dir.display()));
    Ok(())
}

fn compare(evalset: &Path, models: &[String], auto: bool, store: Option<&Path>) -> Result<()> {
    let set = LoadedEvalset::load(evalset)
        .with_context(|| format!("failed to load evalset at {}", evalset.display()))?;

    let mut candidates: Vec<String> = models.to_vec();
    if auto {
        match suggest_candidates(&set.manifest.task) {
            Ok(suggested) if !suggested.is_empty() => {
                ui::hint(&format!("--auto suggested: {}", suggested.join(", ")));
                for s in suggested {
                    if !candidates.contains(&s) {
                        candidates.push(s);
                    }
                }
            }
            Ok(_) => ui::warning(&format!(
                "no auto-suggestions for task {:?}; pass --model explicitly",
                set.manifest.task
            )),
            Err(e) => ui::warning(&format!("registry suggestion unavailable: {e}")),
        }
    }
    if candidates.is_empty() {
        anyhow::bail!("no candidates — pass --model <id> (repeatable) or --auto");
    }

    ui::header(&format!("Eval compare · {}", set.manifest.name));
    let store = open_store(store)?;
    let mut runs: Vec<Run> = Vec::new();
    // The first candidate is the baseline for non-inferiority on the rest.
    let mut baseline_quality: Option<f64> = None;
    for model in &candidates {
        let run = execute_run(&set, model, baseline_quality, true)?;
        if baseline_quality.is_none() {
            baseline_quality = Some(run.scores.quality);
        }
        if let Err(e) = store.save(&run) {
            ui::warning(&format!("failed to save run for {model}: {e}"));
        }
        // Feed the console compare leaderboard (opt-out gated; no-op offline).
        xybrid_sdk::telemetry::publish_eval_run_event(&run);
        runs.push(run);
    }

    print_leaderboard(&runs);
    Ok(())
}

// ============================================================================
// Model execution glue
// ============================================================================

/// Resolve + extract a model and run the evalset through it.
fn execute_run(
    set: &LoadedEvalset,
    model_id: &str,
    baseline_quality: Option<f64>,
    capture: bool,
) -> Result<Run> {
    let client = RegistryClient::from_env().context("failed to initialize registry client")?;
    let sp = ui::spinner(&format!("Loading {model_id}…"));
    let dir = client
        .fetch_extracted(model_id, None, |_p| {})
        .with_context(|| format!("failed to fetch model '{model_id}'"))?;
    let metadata: ModelMetadata = serde_json::from_str(
        &std::fs::read_to_string(dir.join("model_metadata.json"))
            .context("model is missing model_metadata.json")?,
    )
    .context("failed to parse model_metadata.json")?;
    let mut executor = TemplateExecutor::with_base_path(
        dir.to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 model path"))?,
    );
    sp.finish_and_clear();

    let policy = gate_policy(&set.manifest);
    let judge = OverlapJudge::default();
    let use_judge = matches!(
        set.manifest.task,
        TaskType::Chat | TaskType::Summarize | TaskType::Vlm
    );
    let options = RunOptions {
        capture_outputs: capture,
        baseline_quality,
        // Exclude expired cases (quarantined are always excluded).
        today: Some(xybrid_sdk::eval::today_utc()),
    };
    let run_id = format!("run_{}", uuid::Uuid::new_v4().simple());

    let run = run_evalset(
        set,
        CandidateRef::new(model_id),
        current_environment(),
        &policy,
        if use_judge {
            Some(&judge as &dyn xybrid_sdk::eval::Judge)
        } else {
            None
        },
        &options,
        run_id,
        |case| {
            let envelope = case_to_envelope(set, case)?;
            let start = Instant::now();
            let output = executor
                .execute(&metadata, &envelope, None)
                .map_err(|e| e.to_string())?;
            let latency_ms = start.elapsed().as_millis() as u32;
            Ok(CaseOutcome {
                output: envelope_to_output(&output),
                latency_ms: Some(latency_ms),
            })
        },
    );
    Ok(run)
}

/// Build an input `Envelope` from a case, safely resolving any `file:` payload.
fn case_to_envelope(
    set: &LoadedEvalset,
    case: &xybrid_sdk::eval::Case,
) -> Result<Envelope, String> {
    use xybrid_sdk::eval::CaseInput;
    match &case.input {
        CaseInput::Text(t) => Ok(Envelope::new(EnvelopeKind::Text(t.clone()))),
        CaseInput::Audio(raw) => {
            if raw.starts_with("file:") {
                use std::io::Read;
                let path = set
                    .resolve_payload(case)
                    .ok_or_else(|| "audio payload reference missing".to_string())?
                    .map_err(|e| e.to_string())?;
                // DoS guard: a FIFO/device file reports len()==0 (passing a naive
                // size cap) and then blocks/streams unbounded on read. Require a
                // regular file, AND read through a bounded reader so a file that
                // grows past the cap mid-read still can't exhaust memory.
                let meta = std::fs::metadata(&path).map_err(|e| format!("stat audio: {e}"))?;
                if !meta.is_file() {
                    return Err("payload is not a regular file".to_string());
                }
                if meta.len() > MAX_PAYLOAD_BYTES {
                    return Err(format!("audio payload too large: {} bytes", meta.len()));
                }
                let file = std::fs::File::open(&path).map_err(|e| format!("open audio: {e}"))?;
                let mut buf = Vec::new();
                // Read at most MAX_PAYLOAD_BYTES + 1; an over-cap read is rejected.
                file.take(MAX_PAYLOAD_BYTES + 1)
                    .read_to_end(&mut buf)
                    .map_err(|e| format!("read audio: {e}"))?;
                if buf.len() as u64 > MAX_PAYLOAD_BYTES {
                    return Err(format!(
                        "audio payload too large: > {MAX_PAYLOAD_BYTES} bytes"
                    ));
                }
                Ok(Envelope::new(EnvelopeKind::Audio(buf)))
            } else {
                Err("inline audio not supported; use a file: reference".to_string())
            }
        }
        CaseInput::Image(_) => Err("image inputs are not yet supported in eval run".to_string()),
    }
}

/// Map an output envelope to a gradeable output.
fn envelope_to_output(envelope: &Envelope) -> GradeOutput {
    match &envelope.kind {
        EnvelopeKind::Text(t) => GradeOutput::Text(t.clone()),
        EnvelopeKind::Embedding(v) => GradeOutput::Embedding(v.clone()),
        // Audio / image outputs aren't text-gradable yet (tts/vlm golden is
        // deferred); store a marker so the case still records.
        _ => GradeOutput::Json(serde_json::json!({"non_text_output": true})),
    }
}

fn current_environment() -> Environment {
    let ep = ExecutionProviderInfo::current();
    Environment {
        host: ep.platform,
        backend: "local".to_string(),
        execution_provider: ep.name,
        sdk_version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// Ask the registry for comparable candidates by task type (best-effort).
fn suggest_candidates(task: &TaskType) -> Result<Vec<String>> {
    let client = RegistryClient::from_env()?;
    let models = client.list_models()?;
    let wanted = verb_to_hf_tasks(*task);
    let mut matches: Vec<_> = models
        .into_iter()
        .filter(|m| wanted.iter().any(|w| m.task.eq_ignore_ascii_case(w)))
        .collect();
    // Rank smallest-first; surface up to three suggestions (comparable or
    // smaller, plus one upsize).
    matches.sort_by_key(|m| m.parameters);
    Ok(matches.into_iter().take(3).map(|m| m.id).collect())
}

// ============================================================================
// Rendering
// ============================================================================

fn print_run(run: &Run) {
    println!();
    ui::kv("Run", &run.run_id);
    ui::kv(
        "Evalset",
        &format!("{} v{}", run.evalset, run.evalset_version),
    );
    ui::kv("Candidate", &run.candidate.model_id);
    ui::kv(
        "Quality",
        &format!(
            "{:.3}  ({}/{} pass)",
            run.scores.quality,
            run.scores.pass,
            run.scores.pass + run.scores.fail
        ),
    );
    if let Some(ci) = &run.scores.ci {
        ui::kv("95% CI", &format!("[{:.3}, {:.3}]", ci.low, ci.high));
    }
    if let Some(p95) = run.scores.latency_p95_ms {
        ui::kv("p95 latency", &format!("{p95:.0} ms"));
    }
    if run.scores.unblessed > 0 {
        ui::kv("Unblessed", &run.scores.unblessed.to_string());
    }
    if run.scores.crash_or_timeout > 0 {
        ui::kv("Crashes", &run.scores.crash_or_timeout.to_string());
    }
    ui::kv("Verdict", &verdict_label(run.scores.verdict));
}

fn print_leaderboard(runs: &[Run]) {
    println!();
    // The first candidate is the baseline; show each other candidate's quality
    // delta against it (the leaderboard's "+0.06 vs base").
    let baseline = runs.first().map(|r| r.scores.quality);
    let mut table = ui::Table::new(vec!["candidate", "quality", "Δ vs base", "p95", "verdict"]);
    for (i, r) in runs.iter().enumerate() {
        let delta = match (i, baseline) {
            (0, _) => "base".to_string(),
            (_, Some(b)) => format!("{:+.3}", r.scores.quality - b),
            (_, None) => "-".to_string(),
        };
        table.row(vec![
            &r.candidate.model_id,
            &format!("{:.3}", r.scores.quality),
            &delta,
            &r.scores
                .latency_p95_ms
                .map(|p| format!("{p:.0}ms"))
                .unwrap_or_else(|| "-".into()),
            &format!("{:?}", r.scores.verdict).to_lowercase(),
        ]);
    }
    table.print();
    println!();
    match recommend_winner(runs) {
        Some(i) => {
            ui::ok(&format!(
                "recommended: {} (quality {:.3}, {})",
                runs[i].candidate.model_id,
                runs[i].scores.quality,
                verdict_label(runs[i].scores.verdict)
            ));
            ui::hint("hard-constraint failures and flaky candidates are excluded; ties break on p95 latency then bundle size");
        }
        None => ui::warning("no candidate passed the hard constraints"),
    }
}

// ============================================================================
// Utilities
// ============================================================================

fn open_store(store: Option<&Path>) -> Result<EvalRunStore> {
    match store {
        Some(dir) => Ok(EvalRunStore::with_dir(dir)),
        None => EvalRunStore::default_location().map_err(|e| anyhow::anyhow!("{e}")),
    }
}

fn task_default_name(task: TaskType) -> &'static str {
    match task {
        TaskType::Classify => "classifier",
        TaskType::Chat => "chat",
        TaskType::Summarize => "summarize",
        TaskType::Extract => "extract",
        TaskType::Asr => "asr",
        TaskType::Tts => "tts",
        TaskType::Embedding => "embedding",
        TaskType::Vlm => "vlm",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use xybrid_sdk::eval::{CandidateRef, Environment, Run, Scores};

    #[test]
    fn parse_task_accepts_all_verbs() {
        for t in [
            "classify",
            "chat",
            "summarize",
            "extract",
            "asr",
            "tts",
            "embedding",
            "vlm",
        ] {
            assert!(parse_task(t).is_ok(), "{t} should parse");
        }
        assert!(parse_task("telekinesis").is_err());
        assert!(parse_task(" classify ").is_ok()); // trims
    }

    #[test]
    fn gate_exit_codes() {
        assert_eq!(gate_exit_code(GateVerdict::Pass, false), 0);
        assert_eq!(gate_exit_code(GateVerdict::Fail, false), 2);
        assert_eq!(gate_exit_code(GateVerdict::Inconclusive, false), 0);
        assert_eq!(gate_exit_code(GateVerdict::Inconclusive, true), 2);
    }

    #[test]
    fn manifest_has_gate_thresholds_detects_enforceable_gates() {
        use xybrid_sdk::eval::Gate;
        // No gate block → nothing to enforce.
        assert!(!manifest_has_gate_thresholds(&Evalset::new(
            "s",
            TaskType::Classify
        )));
        let with_threshold = |g: Gate| {
            let mut m = Evalset::new("s", TaskType::Classify);
            m.gate = Some(g);
            manifest_has_gate_thresholds(&m)
        };
        // An empty gate block → still nothing enforceable.
        assert!(!with_threshold(Gate::default()));
        // Either threshold makes it enforceable.
        assert!(with_threshold(Gate {
            min_quality: Some(0.9),
            ..Gate::default()
        }));
        assert!(with_threshold(Gate {
            max_p95_latency_ms: Some(800.0),
            ..Gate::default()
        }));
    }

    #[test]
    fn default_grader_labels_are_non_empty() {
        for t in [
            TaskType::Classify,
            TaskType::Asr,
            TaskType::Extract,
            TaskType::Chat,
            TaskType::Tts,
            TaskType::Embedding,
        ] {
            assert!(!default_grader_label(t).is_empty());
        }
    }

    #[test]
    fn verb_to_hf_tasks_cover_known_verbs() {
        assert!(verb_to_hf_tasks(TaskType::Asr).contains(&"automatic-speech-recognition"));
        assert!(verb_to_hf_tasks(TaskType::Tts).contains(&"text-to-speech"));
        assert!(!verb_to_hf_tasks(TaskType::Chat).is_empty());
    }

    fn run_with(
        model: &str,
        quality: f64,
        verdict: GateVerdict,
        p95: Option<f64>,
        flaky: bool,
    ) -> Run {
        let mut scores = Scores {
            quality,
            verdict,
            latency_p95_ms: p95,
            flaky,
            ..Scores::default()
        };
        scores.pass = (quality * 10.0) as usize;
        Run {
            run_id: format!("run_{model}"),
            evalset: "s".into(),
            evalset_version: 1,
            candidate: CandidateRef::new(model),
            environment: Environment {
                host: "h".into(),
                backend: "b".into(),
                execution_provider: "cpu".into(),
                sdk_version: "0".into(),
            },
            scores,
            cases: vec![],
            created: None,
        }
    }

    #[test]
    fn recommend_winner_picks_highest_quality_passing() {
        let runs = vec![
            run_with("a", 0.80, GateVerdict::Pass, Some(400.0), false),
            run_with("b", 0.95, GateVerdict::Pass, Some(600.0), false),
            run_with("c", 0.99, GateVerdict::Fail, Some(300.0), false), // fails hard constraint
        ];
        assert_eq!(recommend_winner(&runs), Some(1)); // b: highest quality among passing
    }

    #[test]
    fn recommend_winner_breaks_ties_on_latency() {
        let runs = vec![
            run_with("a", 0.90, GateVerdict::Pass, Some(600.0), false),
            run_with("b", 0.90, GateVerdict::Pass, Some(400.0), false), // same quality, faster
        ];
        assert_eq!(recommend_winner(&runs), Some(1));
    }

    #[test]
    fn recommend_winner_excludes_flaky() {
        let runs = vec![
            run_with("a", 0.99, GateVerdict::Pass, Some(400.0), true), // flaky → excluded
            run_with("b", 0.85, GateVerdict::Pass, Some(400.0), false),
        ];
        assert_eq!(recommend_winner(&runs), Some(1));
    }

    #[test]
    fn recommend_winner_none_when_all_fail() {
        let runs = vec![
            run_with("a", 0.99, GateVerdict::Fail, None, false),
            run_with("b", 0.80, GateVerdict::Fail, None, false),
        ];
        assert_eq!(recommend_winner(&runs), None);
    }

    #[test]
    fn recommend_winner_excludes_inconclusive_even_if_higher_quality() {
        // C2: an inconclusive candidate is never promoted, even at higher
        // quality. The lower-quality Pass wins.
        let runs = vec![
            run_with("pass", 0.85, GateVerdict::Pass, Some(400.0), false),
            run_with("incon", 0.99, GateVerdict::Inconclusive, Some(400.0), false),
        ];
        assert_eq!(recommend_winner(&runs), Some(0));
    }

    #[test]
    fn recommend_winner_excludes_non_finite_quality() {
        // C8: a candidate whose quality is NaN must never be recommended.
        let runs = vec![
            run_with("nan", f64::NAN, GateVerdict::Pass, Some(400.0), false),
            run_with("ok", 0.80, GateVerdict::Pass, Some(400.0), false),
        ];
        assert_eq!(recommend_winner(&runs), Some(1));
        // …and if the NaN candidate is the only one, nothing qualifies.
        let only_nan = vec![run_with("nan", f64::NAN, GateVerdict::Pass, None, false)];
        assert_eq!(recommend_winner(&only_nan), None);
    }

    #[test]
    fn case_to_envelope_rejects_non_regular_payload_file() {
        // S1: a `file:` audio payload that resolves to a directory (stand-in for
        // a FIFO/device file — both fail is_file) is rejected, never read (so it
        // can't hang on an unbounded read).
        use xybrid_sdk::eval::Case;
        let dir = TempDir::new().unwrap();
        // Create a directory INSIDE the evalset root, referenced as the payload.
        std::fs::create_dir_all(dir.path().join("clip.wav")).unwrap();
        let set = LoadedEvalset {
            manifest: Evalset::new("s", TaskType::Asr),
            cases: vec![],
            root: dir.path().to_path_buf(),
        };
        let case = Case::new(
            "c1",
            xybrid_sdk::eval::CaseInput::Audio("file:clip.wav".into()),
        );
        let err = case_to_envelope(&set, &case).unwrap_err();
        assert!(
            err.contains("not a regular file"),
            "expected regular-file rejection, got: {err}"
        );
    }

    #[test]
    fn init_scaffolds_loadable_evalset() {
        let dir = TempDir::new().unwrap();
        init("classify", Some("intent"), dir.path(), false).unwrap();
        let set = LoadedEvalset::load(dir.path().join("intent")).unwrap();
        assert_eq!(set.manifest.task, TaskType::Classify);
        assert_eq!(set.manifest.name, "intent");
        assert!(set.cases.is_empty());
        // re-init without --force fails
        assert!(init("classify", Some("intent"), dir.path(), false).is_err());
        // with --force succeeds
        assert!(init("classify", Some("intent"), dir.path(), true).is_ok());
    }

    #[test]
    fn select_new_cases_dedupes_by_id() {
        let existing = vec![Case::new(
            "a",
            xybrid_sdk::eval::CaseInput::Text("x".into()),
        )];
        let pulled = vec![
            Case::new("a", xybrid_sdk::eval::CaseInput::Text("x".into())), // dup
            Case::new("b", xybrid_sdk::eval::CaseInput::Text("y".into())), // new
        ];
        let fresh = select_new_cases(&existing, &pulled);
        assert_eq!(fresh.len(), 1);
        assert_eq!(fresh[0].id, "b");
    }

    #[test]
    fn read_inbox_parses_and_skips_blanks() {
        let dir = TempDir::new().unwrap();
        let inbox = dir.path().join("inbox.jsonl");
        std::fs::write(
            &inbox,
            "{\"id\":\"c1\",\"input\":{\"text\":\"x\"},\"source\":\"flagged\",\"trace_id\":\"tr_9\"}\n\n{\"id\":\"c2\",\"input\":{\"text\":\"y\"}}\n",
        )
        .unwrap();
        let cases = read_inbox(&inbox).unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].source, xybrid_sdk::eval::CaseSource::Flagged);
        assert_eq!(cases[0].trace_id.as_deref(), Some("tr_9"));
        // missing inbox → empty
        assert!(read_inbox(&dir.path().join("nope.jsonl"))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn pull_accept_all_lands_flagged_case_and_bumps_version() {
        // flag → inbox → pull → case present with source:flagged + trace_id, v bumped.
        let dir = TempDir::new().unwrap();
        let eval_dir = dir.path().join("intent");
        std::fs::create_dir_all(&eval_dir).unwrap();
        std::fs::write(
            eval_dir.join("evalset.yaml"),
            "name: intent\ntask: classify\nversion: 3\nlabels: [refund, cancel]\n",
        )
        .unwrap();
        std::fs::write(eval_dir.join("cases.jsonl"), "").unwrap();
        let inbox = dir.path().join("inbox.jsonl");
        std::fs::write(
            &inbox,
            "{\"id\":\"f1\",\"input\":{\"text\":\"refund please\"},\"expected\":{\"label\":\"refund\"},\"source\":\"flagged\",\"trace_id\":\"tr_1\"}\n",
        )
        .unwrap();

        pull(&eval_dir, Some(&inbox), true, false).unwrap();

        let set = LoadedEvalset::load(&eval_dir).unwrap();
        assert_eq!(set.cases.len(), 1);
        assert_eq!(set.cases[0].id, "f1");
        assert_eq!(set.cases[0].source, xybrid_sdk::eval::CaseSource::Flagged);
        assert_eq!(set.cases[0].trace_id.as_deref(), Some("tr_1"));
        assert_eq!(set.manifest.version, 4); // bumped from 3
                                             // the inbox is drained (accepted case removed).
        assert!(read_inbox(&inbox).unwrap().is_empty());

        // a second pull with the same (now-empty) inbox is a no-op, no double-add.
        pull(&eval_dir, Some(&inbox), true, false).unwrap();
        assert_eq!(LoadedEvalset::load(&eval_dir).unwrap().cases.len(), 1);
    }

    #[test]
    fn pull_dry_run_writes_nothing() {
        let dir = TempDir::new().unwrap();
        let eval_dir = dir.path().join("s");
        std::fs::create_dir_all(&eval_dir).unwrap();
        std::fs::write(
            eval_dir.join("evalset.yaml"),
            "name: s\ntask: chat\nversion: 1\n",
        )
        .unwrap();
        std::fs::write(eval_dir.join("cases.jsonl"), "").unwrap();
        let inbox = dir.path().join("inbox.jsonl");
        std::fs::write(&inbox, "{\"id\":\"x\",\"input\":{\"text\":\"hi\"}}\n").unwrap();

        pull(&eval_dir, Some(&inbox), false, true).unwrap();
        // nothing pulled, version unchanged, inbox intact.
        let set = LoadedEvalset::load(&eval_dir).unwrap();
        assert!(set.cases.is_empty());
        assert_eq!(set.manifest.version, 1);
        assert_eq!(read_inbox(&inbox).unwrap().len(), 1);
    }

    /// Build + store a run with `n` cases each scoring `score`.
    fn store_run(run_store: &Path, run_id: &str, n: usize, score: f64) {
        let cases: Vec<xybrid_sdk::eval::RunCase> = (0..n)
            .map(|i| xybrid_sdk::eval::RunCase {
                id: format!("c{i}"),
                output: None,
                verdict: if score >= 0.5 {
                    Verdict::Pass
                } else {
                    Verdict::Fail
                },
                score,
                latency_ms: Some(100),
                detail: None,
            })
            .collect();
        let run = Run {
            run_id: run_id.into(),
            evalset: "intent".into(),
            evalset_version: 3,
            candidate: CandidateRef::new("qwen3.5-0.8b"),
            environment: Environment {
                host: "h".into(),
                backend: "b".into(),
                execution_provider: "cpu".into(),
                sdk_version: "0".into(),
            },
            scores: Scores {
                quality: score,
                pass: if score >= 0.5 { n } else { 0 },
                fail: if score >= 0.5 { 0 } else { n },
                verdict: GateVerdict::Pass,
                latency_p95_ms: Some(100.0),
                ..Scores::default()
            },
            cases,
            created: None,
        };
        EvalRunStore::with_dir(run_store).save(&run).unwrap();
    }

    #[test]
    fn ship_records_promotion_only_when_gate_passes() {
        let dir = TempDir::new().unwrap();
        let eval_dir = dir.path().join("intent");
        std::fs::create_dir_all(&eval_dir).unwrap();
        std::fs::write(
            eval_dir.join("evalset.yaml"),
            "name: intent\ntask: classify\nversion: 3\nlabels: [a, b]\ngate:\n  min_quality: 0.9\n  min_cases: 10\n",
        )
        .unwrap();
        std::fs::write(eval_dir.join("cases.jsonl"), "").unwrap();
        let run_store = dir.path().join("runs");
        let dep_store = dir.path().join("deployments");
        store_run(&run_store, "run_pass", 20, 1.0); // all pass → quality 1.0 ≥ 0.9
        store_run(&run_store, "run_fail", 20, 0.0); // all fail → quality 0.0 < 0.9

        // Passing run → promotion recorded with full provenance.
        ship(
            &eval_dir,
            Some("run_pass"),
            None,
            None,
            5,
            vec!["os=ios>=16".into()],
            Some(&run_store),
            Some(&dep_store),
        )
        .unwrap();
        let deps = xybrid_sdk::eval::DeploymentStore::with_dir(&dep_store);
        let ids = deps.list().unwrap();
        assert_eq!(ids.len(), 1);
        let rec = deps.load(&ids[0]).unwrap();
        assert_eq!(rec.gate_verdict, GateVerdict::Pass);
        assert_eq!(rec.evalset_version, 3);
        assert_eq!(rec.run_id, "run_pass");
        assert_eq!(rec.candidate.model_id, "qwen3.5-0.8b");
        assert_eq!(rec.canary_pct, 5);
        assert_eq!(rec.device_constraints, vec!["os=ios>=16".to_string()]);

        // Failing run → refused, no new deployment recorded.
        let err = ship(
            &eval_dir,
            Some("run_fail"),
            None,
            None,
            5,
            vec![],
            Some(&run_store),
            Some(&dep_store),
        );
        assert!(err.is_err());
        assert_eq!(deps.list().unwrap().len(), 1); // unchanged
    }

    #[test]
    fn ship_refuses_evalset_without_gate() {
        let dir = TempDir::new().unwrap();
        let eval_dir = dir.path().join("nogate");
        std::fs::create_dir_all(&eval_dir).unwrap();
        std::fs::write(
            eval_dir.join("evalset.yaml"),
            "name: nogate\ntask: chat\nversion: 1\n",
        )
        .unwrap();
        std::fs::write(eval_dir.join("cases.jsonl"), "").unwrap();
        let run_store = dir.path().join("runs");
        store_run(&run_store, "run_x", 10, 1.0);
        let err = ship(
            &eval_dir,
            Some("run_x"),
            None,
            None,
            5,
            vec![],
            Some(&run_store),
            Some(&dir.path().join("d")),
        );
        assert!(err.is_err(), "must refuse to ship without a gate");
    }

    #[test]
    fn promote_ramps_when_gate_holds_and_rollback_freezes() {
        let dir = TempDir::new().unwrap();
        let eval_dir = dir.path().join("intent");
        std::fs::create_dir_all(&eval_dir).unwrap();
        std::fs::write(
            eval_dir.join("evalset.yaml"),
            "name: intent\ntask: classify\nversion: 3\nlabels: [a, b]\ngate:\n  min_quality: 0.9\n  min_cases: 10\n",
        )
        .unwrap();
        std::fs::write(eval_dir.join("cases.jsonl"), "").unwrap();
        let run_store = dir.path().join("runs");
        let dep_store = dir.path().join("deployments");
        store_run(&run_store, "run_win", 20, 1.0);

        // ship → a Pending deployment at 5%.
        ship(
            &eval_dir,
            Some("run_win"),
            None,
            None,
            5,
            vec![],
            Some(&run_store),
            Some(&dep_store),
        )
        .unwrap();
        let deps = xybrid_sdk::eval::DeploymentStore::with_dir(&dep_store);
        let id = deps.list().unwrap()[0].clone();

        // promote to 50% — re-gate still passes → ramps.
        promote(&id, 50, &eval_dir, Some(&run_store), Some(&dep_store)).unwrap();
        assert_eq!(deps.load(&id).unwrap().canary_pct, 50);

        // rollback → status frozen.
        rollback(&id, Some("manual: customer complaints"), Some(&dep_store)).unwrap();
        let rec = deps.load(&id).unwrap();
        assert_eq!(rec.status, xybrid_sdk::eval::DeploymentStatus::RolledBack);
        assert_eq!(
            rec.rollback_reason.as_deref(),
            Some("manual: customer complaints")
        );

        // promote after rollback is refused.
        assert!(promote(&id, 100, &eval_dir, Some(&run_store), Some(&dep_store)).is_err());
    }

    #[test]
    fn promote_blocked_when_gate_regresses() {
        let dir = TempDir::new().unwrap();
        let eval_dir = dir.path().join("intent");
        std::fs::create_dir_all(&eval_dir).unwrap();
        // ship against a lenient gate, then tighten it so the re-gate fails.
        std::fs::write(
            eval_dir.join("evalset.yaml"),
            "name: intent\ntask: classify\nversion: 1\nlabels: [a, b]\ngate:\n  min_quality: 0.5\n  min_cases: 10\n",
        )
        .unwrap();
        std::fs::write(eval_dir.join("cases.jsonl"), "").unwrap();
        let run_store = dir.path().join("runs");
        let dep_store = dir.path().join("deployments");
        store_run(&run_store, "run_mid", 20, 0.7); // quality 0.7 — passes 0.5, fails 0.9

        ship(
            &eval_dir,
            Some("run_mid"),
            None,
            None,
            5,
            vec![],
            Some(&run_store),
            Some(&dep_store),
        )
        .unwrap();
        let deps = xybrid_sdk::eval::DeploymentStore::with_dir(&dep_store);
        let id = deps.list().unwrap()[0].clone();

        // tighten the gate; re-gate now fails → promote refused.
        std::fs::write(
            eval_dir.join("evalset.yaml"),
            "name: intent\ntask: classify\nversion: 1\nlabels: [a, b]\ngate:\n  min_quality: 0.9\n  min_cases: 10\n",
        )
        .unwrap();
        assert!(promote(&id, 50, &eval_dir, Some(&run_store), Some(&dep_store)).is_err());
        assert_eq!(deps.load(&id).unwrap().canary_pct, 5); // unchanged
    }

    #[test]
    fn ship_with_skill_is_resolvable_per_cohort() {
        let dir = TempDir::new().unwrap();
        let eval_dir = dir.path().join("intent");
        std::fs::create_dir_all(&eval_dir).unwrap();
        std::fs::write(
            eval_dir.join("evalset.yaml"),
            "name: intent\ntask: classify\nversion: 3\nlabels: [a, b]\ngate:\n  min_quality: 0.9\n  min_cases: 10\n",
        )
        .unwrap();
        std::fs::write(eval_dir.join("cases.jsonl"), "").unwrap();
        let run_store = dir.path().join("runs");
        let dep_store = dir.path().join("deployments");
        store_run(&run_store, "run_win", 20, 1.0);

        // ship for the "support-intent" skill at 5% canary.
        ship(
            &eval_dir,
            Some("run_win"),
            None,
            Some("support-intent"),
            5,
            vec![],
            Some(&run_store),
            Some(&dep_store),
        )
        .unwrap();

        let deps = xybrid_sdk::eval::DeploymentStore::with_dir(&dep_store);
        // a device in cohort 2 (< 5%) resolves the new deployment...
        let active = deps.active_for_skill("support-intent", 2).unwrap().unwrap();
        assert_eq!(active.skill.as_deref(), Some("support-intent"));
        assert!(active.created.is_some()); // stamped for ordering
                                           // ...a device in cohort 50 is not yet in the canary.
        assert!(deps
            .active_for_skill("support-intent", 50)
            .unwrap()
            .is_none());
        // the CLI list/resolve handlers run without error.
        assert!(list_deployments(Some("support-intent"), Some(&dep_store)).is_ok());
        assert!(resolve_skill("support-intent", 2, Some(&dep_store)).is_ok());
    }

    #[test]
    fn inspect_reports_valid_evalset() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("evalset.yaml"),
            "name: s\ntask: classify\nlabels: [a, b]\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("cases.jsonl"),
            "{\"id\":\"c1\",\"input\":{\"text\":\"x\"},\"expected\":{\"label\":\"a\"}}\n",
        )
        .unwrap();
        // inspect prints to stdout; assert it doesn't error on a valid set.
        assert!(inspect(dir.path()).is_ok());
    }

    #[test]
    fn shipped_reference_packs_validate_clean() {
        let packs = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference-packs");
        // Three reference packs ship with the CLI (classify, chat, asr).
        for name in [
            "classify-sentiment",
            "chat-helpfulness",
            "asr-commands",
            "safety-prompt-injection",
        ] {
            let set = LoadedEvalset::load(packs.join(name))
                .unwrap_or_else(|e| panic!("reference pack {name} failed to load: {e}"));
            assert!(!set.cases.is_empty(), "{name} has no cases");
            // Every case carries an expected reference (reference packs are
            // curated, not golden).
            assert!(
                set.cases.iter().all(|c| c.expected.is_some()),
                "{name} has unblessed cases"
            );
        }
    }

    #[test]
    fn inbox_fmt_when_formats_iso_and_is_multibyte_safe() {
        // Canonical Postgres `to_char` output → compact MM-DD HH:MM.
        assert_eq!(inbox_fmt_when("2026-06-27T10:00:00.000Z"), "06-27 10:00");
        // Short / odd input falls back to the raw string, no panic.
        assert_eq!(inbox_fmt_when("2026-06"), "2026-06");
        // A malformed multibyte timestamp must NOT panic on a byte slice
        // (regression: byte-indexed `iso[5..16]` panicked mid-char).
        let weird = "20€6-06-27Ti🙂:00:00Z"; // ≥16 chars, multibyte at boundaries
        let _ = inbox_fmt_when(weird); // must not panic
    }

    #[test]
    fn inbox_trunc_and_short_trace_are_char_safe() {
        assert_eq!(inbox_trunc("short", 48), "short");
        assert_eq!(inbox_trunc("abcdef", 3), "abc…");
        assert_eq!(inbox_short_trace("tr_short"), "tr_short");
        assert_eq!(inbox_short_trace("tr_0123456789abcdef"), "tr_0123456…");
        // Multibyte content must not panic.
        let _ = inbox_trunc("café — déjà vu, naïve façade, 🙂🙂🙂", 5);
    }
}
