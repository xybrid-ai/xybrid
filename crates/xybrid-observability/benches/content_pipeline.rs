//! Content pipeline cost: what recording one content value costs the calling
//! thread — capture gate, preflight, clone, redaction (callback hook and
//! declarative rules), truncation to the 16 KiB attribute budget, and the
//! retained-operation bookkeeping (CONTRACT.md "Content pipeline").
//!
//! Measured call: `InferenceOperation::record_input_messages` on a fresh
//! operation of an observer over an SDK tracer with an in-memory exporter.
//! Starting and ending the operation are not timed.
//!
//! Workloads (representative prose with e-mail, phone and key strings):
//!
//! | id | value | path |
//! |---|---|---|
//! | `msg_256b`, `msg_1k`, `msg_4k` | one user message | under the budget, no truncation |
//! | `msg_15k` | one message, JSON just under 16 KiB | largest untruncated |
//! | `msg_17k`, `msg_64k`, `msg_256k` | one message | string halving (truncation steps 2-3) |
//! | `hist_32x4k` | 32 × 4 KiB messages (~128 KiB) | halving across many strings |
//! | `hist_near_budget_4k` | 4 KiB messages up to the 480 KiB traversal budget | largest accepted history, halving |
//! | `hist_near_budget_64b` | 64 B messages up to the budget | halving cannot help: oldest-message removal (step 4) |
//! | `hist_over_budget` | `hist_near_budget_4k` plus one message | preflight fails closed |
//!
//! Policies: `metadata_only` (gate closed: the floor), `default` (no
//! redaction), `rules` (three `replace_literal` rules and an
//! `omit_attribute` rule on another key), `hook_and_rules` (a substring
//! callback redactor, then the same rules), and `unsampled` (the default
//! policy under a tracer whose sampler drops every span: nothing can be
//! recorded, so this measures what content processing costs an operation
//! whose span is not recording — unsampled parents, sampler drops).
//!
//! Each (policy, workload) outcome is checked once before measuring, so a
//! bench never silently measures a different path. Expected outcomes are the
//! table's, except: `metadata_only` and `unsampled` record nothing, and under
//! `hook_and_rules` the near-budget histories fail closed, because the
//! callback's `[REDACTED]` replacement is longer than the words it replaces
//! and pushes them over the traversal budget (CONTRACT.md step 6). The side
//! report `content_pipeline.json` records sizes, preflight costs and
//! outcomes.
//!
//! ```bash
//! cargo bench -p xybrid-observability --bench content_pipeline
//! ```

mod support;

use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use opentelemetry_sdk::trace::{InMemorySpanExporter, Sampler, SdkTracerProvider};
use serde_json::json;
use xybrid_observability::semconv::{attr, DEFAULT_MAX_TRAVERSAL_BYTES};
use xybrid_observability::{
    ContentCapture, ContentPolicy, InferenceRequest, InputMessage, Location, Observer,
    RedactionRule, StartOptions, SubstringRedactor,
};

/// Measured calls between exporter resets (outside the timed region).
const DRAIN_EVERY: u64 = 256;

/// What a workload is expected to do to `gen_ai.input.messages`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expect {
    Kept,
    Truncated,
    FailedClosed,
    /// The capture gate is closed: nothing recorded, nothing marked.
    Skipped,
}

/// The outcome a workload must have under a policy (see the module docs).
fn expected(policy: &str, w: &Workload) -> Expect {
    match policy {
        "metadata_only" | "unsampled" => Expect::Skipped,
        "hook_and_rules" if w.id.starts_with("hist_near_budget") => Expect::FailedClosed,
        _ => w.expect,
    }
}

struct Workload {
    id: &'static str,
    messages: Vec<InputMessage>,
    expect: Expect,
}

fn workloads() -> Vec<Workload> {
    let budget = DEFAULT_MAX_TRAVERSAL_BYTES as u64;
    let near_4k = support::history_within(budget, 4096);
    let mut over = near_4k.clone();
    over.extend(support::history(1, 4096));
    // The largest single message whose JSON stays within 16 KiB.
    let envelope = support::json_len(&support::single_message(0));
    let just_under = 16 * 1024 - envelope;
    vec![
        Workload {
            id: "msg_256b",
            messages: support::single_message(256),
            expect: Expect::Kept,
        },
        Workload {
            id: "msg_1k",
            messages: support::single_message(1024),
            expect: Expect::Kept,
        },
        Workload {
            id: "msg_4k",
            messages: support::single_message(4096),
            expect: Expect::Kept,
        },
        Workload {
            id: "msg_15k",
            messages: support::single_message(just_under),
            expect: Expect::Kept,
        },
        Workload {
            id: "msg_17k",
            messages: support::single_message(17 * 1024),
            expect: Expect::Truncated,
        },
        Workload {
            id: "msg_64k",
            messages: support::single_message(64 * 1024),
            expect: Expect::Truncated,
        },
        Workload {
            id: "msg_256k",
            messages: support::single_message(256 * 1024),
            expect: Expect::Truncated,
        },
        Workload {
            id: "hist_32x4k",
            messages: support::history(32, 4096),
            expect: Expect::Truncated,
        },
        Workload {
            id: "hist_near_budget_4k",
            messages: near_4k,
            expect: Expect::Truncated,
        },
        Workload {
            id: "hist_near_budget_64b",
            messages: support::history_within(budget, 64),
            expect: Expect::Truncated,
        },
        Workload {
            id: "hist_over_budget",
            messages: over,
            expect: Expect::FailedClosed,
        },
    ]
}

fn policies() -> Vec<(&'static str, ContentPolicy)> {
    let rules = || {
        [
            RedactionRule::replace_literal("user@example.com", "[EMAIL]").expect("valid rule"),
            RedactionRule::replace_literal("+1-555-0100", "[PHONE]").expect("valid rule"),
            RedactionRule::replace_literal("sk-live-", "[KEY]-").expect("valid rule"),
            RedactionRule::omit_attribute(attr::TOOL_DEFINITIONS).expect("valid rule"),
        ]
    };
    let span = || ContentPolicy::default().with_capture(ContentCapture::SpanOnly);
    vec![
        ("metadata_only", ContentPolicy::metadata_only()),
        ("default", span()),
        ("rules", span().with_rules(rules())),
        (
            "hook_and_rules",
            span()
                .with_redactor(SubstringRedactor::new(["Lisbon", "Tagus"]))
                .with_rules(rules()),
        ),
        // Harness::new drops every span for this id.
        ("unsampled", span()),
    ]
}

struct Harness {
    observer: Observer,
    spans: InMemorySpanExporter,
    _provider: SdkTracerProvider,
}

impl Harness {
    fn new(policy_id: &str, policy: ContentPolicy) -> Self {
        let spans = InMemorySpanExporter::default();
        let mut builder = SdkTracerProvider::builder().with_simple_exporter(spans.clone());
        if policy_id == "unsampled" {
            builder = builder.with_sampler(Sampler::AlwaysOff);
        }
        let provider = builder.build();
        let observer = Observer::builder()
            .tracer_provider(&provider)
            .content_policy(policy)
            .build();
        Self {
            observer,
            spans,
            _provider: provider,
        }
    }

    fn start(&self) -> xybrid_observability::InferenceOperation {
        self.observer.start_inference(
            InferenceRequest::new("chat", "llama.cpp")
                .with_model("qwen3.5-0.8b")
                .with_location(Location::Local),
            StartOptions::default(),
        )
    }

    /// Records once and reports what reached the span.
    fn outcome(&self, messages: &[InputMessage]) -> (Expect, usize) {
        self.spans.reset();
        let op = self.start();
        op.record_input_messages(messages);
        op.end();
        let spans = self.spans.get_finished_spans().expect("finished spans");
        let Some(span) = spans.last() else {
            // A span that was not sampled is never exported.
            return (Expect::Skipped, 0);
        };
        let value = |key: &str| {
            span.attributes
                .iter()
                .find(|kv| kv.key.as_str() == key)
                .map(|kv| kv.value.clone())
        };
        let recorded = value(attr::INPUT_MESSAGES);
        let marked = value(attr::CONTENT_TRUNCATED_ATTRIBUTES)
            .is_some_and(|v| v.as_str().contains(attr::INPUT_MESSAGES));
        self.spans.reset();
        match (recorded, marked) {
            (Some(v), false) => (Expect::Kept, v.as_str().len()),
            (Some(v), true) => (Expect::Truncated, v.as_str().len()),
            (None, true) => (Expect::FailedClosed, 0),
            (None, false) => (Expect::Skipped, 0),
        }
    }
}

fn main() {
    let workloads = workloads();
    let policies = policies();
    let mut catalog = Vec::new();

    // Check every (policy, workload) path before timing it.
    let mut outcomes = Vec::new();
    for (policy_id, policy) in &policies {
        let check = Harness::new(policy_id, policy.clone());
        for w in &workloads {
            let (outcome, recorded) = check.outcome(&w.messages);
            assert_eq!(
                outcome,
                expected(policy_id, w),
                "workload {} under {policy_id} took an unexpected path (recorded {recorded} bytes)",
                w.id
            );
            outcomes.push(json!({
                "policy": policy_id,
                "workload": w.id,
                "outcome": format!("{outcome:?}"),
                "recorded_attribute_bytes": recorded,
            }));
        }
    }
    for w in &workloads {
        catalog.push(json!({
            "id": w.id,
            "messages": w.messages.len(),
            "text_bytes": w.messages.iter().flat_map(|m| &m.parts).map(|p| match p {
                xybrid_observability::Part::Text { content } => content.len(),
                _ => 0,
            }).sum::<usize>(),
            "json_bytes": support::json_len(&w.messages),
            "preflight_cost": support::cost_of(&w.messages),
            "traversal_budget": DEFAULT_MAX_TRAVERSAL_BYTES,
            "expected_default": format!("{:?}", w.expect),
        }));
    }

    let mut criterion = Criterion::default().configure_from_args();
    let mut group = criterion.benchmark_group("content");
    for (policy_id, policy) in &policies {
        let harness = Harness::new(policy_id, policy.clone());
        for w in &workloads {
            group.throughput(Throughput::Bytes(support::json_len(&w.messages) as u64));
            group.bench_with_input(BenchmarkId::new(*policy_id, w.id), &w.messages, |b, m| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for i in 0..iters {
                        let op = harness.start();
                        let start = Instant::now();
                        op.record_input_messages(m);
                        total += start.elapsed();
                        op.end();
                        if (i + 1).is_multiple_of(DRAIN_EVERY) {
                            harness.spans.reset();
                        }
                    }
                    harness.spans.reset();
                    total
                });
            });
        }
    }
    group.finish();
    criterion.final_summary();
    if support::list_only() {
        return;
    }

    // Individually timed calls for tail percentiles, per policy and workload.
    let samples = if support::bench_mode() && !support::quick() {
        400
    } else {
        10
    };
    let mut measured = Vec::new();
    for (policy_id, policy) in &policies {
        let harness = Harness::new(policy_id, policy.clone());
        for w in &workloads {
            let mut ns = Vec::with_capacity(samples);
            for _ in 0..samples {
                let op = harness.start();
                let start = Instant::now();
                op.record_input_messages(&w.messages);
                ns.push(support::nanos(start.elapsed()));
                op.end();
                harness.spans.reset();
            }
            measured.push(json!({
                "policy": policy_id,
                "workload": w.id,
                "record_input_messages_ns": support::distribution(ns),
            }));
        }
    }
    support::write_report(
        "content_pipeline",
        &json!({
            "bench": "content_pipeline",
            "environment": support::environment(),
            "attribute_budget_bytes": 16 * 1024,
            "workloads": catalog,
            "outcomes": outcomes,
            "policies": policies.iter().map(|(id, p)| json!({"id": id, "policy": format!("{p:?}")})).collect::<Vec<_>>(),
            "measurements": measured,
        }),
    );
}
