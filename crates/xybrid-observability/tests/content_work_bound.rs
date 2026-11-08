//! Amendment A1 (b): replacement rules bound their work. A rule that would
//! expand a string beyond the traversal budget fails closed before the
//! expanded string is built.
//!
//! The binary installs an allocator that records the largest single
//! allocation made while the measured call runs; it holds one test so that
//! nothing else allocates concurrently.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use opentelemetry::Value;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use xybrid_observability::semconv::attr;
use xybrid_observability::{
    ContentPolicy, InferenceRequest, InputMessage, Observer, RedactionRule, StartOptions,
};

static ARMED: AtomicBool = AtomicBool::new(false);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

struct Tracking;

// SAFETY: every call is forwarded unchanged to the system allocator; the
// wrapper only records sizes in atomics (which never allocate).
unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            LARGEST.fetch_max(new_size, Ordering::Relaxed);
        }
        // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Tracking = Tracking;

#[test]
fn an_expanding_replacement_rule_fails_closed_without_building_the_expansion() {
    let memory = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(memory.clone())
        .build();
    // 400 KiB of `a` passes the 480 KiB traversal preflight; replacing every
    // `a` by 64 bytes would build a ~26 MiB string.
    let observer = Observer::builder()
        .tracer_provider(&provider)
        .content_policy(
            ContentPolicy::default()
                .with_rule(RedactionRule::replace_literal("a", "b".repeat(64)).unwrap()),
        )
        .build();
    let input = [InputMessage::text("user", "a".repeat(400 * 1024))];
    let op = observer.start_inference(
        InferenceRequest::new("chat", "fake"),
        StartOptions::default(),
    );

    LARGEST.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    op.record_input_messages(&input);
    ARMED.store(false, Ordering::Relaxed);
    let largest = LARGEST.load(Ordering::Relaxed);
    op.end();

    assert!(
        largest < 4 * 1024 * 1024,
        "the rule built a {largest}-byte allocation before failing closed"
    );
    let spans = memory.get_finished_spans().unwrap();
    let span = &spans[0];
    assert!(
        !span
            .attributes
            .iter()
            .any(|kv| kv.key.as_str() == attr::INPUT_MESSAGES),
        "the content failed closed"
    );
    let truncated = span
        .attributes
        .iter()
        .find(|kv| kv.key.as_str() == attr::CONTENT_TRUNCATED_ATTRIBUTES)
        .map(|kv| kv.value.clone());
    let Some(Value::Array(list)) = truncated else {
        panic!("the omitted attribute is listed: {truncated:?}")
    };
    assert!(list.to_string().contains(attr::INPUT_MESSAGES), "{list}");
}
