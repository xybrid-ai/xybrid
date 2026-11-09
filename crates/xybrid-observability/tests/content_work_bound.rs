//! Amendments A1 (b) and A9: replacement rules bound their work. A rule that
//! would expand a string beyond the traversal budget fails closed before the
//! expanded string is built (A1), and a document whose strings each pass that
//! guard but together outgrow the budget fails closed as soon as the strings
//! rewritten so far prove it, without rewriting the rest (A9).
//!
//! The binary installs an allocator that records, for the thread that armed
//! it only, the largest single allocation and the total bytes requested while
//! the measured call runs. Tracking is per thread, so tests in this binary may
//! run concurrently.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use opentelemetry::Value;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use xybrid_observability::semconv::attr;
use xybrid_observability::{
    ContentCapture, ContentPolicy, InferenceRequest, InputMessage, Observer, RedactionRule,
    StartOptions, ToolCallInfo,
};

thread_local! {
    // Const-initialised `Cell`s with no destructor: reading them never
    // allocates, so the allocator can use them.
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static LARGEST: Cell<usize> = const { Cell::new(0) };
    static TOTAL: Cell<usize> = const { Cell::new(0) };
}

fn record(size: usize) {
    // `try_with` because the allocator also runs while thread locals are
    // being torn down.
    let _ = ARMED.try_with(|armed| {
        if armed.get() {
            let _ = LARGEST.try_with(|l| l.set(l.get().max(size)));
            let _ = TOTAL.try_with(|t| t.set(t.get().saturating_add(size)));
        }
    });
}

struct Tracking;

// SAFETY: every call is forwarded unchanged to the system allocator; the
// wrapper only records sizes in non-allocating thread-local cells.
unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Counts the whole new block, an upper bound of the bytes requested.
        record(new_size);
        // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Tracking = Tracking;

/// Allocation sizes requested on this thread while `f` ran.
struct Measured {
    largest: usize,
    total: usize,
}

fn measure(f: impl FnOnce()) -> Measured {
    LARGEST.with(|l| l.set(0));
    TOTAL.with(|t| t.set(0));
    ARMED.with(|a| a.set(true));
    f();
    ARMED.with(|a| a.set(false));
    Measured {
        largest: LARGEST.with(Cell::get),
        total: TOTAL.with(Cell::get),
    }
}

fn observer(provider: &SdkTracerProvider, rules: Vec<RedactionRule>) -> Observer {
    Observer::builder()
        .tracer_provider(provider)
        .content_policy(
            ContentPolicy::default()
                .with_capture(ContentCapture::SpanOnly)
                .with_rules(rules),
        )
        .build()
}

fn memory() -> (InMemorySpanExporter, SdkTracerProvider) {
    let memory = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(memory.clone())
        .build();
    (memory, provider)
}

/// The single finished span has no `key` and lists it as truncated.
fn assert_failed_closed(memory: &InMemorySpanExporter, key: &str) {
    let spans = memory.get_finished_spans().unwrap();
    let span = &spans[0];
    assert!(
        !span.attributes.iter().any(|kv| kv.key.as_str() == key),
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
    assert!(list.to_string().contains(key), "{list}");
}

#[test]
fn an_expanding_replacement_rule_fails_closed_without_building_the_expansion() {
    let (memory, provider) = memory();
    // 400 KiB of `a` passes the 480 KiB traversal preflight; replacing every
    // `a` by 64 bytes would build a ~26 MiB string.
    let observer = observer(
        &provider,
        vec![RedactionRule::replace_literal("a", "b".repeat(64)).unwrap()],
    );
    let input = [InputMessage::text("user", "a".repeat(400 * 1024))];
    let op = observer.start_inference(
        InferenceRequest::new("chat", "fake"),
        StartOptions::default(),
    );

    let largest = measure(|| op.record_input_messages(&input)).largest;
    op.end();

    assert!(
        largest < 4 * 1024 * 1024,
        "the rule built a {largest}-byte allocation before failing closed"
    );
    assert_failed_closed(&memory, attr::INPUT_MESSAGES);
}

/// 600 strings of 480 `a` (preflight cost 297,616 of the 491,520 default).
/// Replacing every `a` by 1,000 bytes grows each string to 480,000 bytes,
/// which passes the per-string guard, so before amendment A9 every string was
/// rewritten (about 288 MB live, more requested) before the final check.
fn many_strings() -> serde_json::Value {
    serde_json::Value::Array(vec![serde_json::Value::String("a".repeat(480)); 600])
}

#[test]
fn many_strings_that_each_pass_fail_closed_before_rewriting_the_document() {
    let (memory, provider) = memory();
    let observer = observer(
        &provider,
        vec![RedactionRule::replace_literal("a", "b".repeat(1000)).unwrap()],
    );
    let result = many_strings();
    let tool = observer.start_tool(ToolCallInfo::new("lookup"), StartOptions::default());

    let used = measure(|| tool.record_result(&result));
    tool.end();

    // The clone of the 288 KB document plus bookkeeping; no rule rewrite
    // (each would request at least 480,000 bytes).
    assert!(
        used.total < 2 * 1024 * 1024,
        "the rules requested {} bytes before failing closed",
        used.total
    );
    assert!(used.largest < 480_000, "{} bytes", used.largest);
    assert_failed_closed(&memory, attr::TOOL_CALL_RESULT);
}

#[test]
fn a_rule_that_can_shrink_text_still_stops_building_once_rewritten_strings_prove_the_overflow() {
    // A second rule that can shrink text (it never matches here) means any
    // other string could end empty, so the rewritten strings are built and
    // their final lengths decide. The structural cost is 9,616 bytes; after
    // the second rewritten string (2 * 480,000) the bound is over budget and
    // no further string is built.
    let (memory, provider) = memory();
    let observer = observer(
        &provider,
        vec![
            RedactionRule::replace_literal("a", "b".repeat(1000)).unwrap(),
            RedactionRule::replace_literal("\u{0}", "").unwrap(),
        ],
    );
    let result = many_strings();
    let tool = observer.start_tool(ToolCallInfo::new("lookup"), StartOptions::default());

    let used = measure(|| tool.record_result(&result));
    tool.end();

    // Two rewritten strings, each built by appending (growth doubles the
    // buffer, so each requests under 2 * 480,000 bytes in total), plus the
    // clone; 600 strings would request more than 288 MB.
    assert!(
        used.total < 4 * 1024 * 1024,
        "the rules requested {} bytes before failing closed",
        used.total
    );
    assert_failed_closed(&memory, attr::TOOL_CALL_RESULT);
}
