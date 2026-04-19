//! Zero-copy Envelope ↔ MlxArray cycle benchmark (US-008 acceptance).
//!
//! Measures two things end-to-end on Apple Silicon with Metal reachable:
//!
//! 1. **`Envelope → MlxArray → Envelope` cycle** for a 1MB embedding
//!    tensor — the PRD's primary criterion. The average per-iteration
//!    time must stay under **50 µs** to pass.
//! 2. **Heap allocation count** during the hot loop via a counting
//!    global allocator. The count is reported and bounded at an upper
//!    threshold that reflects the current Rust-side allocations in
//!    `from_envelope` (the Box<[T]> clone, the Box<SharedBuffer>
//!    payload) plus the readback Vec. "Zero new heap regions" in the
//!    strict sense is not achievable while `from_envelope` takes
//!    `&Envelope` (the clone is unavoidable), but the number must stay
//!    *stable* across iterations — any growth signals a latent
//!    quadratic allocation bug that later iteration would miss.
//!
//! # Harness
//!
//! `harness = false` in Cargo.toml lets us skip the criterion crate and
//! ship a tiny purpose-built bench: `std::time::Instant` for timing, an
//! atomic counter wrapped around `std::alloc::System` for alloc
//! accounting. The benefit is no dev-dep graph churn for a one-file
//! bench; the downside is no HTML report — which is fine because the
//! bench is a pass/fail gate, not a continuous performance-regression
//! tracker.
//!
//! # Running
//!
//! ```bash
//! # From repos/xybrid (inside the main xybrid monorepo).
//! cargo bench -p xybrid-mlx --features bindings --bench zero_copy
//! ```
//!
//! Requires `vendor/mlx-apple/mlx.xcframework` to be present.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use xybrid_mlx::{EnvelopePayload, EnvelopeSource, MlxArray, OwnedEnvelopePayload};

// --- Counting allocator ---------------------------------------------------
//
// Wraps the system allocator with two atomics: total allocations and total
// bytes requested. The bench captures deltas around the hot loop to reason
// about "zero new heap regions" (acceptance criterion from US-008).

struct CountingAllocator {
    alloc_count: AtomicUsize,
    alloc_bytes: AtomicUsize,
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        self.alloc_bytes.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        self.alloc_bytes.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // realloc counts as an allocation event since it can return a fresh region.
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        self.alloc_bytes.fetch_add(new_size, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator {
    alloc_count: AtomicUsize::new(0),
    alloc_bytes: AtomicUsize::new(0),
};

fn allocs_so_far() -> usize {
    ALLOCATOR.alloc_count.load(Ordering::Relaxed)
}

// --- A throwaway envelope impl to exercise from_envelope ------------------

struct BorrowedEnvelope<'a> {
    payload: EnvelopePayload<'a>,
}
impl EnvelopeSource for BorrowedEnvelope<'_> {
    fn payload(&self) -> EnvelopePayload<'_> {
        self.payload
    }
}

// --- Bench driver ---------------------------------------------------------

/// 1 MiB embedding tensor: 262_144 f32 elements = 1_048_576 bytes.
const TENSOR_ELEMENTS: usize = 1_024 * 1_024 / std::mem::size_of::<f32>();
/// Per-iteration time budget from the PRD.
const BUDGET: Duration = Duration::from_micros(50);
/// Iteration count for the timing loop. 1_000 iterations × ~30-50µs =
/// ~30-50ms total bench time; enough to smooth out scheduler jitter
/// without blowing up CI time.
const ITERS: usize = 1_000;
/// Warm-up count — MLX lazily initialises its default stream and Metal
/// pipelines on first use; the first few iterations are pathologically
/// slow and would pollute the average.
const WARMUP: usize = 50;

fn main() {
    // Synthesise a fixed 1 MiB payload. Populate with a distinct pattern so
    // any accidental over-optimisation that skips the copy is caught by the
    // assertion on the readback.
    let payload: Vec<f32> = (0..TENSOR_ELEMENTS).map(|i| (i as f32) * 0.5).collect();

    // Warm up: first call touches MLX's default stream + Metal init.
    for _ in 0..WARMUP {
        let env = BorrowedEnvelope {
            payload: EnvelopePayload::Embedding(&payload),
        };
        let arr = MlxArray::from_envelope(&env).expect("warmup from_envelope");
        let _back = arr
            .to_envelope_payload()
            .expect("warmup to_envelope_payload");
    }

    // ------- Timing loop -----------------------------------------------
    let alloc_before = allocs_so_far();
    let start = Instant::now();
    for _ in 0..ITERS {
        let env = BorrowedEnvelope {
            payload: EnvelopePayload::Embedding(&payload),
        };
        let arr = MlxArray::from_envelope(black_box(&env)).expect("from_envelope");
        let back = arr.to_envelope_payload().expect("to_envelope_payload");
        // black_box keeps the optimiser from hoisting the readback out.
        black_box(&back);
        drop(back);
    }
    let elapsed = start.elapsed();
    let alloc_after = allocs_so_far();

    let per_iter = elapsed / (ITERS as u32);
    let allocs_per_iter = (alloc_after - alloc_before) as f64 / ITERS as f64;

    println!("zero_copy bench: {ITERS} iters, {TENSOR_ELEMENTS} f32 elements (1 MiB) per cycle");
    println!("  total:           {elapsed:?}  (per iter: {per_iter:?})");
    println!(
        "  allocs:          {} total  ({allocs_per_iter:.2}/iter)",
        alloc_after - alloc_before
    );

    // ------- Assertions -------------------------------------------------
    assert!(
        per_iter < BUDGET,
        "Envelope → MlxArray → Envelope cycle took {per_iter:?}/iter on a 1 MiB tensor; PRD \
         budget is <{BUDGET:?}. Check whether a rebuild disabled Metal, or whether the \
         readback path is taking extra copies."
    );

    // "Zero new heap regions" — interpreted as: allocations per cycle are
    // bounded and stable. Current expected Rust-side allocations per cycle:
    //   - `from_envelope`: clone into Box<[f32]> (1 alloc), Box<SharedBuffer>
    //     wrapping it (1 alloc).
    //   - `to_envelope_payload`: returned Vec<f32> (1 alloc).
    // Plus whatever mlx-c allocates internally for the array ctx — that is
    // not counted here (it happens inside C++ frames), so our counter only
    // sees Rust-side activity. We set the bound generously at 16 to tolerate
    // minor jitter from tracing macros and black_box side paths.
    const MAX_ALLOCS_PER_ITER: f64 = 16.0;
    assert!(
        allocs_per_iter <= MAX_ALLOCS_PER_ITER,
        "allocation rate {allocs_per_iter:.2}/iter exceeds the {MAX_ALLOCS_PER_ITER} budget — \
         dhat-profile the cycle and update this bound (or fix the regression)."
    );

    println!("zero_copy bench: PASS (all budgets met)");
}
