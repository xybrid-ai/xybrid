//! One streamed response, from start to quiescence.
//!
//! The runtime delivers chunks through a C callback on threads it owns. This
//! module turns that into a queue the caller drains on its own thread, and
//! owns the hard part: stopping early and knowing when no callback can still
//! be running.
//!
//! Rules it keeps (upstream's "Threading model", `engine.h`):
//!
//! - The callback copies the chunk, pushes it and returns. It holds no lock
//!   while it reads the chunk and calls nothing on the engine or the
//!   conversation.
//! - Cancel is requested at most once, by the caller's thread.
//! - After a cancel the conversation is not used again; it is only deleted.
//! - A final chunk that carries an error does not prove the runtime is done:
//!   upstream marks every error final, including one raised while it is
//!   still decoding. So a stream that ended with an error is cancelled
//!   before its conversation is deleted, like one the caller stopped early.
//! - When the runtime cannot be shown to be at rest (a stream never ends, or
//!   the wait fails), nothing it may still use is freed: the conversation
//!   and its engine are leaked, and the engine and every conversation on it
//!   refuse further work. See [`Error::Stalled`].
//! - None of this relies on a destructor running. The conversation, not the
//!   [`Stream`] handle, owns the state of its running stream, so a handle
//!   lost with `mem::forget` leaves a conversation that refuses work until
//!   it is dropped, and dropping it tears the stream down.
//!
//! # Late callbacks
//!
//! The C API does not say when callbacks stop. After a cancel "the stream
//! callback can still run", and nothing is promised about a conversation
//! that has been deleted. So what crosses the C boundary as callback data is
//! an id, not a pointer: the callback looks its stream up in a table, and one
//! that arrives after the stream was retired finds nothing and returns.
//! Memory safety therefore does not depend on how the runtime schedules its
//! callbacks.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::{c_char, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::conversation::Conversation;
// Named by the docs below; the code here returns it only through `Result`.
#[cfg(doc)]
use crate::error::Error;
use crate::error::Result;
use crate::sys;

/// One chunk of a streamed response.
///
/// Text, error and finality are independent: the runtime's final chunk
/// normally carries neither text nor error, and a failure arrives as a final
/// chunk with an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// The chunk's payload. For a conversation this is one JSON message.
    pub text: Option<String>,
    /// The runtime's error message, when the stream failed.
    pub error: Option<String>,
    /// Whether this is the last chunk of the stream.
    pub is_final: bool,
    /// When the callback received it.
    pub received_at: Instant,
}

/// How a stream ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamEnd {
    /// A cancel was requested: by the caller, or by teardown because the
    /// stream was stopped early or ended with an error. The conversation can
    /// then only be deleted.
    pub cancelled: bool,
}

/// Where a stream stands, as far as the callback has reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EndState {
    /// No final chunk yet.
    Running,
    /// The final chunk arrived without an error: the runtime ended the turn.
    Clean,
    /// A final chunk arrived with an error. The runtime may still be busy.
    Failed,
}

/// Streams the runtime may still call back into, by the id it was handed as
/// callback data. See "Late callbacks" in the module docs.
///
/// A process-wide table on purpose: the id must mean something to a callback
/// that outlives every object of this crate. Two copies of the crate in one
/// process each have their own table and their own callback, which is
/// consistent.
static STREAMS: Mutex<BTreeMap<usize, Arc<Shared>>> = Mutex::new(BTreeMap::new());
static NEXT_STREAM_ID: AtomicUsize = AtomicUsize::new(1);

fn streams() -> MutexGuard<'static, BTreeMap<usize, Arc<Shared>>> {
    // The table holds reference-counted handles, valid after any panic.
    STREAMS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The id of a registered stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StreamId(usize);

impl StreamId {
    /// Make `shared` reachable from the runtime's callback.
    pub(crate) fn register(shared: &Arc<Shared>) -> Self {
        let id = NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed);
        streams().insert(id, Arc::clone(shared));
        Self(id)
    }

    /// Stop callbacks from reaching the stream. Callbacks already past the
    /// lookup hold their own reference and finish normally.
    pub(crate) fn retire(self) {
        streams().remove(&self.0);
    }

    /// Whether the runtime's callback can still reach the stream.
    #[cfg(all(test, feature = "test-util"))]
    pub(crate) fn is_registered(self) -> bool {
        streams().contains_key(&self.0)
    }

    /// The id in the shape the C API carries callback data in. It is an
    /// address-sized integer and is never dereferenced.
    pub(crate) fn as_callback_data(self) -> *mut c_void {
        ptr::without_provenance_mut(self.0)
    }
}

/// State shared between the caller and the runtime's callback.
pub(crate) struct Shared {
    queue: Mutex<Queue>,
    ready: Condvar,
    panicked: AtomicBool,
    capacity: usize,
    chunk_text: unsafe extern "C" fn(*const sys::StreamChunk, *mut *const c_char) -> sys::Status,
    chunk_error: unsafe extern "C" fn(*const sys::StreamChunk, *mut *const c_char) -> sys::Status,
    chunk_is_final: unsafe extern "C" fn(*const sys::StreamChunk, *mut bool) -> sys::Status,
}

struct Queue {
    chunks: VecDeque<Chunk>,
    /// Running until a final chunk has been received (it may still be
    /// queued). Chunks after it are ignored.
    end: EndState,
    /// A chunk arrived when the queue was full and was dropped.
    overflowed: bool,
}

impl Shared {
    pub(crate) fn new(api: &sys::Api, capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            queue: Mutex::new(Queue {
                chunks: VecDeque::new(),
                end: EndState::Running,
                overflowed: false,
            }),
            ready: Condvar::new(),
            panicked: AtomicBool::new(false),
            capacity: capacity.max(1),
            chunk_text: api.stream_chunk_get_text,
            chunk_error: api.stream_chunk_get_error,
            chunk_is_final: api.stream_chunk_is_final,
        })
    }

    fn lock(&self) -> MutexGuard<'_, Queue> {
        // The queue holds plain data, valid after any panic.
        self.queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Copy one chunk out of the runtime and queue it.
    ///
    /// # Safety
    ///
    /// `chunk` must be the chunk pointer the runtime passed to the callback,
    /// still inside that callback.
    unsafe fn accept(&self, chunk: *const sys::StreamChunk) {
        let received_at = Instant::now();
        let chunk = if chunk.is_null() {
            Chunk {
                text: None,
                error: Some("the runtime passed a null stream chunk".to_string()),
                is_final: true,
                received_at,
            }
        } else {
            let mut is_final = false;
            // A failing accessor leaves its out-parameter untouched, so the
            // defaults (no text, no error, not final) stand.
            let _ = (self.chunk_is_final)(chunk, &mut is_final);
            Chunk {
                text: read_chunk_string(self.chunk_text, chunk),
                error: read_chunk_string(self.chunk_error, chunk),
                is_final,
                received_at,
            }
        };
        self.push(chunk);
    }

    fn push(&self, chunk: Chunk) {
        let mut queue = self.lock();
        if queue.end != EndState::Running {
            // The consumer's stream ended with the first final chunk. A
            // runtime that sends more is ignored rather than trusted.
            return;
        }
        if chunk.is_final {
            // The final chunk is never dropped: it is what ends the wait.
            queue.end = if chunk.error.is_none() {
                EndState::Clean
            } else {
                EndState::Failed
            };
            queue.chunks.push_back(chunk);
        } else if queue.chunks.len() < self.capacity {
            queue.chunks.push_back(chunk);
        } else {
            queue.overflowed = true;
        }
        drop(queue);
        self.ready.notify_all();
    }

    /// Where the stream stands.
    pub(crate) fn end_state(&self) -> EndState {
        self.lock().end
    }

    /// Wait until the runtime has delivered a final chunk.
    pub(crate) fn wait_end(&self, grace: Duration) -> bool {
        let deadline = Instant::now() + grace;
        let mut queue = self.lock();
        while queue.end == EndState::Running {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            queue = self
                .ready
                .wait_timeout(queue, deadline - now)
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .0;
        }
        true
    }

    /// A way the stream broke the protocol this crate relies on, if it did.
    pub(crate) fn protocol_fault(&self) -> Option<&'static str> {
        if self.panicked.load(Ordering::Acquire) {
            return Some("the stream callback panicked");
        }
        if self.lock().overflowed {
            return Some("the runtime sent more chunks than the output budget allows");
        }
        None
    }
}

/// # Safety
///
/// `chunk` must be valid for the duration of the call.
unsafe fn read_chunk_string(
    accessor: unsafe extern "C" fn(*const sys::StreamChunk, *mut *const c_char) -> sys::Status,
    chunk: *const sys::StreamChunk,
) -> Option<String> {
    let mut text: *const c_char = ptr::null();
    if accessor(chunk, &mut text) != sys::STATUS_OK || text.is_null() {
        return None;
    }
    // The string is owned by the chunk and valid only inside the callback,
    // so it is copied here.
    Some(CStr::from_ptr(text).to_string_lossy().into_owned())
}

/// The C callback handed to the runtime.
///
/// # Safety
///
/// `data` must be the value of [`StreamId::as_callback_data`], and `chunk`
/// the chunk pointer the runtime passes to its stream callback, valid for
/// the duration of the call.
pub(crate) unsafe extern "C" fn trampoline(data: *mut c_void, chunk: *const sys::StreamChunk) {
    // A panic must not unwind into the runtime's C++ frames.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // The reference taken here keeps the state alive for this
        // invocation, whatever the owner does meanwhile.
        let Some(shared) = streams().get(&data.addr()).cloned() else {
            // The stream was retired; this callback is late.
            return;
        };
        // SAFETY: `chunk` is the runtime's chunk, inside its callback.
        let accepted = catch_unwind(AssertUnwindSafe(|| unsafe { shared.accept(chunk) }));
        if accepted.is_err() {
            shared.panicked.store(true, Ordering::Release);
            // End the caller's wait.
            shared.push(Chunk {
                text: None,
                error: Some("the stream callback panicked".to_string()),
                is_final: true,
                received_at: Instant::now(),
            });
        }
    }));
}

/// A response being streamed.
///
/// Borrows its [`Conversation`] for as long as it lives. Dropping it tears
/// the stream down the same way [`Stream::finish`] does, without reporting
/// errors; call `finish` to see them.
///
/// The handle only reads chunks. The conversation owns the stream's state,
/// so losing the handle (`mem::forget`) loses nothing the teardown needs.
pub struct Stream<'c> {
    conversation: &'c mut Conversation,
    shared: Arc<Shared>,
    final_delivered: bool,
}

impl<'c> Stream<'c> {
    pub(crate) fn new(conversation: &'c mut Conversation, shared: Arc<Shared>) -> Self {
        Self {
            conversation,
            shared,
            final_delivered: false,
        }
    }

    /// Wait up to `timeout` for the next chunk.
    ///
    /// Returns `None` when no chunk arrived in time, and always after the
    /// final chunk has been returned.
    pub fn next_chunk(&mut self, timeout: Duration) -> Option<Chunk> {
        if self.final_delivered {
            return None;
        }
        let deadline = Instant::now() + timeout;
        let mut queue = self.shared.lock();
        loop {
            if let Some(chunk) = queue.chunks.pop_front() {
                drop(queue);
                if chunk.is_final {
                    self.final_delivered = true;
                }
                return Some(chunk);
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            queue = self
                .shared
                .ready
                .wait_timeout(queue, deadline - now)
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .0;
        }
    }

    /// Whether the final chunk has been returned by [`Stream::next_chunk`].
    pub fn is_finished(&self) -> bool {
        self.final_delivered
    }

    /// Ask the runtime to stop generating. Only the first call does
    /// anything. Chunks already produced may still arrive, followed by a
    /// final chunk that reports the cancellation as an error.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if the runtime rejects the request, and
    /// [`Error::EngineUnusable`] if the engine was marked unusable meanwhile.
    pub fn cancel(&mut self) -> Result<()> {
        self.conversation.cancel_stream()
    }

    /// End the stream and bring the runtime to rest.
    ///
    /// - If the runtime ended the turn itself, this waits for it to settle.
    ///   The conversation stays usable.
    /// - If the stream was stopped early or ended with an error, generation
    ///   is cancelled, and the runtime gets `grace` to deliver a final chunk
    ///   if it has not sent one. The conversation can then only be dropped.
    ///
    /// # Errors
    ///
    /// - [`Error::Stalled`] when a final chunk never arrives, and
    ///   [`Error::Call`] when the runtime fails the wait. In both cases the
    ///   conversation and its engine are leaked and refuse further work.
    /// - [`Error::Protocol`] when the callback panicked or the runtime sent
    ///   more chunks than the queue was sized for.
    /// - [`Error::EngineUnusable`] when another conversation made the engine
    ///   unusable while this stream was running.
    pub fn finish(self, grace: Duration) -> Result<StreamEnd> {
        // Dropping `self` afterwards finds no running stream and does nothing.
        self.conversation.finish_stream(grace)
    }
}

impl Drop for Stream<'_> {
    fn drop(&mut self) {
        // Errors cannot be reported from here; `finish` is the reporting
        // path. A stall still leaks instead of freeing.
        let _ = self.conversation.finish_stream(DEFAULT_GRACE);
    }
}

/// How long a dropped stream waits for the runtime's final chunk after
/// cancelling.
pub const DEFAULT_GRACE: Duration = Duration::from_secs(30);

#[cfg(test)]
mod tests {
    use super::*;

    fn shared() -> Arc<Shared> {
        Shared::new(&sys::Api::stub(), 4)
    }

    #[test]
    fn a_retired_stream_is_no_longer_reachable() {
        let shared = shared();
        let id = StreamId::register(&shared);
        assert!(streams().contains_key(&id.as_callback_data().addr()));
        assert_eq!(Arc::strong_count(&shared), 2, "the table holds a reference");

        id.retire();

        assert!(!streams().contains_key(&id.as_callback_data().addr()));
        assert_eq!(Arc::strong_count(&shared), 1);
        id.retire(); // retiring twice is harmless
    }

    #[test]
    fn stream_ids_are_never_reused() {
        let (a, b) = (shared(), shared());
        let first = StreamId::register(&a);
        first.retire();
        let second = StreamId::register(&b);
        assert_ne!(first, second);
        second.retire();
    }

    #[test]
    fn a_callback_for_an_unknown_stream_does_nothing() {
        // Callback data that names no stream, and a chunk pointer that must
        // not be read.
        // SAFETY: the trampoline looks the id up before it touches the chunk.
        unsafe { trampoline(ptr::without_provenance_mut(usize::MAX), ptr::null()) };
    }

    #[test]
    fn a_final_chunk_with_an_error_is_not_a_clean_end() {
        let shared = shared();
        assert_eq!(shared.end_state(), EndState::Running);
        shared.push(Chunk {
            text: None,
            error: Some("INTERNAL: boom".to_string()),
            is_final: true,
            received_at: Instant::now(),
        });
        assert_eq!(shared.end_state(), EndState::Failed);
        // What the runtime sends afterwards never reaches the consumer.
        shared.push(Chunk {
            text: None,
            error: None,
            is_final: true,
            received_at: Instant::now(),
        });
        assert_eq!(shared.end_state(), EndState::Failed);
        assert_eq!(shared.lock().chunks.len(), 1);
        assert!(shared.wait_end(Duration::ZERO));
    }

    #[test]
    fn the_queue_keeps_the_final_chunk_when_it_is_full() {
        let shared = shared();
        for _ in 0..10 {
            shared.push(Chunk {
                text: Some("x".to_string()),
                error: None,
                is_final: false,
                received_at: Instant::now(),
            });
        }
        shared.push(Chunk {
            text: None,
            error: None,
            is_final: true,
            received_at: Instant::now(),
        });
        assert_eq!(shared.end_state(), EndState::Clean);
        assert_eq!(
            shared.protocol_fault(),
            Some("the runtime sent more chunks than the output budget allows")
        );
        let queue = shared.lock();
        assert!(queue.overflowed);
        assert_eq!(queue.chunks.len(), 5, "four chunks and the final one");
        assert!(queue.chunks.back().expect("a chunk").is_final);
    }
}
