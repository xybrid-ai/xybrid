//! Paged KV cache for MLX LLM inference.
//!
//! The cache pre-allocates page descriptors for `max_seq_len` tokens, divided
//! into fixed [`KV_CACHE_PAGE_TOKENS`]-token chunks. Each [`KvCachePage`]
//! carries the slot metadata for one chunk; the actual K/V tensor backing —
//! one `xybrid_mlx::MlxArray` per layer per page — is materialised lazily
//! when an architecture builder writes to it (US-011..US-014).
//!
//! Paging — instead of one monolithic `[batch, n_kv_heads, max_seq_len, head_dim]`
//! tensor per layer — keeps the device-side allocation footprint flat under
//! growing context windows: the runtime materialises only as many pages as it
//! has actually filled. The 256-token chunk size matches the page granularity
//! used by MLX-LM upstream and trades a small fixed-shape overhead for a 16×
//! reduction in worst-case allocation vs. naive per-token caches.

/// Number of tokens stored in a single KV-cache page. Matches MLX-LM's
/// upstream paging granularity. Picked once at the crate level so that the
/// architecture builders, the page index math, and any future wire-format
/// describing cache layout stay in sync.
pub const KV_CACHE_PAGE_TOKENS: usize = 256;

/// Per-page metadata. The actual `K` and `V` tensors live on the GPU and are
/// owned by [`KvCache`] in subsequent stories — this struct is the
/// CPU-side bookkeeping for one chunk.
#[derive(Debug, Clone)]
pub struct KvCachePage {
    /// Token offset of the first slot in this page (multiple of
    /// [`KV_CACHE_PAGE_TOKENS`]).
    pub start_token: usize,
    /// Number of slots actually written. `0..=KV_CACHE_PAGE_TOKENS`. Pages
    /// are filled left-to-right; once `filled == KV_CACHE_PAGE_TOKENS` the
    /// page is full and the next write spills into the next page.
    pub filled: usize,
}

impl KvCachePage {
    /// Construct an empty page anchored at the given token offset.
    pub fn new(start_token: usize) -> Self {
        Self {
            start_token,
            filled: 0,
        }
    }

    /// `true` when no further tokens fit in this page.
    pub fn is_full(&self) -> bool {
        self.filled >= KV_CACHE_PAGE_TOKENS
    }

    /// Remaining slot capacity in this page.
    pub fn remaining(&self) -> usize {
        KV_CACHE_PAGE_TOKENS.saturating_sub(self.filled)
    }
}

/// Paged KV cache for one model instance.
///
/// The cache stores `n_layers × n_pages` page descriptors. Each layer-page
/// will own a pair of `MlxArray`s (K and V) once the architecture builders
/// (US-011..US-013) and the generate loop (US-014) wire them in. For now,
/// only the page descriptors and the per-layer count are tracked, so the
/// pagination math (which page does token `t` belong to? how many pages
/// are needed for `max_seq_len`? what's the next free slot?) can be tested
/// in isolation.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Number of transformer layers. One K/V page descriptor per layer per
    /// page index.
    pub n_layers: usize,
    /// Maximum sequence length the cache was sized for.
    pub max_seq_len: usize,
    /// Total number of pages — `ceil(max_seq_len / KV_CACHE_PAGE_TOKENS)`.
    pub n_pages: usize,
    /// Number of tokens written so far. `0..=max_seq_len`.
    pub current_position: usize,
    /// Per-layer page descriptors. `pages[layer_idx][page_idx]`.
    pages: Vec<Vec<KvCachePage>>,
}

impl KvCache {
    /// Pre-allocate the page index for a model with `n_layers` and a context
    /// window of `max_seq_len` tokens.
    ///
    /// `max_seq_len == 0` produces a zero-page cache; callers that need a
    /// non-empty cache should validate `max_seq_len` before calling.
    pub fn new(n_layers: usize, max_seq_len: usize) -> Self {
        let n_pages = max_seq_len.div_ceil(KV_CACHE_PAGE_TOKENS);
        let pages = (0..n_layers)
            .map(|_| {
                (0..n_pages)
                    .map(|p| KvCachePage::new(p * KV_CACHE_PAGE_TOKENS))
                    .collect()
            })
            .collect();
        Self {
            n_layers,
            max_seq_len,
            n_pages,
            current_position: 0,
            pages,
        }
    }

    /// Returns the page index that holds token `pos`. Saturates at the last
    /// page if `pos >= max_seq_len` so callers can clamp without panicking.
    pub fn page_index(&self, pos: usize) -> usize {
        let raw = pos / KV_CACHE_PAGE_TOKENS;
        if self.n_pages == 0 {
            0
        } else {
            raw.min(self.n_pages - 1)
        }
    }

    /// Returns the slot offset within the page for token `pos`.
    pub fn slot_in_page(&self, pos: usize) -> usize {
        pos % KV_CACHE_PAGE_TOKENS
    }

    /// Read-only access to the page descriptor at `(layer, page)`. Returns
    /// `None` if either index is out of bounds.
    pub fn page(&self, layer: usize, page: usize) -> Option<&KvCachePage> {
        self.pages.get(layer).and_then(|pages| pages.get(page))
    }

    /// Mutable access to the page descriptor at `(layer, page)`. Returns
    /// `None` if either index is out of bounds.
    pub fn page_mut(&mut self, layer: usize, page: usize) -> Option<&mut KvCachePage> {
        self.pages
            .get_mut(layer)
            .and_then(|pages| pages.get_mut(page))
    }

    /// Clear all per-layer page state (resets `filled` to 0 and rewinds the
    /// write cursor). Allocations are kept so the next prompt can reuse the
    /// existing page descriptors without re-allocating.
    pub fn reset(&mut self) {
        self.current_position = 0;
        for layer_pages in &mut self.pages {
            for page in layer_pages {
                page.filled = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_count_rounds_up() {
        assert_eq!(KvCache::new(1, 0).n_pages, 0);
        assert_eq!(KvCache::new(1, 1).n_pages, 1);
        assert_eq!(KvCache::new(1, KV_CACHE_PAGE_TOKENS).n_pages, 1);
        assert_eq!(KvCache::new(1, KV_CACHE_PAGE_TOKENS + 1).n_pages, 2);
        // 2048 tokens → 8 pages of 256 each.
        assert_eq!(KvCache::new(1, 2048).n_pages, 2048 / KV_CACHE_PAGE_TOKENS);
    }

    #[test]
    fn page_descriptors_are_anchored_at_chunk_boundaries() {
        let cache = KvCache::new(2, 1024);
        assert_eq!(cache.n_layers, 2);
        assert_eq!(cache.n_pages, 4);
        for layer in 0..cache.n_layers {
            for page_idx in 0..cache.n_pages {
                let page = cache.page(layer, page_idx).expect("page in range");
                assert_eq!(page.start_token, page_idx * KV_CACHE_PAGE_TOKENS);
                assert_eq!(page.filled, 0);
                assert_eq!(page.remaining(), KV_CACHE_PAGE_TOKENS);
                assert!(!page.is_full());
            }
        }
    }

    #[test]
    fn page_index_round_trip() {
        let cache = KvCache::new(1, 1024);
        // Token 0..255 → page 0, slot 0..255.
        assert_eq!(cache.page_index(0), 0);
        assert_eq!(cache.slot_in_page(0), 0);
        assert_eq!(cache.page_index(255), 0);
        assert_eq!(cache.slot_in_page(255), 255);
        // Token 256 spills into page 1, slot 0.
        assert_eq!(cache.page_index(256), 1);
        assert_eq!(cache.slot_in_page(256), 0);
        // Token 1023 → last page (3), slot 255.
        assert_eq!(cache.page_index(1023), 3);
        assert_eq!(cache.slot_in_page(1023), 255);
    }

    #[test]
    fn page_index_saturates_past_capacity() {
        let cache = KvCache::new(1, 512);
        // 2 pages total (indices 0 and 1). Out-of-range tokens clamp to the
        // last page rather than panicking — callers can detect overflow via
        // `current_position` against `max_seq_len`.
        assert_eq!(cache.page_index(99_999), 1);
    }

    #[test]
    fn reset_clears_filled_but_preserves_layout() {
        let mut cache = KvCache::new(2, 512);
        cache.current_position = 300;
        cache.page_mut(0, 0).unwrap().filled = KV_CACHE_PAGE_TOKENS;
        cache.page_mut(0, 1).unwrap().filled = 44;
        cache.page_mut(1, 0).unwrap().filled = KV_CACHE_PAGE_TOKENS;
        cache.page_mut(1, 1).unwrap().filled = 44;

        cache.reset();

        assert_eq!(cache.current_position, 0);
        assert_eq!(cache.n_layers, 2);
        assert_eq!(cache.n_pages, 2);
        for layer in 0..2 {
            for page in 0..2 {
                let p = cache.page(layer, page).unwrap();
                assert_eq!(p.filled, 0);
                assert_eq!(p.start_token, page * KV_CACHE_PAGE_TOKENS);
            }
        }
    }

    #[test]
    fn page_remaining_decreases_with_fills() {
        let mut page = KvCachePage::new(0);
        assert_eq!(page.remaining(), KV_CACHE_PAGE_TOKENS);
        page.filled = 100;
        assert_eq!(page.remaining(), KV_CACHE_PAGE_TOKENS - 100);
        page.filled = KV_CACHE_PAGE_TOKENS;
        assert_eq!(page.remaining(), 0);
        assert!(page.is_full());
        page.filled = KV_CACHE_PAGE_TOKENS + 50; // overfilled — saturating subtract
        assert_eq!(page.remaining(), 0);
    }
}
