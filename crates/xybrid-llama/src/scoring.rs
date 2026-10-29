//! Final-position logits for label scoring.
//!
//! [`LlamaContext::score_final_position`] prefills a prompt and reads the
//! logits llama.cpp produces at its last token: the next-token distribution a
//! label-token scorer compares its candidate labels against. It returns only
//! what scoring needs, the requested labels' logits and the log-sum-exp of the
//! whole row, so callers never hold a pointer into native memory.
//!
//! The context's memory (the KV cache and, on hybrid models, the recurrent
//! state) is cleared at the start of every call and again on every return, a
//! refused request included: each prompt is scored from a clean state, and
//! nothing it leaves behind reaches the next generation.

use std::ffi::c_int;

use crate::context::LlamaContext;
use crate::error::{LlamaError, LlamaResult};
use crate::ffi::{self, TokenBatch};

/// The logits at a prompt's final position, reduced to what scoring needs.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FinalLogits {
    /// The logit of each requested label token, in request order.
    pub label_logits: Vec<f32>,
    /// `ln Σ exp(logit)` over the whole vocabulary row, so a label's
    /// log-probability is its logit minus this.
    pub logsumexp: f32,
}

impl LlamaContext {
    /// Prefill `tokens` from an empty state and return the logits of
    /// `label_ids` at the last token, with the log-sum-exp of that whole row.
    ///
    /// The prompt is decoded in chunks of at most llama.cpp's own batch size
    /// for this context, with output requested for the final token only: the
    /// way generation prefills. The context's memory is cleared first and
    /// again on every return, a refused request included, so any cached
    /// prefix is gone afterwards.
    ///
    /// Where the prompt is cut changes llama.cpp's arithmetic, so scores
    /// compare exactly only between runs that cut it at the same points: a
    /// smaller `n_batch` that splits the prompt into more chunks moves the
    /// logits slightly.
    ///
    /// Needs a decoder-only language model with causal attention. A context
    /// without memory, which is how llama.cpp runs encoder-only and diffusion
    /// models, is refused. Encoder-decoder models such as T5, and decoders
    /// without causal attention, can't be detected with the functions this
    /// crate binds: don't score with them (llama.cpp aborts on T5, as it does
    /// when generating).
    ///
    /// # Errors
    ///
    /// [`LlamaError::InvalidInput`], before any decode, when the context has
    /// no memory, `tokens` or `label_ids` is empty, the prompt leaves no free
    /// position in the context window, or an id is outside the model's
    /// vocabulary. [`LlamaError::DecodeFailed`] when llama.cpp fails to
    /// decode a chunk, and [`LlamaError::Internal`] when it returns no
    /// logits, a row without a finite log-sum-exp, or a non-finite logit for
    /// a requested label.
    pub fn score_final_position(
        &self,
        tokens: &[i32],
        label_ids: &[i32],
    ) -> LlamaResult<FinalLogits> {
        // SAFETY: the context is live.
        if !unsafe { ffi::clear_memory(self.as_ptr()) } {
            return Err(LlamaError::InvalidInput(
                "the context has no memory: llama.cpp runs this model as an encoder, \
                 which has no next-token row to score"
                    .to_string(),
            ));
        }
        let _clear_on_exit = ClearOnDrop(self);
        let n_vocab = self.n_vocab();
        validate(tokens, label_ids, self.n_ctx(), n_vocab)?;

        let mut batch = TokenBatch::new(self.decode_limit());
        prefill(tokens, &mut batch, |batch| {
            // SAFETY: the context is live. The batch holds at most the
            // context's decode limit of tokens, with ids inside the
            // vocabulary and positions inside the window (`validate`).
            unsafe { ffi::decode(self.as_ptr(), batch) }
        })?;

        // SAFETY: the context is live, and its last decode flagged exactly
        // one token, the prompt's last, for output.
        let logits = unsafe { ffi::logits(self.as_ptr()) };
        if logits.is_null() {
            return Err(LlamaError::Internal(
                "llama_get_logits returned no logits".to_string(),
            ));
        }
        // SAFETY: after that decode the buffer holds one row of n_vocab
        // floats, n_vocab being the vocabulary of the model this context was
        // created from. It stays valid until the next decode; the slice does
        // not outlive this function.
        let row = unsafe { std::slice::from_raw_parts(logits, n_vocab) };
        reduce(row, label_ids)
    }
}

/// Clears the context's memory when dropped, so every return path leaves it
/// empty.
struct ClearOnDrop<'a>(&'a LlamaContext);

impl Drop for ClearOnDrop<'_> {
    fn drop(&mut self) {
        self.0.kv_cache_clear();
    }
}

/// Refuses a request before any native call reads it.
fn validate(tokens: &[i32], label_ids: &[i32], n_ctx: usize, n_vocab: usize) -> LlamaResult<()> {
    if tokens.is_empty() {
        return Err(LlamaError::InvalidInput(
            "cannot score an empty prompt".to_string(),
        ));
    }
    if label_ids.is_empty() {
        return Err(LlamaError::InvalidInput(
            "no label tokens to score".to_string(),
        ));
    }
    if tokens.len() >= n_ctx {
        return Err(LlamaError::InvalidInput(format!(
            "a {}-token prompt leaves no free position in a {n_ctx}-token context",
            tokens.len()
        )));
    }
    let in_vocab = |id: &i32| usize::try_from(*id).is_ok_and(|id| id < n_vocab);
    if let Some(index) = tokens.iter().position(|id| !in_vocab(id)) {
        return Err(LlamaError::InvalidInput(format!(
            "prompt token {index} is outside the {n_vocab}-token vocabulary"
        )));
    }
    if let Some(index) = label_ids.iter().position(|id| !in_vocab(id)) {
        return Err(LlamaError::InvalidInput(format!(
            "label {index} is outside the {n_vocab}-token vocabulary"
        )));
    }
    Ok(())
}

/// Decodes `tokens` in chunks the size of `batch`, flagging the last token
/// for output. `decode` runs one chunk and returns llama.cpp's status; the
/// first failure stops the prefill.
fn prefill(
    tokens: &[i32],
    batch: &mut TokenBatch,
    mut decode: impl FnMut(&mut TokenBatch) -> c_int,
) -> LlamaResult<()> {
    for (start, end) in chunk_ranges(tokens.len(), batch.capacity()) {
        batch.set(&tokens[start..end], start, end == tokens.len());
        let code = decode(batch);
        if code != 0 {
            return Err(LlamaError::DecodeFailed {
                code,
                n_past_in: start,
                detail: format!(
                    "llama_decode failed on scoring chunk [{start}, {end}) of {}",
                    tokens.len()
                ),
            });
        }
    }
    Ok(())
}

/// Consecutive `[start, end)` ranges of at most `chunk` tokens covering
/// `0..len`. `chunk` must be positive.
fn chunk_ranges(len: usize, chunk: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..len)
        .step_by(chunk)
        .map(move |start| (start, (start + chunk).min(len)))
}

/// The labels' logits and the log-sum-exp of `row`, refusing non-finite
/// results. Every label id must already be an index into `row`.
fn reduce(row: &[f32], label_ids: &[i32]) -> LlamaResult<FinalLogits> {
    let logsumexp = logsumexp(row).ok_or_else(|| {
        LlamaError::Internal("the logits row has no finite log-sum-exp".to_string())
    })?;
    let label_logits: Vec<f32> = label_ids
        .iter()
        .map(|&id| row[usize::try_from(id).expect("label ids are validated")])
        .collect();
    if label_logits.iter().any(|logit| !logit.is_finite()) {
        return Err(LlamaError::Internal(
            "a label logit is not finite".to_string(),
        ));
    }
    Ok(FinalLogits {
        label_logits,
        logsumexp,
    })
}

/// `ln Σ exp(x)` over `row`, summed in `f64` around the maximum in index
/// order. `None` for an empty row or a non-finite result.
fn logsumexp(row: &[f32]) -> Option<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None;
    }
    let max = f64::from(max);
    let sum: f64 = row.iter().map(|&x| (f64::from(x) - max).exp()).sum();
    let lse = (max + sum.ln()) as f32;
    lse.is_finite().then_some(lse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_the_prompt_without_overlap() {
        let ranges = |len, chunk| chunk_ranges(len, chunk).collect::<Vec<_>>();
        assert_eq!(ranges(1, 512), [(0, 1)]);
        assert_eq!(ranges(512, 512), [(0, 512)]);
        assert_eq!(ranges(513, 512), [(0, 512), (512, 513)]);
        assert_eq!(ranges(1024, 512), [(0, 512), (512, 1024)]);
        assert_eq!(ranges(3, 1), [(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn requests_are_refused_before_any_decode() {
        let refused = |tokens: &[i32], labels: &[i32], n_ctx, n_vocab| {
            matches!(
                validate(tokens, labels, n_ctx, n_vocab),
                Err(LlamaError::InvalidInput(_))
            )
        };
        assert!(validate(&[1, 2], &[3], 8, 10).is_ok());
        assert!(refused(&[], &[3], 8, 10), "empty prompt");
        assert!(refused(&[1], &[], 8, 10), "no labels");
        assert!(refused(&[1; 8], &[3], 8, 10), "no free position");
        assert!(validate(&[1; 7], &[3], 8, 10).is_ok(), "one position free");
        assert!(
            refused(&[1, 10], &[3], 8, 10),
            "prompt id past the vocabulary"
        );
        assert!(refused(&[1, -1], &[3], 8, 10), "negative prompt id");
        assert!(refused(&[1], &[10], 8, 10), "label past the vocabulary");
        assert!(refused(&[1], &[-1], 8, 10), "negative label");
    }

    #[test]
    fn logsumexp_is_stable_and_refuses_non_finite_rows() {
        // Computed in f64 and rounded once to f32.
        assert_eq!(logsumexp(&[0.0]), Some(0.0));
        assert_eq!(logsumexp(&[0.0, 0.0]), Some(2f64.ln() as f32));
        assert_eq!(
            logsumexp(&[1000.0, 1000.0]),
            Some((1000.0 + 2f64.ln()) as f32)
        );
        assert_eq!(logsumexp(&[f32::NEG_INFINITY, 0.0]), Some(0.0));
        assert_eq!(logsumexp(&[]), None);
        assert_eq!(logsumexp(&[f32::NEG_INFINITY]), None);
        assert_eq!(logsumexp(&[0.0, f32::INFINITY]), None);
        assert_eq!(logsumexp(&[0.0, f32::NAN]), None);
    }

    #[test]
    fn reduce_returns_labels_in_request_order() {
        let row = [0.5, -1.0, 2.0, 0.0];
        let reduced = reduce(&row, &[2, 0, 2]).unwrap();
        assert_eq!(reduced.label_logits, [2.0, 0.5, 2.0]);
        assert_eq!(reduced.logsumexp, logsumexp(&row).unwrap());

        let with_nan = [0.5, f32::NAN];
        assert!(matches!(
            reduce(&with_nan, &[0]),
            Err(LlamaError::Internal(_))
        ));
    }

    /// The (token, position, output flag) of each entry llama.cpp would read
    /// from `batch`, checking each is in sequence 0 alone.
    fn entries(batch: &mut TokenBatch) -> Vec<(i32, i32, i8)> {
        let view = batch.view();
        (0..usize::try_from(view.n_tokens).unwrap())
            .map(|i| {
                // SAFETY: the view points into `batch`'s arrays, each holding
                // n_tokens entries, and `batch` is not changed while reading.
                unsafe {
                    assert_eq!(*view.n_seq_id.add(i), 1);
                    assert_eq!(**view.seq_id.add(i), 0);
                    (*view.token.add(i), *view.pos.add(i), *view.logits.add(i))
                }
            })
            .collect()
    }

    /// Decoding is injected, so these need no model.
    #[test]
    fn the_prefill_cuts_chunks_and_flags_only_the_last_token() {
        let mut batch = TokenBatch::new(2);
        let mut chunks = Vec::new();
        prefill(&[10, 11, 12, 13, 14], &mut batch, |batch| {
            chunks.push(entries(batch));
            0
        })
        .unwrap();
        assert_eq!(
            chunks,
            [
                vec![(10, 0, 0), (11, 1, 0)],
                vec![(12, 2, 0), (13, 3, 0)],
                vec![(14, 4, 1)],
            ]
        );
        assert!(
            batch.view().embd.is_null(),
            "a token batch has no embeddings"
        );
    }

    #[test]
    fn a_failed_decode_stops_the_prefill_and_names_its_chunk() {
        let mut batch = TokenBatch::new(2);
        let mut calls = 0;
        let result = prefill(&[10, 11, 12, 13, 14], &mut batch, |_| {
            calls += 1;
            if calls == 2 {
                -3
            } else {
                0
            }
        });
        assert!(
            matches!(
                result,
                Err(LlamaError::DecodeFailed {
                    code: -3,
                    n_past_in: 2,
                    ..
                })
            ),
            "{result:?}"
        );
        assert_eq!(calls, 2, "no chunk is decoded after a failure");
    }

    #[test]
    #[should_panic(expected = "a batch holds between one token")]
    fn a_token_batch_needs_a_positive_capacity() {
        TokenBatch::new(0);
    }

    #[test]
    #[should_panic(expected = "do not fit a batch of 2")]
    fn a_token_batch_refuses_more_tokens_than_it_holds() {
        TokenBatch::new(2).set(&[1, 2, 3], 0, true);
    }
}
