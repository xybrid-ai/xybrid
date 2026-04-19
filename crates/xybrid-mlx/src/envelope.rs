//! Envelope payload abstraction for the zero-copy conversion path.
//!
//! `xybrid-core` owns the canonical [`Envelope`][core-env] type (and all
//! its serde / metadata baggage). Replicating that dep here would create a
//! cycle once `xybrid-core` starts depending on `xybrid-mlx` for the MLX
//! runtime adapter (US-010), so `xybrid-mlx` instead exposes a minimal
//! trait ([`EnvelopeSource`]) that `xybrid-core::ir::Envelope` can
//! implement downstream. The trait only surfaces the information the
//! zero-copy path actually needs: the typed payload bytes as a borrowed
//! slice.
//!
//! # Mapping to `xybrid-core::ir::Envelope`
//!
//! | `EnvelopeKind` variant (in `xybrid-core`) | [`EnvelopePayload`] variant (here) | Dtype |
//! | --- | --- | --- |
//! | `Audio(Vec<u8>)` | [`EnvelopePayload::Audio`] | [`MlxDtype::U8`] |
//! | `Embedding(Vec<f32>)` | [`EnvelopePayload::Embedding`] | [`MlxDtype::F32`] |
//! | `Text(String)` | — (rejected; tokenize first) | — |
//! | `TokenIds(Vec<i64>)` *(US-009)* | [`EnvelopePayload::TokenIds`] | — (i64 is not in xybrid-mlx's supported dtype set yet) |
//!
//! The `TokenIds` variant is defined here ahead of US-009 so the
//! `MlxArray::from_envelope` dispatcher can give a structurally-meaningful
//! error when the caller passes one through early — and so the downstream
//! `impl EnvelopeSource for Envelope` in US-009 can land without touching
//! this crate.
//!
//! [core-env]: ../xybrid_core/ir/struct.Envelope.html

use crate::dtype::MlxDtype;

/// A borrowed view onto an envelope's typed payload.
///
/// Variants correspond 1:1 with the xybrid-core `EnvelopeKind` payload
/// variants that can be mapped into a tensor. The lifetime parameter
/// borrows from the owning envelope for the duration of a
/// `MlxArray::from_envelope` call — we clone the slice into a `Box<[T]>`
/// that the `SharedBuffer` takes ownership of, so the envelope is safe
/// to drop immediately after the call. (Those two types are cfg-gated to
/// Apple + bindings builds, which is why they are referenced with plain
/// backticks here rather than intra-doc links — the no-bindings rustdoc
/// default run wouldn't resolve them.)
#[derive(Debug, Clone, Copy)]
pub enum EnvelopePayload<'a> {
    /// Raw audio bytes — mapped to a 1-D `u8` tensor.
    Audio(&'a [u8]),
    /// Dense float embedding — mapped to a 1-D `f32` tensor.
    Embedding(&'a [f32]),
    /// Token IDs — reserved for US-009. `i64` is not yet in xybrid-mlx's
    /// supported dtype set; feeding this variant in today triggers a
    /// descriptive error from `MlxArray::from_envelope`.
    TokenIds(&'a [i64]),
}

impl EnvelopePayload<'_> {
    /// Dtype the payload maps to inside MLX, if one exists today.
    ///
    /// Returns `None` for [`EnvelopePayload::TokenIds`] because `i64` is
    /// outside the current [`MlxDtype`] surface — US-009 will either
    /// extend the enum or add an explicit int-tokens constructor.
    #[must_use]
    pub fn dtype(&self) -> Option<MlxDtype> {
        match self {
            Self::Audio(_) => Some(MlxDtype::U8),
            Self::Embedding(_) => Some(MlxDtype::F32),
            Self::TokenIds(_) => None,
        }
    }

    /// Element count for the payload.
    #[must_use]
    pub fn element_count(&self) -> usize {
        match self {
            Self::Audio(s) => s.len(),
            Self::Embedding(s) => s.len(),
            Self::TokenIds(s) => s.len(),
        }
    }
}

/// Trait implemented by envelope-like types that can feed the zero-copy
/// `MlxArray::from_envelope` path.
///
/// The only method, [`EnvelopeSource::payload`], returns a borrowed
/// [`EnvelopePayload`]. Implementers in `xybrid-core` simply match on
/// `EnvelopeKind` and hand back the relevant slice — the heavy lifting
/// (copy to a `Box<[T]>`, MTLBuffer wrap, mlx_array construction) happens
/// inside `MlxArray::from_envelope`.
///
/// Downstream consumers that want to construct an `MlxArray` from a
/// non-xybrid-core type (e.g. a bespoke message class in an embedder) can
/// also implement this trait directly.
pub trait EnvelopeSource {
    /// Borrow the typed payload. Must return a stable slice for the
    /// duration of the call.
    fn payload(&self) -> EnvelopePayload<'_>;
}

/// Owned form of [`EnvelopePayload`]. Returned from
/// `MlxArray::to_envelope_payload` so callers can reconstitute a concrete
/// [`Envelope`][core-env] (or equivalent type) without `xybrid-mlx` having
/// to know about `xybrid-core`. (`MlxArray` is cfg-gated to Apple +
/// bindings; hence plain backticks rather than an intra-doc link.)
///
/// The two owned variants mirror the two readable-today payload kinds. We
/// do not attempt to recover from the "MLX refused to evaluate" case — the
/// caller receives an error from `to_envelope_payload` in that situation
/// and no `OwnedEnvelopePayload` is constructed.
///
/// [core-env]: ../xybrid_core/ir/struct.Envelope.html
#[derive(Debug, Clone, PartialEq)]
pub enum OwnedEnvelopePayload {
    /// Readback of a `u8` tensor — matches `EnvelopeKind::Audio` upstream.
    Audio(Vec<u8>),
    /// Readback of an `f32` tensor — matches `EnvelopeKind::Embedding`
    /// upstream.
    Embedding(Vec<f32>),
}
