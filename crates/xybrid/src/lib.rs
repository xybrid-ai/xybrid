//! # xybrid
//!
//! Hybrid cloud-edge AI inference SDK. This is the umbrella crate — it
//! re-exports the public API of [`xybrid_sdk`] so users can depend on a
//! single `xybrid` crate from crates.io.
//!
//! ```toml
//! [dependencies]
//! xybrid = "0.1"
//! ```
//!
//! Feature flags map 1:1 to the underlying [`xybrid_sdk`] features — e.g.
//! `xybrid/platform-macos` forwards to `xybrid-sdk/platform-macos`.
//!
//! See the [`xybrid_sdk`] docs for the API surface.

pub use xybrid_sdk::*;
