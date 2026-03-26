//! CLI command handlers organized by subcommand.
//!
//! Each module handles a specific set of CLI commands:
//!
//! | Module | Commands |
//! |--------|----------|
//! | [`run`] | `run` - Execute pipelines, bundles, and models |
//! | [`repl`] | `repl` - Interactive REPL mode |
//! | [`models`] | `models list`, `models info`, `models voices` |
//! | [`cache`] | `cache list`, `cache status`, `cache clear` |
//! | [`fetch`] | `fetch` - Model downloading |
//! | [`bundle`] | `bundle` - Create .xyb bundles from registry |
//! | [`pack`] | `pack` - Package local model artifacts |
//! | [`pipeline`] | `prepare`, `plan` - Pipeline validation and planning |
//! | [`trace`] | `trace` - Session telemetry analysis |
//! | [`types`] | Shared CLI enum types (ModelsCommand, CacheCommand) |
//! | [`utils`] | Shared utility functions |

pub mod bundle;
pub mod cache;
pub mod fetch;
pub mod init;
pub mod models;
pub mod pack;
pub mod pipeline;
pub mod repl;
pub mod run;
pub mod trace;
pub mod types;
pub mod utils;

// Re-export shared types
pub(crate) use types::{CacheCommand, ModelsCommand};
