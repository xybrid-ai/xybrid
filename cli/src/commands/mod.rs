//! CLI command handlers organized by subcommand.
//!
//! Each module handles a specific set of CLI commands:
//!
//! | Module | Commands |
//! |--------|----------|
//! | [`run`] | `run` - Execute pipelines |
//! | [`models`] | `models list`, `models info`, etc. |
//! | [`cache`] | `cache status`, `cache clear`, etc. |
//! | [`trace`] | `trace` - Session analysis |
//!
//! ## Future Refactoring
//!
//! The remaining commands can be extracted as needed:
//! - `prepare` - Pipeline validation
//! - `plan` - Execution planning
//! - `fetch` - Model downloading
//! - `pack` - Bundle creation
//! - `deploy` - Bundle deployment

// Currently commands are still in main.rs
// This module structure is prepared for incremental migration

pub mod utils;

// Re-export utility functions for use in main.rs
pub use utils::*;
