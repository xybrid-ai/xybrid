//! Shared CLI types used across command modules.

use clap::Subcommand;

/// Subcommands for `xybrid models`
#[derive(Subcommand)]
pub(crate) enum ModelsCommand {
    /// List all available models in the registry
    List {
        /// Filter variants by local backend compatibility (auto|mlx|llamacpp|mistral)
        #[arg(long, value_name = "BACKEND")]
        backend: Option<String>,
    },
    /// Search for models by name or task
    Search {
        /// Search query (matches model ID, family, task, or description)
        #[arg(value_name = "QUERY")]
        query: String,
    },
    /// Show details about a specific model
    Info {
        /// Model ID (e.g., "kokoro-82m")
        #[arg(value_name = "ID")]
        model_id: String,
    },
    /// List available voices for a TTS model
    Voices {
        /// Model ID (e.g., "kokoro-82m")
        #[arg(value_name = "ID")]
        model_id: String,
    },
}

/// Subcommands for `xybrid cache`
#[derive(Subcommand)]
pub(crate) enum CacheCommand {
    /// List all cached models
    List,
    /// Show cache statistics
    Status,
    /// Clear cached models
    Clear {
        /// Model ID to clear (clears all if not specified)
        #[arg(value_name = "ID")]
        model_id: Option<String>,
    },
}
