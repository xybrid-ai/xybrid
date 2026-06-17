//! `xybrid prompt-opt` — model-aware prompt lint + rewrite suggestions.
//!
//! Lints a prompt against the target model's profile and prints actionable,
//! rule-cited suggestions. Scoring those suggestions against an evalset (so only
//! score-backed suggestions are surfaced) rides the normal run path; writing an
//! applied suggestion back as a new prompt version is a remote backend's prompt
//! library job — both are noted where they'd plug in.

use std::path::Path;

use anyhow::{Context, Result};

use xybrid_sdk::eval::{lint_prompt, model_profile, ModelProfile, PromptLibrary, SuggestionLevel};

use crate::ui;

/// Handle `xybrid prompt-opt <prompt-file> [--model <id>] [--system <file>]
/// [--write <prompt-id>] [--note <n>] [--library <dir>]`.
#[allow(clippy::too_many_arguments)]
pub fn handle_prompt_opt(
    prompt_path: &Path,
    model: Option<&str>,
    system_path: Option<&Path>,
    write: Option<&str>,
    note: Option<&str>,
    library: Option<&Path>,
) -> Result<()> {
    let prompt = std::fs::read_to_string(prompt_path)
        .with_context(|| format!("reading prompt {}", prompt_path.display()))?;
    let system = match system_path {
        Some(p) => Some(
            std::fs::read_to_string(p)
                .with_context(|| format!("reading system prompt {}", p.display()))?,
        ),
        None => None,
    };
    let profile = match model {
        Some(id) => model_profile(id),
        None => ModelProfile::generic("generic"),
    };

    ui::header(&format!("prompt-opt · {}", profile.model_id));
    println!();
    ui::kv(
        "Model profile",
        &format!(
            "{} ctx · system-prompt {} · {}",
            profile.ctx_window,
            if profile.supports_system_prompt {
                "yes"
            } else {
                "no"
            },
            profile.chat_template
        ),
    );

    let suggestions = lint_prompt(&prompt, system.as_deref(), &profile);
    if suggestions.is_empty() {
        println!();
        ui::ok("No issues found — the prompt looks well-formed for this model.");
    } else {
        ui::section(&format!("{} suggestion(s)", suggestions.len()));
        println!();
        for s in &suggestions {
            let badge = match s.level {
                SuggestionLevel::Error => ui::error("error"),
                SuggestionLevel::Warn => ui::warn("warn"),
                SuggestionLevel::Hint => ui::dim("hint"),
            };
            println!("  {} {}  {}", badge, ui::secondary(&s.rule), s.message);
            ui::sub(&format!("→ {}", s.fix));
        }
    }

    // Write-back: commit this prompt as a new version in the library.
    if let Some(prompt_id) = write {
        // Surface, but don't block, committing a prompt that still has
        // error-level issues — the dev may be versioning a work-in-progress.
        if suggestions
            .iter()
            .any(|s| s.level == SuggestionLevel::Error)
        {
            println!();
            println!(
                "  {}  committing despite the error-level issue(s) above",
                ui::warn("warn")
            );
        }
        let lib = match library {
            Some(dir) => PromptLibrary::with_dir(dir),
            None => PromptLibrary::default_location().map_err(|e| anyhow::anyhow!("{e}"))?,
        };
        let created = Some(xybrid_sdk::eval::now_rfc3339());
        let version = lib
            .write(prompt_id, &prompt, system.as_deref(), note, created)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        println!();
        ui::ok(&format!(
            "Wrote {} v{} to the prompt library",
            version.prompt_id, version.version
        ));
        ui::hint("Compare it against your evalset; a remote backend hosts the shared library.");
    } else if !suggestions.is_empty() {
        println!();
        ui::hint("Apply the fixes, then write a new version: prompt-opt … --write <prompt-id>");
        ui::hint("Score variants with `xybrid eval compare` before promoting.");
    }
    Ok(())
}
