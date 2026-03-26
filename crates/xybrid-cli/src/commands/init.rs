//! `xybrid init` command handler.
//!
//! Inspects a model directory and generates `model_metadata.json` by analyzing
//! model files, supporting configs, and ONNX tensor metadata.

use anyhow::{Context, Result};
use colored::*;
use std::path::{Path, PathBuf};

use xybrid_sdk::metadata_gen::{self, Confidence, TaskInference};
use xybrid_sdk::SdkError;

/// Flags parsed from CLI arguments for the init command.
pub(crate) struct InitFlags {
    pub force: bool,
    pub yes: bool,
    pub task: Option<String>,
    pub json: bool,
    pub model_id: Option<String>,
}

/// Handle the `xybrid init <directory>` command.
///
/// Exit codes: 0=success, 1=no model files, 2=inspection failed, 3=user cancelled
pub(crate) fn handle_init_command(dir: &str, flags: InitFlags) -> Result<()> {
    let dir_path = PathBuf::from(dir)
        .canonicalize()
        .with_context(|| format!("Directory not found: {}", dir))?;

    if !dir_path.is_dir() {
        let err_msg = format!("'{}' is not a directory", dir);
        if flags.json {
            print_json_error(1, &err_msg);
        } else {
            eprintln!("{} {}", "error:".bright_red().bold(), err_msg);
        }
        std::process::exit(1);
    }

    // Check if model_metadata.json already exists
    let metadata_path = dir_path.join("model_metadata.json");
    if metadata_path.exists() && !flags.force {
        let err_msg = format!(
            "model_metadata.json already exists in '{}'. Use --force to overwrite.",
            dir_path.display()
        );
        if flags.json {
            print_json_error(1, &err_msg);
        } else {
            eprintln!("{} {}", "warning:".yellow().bold(), err_msg);
        }
        std::process::exit(1);
    }

    // --json implies --yes (non-interactive)
    let non_interactive = flags.yes || flags.json;

    if !flags.json {
        println!("🔍 Scanning directory: {}", dir_path.display());
        println!("{}", "=".repeat(60));
        println!();
    }

    // Run inspection and generation
    let result = metadata_gen::inspect_and_generate(
        &dir_path,
        "", // empty repo — derive model_id from directory name
        flags.model_id.as_deref(),
    );

    let (mut metadata, task_inference) = match result {
        Ok(pair) => pair,
        Err(SdkError::LoadError(msg)) => {
            if flags.json {
                print_json_error(1, &msg);
            } else {
                eprintln!("{} {}", "error:".bright_red().bold(), msg);
            }
            std::process::exit(1);
        }
        Err(e) => {
            let msg = format!("Inspection failed: {}", e);
            if flags.json {
                print_json_error(2, &msg);
            } else {
                eprintln!("{} {}", "error:".bright_red().bold(), msg);
            }
            std::process::exit(2);
        }
    };

    // Override task if --task flag is set
    if let Some(ref task_override) = flags.task {
        metadata.metadata.insert(
            "task".to_string(),
            serde_json::Value::String(task_override.clone()),
        );
    }

    let confidence = task_inference
        .as_ref()
        .map(|ti| ti.confidence)
        .unwrap_or(Confidence::Low);
    let detected_task = task_inference
        .as_ref()
        .map(|ti| ti.task.clone())
        .or_else(|| {
            metadata
                .metadata
                .get("task")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    // Handle ambiguous cases with interactive prompt
    if confidence == Confidence::Low && !non_interactive {
        if let Some(ref ti) = task_inference {
            if !ti.alternatives.is_empty() {
                match prompt_task_selection(ti) {
                    Some(selected) => {
                        metadata.preprocessing = selected.preprocessing;
                        metadata.postprocessing = selected.postprocessing;
                        metadata.metadata.insert(
                            "task".to_string(),
                            serde_json::Value::String(selected.task.clone()),
                        );
                    }
                    None => {
                        eprintln!("{} User cancelled.", "info:".bright_blue().bold());
                        std::process::exit(3);
                    }
                }
            }
        }
    }

    // Print summary (unless --json)
    if !flags.json {
        print_summary(&dir_path, &metadata, &detected_task, confidence);
    }

    // Write model_metadata.json
    metadata_gen::write_metadata(&dir_path, &metadata)
        .context("Failed to write model_metadata.json")?;

    if flags.json {
        let json_output = serde_json::json!({
            "status": "ok",
            "path": metadata_path.display().to_string(),
            "task": detected_task,
            "confidence": match confidence {
                Confidence::High => "high",
                Confidence::Medium => "medium",
                Confidence::Low => "low",
            }
        });
        println!("{}", serde_json::to_string(&json_output).unwrap());
    } else {
        println!();
        println!(
            "{}",
            format!("✅ Written: {}", metadata_path.display())
                .green()
                .bold()
        );
        println!();
        println!(
            "Test it: {} {} {} {}",
            "xybrid run".bright_cyan(),
            "--directory".bright_cyan(),
            dir_path.display().to_string().bright_cyan(),
            "--input-text \"Hello world\"".bright_cyan()
        );
    }

    Ok(())
}

/// Print a human-readable summary of the generated metadata.
fn print_summary(
    dir: &Path,
    metadata: &xybrid_core::execution::ModelMetadata,
    task: &str,
    confidence: Confidence,
) {
    println!(
        "  {} {}",
        "Model ID:".bright_white().bold(),
        metadata.model_id.cyan()
    );
    println!(
        "  {} {}",
        "Template:".bright_white().bold(),
        format_template(&metadata.execution_template)
    );
    println!("  {} {}", "Task:    ".bright_white().bold(), task.yellow());

    let confidence_str = match confidence {
        Confidence::High => "high".green(),
        Confidence::Medium => "medium".yellow(),
        Confidence::Low => "low".red(),
    };
    println!(
        "  {} {}",
        "Confidence:".bright_white().bold(),
        confidence_str
    );
    println!();

    // Files found
    let files = metadata_gen::list_model_files_pub(dir);
    println!(
        "  {} ({} files)",
        "Files found:".bright_white().bold(),
        files.len()
    );
    for f in &files {
        println!("    {} {}", "•".bright_cyan(), f);
    }
    println!();

    // Preprocessing
    println!("  {}", "Preprocessing:".bright_white().bold());
    if metadata.preprocessing.is_empty() {
        println!("    {} (none)", "•".bright_black());
    } else {
        for step in &metadata.preprocessing {
            println!(
                "    {} {}",
                "•".bright_cyan(),
                format_preprocessing_step(step)
            );
        }
    }
    println!();

    // Postprocessing
    println!("  {}", "Postprocessing:".bright_white().bold());
    if metadata.postprocessing.is_empty() {
        println!("    {} (none)", "•".bright_black());
    } else {
        for step in &metadata.postprocessing {
            println!(
                "    {} {}",
                "•".bright_cyan(),
                format_postprocessing_step(step)
            );
        }
    }
}

/// Format an ExecutionTemplate variant name for display.
fn format_template(template: &xybrid_core::execution::ExecutionTemplate) -> String {
    match template {
        xybrid_core::execution::ExecutionTemplate::Onnx { model_file, .. } => {
            format!("Onnx ({})", model_file)
        }
        xybrid_core::execution::ExecutionTemplate::Gguf {
            model_file,
            context_length,
            ..
        } => {
            format!("Gguf ({}, ctx={})", model_file, context_length)
        }
        xybrid_core::execution::ExecutionTemplate::SafeTensors { model_file, .. } => {
            format!("SafeTensors ({})", model_file)
        }
        _ => format!("{:?}", template),
    }
}

/// Format a preprocessing step for display.
fn format_preprocessing_step(step: &xybrid_core::execution::PreprocessingStep) -> String {
    match step {
        xybrid_core::execution::PreprocessingStep::AudioDecode { sample_rate, .. } => {
            format!("AudioDecode ({}Hz)", sample_rate)
        }
        xybrid_core::execution::PreprocessingStep::Phonemize { backend, .. } => {
            format!("Phonemize ({:?})", backend)
        }
        xybrid_core::execution::PreprocessingStep::Tokenize {
            vocab_file,
            max_length,
            ..
        } => {
            let max = max_length.map_or("none".to_string(), |m| m.to_string());
            format!("Tokenize ({}, max_length={})", vocab_file, max)
        }
        xybrid_core::execution::PreprocessingStep::Normalize { mean, std, .. } => {
            format!("Normalize (mean={:?}, std={:?})", mean, std)
        }
        xybrid_core::execution::PreprocessingStep::Reshape { shape, .. } => {
            format!("Reshape ({:?})", shape)
        }
        other => format!("{:?}", other),
    }
}

/// Format a postprocessing step for display.
fn format_postprocessing_step(step: &xybrid_core::execution::PostprocessingStep) -> String {
    match step {
        xybrid_core::execution::PostprocessingStep::CTCDecode { .. } => "CTCDecode".to_string(),
        xybrid_core::execution::PostprocessingStep::TTSAudioEncode { sample_rate, .. } => {
            format!("TTSAudioEncode ({}Hz)", sample_rate)
        }
        xybrid_core::execution::PostprocessingStep::Softmax { .. } => "Softmax".to_string(),
        xybrid_core::execution::PostprocessingStep::Argmax { .. } => "Argmax".to_string(),
        xybrid_core::execution::PostprocessingStep::MeanPool { .. } => "MeanPool".to_string(),
        other => format!("{:?}", other),
    }
}

/// Prompt user to select from alternative task interpretations using dialoguer.
fn prompt_task_selection(ti: &TaskInference) -> Option<TaskInference> {
    use dialoguer::Select;

    let mut options = vec![format!("{} (primary)", ti.task)];
    for alt in &ti.alternatives {
        options.push(alt.task.clone());
    }
    options.push("Cancel".to_string());

    println!(
        "  {} Multiple task interpretations detected (confidence: {}):",
        "⚠️".yellow(),
        "low".red()
    );
    println!();

    let selection = Select::new()
        .with_prompt("Select the correct task")
        .items(&options)
        .default(0)
        .interact_opt()
        .ok()
        .flatten();

    match selection {
        Some(0) => Some(ti.clone()),
        Some(i) if i <= ti.alternatives.len() => Some(ti.alternatives[i - 1].clone()),
        _ => None, // Cancel or error
    }
}

/// Print a JSON error message to stderr.
fn print_json_error(code: i32, message: &str) {
    let json = serde_json::json!({
        "status": "error",
        "code": code,
        "message": message,
    });
    eprintln!("{}", serde_json::to_string(&json).unwrap());
}
