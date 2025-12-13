//! Create real .xyb bundles with proper model files for testing
//!
//! By default, bundles are created in the local cache (~/.xybrid/registry/).
//! Use --output to specify a different output directory.
//!
//! Usage:
//!   cargo run --bin create-bundles -p registry                         # Create all bundles to ~/.xybrid/registry/
//!   cargo run --bin create-bundles -p registry -- whisper-tiny         # Create specific bundle
//!   cargo run --bin create-bundles -p registry -- --output ./bundles   # Create to custom path
//!   cargo run --bin create-bundles -p registry -- --list               # List available bundles
//!   cargo run --bin create-bundles -p registry -- --clean              # Clean up bundles

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use xybrid_core::bundler::XyBundle;

/// Index entry for the registry's index.json file
/// This matches the format expected by the registry server
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexEntry {
    model_id: String,
    version: String,
    target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    hash: Option<String>,
    size_bytes: u64,
    path: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct BundleConfig {
    bundles: std::collections::HashMap<String, BundleDefinition>,
}

#[derive(Debug, Deserialize, Serialize)]
struct BundleDefinition {
    model_id: String,
    version: String,
    targets: Vec<TargetDefinition>,
    description: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct TargetDefinition {
    platform: String,
    model_file: String,
    model_type: String,
    /// Path to a single model file (legacy format)
    #[serde(default)]
    source_path: Option<String>,
    /// Path to a directory containing all model files (preferred for metadata-driven models)
    #[serde(default)]
    source_dir: Option<String>,
    fallback: String,
}

/// Get the default local cache directory (~/.xybrid/registry/)
fn get_local_cache_dir() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".xybrid").join("registry"))
        .unwrap_or_else(|| PathBuf::from(".xybrid/registry"))
}

/// Load the existing index.json or create a new empty index
fn load_index(registry_dir: &Path) -> HashMap<String, IndexEntry> {
    let index_path = registry_dir.join("index.json");
    if index_path.exists() {
        if let Ok(content) = fs::read_to_string(&index_path) {
            if let Ok(index) = serde_json::from_str(&content) {
                return index;
            }
        }
    }
    HashMap::new()
}

/// Save the index to index.json
fn save_index(registry_dir: &Path, index: &HashMap<String, IndexEntry>) -> Result<(), Box<dyn std::error::Error>> {
    let index_path = registry_dir.join("index.json");
    let content = serde_json::to_string_pretty(index)?;
    fs::write(&index_path, content)?;
    Ok(())
}

/// Add or update an entry in the index
fn update_index_entry(
    index: &mut HashMap<String, IndexEntry>,
    model_id: &str,
    version: &str,
    target: &str,
    bundle_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let key = format!("{}/{}/{}", model_id, version, target);
    let size_bytes = bundle_path.metadata()?.len();

    index.insert(key, IndexEntry {
        model_id: model_id.to_string(),
        version: version.to_string(),
        target: target.to_string(),
        hash: None, // Could add SHA256 hash here in the future
        size_bytes,
        path: bundle_path.to_string_lossy().to_string(),
    });

    Ok(())
}

/// Parse command-line arguments
fn parse_args() -> (Option<String>, PathBuf, bool, bool) {
    let args: Vec<String> = env::args().collect();

    let mut bundle_id: Option<String> = None;
    let mut output_dir = get_local_cache_dir();
    let mut list_mode = false;
    let mut clean_mode = false;

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];

        match arg.as_str() {
            "--list" | "-l" => {
                list_mode = true;
            }
            "--clean" | "-c" => {
                clean_mode = true;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    let path = &args[i];
                    // Expand ~ to home directory
                    output_dir = if path.starts_with("~") {
                        dirs::home_dir()
                            .map(|h| h.join(&path[2..]))
                            .unwrap_or_else(|| PathBuf::from(path))
                    } else {
                        PathBuf::from(path)
                    };
                } else {
                    eprintln!("❌ Error: --output requires a path argument");
                    std::process::exit(1);
                }
            }
            _ => {
                // Treat as bundle ID if not a flag
                if !arg.starts_with('-') {
                    bundle_id = Some(arg.clone());
                }
            }
        }

        i += 1;
    }

    (bundle_id, output_dir, list_mode, clean_mode)
}

/// Find the bundles.json config file
/// Searches in multiple locations to support running from different directories
fn find_config_file() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("registry/bundles/bundles.json"), // From project root
        PathBuf::from("bundles/bundles.json"),          // From registry directory
        PathBuf::from("bundles.json"),                  // From bundles directory
    ];

    for path in &candidates {
        if path.exists() {
            return Some(path.clone());
        }
    }
    None
}

/// Find a source path (file or directory) by searching in multiple locations
/// Supports running from project root or from registry directory
fn find_source_path(source: &str) -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(source),              // As specified (from project root)
        PathBuf::from("..").join(source),   // From registry directory (go up one level)
    ];

    for path in &candidates {
        if path.exists() {
            return Some(path.clone());
        }
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (bundle_id, output_dir, list_mode, clean_mode) = parse_args();

    // Load bundle configuration
    let config_path = match find_config_file() {
        Some(path) => path,
        None => {
            eprintln!("❌ Error: bundles.json not found");
            eprintln!("   Searched in:");
            eprintln!("     - registry/bundles/bundles.json (from project root)");
            eprintln!("     - bundles/bundles.json (from registry directory)");
            eprintln!("     - bundles.json (from bundles directory)");
            std::process::exit(1);
        }
    };

    let config_content = fs::read_to_string(&config_path)?;
    let config: BundleConfig = serde_json::from_str(&config_content)?;

    // Handle list mode
    if list_mode {
        list_bundles(&config);
        return Ok(());
    }

    // Handle clean mode
    if clean_mode {
        cleanup_bundles(&output_dir)?;
        return Ok(());
    }

    // Create output directory if it doesn't exist
    fs::create_dir_all(&output_dir)?;

    // Load existing index
    let mut index = load_index(&output_dir);

    println!("🔨 Xybrid Bundle Creator");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📂 Output: {}", output_dir.display());
    println!();

    // Create specific bundle or all bundles
    if let Some(id) = bundle_id {
        if let Some(bundle_def) = config.bundles.get(&id) {
            create_bundle_from_config(bundle_def, &output_dir, &mut index)?;
            println!();
            println!("✅ Bundle '{}' created successfully!", id);
        } else {
            eprintln!("❌ Error: Bundle '{}' not found", id);
            eprintln!();
            list_bundles(&config);
            std::process::exit(1);
        }
    } else {
        // Create all bundles
        println!("Creating all bundles...");
        println!();

        let mut success_count = 0;
        let mut fail_count = 0;

        for (bundle_id, bundle_def) in &config.bundles {
            match create_bundle_from_config(bundle_def, &output_dir, &mut index) {
                Ok(_) => {
                    success_count += 1;
                }
                Err(e) => {
                    eprintln!("  ❌ Failed to create bundle '{}': {}", bundle_id, e);
                    fail_count += 1;
                }
            }
            println!();
        }

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("✅ Bundle creation complete!");
        println!("   Success: {}, Failed: {}", success_count, fail_count);

        if fail_count > 0 {
            std::process::exit(1);
        }
    }

    // Save updated index
    save_index(&output_dir, &index)?;
    println!("📋 Updated index.json with {} entries", index.len());

    println!();
    println!("To serve these bundles:");
    println!("  just registry-local");
    println!("  # or: cargo run --bin registry-server -p registry -- --local-cache");

    Ok(())
}

fn list_bundles(config: &BundleConfig) {
    println!("📋 Available Bundles:");
    println!();
    for (bundle_id, bundle_def) in &config.bundles {
        println!("  {} (v{})", bundle_id, bundle_def.version);
        println!("    Description: {}", bundle_def.description);
        println!("    Targets: {}", bundle_def.targets.len());
        for target in &bundle_def.targets {
            println!("      - {} ({})", target.platform, target.model_type);
        }
        println!();
    }
}

fn print_help() {
    println!("Xybrid Bundle Creator");
    println!();
    println!("Creates .xyb bundles from model files. By default, bundles are created");
    println!("in the local cache (~/.xybrid/registry/).");
    println!();
    println!("Usage:");
    println!("  create-bundles [OPTIONS] [BUNDLE_ID]");
    println!();
    println!("Arguments:");
    println!("  BUNDLE_ID              Create only this specific bundle (optional)");
    println!();
    println!("Options:");
    println!("  -o, --output <PATH>    Output directory (default: ~/.xybrid/registry/)");
    println!("  -l, --list             List available bundles");
    println!("  -c, --clean            Clean up all .xyb files in output directory");
    println!("  -h, --help             Show this help message");
    println!();
    println!("Examples:");
    println!("  create-bundles                          # Create all bundles to local cache");
    println!("  create-bundles whisper-tiny             # Create specific bundle");
    println!("  create-bundles --output ./my-bundles    # Create to custom directory");
    println!("  create-bundles --list                   # List available bundles");
    println!("  create-bundles --clean                  # Clean local cache");
    println!();
}

fn cleanup_bundles(registry_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧹 Cleaning up bundles in {}...", registry_dir.display());

    if !registry_dir.exists() {
        println!("  ℹ️  Directory does not exist");
        return Ok(());
    }

    let mut cleaned_files = 0;
    let mut cleaned_dirs = 0;

    // Remove all .xyb files and empty directories
    for entry in fs::read_dir(registry_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Recursively find and remove .xyb files, then clean empty directories
            remove_xyb_files_and_empty_dirs(&path, &mut cleaned_files, &mut cleaned_dirs)?;
        }
    }

    // Remove index.json if it exists
    let index_path = registry_dir.join("index.json");
    if index_path.exists() {
        fs::remove_file(&index_path)?;
        println!("    Removed: index.json");
    }

    println!("  ✅ Cleaned {} bundle files, {} directories", cleaned_files, cleaned_dirs);
    Ok(())
}

/// Recursively remove .xyb files and clean up empty directories
fn remove_xyb_files_and_empty_dirs(
    dir: &Path,
    file_count: &mut usize,
    dir_count: &mut usize,
) -> Result<bool, Box<dyn std::error::Error>> {
    let mut has_remaining_files = false;

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Recursively process subdirectory
            let subdir_has_files = remove_xyb_files_and_empty_dirs(&path, file_count, dir_count)?;
            if subdir_has_files {
                has_remaining_files = true;
            }
        } else if path.extension().and_then(|s| s.to_str()) == Some("xyb") {
            fs::remove_file(&path)?;
            *file_count += 1;
            println!("    Removed: {}", path.display());
        } else {
            // There's a non-.xyb file, so this directory shouldn't be removed
            has_remaining_files = true;
        }
    }

    // If directory is now empty (no remaining files), remove it
    if !has_remaining_files {
        // Check if directory is truly empty
        if fs::read_dir(dir)?.next().is_none() {
            fs::remove_dir(dir)?;
            *dir_count += 1;
            println!("    Removed dir: {}", dir.display());
        }
    }

    Ok(has_remaining_files)
}

fn create_bundle_from_config(
    bundle_def: &BundleDefinition,
    registry_dir: &Path,
    index: &mut HashMap<String, IndexEntry>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "📦 Creating bundle: {} (v{})",
        bundle_def.model_id, bundle_def.version
    );

    for target in &bundle_def.targets {
        // Check if source_dir is specified (new format - bundles entire directory)
        if let Some(source_dir) = &target.source_dir {
            create_bundle_from_directory(
                registry_dir,
                &bundle_def.model_id,
                &bundle_def.version,
                &target.platform,
                source_dir,
                &target.fallback,
                index,
            )?;
        } else if let Some(source_path) = &target.source_path {
            // Legacy format - single model file
            create_bundle(
                registry_dir,
                &bundle_def.model_id,
                &bundle_def.version,
                &target.platform,
                &target.model_file,
                &target.model_type,
                source_path,
                &target.fallback,
                index,
            )?;
        } else {
            eprintln!(
                "  ⚠️  Target {} has no source_path or source_dir",
                target.platform
            );
        }
    }

    Ok(())
}

/// Create a bundle from all files in a directory (for metadata-driven models)
/// If model_metadata.json exists and has a "files" array, only those files are included.
/// Otherwise, all files in the directory are included recursively.
fn create_bundle_from_directory(
    registry_dir: &Path,
    model_id: &str,
    version: &str,
    platform: &str,
    source_dir: &str,
    fallback: &str,
    index: &mut HashMap<String, IndexEntry>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Find source directory (supports running from project root or registry directory)
    let source_path = match find_source_path(source_dir) {
        Some(path) if path.is_dir() => path,
        _ => {
            if fallback == "error" {
                return Err(format!("Source directory not found: {}", source_dir).into());
            }
            println!("  ⚠️  Source directory not found: {}", source_dir);
            return Ok(());
        }
    };

    // Create output directory: {registry}/{model_id}/{version}/{platform}/
    let bundle_dir = registry_dir.join(model_id).join(version).join(platform);
    fs::create_dir_all(&bundle_dir)?;

    // Create bundle
    let mut bundle = XyBundle::new(model_id, version, platform);
    let mut file_count = 0;

    // Check if model_metadata.json exists and has a "files" list
    let metadata_path = source_path.join("model_metadata.json");
    let files_to_include = if metadata_path.exists() {
        // Parse metadata to get explicit file list
        let metadata_content = fs::read_to_string(&metadata_path)?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;

        if let Some(files_array) = metadata.get("files").and_then(|f| f.as_array()) {
            let mut files: Vec<String> = files_array
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            // Always include model_metadata.json itself
            if !files.contains(&"model_metadata.json".to_string()) {
                files.push("model_metadata.json".to_string());
            }
            Some(files)
        } else {
            None
        }
    } else {
        None
    };

    if let Some(files) = files_to_include {
        // Add only the specified files
        println!("    Using files list from model_metadata.json ({} files)", files.len());
        for rel_path in &files {
            let file_path = source_path.join(rel_path);
            if file_path.exists() {
                bundle.add_file_with_relative_path(&file_path, rel_path)?;
                file_count += 1;
            } else {
                println!("    ⚠️  File not found: {}", rel_path);
            }
        }
    } else {
        // Fall back to adding all files recursively
        add_directory_files_recursive(&source_path, &source_path, &mut bundle, &mut file_count)?;
    }

    if file_count == 0 {
        return Err(format!("No files found in source directory: {}", source_dir).into());
    }

    // Write bundle
    let bundle_path = bundle_dir.join(format!("{}.xyb", model_id));
    bundle.write(&bundle_path)?;

    // Update index
    update_index_entry(index, model_id, version, platform, &bundle_path)?;

    let size_mb = bundle_path.metadata()?.len() as f64 / 1_000_000.0;
    println!(
        "  ✅ Created: {} ({} files, {:.1} MB)",
        bundle_path.display(),
        file_count,
        size_mb
    );

    Ok(())
}

/// Recursively add files from a directory to the bundle, preserving relative paths
fn add_directory_files_recursive(
    base_path: &Path,
    current_path: &Path,
    bundle: &mut XyBundle,
    file_count: &mut usize,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(current_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            // Calculate relative path from base directory
            let rel_path = path
                .strip_prefix(base_path)
                .map_err(|e| format!("Failed to get relative path: {}", e))?
                .to_string_lossy()
                .to_string();

            bundle.add_file_with_relative_path(&path, &rel_path)?;
            *file_count += 1;
        } else if path.is_dir() {
            // Recursively process subdirectories
            add_directory_files_recursive(base_path, &path, bundle, file_count)?;
        }
    }

    Ok(())
}

fn create_bundle(
    registry_dir: &Path,
    model_id: &str,
    version: &str,
    platform: &str,
    model_filename: &str,
    model_type: &str,
    source_path_str: &str,
    fallback: &str,
    index: &mut HashMap<String, IndexEntry>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create directory structure: {registry}/{model_id}/{version}/{platform}/
    let bundle_dir = registry_dir.join(model_id).join(version).join(platform);
    fs::create_dir_all(&bundle_dir)?;

    // Create a temporary directory for model files
    let temp_dir = std::env::temp_dir().join(format!("xybrid_bundle_{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&temp_dir)?;

    let model_path = temp_dir.join(model_filename);

    // Try to use real model file if it exists (supports running from project root or registry directory)
    let real_model_path = find_source_path(source_path_str);
    let use_real_model = real_model_path
        .as_ref()
        .map(|p| p.is_file() && p.metadata().map(|m| m.len() > 1000).unwrap_or(false))
        .unwrap_or(false);

    if use_real_model {
        let real_path = real_model_path.as_ref().unwrap();
        println!("  📦 Using real model: {}", real_path.display());
        fs::copy(real_path, &model_path)?;
    } else {
        // Create placeholder model file
        match model_type {
            "coreml" => create_minimal_coreml_model(&model_path)?,
            "onnx" => create_minimal_onnx_file(&model_path)?,
            _ => {
                // Generic placeholder
                fs::write(
                    &model_path,
                    format!("{} model placeholder for testing", model_type),
                )?;
            }
        }

        if fallback == "placeholder" {
            println!(
                "  ⚠️  Using placeholder (real model not found at: {})",
                source_path_str
            );
        }
    }

    // Create bundle
    let mut bundle = XyBundle::new(model_id, version, platform);
    bundle.add_file(&model_path)?;

    // Write bundle
    let bundle_path = bundle_dir.join(format!("{}.xyb", model_id));
    bundle.write(&bundle_path)?;

    // Update index
    update_index_entry(index, model_id, version, platform, &bundle_path)?;

    let size_kb = bundle_path.metadata()?.len() as f64 / 1024.0;
    println!("  ✅ Created: {} ({:.1} KB)", bundle_path.display(), size_kb);

    // Cleanup temp directory
    fs::remove_dir_all(&temp_dir)?;

    Ok(())
}

/// Creates a minimal CoreML .mlmodel file
fn create_minimal_coreml_model(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(path, b"CoreML model placeholder for testing")?;
    Ok(())
}

/// Creates a minimal ONNX file
fn create_minimal_onnx_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(path, b"ONNX model placeholder for testing")?;
    Ok(())
}
