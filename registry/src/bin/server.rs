//! HTTP registry server for serving Xybrid model bundles.
//!
//! This server can serve bundles from:
//! - The `registry/bundles` directory (for development)
//! - The local cache at `~/.xybrid/registry/` (for testing with published models)
//! - Any custom directory path
//!
//! Endpoints:
//! - GET /health - Health check endpoint (returns "ok")
//! - GET /index - Returns JSON array of BundleDescriptor
//! - GET /bundles/{id}/{version}/... - Returns bundle file bytes
//!
//! Usage:
//!   cargo run --bin registry-server [PORT] [REGISTRY_PATH]
//!
//! Examples:
//!   cargo run --bin registry-server                           # Default: port 8080, registry/bundles/
//!   cargo run --bin registry-server 8080                      # Custom port, registry/bundles/
//!   cargo run --bin registry-server 8080 ~/.xybrid/registry   # Serve local cache
//!   cargo run --bin registry-server 8080 --local-cache        # Shorthand for ~/.xybrid/registry

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};

/// Bundle descriptor matching the registry API format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BundleDescriptor {
    id: String,
    version: String,
    /// Target platform (onnx, coreml, tflite, generic)
    #[serde(default = "default_target")]
    target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    hash: Option<String>,
    size_bytes: u64,
    location: BundleLocation,
}

fn default_target() -> String {
    "onnx".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BundleLocation {
    Remote { url: String },
}

/// Registry server state
#[derive(Clone)]
struct RegistryState {
    base_dir: PathBuf,
    bundles: Arc<Mutex<Vec<BundleDescriptor>>>,
}

/// Index entry from local cache index.json
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LocalCacheEntry {
    model_id: String,
    version: String,
    #[serde(default = "default_target")]
    target: String,
    #[serde(default)]
    hash: Option<String>,
    #[serde(default)]
    size_bytes: u64,
    path: String,
}

impl RegistryState {
    fn new(base_dir: PathBuf) -> Self {
        Self {
            base_dir,
            bundles: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Scan the registry directory and build bundle index.
    /// Supports both registry/bundles structure and local cache structure.
    fn scan_bundles(&self) -> Result<Vec<BundleDescriptor>, Box<dyn std::error::Error>> {
        let registry_dir = &self.base_dir;

        if !registry_dir.exists() {
            return Ok(Vec::new());
        }

        // First, try to read existing index.json (local cache format)
        let index_path = registry_dir.join("index.json");
        if index_path.exists() {
            if let Ok(descriptors) = self.load_from_index_json(&index_path) {
                if !descriptors.is_empty() {
                    println!("  Loaded {} bundles from index.json", descriptors.len());
                    return Ok(descriptors);
                }
            }
        }

        // Fallback: scan directory structure
        self.scan_directory_structure()
    }

    /// Load bundles from local cache index.json format
    fn load_from_index_json(&self, index_path: &PathBuf) -> Result<Vec<BundleDescriptor>, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(index_path)?;
        let entries: HashMap<String, LocalCacheEntry> = serde_json::from_str(&content)?;

        let mut descriptors = Vec::new();
        for (_key, entry) in entries {
            // Convert local path to relative URL
            let bundle_path = PathBuf::from(&entry.path);

            // Get actual file size if path exists
            let size_bytes = if bundle_path.exists() {
                bundle_path.metadata().map(|m| m.len()).unwrap_or(entry.size_bytes)
            } else {
                entry.size_bytes
            };

            // Build URL: /bundles/{model_id}/{version}/{target}/{model_id}.xyb (actual structure)
            let url = format!(
                "/bundles/{}/{}/{}/{}.xyb",
                entry.model_id,
                entry.version,
                entry.target,
                entry.model_id
            );

            descriptors.push(BundleDescriptor {
                id: entry.model_id,
                version: entry.version,
                target: entry.target,
                hash: entry.hash,
                size_bytes,
                location: BundleLocation::Remote { url },
            });
        }

        Ok(descriptors)
    }

    /// Scan directory structure for bundles
    /// Supports multiple structures:
    /// - New: {id}/{version}/{target}.xyb
    /// - Legacy: {id}/{version}.bundle or {id}/{version}.xyb
    /// - Old: {id}/{version}/{target}/bundle.xyb
    fn scan_directory_structure(&self) -> Result<Vec<BundleDescriptor>, Box<dyn std::error::Error>> {
        let mut descriptors = Vec::new();
        let registry_dir = &self.base_dir;

        for id_entry in fs::read_dir(registry_dir)? {
            let id_entry = id_entry?;
            if !id_entry.file_type()?.is_dir() {
                continue;
            }

            let id = id_entry.file_name().to_string_lossy().to_string();
            let id_path = id_entry.path();

            // Scan contents of model directory
            for entry in fs::read_dir(&id_path)? {
                let entry = entry?;
                let entry_path = entry.path();

                if entry.file_type()?.is_dir() {
                    // Version directory - scan for targets
                    let version = entry.file_name().to_string_lossy().to_string();
                    self.scan_version_directory(&mut descriptors, registry_dir, &id, &version, &entry_path)?;
                } else {
                    // Legacy: {version}.bundle or {version}.xyb file directly in model dir
                    let ext = entry_path.extension().and_then(|s| s.to_str());
                    if ext == Some("bundle") || ext == Some("xyb") {
                        let filename = entry_path.file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown");
                        let version = filename.to_string();
                        let size_bytes = entry_path.metadata()?.len();

                        let relative_path = entry_path
                            .strip_prefix(registry_dir)
                            .unwrap_or(&entry_path);
                        let url = format!(
                            "/bundles/{}",
                            relative_path.to_string_lossy().replace('\\', "/")
                        );

                        descriptors.push(BundleDescriptor {
                            id: id.clone(),
                            version,
                            target: "onnx".to_string(), // Legacy bundles default to onnx
                            hash: None,
                            size_bytes,
                            location: BundleLocation::Remote { url },
                        });
                    }
                }
            }
        }

        Ok(descriptors)
    }

    /// Scan a version directory for target-specific bundles
    /// Supports:
    /// - New structure: {target}.xyb files directly in version dir
    /// - Old structure: {target}/bundle.xyb (target subdirectory with bundle file)
    fn scan_version_directory(
        &self,
        descriptors: &mut Vec<BundleDescriptor>,
        registry_dir: &PathBuf,
        id: &str,
        version: &str,
        version_path: &PathBuf,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for target_entry in fs::read_dir(version_path)? {
            let target_entry = target_entry?;
            let target_path = target_entry.path();

            if target_entry.file_type()?.is_dir() {
                // Old structure: target directory containing bundle files
                let target = target_entry.file_name().to_string_lossy().to_string();

                for bundle_file in fs::read_dir(&target_path)? {
                    let bundle_file = bundle_file?;
                    let bundle_path = bundle_file.path();
                    let ext = bundle_path.extension().and_then(|s| s.to_str());

                    if ext == Some("xyb") || ext == Some("bundle") {
                        let size_bytes = bundle_path.metadata()?.len();
                        let relative_path = bundle_path
                            .strip_prefix(registry_dir)
                            .unwrap_or(&bundle_path);
                        let url = format!(
                            "/bundles/{}",
                            relative_path.to_string_lossy().replace('\\', "/")
                        );

                        descriptors.push(BundleDescriptor {
                            id: id.to_string(),
                            version: version.to_string(),
                            target: target.clone(),
                            hash: None,
                            size_bytes,
                            location: BundleLocation::Remote { url },
                        });
                    }
                }
            } else {
                // New structure: {target}.xyb file directly in version directory
                let ext = target_path.extension().and_then(|s| s.to_str());
                if ext == Some("xyb") {
                    // Target name is the filename without extension
                    let target = target_path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("onnx")
                        .to_string();

                    let size_bytes = target_path.metadata()?.len();
                    let relative_path = target_path
                        .strip_prefix(registry_dir)
                        .unwrap_or(&target_path);
                    let url = format!(
                        "/bundles/{}",
                        relative_path.to_string_lossy().replace('\\', "/")
                    );

                    descriptors.push(BundleDescriptor {
                        id: id.to_string(),
                        version: version.to_string(),
                        target,
                        hash: None,
                        size_bytes,
                        location: BundleLocation::Remote { url },
                    });
                }
            }
        }

        Ok(())
    }
}

/// GET /health - Health check endpoint
async fn health_check() -> impl IntoResponse {
    "ok"
}

/// GET /index - Returns bundle index
async fn get_index(State(state): State<RegistryState>) -> impl IntoResponse {
    let bundles = state.bundles.lock().unwrap();
    Json(bundles.clone())
}

/// GET /bundles/{id}/{version} - Returns bundle file
async fn get_bundle(
    AxumPath(path): AxumPath<String>,
    State(state): State<RegistryState>,
) -> impl IntoResponse {
    // Path format: "whisper-tiny/1.2/ios-aarch64/whisper-tiny.xyb" or similar
    let bundle_path = state.base_dir.join(&path);

    if !bundle_path.exists() {
        return (StatusCode::NOT_FOUND, "Bundle not found").into_response();
    }

    match fs::read(&bundle_path) {
        Ok(bytes) => {
            (
                StatusCode::OK,
                [("Content-Type", "application/octet-stream")],
                bytes,
            )
                .into_response()
        }
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Failed to read bundle").into_response(),
    }
}

/// Get the local cache directory (~/.xybrid/registry/)
fn get_local_cache_dir() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".xybrid").join("registry"))
        .unwrap_or_else(|| PathBuf::from(".xybrid/registry"))
}

/// Parse command-line arguments
fn parse_args() -> (u16, PathBuf) {
    let args: Vec<String> = std::env::args().collect();

    // Default values
    let mut port = 8080u16;
    let mut registry_dir = PathBuf::from("registry/bundles");

    // Parse arguments
    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];

        if arg == "--local-cache" || arg == "-l" {
            registry_dir = get_local_cache_dir();
        } else if arg == "--help" || arg == "-h" {
            print_usage();
            std::process::exit(0);
        } else if let Ok(p) = arg.parse::<u16>() {
            port = p;
        } else {
            // Treat as registry path
            let path = if arg.starts_with("~") {
                // Expand ~ to home directory
                dirs::home_dir()
                    .map(|h| h.join(&arg[2..]))
                    .unwrap_or_else(|| PathBuf::from(arg))
            } else {
                PathBuf::from(arg)
            };
            registry_dir = path;
        }

        i += 1;
    }

    (port, registry_dir)
}

fn print_usage() {
    println!("Xybrid Registry Server");
    println!();
    println!("Usage: registry-server [PORT] [REGISTRY_PATH]");
    println!();
    println!("Arguments:");
    println!("  PORT              Port to listen on (default: 8080)");
    println!("  REGISTRY_PATH     Path to registry directory");
    println!();
    println!("Options:");
    println!("  --local-cache, -l  Use local cache (~/.xybrid/registry/)");
    println!("  --help, -h         Show this help message");
    println!();
    println!("Examples:");
    println!("  registry-server                           # port 8080, registry/bundles/");
    println!("  registry-server 8080                      # port 8080, registry/bundles/");
    println!("  registry-server 8080 ~/.xybrid/registry   # port 8080, local cache");
    println!("  registry-server --local-cache             # port 8080, local cache");
    println!("  registry-server 9000 -l                   # port 9000, local cache");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (port, registry_dir) = parse_args();

    if !registry_dir.exists() {
        eprintln!("❌ Error: Registry directory not found: {}", registry_dir.display());
        eprintln!();
        eprintln!("Make sure the directory exists, or use --local-cache to serve ~/.xybrid/registry/");
        std::process::exit(1);
    }

    let state = RegistryState::new(registry_dir.clone());
    let bundles = state.scan_bundles()?;
    let bundle_count = bundles.len();
    *state.bundles.lock().unwrap() = bundles;

    println!("📦 Xybrid Registry Server");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📂 Serving from: {}", registry_dir.display());
    println!("🌐 Listening on: http://localhost:{}", port);
    println!("📊 Bundles found: {}", bundle_count);
    println!();
    println!("Endpoints:");
    println!("  GET /health                   - Health check");
    println!("  GET /index                    - Bundle index (JSON)");
    println!("  GET /bundles/{{id}}/{{ver}}/... - Download bundle");
    println!();

    if bundle_count > 0 {
        let bundles_guard = state.bundles.lock().unwrap();
        println!("Available bundles:");
        for bundle in bundles_guard.iter() {
            let size_mb = bundle.size_bytes as f64 / 1_000_000.0;
            println!("  • {}@{}:{} - {:.1} MB", bundle.id, bundle.version, bundle.target, size_mb);
        }
        println!();
    }

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/index", get(get_index))
        .route("/bundles/*path", get(get_bundle))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

