mod setup_env;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Checks if a target is an iOS target.
fn is_ios_target(target: &str) -> bool {
    target.contains("apple-ios")
}

/// Sets fp16 rustflags for iOS targets via CARGO_TARGET_<TRIPLE>_RUSTFLAGS env var.
/// Required by gemm-f16 crate (used by Candle for Whisper).
/// See: https://github.com/sarah-quinones/gemm/issues/31
fn set_ios_rustflags(cmd: &mut Command, target: &str) {
    if !is_ios_target(target) {
        return;
    }
    let env_key = format!(
        "CARGO_TARGET_{}_RUSTFLAGS",
        target.to_uppercase().replace('-', "_")
    );
    cmd.env(&env_key, "-Ctarget-feature=+fp16");
    println!("  Setting {}=\"-Ctarget-feature=+fp16\"", env_key);
}

/// Maps a Rust target triple to the corresponding xcframework subdirectory name.
/// Returns None if the target is not an iOS target or has no known mapping.
fn xcframework_subdir_for_target(target: &str) -> Option<&'static str> {
    match target {
        "aarch64-apple-ios" => Some("ios-arm64"),
        "aarch64-apple-ios-sim" => Some("ios-arm64-simulator"),
        "x86_64-apple-ios" => Some("ios-x86_64-simulator"),
        _ if target.contains("apple-ios") => Some("ios-arm64"), // fallback for unknown iOS
        _ => None,
    }
}

/// Resolves the ORT iOS library location for iOS targets.
///
/// Resolution order:
/// 1. ORT_LIB_LOCATION env var if set and contains libonnxruntime.a
/// 2. vendor/ort-ios/onnxruntime.xcframework/<subdir>/ matching the target
/// 3. None if not found
///
/// Returns None for non-iOS targets or when no library exists for the target.
fn resolve_ort_lib_location(target: &str) -> Option<PathBuf> {
    // Only applies to iOS targets
    if !is_ios_target(target) {
        return None;
    }

    // Check ORT_LIB_LOCATION env var first (explicit override, any target)
    if let Ok(env_path) = std::env::var("ORT_LIB_LOCATION") {
        let lib_path = PathBuf::from(&env_path).join("libonnxruntime.a");
        if lib_path.exists() {
            return Some(PathBuf::from(env_path));
        } else {
            eprintln!(
                "Warning: ORT_LIB_LOCATION is set to '{}' but libonnxruntime.a not found there",
                env_path
            );
        }
    }

    // Map target to xcframework subdirectory
    let subdir = xcframework_subdir_for_target(target)?;

    // Check vendored location (must return absolute path since ort-sys build script
    // runs from a different working directory)
    let vendor_path = PathBuf::from(format!(
        "vendor/ort-ios/onnxruntime.xcframework/{}",
        subdir
    ));
    let vendor_lib = vendor_path.join("libonnxruntime.a");
    if vendor_lib.exists() {
        let abs_path = vendor_path.canonicalize().unwrap_or(vendor_path);
        println!(
            "Using ORT iOS library: {}",
            abs_path.display()
        );
        return Some(abs_path);
    }

    None
}

/// Maps a Rust target triple to the appropriate xybrid platform preset feature.
///
/// Platform presets configure ORT execution providers and LLM backends correctly
/// for each target platform. Unknown targets fall back to `platform-desktop`.
///
/// # Arguments
/// * `target` - A Rust target triple (e.g., "aarch64-apple-darwin")
///
/// # Returns
/// The platform preset feature name (e.g., "platform-macos")
fn platform_preset_for_target(target: &str) -> &'static str {
    match target {
        // macOS targets
        "aarch64-apple-darwin" | "x86_64-apple-darwin" => "platform-macos",

        // iOS targets
        "aarch64-apple-ios" | "aarch64-apple-ios-sim" | "x86_64-apple-ios" => "platform-ios",

        // Android targets
        "aarch64-linux-android" | "armv7-linux-androideabi" | "x86_64-linux-android" => {
            "platform-android"
        }

        // Desktop/unknown targets
        "x86_64-unknown-linux-gnu" | "x86_64-pc-windows-msvc" => "platform-desktop",

        // Unknown target - warn and fallback to desktop
        _ => {
            eprintln!(
                "Warning: Unknown target '{}', falling back to platform-desktop",
                target
            );
            "platform-desktop"
        }
    }
}

/// Returns the platform preset for the current build machine.
///
/// Uses compile-time detection to determine the host platform.
fn host_platform_preset() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "platform-macos"
    }
    #[cfg(target_os = "windows")]
    {
        "platform-desktop"
    }
    #[cfg(target_os = "linux")]
    {
        "platform-desktop"
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        "platform-desktop"
    }
}

/// Get the version from Cargo.toml workspace or git tag
fn get_version(override_version: Option<&str>) -> String {
    // If explicit version provided, use it
    if let Some(v) = override_version {
        return v.to_string();
    }

    // Try to get version from git tag first (e.g., "v0.1.0" -> "0.1.0")
    if let Ok(output) = Command::new("git")
        .args(["describe", "--tags", "--exact-match"])
        .output()
    {
        if output.status.success() {
            let tag = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if let Some(version) = tag.strip_prefix('v') {
                return version.to_string();
            }
            return tag;
        }
    }

    // Fall back to reading from Cargo.toml
    let cargo_toml_path = PathBuf::from("Cargo.toml");
    if cargo_toml_path.exists() {
        if let Ok(contents) = std::fs::read_to_string(&cargo_toml_path) {
            // Simple parsing to find version in [workspace.package] section
            let mut in_workspace_package = false;
            for line in contents.lines() {
                let trimmed = line.trim();
                if trimmed == "[workspace.package]" {
                    in_workspace_package = true;
                    continue;
                }
                if trimmed.starts_with('[') {
                    in_workspace_package = false;
                }
                if in_workspace_package && trimmed.starts_with("version") {
                    if let Some(pos) = trimmed.find('=') {
                        let value = trimmed[pos + 1..].trim();
                        // Remove quotes
                        let version = value.trim_matches('"').trim_matches('\'');
                        return version.to_string();
                    }
                }
            }
        }
    }

    // Default fallback
    "0.0.0".to_string()
}

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "Development tasks for Xybrid", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Setup the integration test environment (download models etc)
    SetupTestEnv {
        /// Registry URL to download models from
        #[arg(long)]
        registry: Option<String>,
    },

    /// Build the xybrid-uniffi library (Swift/Kotlin FFI via UniFFI)
    BuildUniffi {
        /// Target triple (e.g., aarch64-apple-darwin, aarch64-linux-android)
        #[arg(long)]
        target: Option<String>,

        /// Build in release mode
        #[arg(long)]
        release: bool,

        /// Override the platform preset (e.g., platform-macos, platform-ios, platform-android, platform-desktop)
        /// If not specified, auto-detected from target or host
        #[arg(long)]
        platform_preset: Option<String>,
    },

    /// Build the xybrid-ffi library (C ABI for Unity/C++)
    BuildFfi {
        /// Target triple (e.g., aarch64-apple-darwin, x86_64-unknown-linux-gnu)
        #[arg(long)]
        target: Option<String>,

        /// Build in release mode
        #[arg(long)]
        release: bool,

        /// Override the platform preset (e.g., platform-macos, platform-ios, platform-android, platform-desktop)
        /// If not specified, auto-detected from target or host
        #[arg(long)]
        platform_preset: Option<String>,

        /// Generate C# bindings for Unity (enables csharp feature)
        #[arg(long)]
        csharp: bool,

        /// Copy built library to Unity bindings directory
        #[arg(long)]
        deploy_unity: bool,
    },

    /// Generate Swift/Kotlin bindings from xybrid-uniffi
    GenerateBindings {
        /// Language to generate bindings for
        #[arg(long, value_enum, default_value = "all")]
        language: BindingsLanguage,

        /// Output directory for generated bindings
        #[arg(long)]
        out_dir: Option<PathBuf>,
    },

    /// Build Apple XCFramework for iOS and macOS platforms
    BuildXcframework {
        /// Build in release mode (default: true)
        #[arg(long, default_value = "true")]
        release: bool,

        /// Build in debug mode (overrides --release)
        #[arg(long)]
        debug: bool,

        /// Override the version (defaults to Cargo.toml version or git tag)
        #[arg(long)]
        version: Option<String>,
    },

    /// Build Android .so files for all ABIs (armeabi-v7a, arm64-v8a, x86_64)
    BuildAndroid {
        /// Build in release mode (default: true)
        #[arg(long, default_value = "true")]
        release: bool,

        /// Build in debug mode (overrides --release)
        #[arg(long)]
        debug: bool,

        /// Build only specific ABI(s). Can be specified multiple times.
        #[arg(long, value_enum)]
        abi: Vec<AndroidAbi>,

        /// Override the version (defaults to Cargo.toml version or git tag)
        #[arg(long)]
        version: Option<String>,
    },

    /// Build Flutter native libraries for a specific platform
    BuildFlutter {
        /// Target platform to build for
        #[arg(long, value_enum)]
        platform: FlutterPlatform,

        /// Build in release mode (default: true)
        #[arg(long, default_value = "true")]
        release: bool,

        /// Build in debug mode (overrides --release)
        #[arg(long)]
        debug: bool,

        /// Override the version (defaults to Cargo.toml version or git tag)
        #[arg(long)]
        version: Option<String>,
    },

    /// Install required Rust cross-compilation targets for iOS, macOS, and Android
    SetupTargets,

    /// Build all platforms with one command
    BuildAll {
        /// Build in release mode (default: true)
        #[arg(long, default_value = "true")]
        release: bool,

        /// Build in debug mode (overrides --release)
        #[arg(long)]
        debug: bool,

        /// Run builds concurrently where possible (experimental)
        #[arg(long)]
        parallel: bool,

        /// Override the version (defaults to Cargo.toml version or git tag)
        #[arg(long)]
        version: Option<String>,
    },

    /// Package build artifacts for distribution (creates dist/ with .zip files and checksums)
    Package {
        /// Override the version (defaults to Cargo.toml version or git tag)
        #[arg(long)]
        version: Option<String>,

        /// Output directory for packages (default: dist/)
        #[arg(long, default_value = "dist")]
        output_dir: PathBuf,

        /// Skip packaging XCFramework (Apple artifacts)
        #[arg(long)]
        skip_apple: bool,

        /// Skip packaging Android .so files
        #[arg(long)]
        skip_android: bool,

        /// Skip packaging Flutter plugin
        #[arg(long)]
        skip_flutter: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum BindingsLanguage {
    Swift,
    Kotlin,
    All,
}

#[derive(Clone, Copy, ValueEnum, Debug, PartialEq, Eq)]
enum AndroidAbi {
    /// ARM 32-bit (armeabi-v7a)
    #[value(name = "armeabi-v7a")]
    ArmeabiV7a,
    /// ARM 64-bit (arm64-v8a)
    #[value(name = "arm64-v8a")]
    Arm64V8a,
    /// x86_64 (x86_64)
    #[value(name = "x86_64")]
    X86_64,
}

impl AndroidAbi {
    fn rust_target(&self) -> &'static str {
        match self {
            AndroidAbi::ArmeabiV7a => "armv7-linux-androideabi",
            AndroidAbi::Arm64V8a => "aarch64-linux-android",
            AndroidAbi::X86_64 => "x86_64-linux-android",
        }
    }

    fn ndk_arch(&self) -> &'static str {
        match self {
            AndroidAbi::ArmeabiV7a => "armeabi-v7a",
            AndroidAbi::Arm64V8a => "arm64-v8a",
            AndroidAbi::X86_64 => "x86_64",
        }
    }

    fn all() -> Vec<AndroidAbi> {
        vec![
            AndroidAbi::ArmeabiV7a,
            AndroidAbi::Arm64V8a,
            AndroidAbi::X86_64,
        ]
    }
}

#[derive(Clone, Copy, ValueEnum, Debug, PartialEq, Eq)]
enum FlutterPlatform {
    /// iOS (requires macOS)
    #[value(name = "ios")]
    Ios,
    /// Android (requires Android NDK)
    #[value(name = "android")]
    Android,
    /// macOS (requires macOS)
    #[value(name = "macos")]
    Macos,
    /// Windows
    #[value(name = "windows")]
    Windows,
    /// Linux
    #[value(name = "linux")]
    Linux,
}

impl FlutterPlatform {
    /// Returns the Rust targets to build for this platform
    fn rust_targets(&self) -> Vec<&'static str> {
        match self {
            FlutterPlatform::Ios => vec!["aarch64-apple-ios", "aarch64-apple-ios-sim"],
            FlutterPlatform::Android => vec![
                "aarch64-linux-android",
                "armv7-linux-androideabi",
                "x86_64-linux-android",
            ],
            FlutterPlatform::Macos => vec!["aarch64-apple-darwin", "x86_64-apple-darwin"],
            FlutterPlatform::Windows => vec!["x86_64-pc-windows-msvc"],
            FlutterPlatform::Linux => vec!["x86_64-unknown-linux-gnu"],
        }
    }

    /// Returns the platform name as a string
    fn name(&self) -> &'static str {
        match self {
            FlutterPlatform::Ios => "ios",
            FlutterPlatform::Android => "android",
            FlutterPlatform::Macos => "macos",
            FlutterPlatform::Windows => "windows",
            FlutterPlatform::Linux => "linux",
        }
    }

    /// Check if the platform can be built on the current OS
    fn can_build_on_current_os(&self) -> bool {
        match self {
            FlutterPlatform::Ios | FlutterPlatform::Macos => cfg!(target_os = "macos"),
            FlutterPlatform::Windows => cfg!(target_os = "windows"),
            FlutterPlatform::Linux => cfg!(target_os = "linux"),
            FlutterPlatform::Android => true, // Android can be cross-compiled from any OS
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::SetupTestEnv { registry } => {
            setup_env::run(registry)?;
        }
        Commands::BuildUniffi {
            target,
            release,
            platform_preset,
        } => {
            build_uniffi(target, release, platform_preset)?;
        }
        Commands::BuildFfi {
            target,
            release,
            platform_preset,
            csharp,
            deploy_unity,
        } => {
            build_ffi(target, release, platform_preset, csharp, deploy_unity)?;
        }
        Commands::GenerateBindings { language, out_dir } => {
            generate_bindings(language, out_dir)?;
        }
        Commands::BuildXcframework {
            release,
            debug,
            version,
        } => {
            let is_release = !debug && release;
            let ver = get_version(version.as_deref());
            build_xcframework(is_release, &ver)?;
        }
        Commands::BuildAndroid {
            release,
            debug,
            abi,
            version,
        } => {
            let is_release = !debug && release;
            let ver = get_version(version.as_deref());
            build_android(is_release, abi, &ver)?;
        }
        Commands::BuildFlutter {
            platform,
            release,
            debug,
            version,
        } => {
            let is_release = !debug && release;
            let ver = get_version(version.as_deref());
            build_flutter(platform, is_release, &ver)?;
        }
        Commands::SetupTargets => {
            setup_targets()?;
        }
        Commands::BuildAll {
            release,
            debug,
            parallel,
            version,
        } => {
            let is_release = !debug && release;
            let ver = get_version(version.as_deref());
            build_all(is_release, parallel, &ver)?;
        }
        Commands::Package {
            version,
            output_dir,
            skip_apple,
            skip_android,
            skip_flutter,
        } => {
            let ver = get_version(version.as_deref());
            package_artifacts(&ver, &output_dir, skip_apple, skip_android, skip_flutter)?;
        }
    }

    Ok(())
}

/// Build the xybrid-uniffi library
fn build_uniffi(
    target: Option<String>,
    release: bool,
    platform_preset: Option<String>,
) -> Result<()> {
    // Resolve the platform preset: use override if provided, otherwise auto-detect
    let preset = if let Some(ref p) = platform_preset {
        p.clone()
    } else if let Some(ref t) = target {
        platform_preset_for_target(t).to_string()
    } else {
        host_platform_preset().to_string()
    };

    println!("Building xybrid-uniffi with features: {}", preset);

    let mut cmd = Command::new("cargo");
    cmd.arg("build").arg("-p").arg("xybrid-uniffi");

    // Pass the platform preset as a feature
    cmd.arg("--features").arg(&preset);

    if release {
        cmd.arg("--release");
    }

    if let Some(ref t) = target {
        cmd.arg("--target").arg(t);

        // For iOS targets, resolve and set ORT_LIB_LOCATION + fp16 rustflags
        if is_ios_target(t) {
            if let Some(ort_path) = resolve_ort_lib_location(t) {
                cmd.env("ORT_LIB_LOCATION", &ort_path);
            } else {
                anyhow::bail!(
                    "ORT iOS library not found. To build for iOS, either:\n\
                     1. Place the ORT iOS xcframework at vendor/ort-ios/onnxruntime.xcframework/\n\
                     2. Set ORT_LIB_LOCATION env var to a directory containing libonnxruntime.a\n\n\
                     Download from: https://huggingface.co/csukuangfj/ios-onnxruntime"
                );
            }
            set_ios_rustflags(&mut cmd, t);
        }
    }

    let status = cmd.status().context("Failed to run cargo build")?;

    if !status.success() {
        anyhow::bail!("cargo build failed");
    }

    // Print output location
    let profile = if release { "release" } else { "debug" };
    let lib_name = if cfg!(target_os = "macos") {
        "libxybrid_uniffi.dylib"
    } else if cfg!(target_os = "windows") {
        "xybrid_uniffi.dll"
    } else {
        "libxybrid_uniffi.so"
    };

    let output_path = if let Some(ref t) = target {
        format!("target/{}/{}/{}", t, profile, lib_name)
    } else {
        format!("target/{}/{}", profile, lib_name)
    };

    println!("\n✓ Build successful!");
    println!("  Output: {}", output_path);

    Ok(())
}

/// Build the xybrid-ffi library (C ABI)
fn build_ffi(
    target: Option<String>,
    release: bool,
    platform_preset: Option<String>,
    csharp: bool,
    deploy_unity: bool,
) -> Result<()> {
    // Resolve the platform preset: use override if provided, otherwise auto-detect
    let preset = if let Some(ref p) = platform_preset {
        p.clone()
    } else if let Some(ref t) = target {
        platform_preset_for_target(t).to_string()
    } else {
        host_platform_preset().to_string()
    };

    // Build features list
    let mut features = vec![preset.clone()];
    if csharp {
        features.push("csharp".to_string());
    }
    let features_str = features.join(",");

    println!(
        "Building xybrid-ffi{}...",
        if csharp { " with C# bindings" } else { "" }
    );
    println!("  Features: {}", features_str);

    let mut cmd = Command::new("cargo");
    cmd.arg("build").arg("-p").arg("xybrid-ffi");

    // Pass all features
    cmd.arg("--features").arg(&features_str);

    if release {
        cmd.arg("--release");
    }

    if let Some(ref t) = target {
        cmd.arg("--target").arg(t);

        // For iOS targets, resolve and set ORT_LIB_LOCATION + fp16 rustflags
        if is_ios_target(t) {
            if let Some(ort_path) = resolve_ort_lib_location(t) {
                cmd.env("ORT_LIB_LOCATION", &ort_path);
            } else {
                anyhow::bail!(
                    "ORT iOS library not found. To build for iOS, either:\n\
                     1. Place the ORT iOS xcframework at vendor/ort-ios/onnxruntime.xcframework/\n\
                     2. Set ORT_LIB_LOCATION env var to a directory containing libonnxruntime.a\n\n\
                     Download from: https://huggingface.co/csukuangfj/ios-onnxruntime"
                );
            }
            set_ios_rustflags(&mut cmd, t);
        }
    }

    let status = cmd.status().context("Failed to run cargo build")?;

    if !status.success() {
        anyhow::bail!("cargo build failed");
    }

    // Print output location
    let profile = if release { "release" } else { "debug" };
    let dylib_name = if cfg!(target_os = "macos") {
        "libxybrid_ffi.dylib"
    } else if cfg!(target_os = "windows") {
        "xybrid_ffi.dll"
    } else {
        "libxybrid_ffi.so"
    };
    let staticlib_name = if cfg!(target_os = "windows") {
        "xybrid_ffi.lib"
    } else {
        "libxybrid_ffi.a"
    };

    let (dylib_path, staticlib_path) = if let Some(ref t) = target {
        (
            format!("target/{}/{}/{}", t, profile, dylib_name),
            format!("target/{}/{}/{}", t, profile, staticlib_name),
        )
    } else {
        (
            format!("target/{}/{}", profile, dylib_name),
            format!("target/{}/{}", profile, staticlib_name),
        )
    };

    println!("\n✓ Build successful!");
    println!("  Dynamic library: {}", dylib_path);
    println!("  Static library:  {}", staticlib_path);
    println!("  C header:        crates/xybrid-ffi/include/xybrid.h");

    if csharp {
        println!("  C# bindings:     bindings/unity/Runtime/Native/NativeMethods.g.cs");
    }

    // Deploy to Unity if requested
    if deploy_unity {
        deploy_ffi_to_unity(&dylib_path)?;
    }

    Ok(())
}

/// Deploy the built xybrid-ffi library to the Unity bindings directory
fn deploy_ffi_to_unity(dylib_path: &str) -> Result<()> {
    println!("\nDeploying to Unity...");

    // Determine the platform subdirectory
    let platform_dir = if cfg!(target_os = "macos") {
        "macOS"
    } else if cfg!(target_os = "windows") {
        "Windows"
    } else {
        "Linux"
    };

    // Unity native plugins directory
    let unity_plugins_dir = PathBuf::from("bindings/unity/Runtime/Plugins").join(platform_dir);
    std::fs::create_dir_all(&unity_plugins_dir)
        .with_context(|| format!("Failed to create Unity plugins directory: {:?}", unity_plugins_dir))?;

    // Copy the dynamic library
    let src = PathBuf::from(dylib_path);
    if !src.exists() {
        anyhow::bail!("Built library not found at: {}", dylib_path);
    }

    let lib_name = src.file_name().unwrap();
    let dst = unity_plugins_dir.join(lib_name);

    std::fs::copy(&src, &dst)
        .with_context(|| format!("Failed to copy library to {:?}", dst))?;

    println!("  ✓ Deployed to: {}", dst.display());

    // Also copy the C header
    let header_src = PathBuf::from("crates/xybrid-ffi/include/xybrid.h");
    if header_src.exists() {
        let header_dst = PathBuf::from("bindings/unity/Runtime/Native/xybrid.h");
        if let Some(parent) = header_dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(&header_src, &header_dst)
            .with_context(|| format!("Failed to copy header to {:?}", header_dst))?;
        println!("  ✓ Header copied to: {}", header_dst.display());
    }

    Ok(())
}

/// Generate Swift/Kotlin bindings using uniffi-bindgen
fn generate_bindings(language: BindingsLanguage, out_dir: Option<PathBuf>) -> Result<()> {
    // First, ensure the library is built with host platform preset
    println!("Building xybrid-uniffi (release)...");
    build_uniffi(None, true, None)?;

    let lib_path = if cfg!(target_os = "macos") {
        "target/release/libxybrid_uniffi.dylib"
    } else if cfg!(target_os = "windows") {
        "target/release/xybrid_uniffi.dll"
    } else {
        "target/release/libxybrid_uniffi.so"
    };

    // Check if the library exists
    if !std::path::Path::new(lib_path).exists() {
        anyhow::bail!("Library not found at {}. Build may have failed.", lib_path);
    }

    let languages = match language {
        BindingsLanguage::Swift => vec!["swift"],
        BindingsLanguage::Kotlin => vec!["kotlin"],
        BindingsLanguage::All => vec!["swift", "kotlin"],
    };

    for lang in languages {
        // Default output directories for platform-specific bindings
        let default_out = match lang {
            "swift" => PathBuf::from("bindings/apple/Sources/Xybrid"),
            "kotlin" => PathBuf::from("bindings/kotlin/src/main/kotlin/ai/xybrid"),
            _ => PathBuf::from(format!("bindings/{}", lang)),
        };
        let output = out_dir.clone().unwrap_or(default_out);

        // Create output directory
        std::fs::create_dir_all(&output)
            .with_context(|| format!("Failed to create output directory: {:?}", output))?;

        println!("\nGenerating {} bindings to {:?}...", lang, output);

        let status = Command::new("cargo")
            .arg("run")
            .arg("-p")
            .arg("xybrid-uniffi")
            .arg("--bin")
            .arg("uniffi-bindgen")
            .arg("--")
            .arg("generate")
            .arg("--library")
            .arg(lib_path)
            .arg("--language")
            .arg(lang)
            .arg("--out-dir")
            .arg(&output)
            .status()
            .context("Failed to run uniffi-bindgen")?;

        if !status.success() {
            anyhow::bail!("uniffi-bindgen failed for {}", lang);
        }

        // For Swift, move FFI files to XybridFFI target directory
        if lang == "swift" && out_dir.is_none() {
            let ffi_include_dir = PathBuf::from("bindings/apple/Sources/xybrid_uniffiFFI/include");
            let ffi_dir = PathBuf::from("bindings/apple/Sources/xybrid_uniffiFFI");

            std::fs::create_dir_all(&ffi_include_dir)
                .context("Failed to create XybridFFI include directory")?;

            // Move header to include directory
            let header_src = output.join("xybrid_uniffiFFI.h");
            let header_dst = ffi_include_dir.join("xybrid_uniffiFFI.h");
            if header_src.exists() {
                std::fs::rename(&header_src, &header_dst).with_context(|| {
                    format!("Failed to move {:?} to {:?}", header_src, header_dst)
                })?;
                println!("  Moved header to {:?}", header_dst);
            }

            // Move modulemap to FFI directory
            let modulemap_src = output.join("xybrid_uniffiFFI.modulemap");
            let modulemap_dst = ffi_dir.join("xybrid_uniffiFFI.modulemap");
            if modulemap_src.exists() {
                std::fs::rename(&modulemap_src, &modulemap_dst).with_context(|| {
                    format!("Failed to move {:?} to {:?}", modulemap_src, modulemap_dst)
                })?;
                println!("  Moved modulemap to {:?}", modulemap_dst);
            }
        }

        println!("✓ {} bindings generated to {:?}", lang, output);
    }

    Ok(())
}

/// Apple target architectures for XCFramework
const IOS_ARM64: &str = "aarch64-apple-ios";
const IOS_SIM_ARM64: &str = "aarch64-apple-ios-sim";
const IOS_SIM_X86_64: &str = "x86_64-apple-ios";
const MACOS_ARM64: &str = "aarch64-apple-darwin";
const MACOS_X86_64: &str = "x86_64-apple-darwin";

/// Build Apple XCFramework for iOS and macOS platforms
fn build_xcframework(release: bool, version: &str) -> Result<()> {
    if !cfg!(target_os = "macos") {
        anyhow::bail!("XCFramework builds are only supported on macOS");
    }

    let profile = if release { "release" } else { "debug" };

    println!("Building XCFramework ({} mode, version {})...", profile, version);
    println!();

    // Check for ONNX Runtime iOS library using the unified resolution
    // Use aarch64-apple-ios as the probe target to check for ORT availability
    let ort_lib_location = resolve_ort_lib_location(IOS_ARM64);
    if ort_lib_location.is_none() {
        println!("⚠️  Warning: ORT iOS library not found");
        println!();
        println!("   iOS builds require ONNX Runtime iOS static library. Options:");
        println!();
        println!("   1. Use the vendored library (recommended):");
        println!("      - Ensure vendor/ort-ios/onnxruntime.xcframework/ios-arm64/libonnxruntime.a exists");
        println!();
        println!("   2. Download prebuilt XCFramework:");
        println!("      - From HuggingFace: https://huggingface.co/csukuangfj/ios-onnxruntime");
        println!("      - Extract to vendor/ort-ios/onnxruntime.xcframework/");
        println!();
        println!("   3. Set ORT_LIB_LOCATION environment variable:");
        println!("      export ORT_LIB_LOCATION=/path/to/directory/containing/libonnxruntime.a");
        println!();
        println!("   Building macOS-only (skipping iOS targets)...");
        println!();

        return build_xcframework_macos_only(release, version);
    }

    println!();

    // Define targets
    let targets = [
        (IOS_ARM64, "iOS arm64"),
        (IOS_SIM_ARM64, "iOS Simulator arm64"),
        (IOS_SIM_X86_64, "iOS Simulator x86_64"),
        (MACOS_ARM64, "macOS arm64"),
        (MACOS_X86_64, "macOS x86_64"),
    ];

    let mut built_targets: Vec<(&str, &str)> = Vec::new();

    // Build for each target
    for (target, description) in targets.iter() {
        // For iOS targets, resolve ORT per-target (device vs simulator have different libs)
        if is_ios_target(target) {
            if resolve_ort_lib_location(target).is_none() {
                println!("Skipping {} (no ORT library for this target)", description);
                continue;
            }
        }

        // Resolve platform preset for this target
        let preset = platform_preset_for_target(target);
        println!("Building for {} with features: {}...", description, preset);

        let mut cmd = Command::new("cargo");
        cmd.arg("build")
            .arg("-p")
            .arg("xybrid-uniffi")
            .arg("--target")
            .arg(target)
            .arg("--features")
            .arg(preset);

        if release {
            cmd.arg("--release");
        }

        // Set ORT_LIB_LOCATION and fp16 rustflags for iOS targets
        if is_ios_target(target) {
            if let Some(ort_path) = resolve_ort_lib_location(target) {
                cmd.env("ORT_LIB_LOCATION", &ort_path);
            }
            set_ios_rustflags(&mut cmd, target);
        }

        let status = cmd
            .status()
            .with_context(|| format!("Failed to run cargo build for {}", target))?;

        if !status.success() {
            anyhow::bail!("cargo build failed for {}", target);
        }

        built_targets.push((target, description));
        println!("  ✓ {} ({})", description, preset);
    }

    let has_ios_device = built_targets.iter().any(|(t, _)| *t == IOS_ARM64);
    let has_ios_sim_arm64 = built_targets.iter().any(|(t, _)| *t == IOS_SIM_ARM64);
    let has_ios_sim_x86 = built_targets.iter().any(|(t, _)| *t == IOS_SIM_X86_64);

    println!();
    println!("Creating XCFramework...");

    // Create output directory
    let xcframework_dir = PathBuf::from("bindings/apple/XCFrameworks");
    // Use versioned filename (e.g., XybridFFI-0.1.0.xcframework)
    let xcframework_name = format!("XybridFFI-{}.xcframework", version);
    let xcframework_path = xcframework_dir.join(&xcframework_name);
    // Also create a symlink/copy without version for backwards compatibility
    let xcframework_latest = xcframework_dir.join("XybridFFI.xcframework");

    // Remove existing XCFramework if present
    if xcframework_path.exists() {
        std::fs::remove_dir_all(&xcframework_path)
            .context("Failed to remove existing XCFramework")?;
    }
    if xcframework_latest.exists() {
        std::fs::remove_dir_all(&xcframework_latest)
            .context("Failed to remove existing XCFramework symlink")?;
    }

    // Create lipo'd iOS Simulator library if both sim targets were built
    let ios_sim_lib_path = if has_ios_sim_arm64 && has_ios_sim_x86 {
        let ios_sim_dir = PathBuf::from(format!("target/ios-simulator-universal/{}", profile));
        std::fs::create_dir_all(&ios_sim_dir)
            .context("Failed to create iOS Simulator universal directory")?;

        let ios_sim_lib = ios_sim_dir.join("libxybrid_uniffi.a");
        let sim_arm64_lib = format!("target/{}/{}/libxybrid_uniffi.a", IOS_SIM_ARM64, profile);
        let sim_x86_lib = format!("target/{}/{}/libxybrid_uniffi.a", IOS_SIM_X86_64, profile);

        println!("  Creating universal iOS Simulator library...");
        let lipo_status = Command::new("lipo")
            .args(["-create", "-output"])
            .arg(&ios_sim_lib)
            .arg(&sim_arm64_lib)
            .arg(&sim_x86_lib)
            .status()
            .context("Failed to run lipo for iOS Simulator")?;

        if !lipo_status.success() {
            anyhow::bail!("lipo failed for iOS Simulator libraries");
        }
        Some(ios_sim_lib)
    } else {
        None
    };

    // Create lipo'd macOS library (combine arm64 and x86_64)
    let macos_dir = PathBuf::from(format!("target/macos-universal/{}", profile));
    std::fs::create_dir_all(&macos_dir).context("Failed to create macOS universal directory")?;

    let macos_lib = macos_dir.join("libxybrid_uniffi.a");
    let macos_arm64_lib = format!("target/{}/{}/libxybrid_uniffi.a", MACOS_ARM64, profile);
    let macos_x86_lib = format!("target/{}/{}/libxybrid_uniffi.a", MACOS_X86_64, profile);

    println!("  Creating universal macOS library...");
    let lipo_status = Command::new("lipo")
        .args(["-create", "-output"])
        .arg(&macos_lib)
        .arg(&macos_arm64_lib)
        .arg(&macos_x86_lib)
        .status()
        .context("Failed to run lipo for macOS")?;

    if !lipo_status.success() {
        anyhow::bail!("lipo failed for macOS libraries");
    }

    // Create XCFramework using xcodebuild with available slices
    println!("  Packaging XCFramework...");
    let mut xcbuild = Command::new("xcodebuild");
    xcbuild.arg("-create-xcframework");

    let mut arch_descriptions: Vec<&str> = Vec::new();

    if has_ios_device {
        let ios_arm64_lib = format!("target/{}/{}/libxybrid_uniffi.a", IOS_ARM64, profile);
        xcbuild.arg("-library").arg(&ios_arm64_lib);
        arch_descriptions.push("iOS arm64");
    }

    if let Some(ref sim_lib) = ios_sim_lib_path {
        xcbuild.arg("-library").arg(sim_lib);
        arch_descriptions.push("iOS Simulator arm64 + x86_64 (universal)");
    }

    xcbuild.arg("-library").arg(&macos_lib);
    arch_descriptions.push("macOS arm64 + x86_64 (universal)");

    xcbuild.arg("-output").arg(&xcframework_path);

    let xcbuild_status = xcbuild
        .status()
        .context("Failed to run xcodebuild -create-xcframework")?;

    if !xcbuild_status.success() {
        anyhow::bail!("xcodebuild -create-xcframework failed");
    }

    // Create a copy at XybridFFI.xcframework for backwards compatibility
    println!("  Creating unversioned copy for compatibility...");
    copy_dir_recursive(&xcframework_path, &xcframework_latest)
        .context("Failed to create unversioned XCFramework copy")?;

    println!();
    println!("✓ XCFramework build successful!");
    println!("  Version: {}", version);
    println!("  Output: {}", xcframework_path.display());
    println!("  Also:   {}", xcframework_latest.display());
    println!();
    println!("Architectures included:");
    for desc in &arch_descriptions {
        println!("  - {}", desc);
    }

    Ok(())
}

/// Build macOS-only XCFramework (fallback when ORT iOS library is not available)
fn build_xcframework_macos_only(release: bool, version: &str) -> Result<()> {
    let profile = if release { "release" } else { "debug" };

    // Build for macOS targets only
    let targets = [(MACOS_ARM64, "macOS arm64"), (MACOS_X86_64, "macOS x86_64")];

    for (target, description) in targets.iter() {
        // Resolve platform preset for this target (will be platform-macos)
        let preset = platform_preset_for_target(target);
        println!("Building for {} with features: {}...", description, preset);

        let mut cmd = Command::new("cargo");
        cmd.arg("build")
            .arg("-p")
            .arg("xybrid-uniffi")
            .arg("--target")
            .arg(target)
            .arg("--features")
            .arg(preset);

        if release {
            cmd.arg("--release");
        }

        let status = cmd
            .status()
            .with_context(|| format!("Failed to run cargo build for {}", target))?;

        if !status.success() {
            anyhow::bail!("cargo build failed for {}", target);
        }

        println!("  ✓ {} ({})", description, preset);
    }

    println!();
    println!("Creating XCFramework (macOS only)...");

    // Create output directory
    let xcframework_dir = PathBuf::from("bindings/apple/XCFrameworks");
    let xcframework_name = format!("XybridFFI-{}.xcframework", version);
    let xcframework_path = xcframework_dir.join(&xcframework_name);
    let xcframework_latest = xcframework_dir.join("XybridFFI.xcframework");

    // Remove existing XCFramework if present
    if xcframework_path.exists() {
        std::fs::remove_dir_all(&xcframework_path)
            .context("Failed to remove existing XCFramework")?;
    }
    if xcframework_latest.exists() {
        std::fs::remove_dir_all(&xcframework_latest)
            .context("Failed to remove existing XCFramework symlink")?;
    }

    // Create lipo'd macOS library (combine arm64 and x86_64)
    let macos_dir = PathBuf::from(format!("target/macos-universal/{}", profile));
    std::fs::create_dir_all(&macos_dir).context("Failed to create macOS universal directory")?;

    let macos_lib = macos_dir.join("libxybrid_uniffi.a");
    let macos_arm64_lib = format!("target/{}/{}/libxybrid_uniffi.a", MACOS_ARM64, profile);
    let macos_x86_lib = format!("target/{}/{}/libxybrid_uniffi.a", MACOS_X86_64, profile);

    println!("  Creating universal macOS library...");
    let lipo_status = Command::new("lipo")
        .args(["-create", "-output"])
        .arg(&macos_lib)
        .arg(&macos_arm64_lib)
        .arg(&macos_x86_lib)
        .status()
        .context("Failed to run lipo for macOS")?;

    if !lipo_status.success() {
        anyhow::bail!("lipo failed for macOS libraries");
    }

    // Create XCFramework with macOS only
    println!("  Packaging XCFramework...");
    let xcbuild_status = Command::new("xcodebuild")
        .arg("-create-xcframework")
        .arg("-library")
        .arg(&macos_lib)
        .arg("-output")
        .arg(&xcframework_path)
        .status()
        .context("Failed to run xcodebuild -create-xcframework")?;

    if !xcbuild_status.success() {
        anyhow::bail!("xcodebuild -create-xcframework failed");
    }

    // Create a copy at XybridFFI.xcframework for backwards compatibility
    println!("  Creating unversioned copy for compatibility...");
    copy_dir_recursive(&xcframework_path, &xcframework_latest)
        .context("Failed to create unversioned XCFramework copy")?;

    println!();
    println!("✓ XCFramework build successful (macOS only)!");
    println!("  Version: {}", version);
    println!("  Output: {}", xcframework_path.display());
    println!("  Also:   {}", xcframework_latest.display());
    println!();
    println!("Architectures included:");
    println!("  - macOS arm64 + x86_64 (universal)");
    println!();
    println!("⚠️  Note: iOS targets were skipped because ORT iOS library was not found.");
    println!("   To include iOS, ensure vendor/ort-ios/onnxruntime.xcframework/ios-arm64/libonnxruntime.a exists,");
    println!("   or set ORT_LIB_LOCATION to a directory containing libonnxruntime.a.");

    Ok(())
}

/// Recursively copy a directory
fn copy_dir_recursive(src: &PathBuf, dst: &PathBuf) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Build Android .so files for specified ABIs
fn build_android(release: bool, abis: Vec<AndroidAbi>, version: &str) -> Result<()> {
    let profile = if release { "release" } else { "debug" };

    // Determine which ABIs to build
    let target_abis = if abis.is_empty() {
        AndroidAbi::all()
    } else {
        abis
    };

    println!("Building Android .so files ({} mode, version {})...", profile, version);
    println!();

    // Check for ANDROID_NDK_HOME or try to detect cargo-ndk
    let ndk_home = std::env::var("ANDROID_NDK_HOME").ok();
    let has_cargo_ndk = Command::new("cargo")
        .args(["ndk", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if ndk_home.is_none() && !has_cargo_ndk {
        eprintln!("Warning: ANDROID_NDK_HOME not set and cargo-ndk not found.");
        eprintln!("  Install cargo-ndk: cargo install cargo-ndk");
        eprintln!("  Or set ANDROID_NDK_HOME to your Android NDK path.");
        eprintln!();
    }

    // Build for each ABI
    let mut built_abis = Vec::new();
    for abi in &target_abis {
        let target = abi.rust_target();
        let ndk_arch = abi.ndk_arch();

        println!("Building for {} ({})...", ndk_arch, target);

        let build_result = if has_cargo_ndk {
            // Use cargo-ndk for building
            build_android_with_cargo_ndk(target, release)
        } else if ndk_home.is_some() {
            // Use manual cross-compilation with NDK
            build_android_with_ndk(target, release)
        } else {
            // Try plain cargo build (may fail without proper linker setup)
            build_android_plain(target, release)
        };

        if let Err(e) = build_result {
            eprintln!("  ✗ Failed to build for {}: {}", ndk_arch, e);
            continue;
        }

        // Copy the .so file to bindings/kotlin/libs/{version}/{abi}/
        let profile_dir = if release { "release" } else { "debug" };
        let src_path = PathBuf::from(format!(
            "target/{}/{}/libxybrid_uniffi.so",
            target, profile_dir
        ));

        // Versioned output path: bindings/kotlin/libs/{version}/{abi}/
        let versioned_dir = PathBuf::from(format!("bindings/kotlin/libs/{}/{}", version, ndk_arch));
        std::fs::create_dir_all(&versioned_dir)
            .with_context(|| format!("Failed to create directory: {:?}", versioned_dir))?;

        let versioned_path = versioned_dir.join("libxybrid_uniffi.so");

        // Also maintain backwards-compatible path: bindings/kotlin/libs/{abi}/
        let compat_dir = PathBuf::from(format!("bindings/kotlin/libs/{}", ndk_arch));
        std::fs::create_dir_all(&compat_dir)
            .with_context(|| format!("Failed to create directory: {:?}", compat_dir))?;

        let compat_path = compat_dir.join("libxybrid_uniffi.so");

        if src_path.exists() {
            std::fs::copy(&src_path, &versioned_path)
                .with_context(|| format!("Failed to copy {:?} to {:?}", src_path, versioned_path))?;
            std::fs::copy(&src_path, &compat_path)
                .with_context(|| format!("Failed to copy {:?} to {:?}", src_path, compat_path))?;
            println!("  ✓ {} -> {:?}", ndk_arch, versioned_path);
            built_abis.push(ndk_arch.to_string());
        } else {
            eprintln!("  ✗ Library not found at {:?}", src_path);
        }
    }

    println!();

    if built_abis.is_empty() {
        anyhow::bail!("No ABIs were built successfully");
    }

    println!("✓ Android build successful!");
    println!("  Version: {}", version);
    println!("  Versioned output: bindings/kotlin/libs/{}/", version);
    println!("  Compat output:    bindings/kotlin/libs/");
    println!();
    println!("ABIs built:");
    for abi in &built_abis {
        println!("  - {}", abi);
    }

    Ok(())
}

/// Build using cargo-ndk
fn build_android_with_cargo_ndk(target: &str, release: bool) -> Result<()> {
    println!("  Building with features: platform-android");

    let mut cmd = Command::new("cargo");
    cmd.arg("ndk")
        .arg("--target")
        .arg(target)
        // Use API level 28 for Android builds
        // Required because:
        // - POSIX_MADV_* constants (used by llama.cpp) require API 23+
        // - aws-lc-sys (used for TLS) requires API 28+ for getentropy()
        // This doesn't affect app minSdkVersion - only the NDK headers used during compilation
        .arg("--platform")
        .arg("28")
        .arg("build")
        .arg("-p")
        .arg("xybrid-uniffi")
        .arg("--features")
        .arg("platform-android");

    if release {
        cmd.arg("--release");
    }

    let status = cmd.status().context("Failed to run cargo ndk build")?;

    if !status.success() {
        anyhow::bail!("cargo ndk build failed");
    }

    Ok(())
}

/// Build using ANDROID_NDK_HOME for cross-compilation
fn build_android_with_ndk(target: &str, release: bool) -> Result<()> {
    println!("  Building with features: platform-android");

    let ndk_home = std::env::var("ANDROID_NDK_HOME").context("ANDROID_NDK_HOME not set")?;

    // Determine the linker name based on target
    // Use API level 28 for all targets because:
    // - POSIX_MADV_* constants (used by llama.cpp) require API 23+
    // - aws-lc-sys (used for TLS) requires API 28+ for getentropy()
    let (clang_target, api_level) = match target {
        "armv7-linux-androideabi" => ("armv7a-linux-androideabi", "28"),
        "aarch64-linux-android" => ("aarch64-linux-android", "28"),
        "x86_64-linux-android" => ("x86_64-linux-android", "28"),
        _ => anyhow::bail!("Unsupported Android target: {}", target),
    };

    // Find the NDK toolchain bin directory
    let toolchain_bin = PathBuf::from(&ndk_home)
        .join("toolchains/llvm/prebuilt")
        .join(if cfg!(target_os = "macos") {
            "darwin-x86_64"
        } else if cfg!(target_os = "linux") {
            "linux-x86_64"
        } else {
            "windows-x86_64"
        })
        .join("bin");

    let clang_path = toolchain_bin.join(format!("{}{}-clang", clang_target, api_level));

    // Set up environment for cross-compilation
    let linker_env = format!(
        "CARGO_TARGET_{}_LINKER",
        target.to_uppercase().replace('-', "_")
    );

    let mut cmd = Command::new("cargo");
    cmd.arg("build")
        .arg("-p")
        .arg("xybrid-uniffi")
        .arg("--target")
        .arg(target)
        .arg("--features")
        .arg("platform-android")
        .env(&linker_env, &clang_path);

    if release {
        cmd.arg("--release");
    }

    let status = cmd.status().context("Failed to run cargo build with NDK")?;

    if !status.success() {
        anyhow::bail!("cargo build with NDK failed");
    }

    Ok(())
}

/// Attempt plain cargo build (requires manual toolchain setup)
fn build_android_plain(target: &str, release: bool) -> Result<()> {
    println!("  Building with features: platform-android");

    let mut cmd = Command::new("cargo");
    cmd.arg("build")
        .arg("-p")
        .arg("xybrid-uniffi")
        .arg("--target")
        .arg(target)
        .arg("--features")
        .arg("platform-android");

    if release {
        cmd.arg("--release");
    }

    let status = cmd.status().context("Failed to run cargo build")?;

    if !status.success() {
        anyhow::bail!("cargo build failed for target {}", target);
    }

    Ok(())
}

/// Run flutter_rust_bridge_codegen to generate Dart bindings
fn run_frb_codegen() -> Result<()> {
    let flutter_dir = PathBuf::from("bindings/flutter");
    let config_file = flutter_dir.join("flutter_rust_bridge.yaml");

    // Check if config file exists
    if !config_file.exists() {
        anyhow::bail!(
            "FRB config not found at {:?}. Please create flutter_rust_bridge.yaml",
            config_file
        );
    }

    println!("Running flutter_rust_bridge_codegen...");

    // Check if flutter_rust_bridge_codegen is available
    let frb_available = Command::new("flutter_rust_bridge_codegen")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !frb_available {
        eprintln!("Warning: flutter_rust_bridge_codegen not found.");
        eprintln!("  Install with: cargo install flutter_rust_bridge_codegen");
        eprintln!("  Or: dart pub global activate flutter_rust_bridge");
        eprintln!();
        eprintln!("Skipping codegen - Dart bindings may be out of date.");
        return Ok(());
    }

    // Run codegen from the flutter directory
    let status = Command::new("flutter_rust_bridge_codegen")
        .arg("generate")
        .current_dir(&flutter_dir)
        .status()
        .context("Failed to run flutter_rust_bridge_codegen")?;

    if !status.success() {
        anyhow::bail!("flutter_rust_bridge_codegen failed");
    }

    println!("  ✓ FRB codegen complete");
    println!();

    Ok(())
}

/// Build Flutter native libraries for a specific platform
fn build_flutter(platform: FlutterPlatform, release: bool, version: &str) -> Result<()> {
    let profile = if release { "release" } else { "debug" };

    println!(
        "Building Flutter native libraries for {} ({} mode, version {})...",
        platform.name(),
        profile,
        version
    );
    println!();

    // Check if the platform can be built on the current OS
    if !platform.can_build_on_current_os() {
        anyhow::bail!(
            "Platform '{}' cannot be built on the current operating system.\n\
             iOS and macOS builds require macOS.\n\
             Windows builds require Windows.\n\
             Linux builds require Linux.",
            platform.name()
        );
    }

    // Run FRB codegen first
    run_frb_codegen()?;

    // Get the targets to build
    let targets = platform.rust_targets();
    let flutter_rust_dir = PathBuf::from("bindings/flutter/rust");

    // Build for each target
    let mut built_targets = Vec::new();
    for target in &targets {
        // Resolve the platform preset for this target
        let preset = platform_preset_for_target(target);
        println!("Building for {} with features: {}...", target, preset);

        let build_result = match platform {
            FlutterPlatform::Android => {
                // Android requires cargo-ndk or manual NDK setup
                build_flutter_android(target, release)
            }
            _ => {
                // Other platforms use regular cargo build
                build_flutter_native(target, release, preset)
            }
        };

        match build_result {
            Ok(()) => {
                println!("  ✓ {} ({})", target, preset);
                built_targets.push(*target);
            }
            Err(e) => {
                eprintln!("  ✗ Failed to build for {}: {}", target, e);
            }
        }
    }

    println!();

    if built_targets.is_empty() {
        anyhow::bail!(
            "No targets were built successfully for platform '{}'",
            platform.name()
        );
    }

    // Print output location
    let output_dir = flutter_rust_dir.join("target");

    println!("✓ Flutter {} build successful!", platform.name());
    println!("  Version: {}", version);
    println!("  Output: {}", output_dir.display());
    println!();
    println!("Targets built:");
    for target in &built_targets {
        let profile_dir = if release { "release" } else { "debug" };
        let lib_name = get_flutter_lib_name(target);
        println!(
            "  - {} -> target/{}/{}/{}",
            target, target, profile_dir, lib_name
        );
    }

    Ok(())
}

/// Build Flutter FFI for a native platform (not Android)
fn build_flutter_native(target: &str, release: bool, features: &str) -> Result<()> {
    let mut cmd = Command::new("cargo");
    cmd.arg("build")
        .arg("-p")
        .arg("xybrid_flutter")
        .arg("--target")
        .arg(target)
        .arg("--features")
        .arg(features);

    if release {
        cmd.arg("--release");
    }

    // Set ORT_LIB_LOCATION and fp16 rustflags for iOS targets
    if is_ios_target(target) {
        if let Some(ort_lib_path) = resolve_ort_lib_location(target) {
            println!(
                "  Using ORT iOS library: {}",
                ort_lib_path.display()
            );
            cmd.env("ORT_LIB_LOCATION", &ort_lib_path);
        } else {
            // Warn but don't bail - Flutter iOS via xtask is less common than via flutter build ios
            println!(
                "  Warning: ORT iOS library not found. Build may fail during linking.\n  \
                 Set ORT_LIB_LOCATION env var or place library in vendor/ort-ios/"
            );
        }
        set_ios_rustflags(&mut cmd, target);
    }

    let status = cmd.status().context("Failed to run cargo build")?;

    if !status.success() {
        anyhow::bail!("cargo build failed for target {}", target);
    }

    Ok(())
}

/// Build Flutter FFI for Android using cargo-ndk or manual NDK setup
fn build_flutter_android(target: &str, release: bool) -> Result<()> {
    println!("  Building with features: platform-android");

    // Check for cargo-ndk first
    let has_cargo_ndk = Command::new("cargo")
        .args(["ndk", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if has_cargo_ndk {
        let mut cmd = Command::new("cargo");
        cmd.arg("ndk")
            .arg("--target")
            .arg(target)
            // Use API level 28 for Android builds
            // Required because:
            // - POSIX_MADV_* constants (used by llama.cpp) require API 23+
            // - aws-lc-sys (used for TLS) requires API 28+ for getentropy()
            // This doesn't affect app minSdkVersion - only the NDK headers used during compilation
            .arg("--platform")
            .arg("28")
            .arg("build")
            .arg("-p")
            .arg("xybrid_flutter")
            .arg("--features")
            .arg("platform-android");

        if release {
            cmd.arg("--release");
        }

        let status = cmd.status().context("Failed to run cargo ndk build")?;

        if !status.success() {
            anyhow::bail!("cargo ndk build failed for target {}", target);
        }
    } else {
        // Try using ANDROID_NDK_HOME
        let ndk_home = std::env::var("ANDROID_NDK_HOME").context(
            "ANDROID_NDK_HOME not set and cargo-ndk not found. \
                      Install cargo-ndk: cargo install cargo-ndk",
        )?;

        // Determine the linker name based on target
        // Use API level 28 for Android builds because:
        // - POSIX_MADV_* constants (used by llama.cpp) require API 23+
        // - aws-lc-sys (used for TLS) requires API 28+ for getentropy()
        let (clang_target, api_level) = match target {
            "armv7-linux-androideabi" => ("armv7a-linux-androideabi", "28"),
            "aarch64-linux-android" => ("aarch64-linux-android", "28"),
            "x86_64-linux-android" => ("x86_64-linux-android", "28"),
            _ => anyhow::bail!("Unsupported Android target: {}", target),
        };

        // Find the NDK toolchain bin directory
        let toolchain_bin = PathBuf::from(&ndk_home)
            .join("toolchains/llvm/prebuilt")
            .join(if cfg!(target_os = "macos") {
                "darwin-x86_64"
            } else if cfg!(target_os = "linux") {
                "linux-x86_64"
            } else {
                "windows-x86_64"
            })
            .join("bin");

        let clang_path = toolchain_bin.join(format!("{}{}-clang", clang_target, api_level));

        // Set up environment for cross-compilation
        let linker_env = format!(
            "CARGO_TARGET_{}_LINKER",
            target.to_uppercase().replace('-', "_")
        );

        let mut cmd = Command::new("cargo");
        cmd.arg("build")
            .arg("-p")
            .arg("xybrid_flutter")
            .arg("--target")
            .arg(target)
            .arg("--features")
            .arg("platform-android")
            .env(&linker_env, &clang_path);

        if release {
            cmd.arg("--release");
        }

        let status = cmd.status().context("Failed to run cargo build with NDK")?;

        if !status.success() {
            anyhow::bail!("cargo build with NDK failed for target {}", target);
        }
    }

    Ok(())
}

/// Get the library file name for a given target
fn get_flutter_lib_name(target: &str) -> &'static str {
    if target.contains("apple") {
        "libxybrid_flutter_ffi.a"
    } else if target.contains("windows") {
        "xybrid_flutter_ffi.dll"
    } else if target.contains("android") || target.contains("linux") {
        "libxybrid_flutter_ffi.so"
    } else {
        "libxybrid_flutter_ffi.a"
    }
}

/// All required targets for cross-compilation
const ALL_TARGETS: &[(&str, &str)] = &[
    // iOS targets
    ("aarch64-apple-ios", "iOS arm64"),
    ("x86_64-apple-ios", "iOS Simulator x86_64"),
    ("aarch64-apple-ios-sim", "iOS Simulator arm64"),
    // macOS targets
    ("aarch64-apple-darwin", "macOS arm64"),
    ("x86_64-apple-darwin", "macOS x86_64"),
    // Android targets
    ("aarch64-linux-android", "Android arm64-v8a"),
    ("armv7-linux-androideabi", "Android armeabi-v7a"),
    ("x86_64-linux-android", "Android x86_64"),
];

/// Install required Rust cross-compilation targets
fn setup_targets() -> Result<()> {
    println!("Setting up Rust cross-compilation targets...");
    println!();

    // Get currently installed targets
    let output = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .context("Failed to run rustup target list")?;

    if !output.status.success() {
        anyhow::bail!("rustup target list failed");
    }

    let installed_targets: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|s| s.trim().to_string())
        .collect();

    let mut installed_count = 0;
    let mut skipped_count = 0;

    // Check and install each target
    for (target, description) in ALL_TARGETS {
        if installed_targets.contains(&target.to_string()) {
            println!("  ✓ {} ({}) - already installed", target, description);
            skipped_count += 1;
            continue;
        }

        print!("  Installing {} ({})...", target, description);

        let status = Command::new("rustup")
            .args(["target", "add", target])
            .output()
            .with_context(|| format!("Failed to run rustup target add {}", target))?;

        if status.status.success() {
            println!(" ✓");
            installed_count += 1;
        } else {
            let stderr = String::from_utf8_lossy(&status.stderr);
            println!(" ✗");
            eprintln!("    Error: {}", stderr.trim());
        }
    }

    println!();
    println!("✓ Setup complete!");
    println!();
    println!("Summary:");
    println!("  - {} targets newly installed", installed_count);
    println!("  - {} targets already installed", skipped_count);
    println!();
    println!("Installed targets:");

    // List all targets grouped by platform
    println!();
    println!("  iOS:");
    println!("    - aarch64-apple-ios (device)");
    println!("    - x86_64-apple-ios (simulator x86_64)");
    println!("    - aarch64-apple-ios-sim (simulator arm64)");
    println!();
    println!("  macOS:");
    println!("    - aarch64-apple-darwin (arm64)");
    println!("    - x86_64-apple-darwin (x86_64)");
    println!();
    println!("  Android:");
    println!("    - aarch64-linux-android (arm64-v8a)");
    println!("    - armv7-linux-androideabi (armeabi-v7a)");
    println!("    - x86_64-linux-android (x86_64)");

    Ok(())
}

/// Represents a platform build task
#[derive(Debug, Clone, Copy)]
enum BuildPlatform {
    XCFramework,
    Android,
    FlutterIos,
    FlutterAndroid,
    FlutterMacos,
    FlutterWindows,
    FlutterLinux,
}

impl BuildPlatform {
    /// Returns the name of the platform
    fn name(&self) -> &'static str {
        match self {
            BuildPlatform::XCFramework => "XCFramework (iOS/macOS)",
            BuildPlatform::Android => "Android",
            BuildPlatform::FlutterIos => "Flutter iOS",
            BuildPlatform::FlutterAndroid => "Flutter Android",
            BuildPlatform::FlutterMacos => "Flutter macOS",
            BuildPlatform::FlutterWindows => "Flutter Windows",
            BuildPlatform::FlutterLinux => "Flutter Linux",
        }
    }

    /// Check if this platform can be built on the current OS
    fn can_build_on_current_os(&self) -> bool {
        match self {
            BuildPlatform::XCFramework => cfg!(target_os = "macos"),
            BuildPlatform::Android | BuildPlatform::FlutterAndroid => true, // Cross-compile from any OS
            BuildPlatform::FlutterIos | BuildPlatform::FlutterMacos => cfg!(target_os = "macos"),
            BuildPlatform::FlutterWindows => cfg!(target_os = "windows"),
            BuildPlatform::FlutterLinux => cfg!(target_os = "linux"),
        }
    }

    /// Returns the skip reason if this platform cannot be built
    fn skip_reason(&self) -> &'static str {
        match self {
            BuildPlatform::XCFramework => "XCFramework builds require macOS",
            BuildPlatform::Android | BuildPlatform::FlutterAndroid => {
                "Android builds require Android NDK"
            }
            BuildPlatform::FlutterIos => "iOS builds require macOS",
            BuildPlatform::FlutterMacos => "macOS builds require macOS",
            BuildPlatform::FlutterWindows => "Windows builds require Windows",
            BuildPlatform::FlutterLinux => "Linux builds require Linux",
        }
    }

    /// Get all platforms
    fn all() -> Vec<BuildPlatform> {
        vec![
            BuildPlatform::XCFramework,
            BuildPlatform::Android,
            BuildPlatform::FlutterIos,
            BuildPlatform::FlutterAndroid,
            BuildPlatform::FlutterMacos,
            BuildPlatform::FlutterWindows,
            BuildPlatform::FlutterLinux,
        ]
    }
}

/// Build all platforms with one command
fn build_all(release: bool, parallel: bool, version: &str) -> Result<()> {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let profile = if release { "release" } else { "debug" };

    println!("Building all platforms ({} mode, version {})...", profile, version);
    println!();

    // Categorize platforms into buildable and skipped
    let all_platforms = BuildPlatform::all();
    let (buildable, skipped): (Vec<_>, Vec<_>) = all_platforms
        .into_iter()
        .partition(|p| p.can_build_on_current_os());

    // Report skipped platforms upfront
    if !skipped.is_empty() {
        println!("Skipping (not supported on this OS):");
        for platform in &skipped {
            println!("  • {} - {}", platform.name(), platform.skip_reason());
        }
        println!();
    }

    if buildable.is_empty() {
        println!("No platforms can be built on the current OS.");
        return Ok(());
    }

    println!("Building:");
    for platform in &buildable {
        println!("  • {}", platform.name());
    }
    println!();

    // Track results
    let built: Arc<Mutex<Vec<BuildPlatform>>> = Arc::new(Mutex::new(Vec::new()));
    let failed: Arc<Mutex<Vec<(BuildPlatform, String)>>> = Arc::new(Mutex::new(Vec::new()));

    if parallel {
        // Parallel builds (experimental)
        println!("Running builds in parallel (experimental)...");
        println!();

        let version_owned = version.to_string();
        let handles: Vec<_> = buildable
            .iter()
            .map(|&platform| {
                let built = Arc::clone(&built);
                let failed = Arc::clone(&failed);
                let ver = version_owned.clone();

                thread::spawn(move || {
                    let result = run_platform_build(platform, release, &ver);
                    match result {
                        Ok(()) => {
                            built.lock().unwrap().push(platform);
                        }
                        Err(e) => {
                            failed.lock().unwrap().push((platform, e.to_string()));
                        }
                    }
                })
            })
            .collect();

        // Wait for all builds to complete
        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    } else {
        // Sequential builds
        for platform in &buildable {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Building {}...", platform.name());
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!();

            match run_platform_build(*platform, release, version) {
                Ok(()) => {
                    built.lock().unwrap().push(*platform);
                    println!();
                    println!("✓ {} build complete", platform.name());
                    println!();
                }
                Err(e) => {
                    failed.lock().unwrap().push((*platform, e.to_string()));
                    println!();
                    eprintln!("✗ {} build failed: {}", platform.name(), e);
                    println!();
                }
            }
        }
    }

    // Print summary
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Build Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let built_list = built.lock().unwrap();
    let failed_list = failed.lock().unwrap();

    if !built_list.is_empty() {
        println!("✓ Successfully built ({}):", built_list.len());
        for platform in built_list.iter() {
            println!("    • {}", platform.name());
        }
        println!();
    }

    if !failed_list.is_empty() {
        println!("✗ Failed ({}):", failed_list.len());
        for (platform, error) in failed_list.iter() {
            println!("    • {}: {}", platform.name(), error);
        }
        println!();
    }

    if !skipped.is_empty() {
        println!("⊘ Skipped ({}):", skipped.len());
        for platform in &skipped {
            println!("    • {}", platform.name());
        }
        println!();
    }

    // Summary line
    let total = buildable.len();
    let success = built_list.len();
    let fail = failed_list.len();
    let skip = skipped.len();

    println!(
        "Total: {} built, {} failed, {} skipped",
        success, fail, skip
    );

    if fail > 0 {
        anyhow::bail!("{} of {} builds failed", fail, total);
    }

    Ok(())
}

/// Execute the build for a specific platform
fn run_platform_build(platform: BuildPlatform, release: bool, version: &str) -> Result<()> {
    match platform {
        BuildPlatform::XCFramework => build_xcframework(release, version),
        BuildPlatform::Android => build_android(release, vec![], version),
        BuildPlatform::FlutterIos => build_flutter(FlutterPlatform::Ios, release, version),
        BuildPlatform::FlutterAndroid => build_flutter(FlutterPlatform::Android, release, version),
        BuildPlatform::FlutterMacos => build_flutter(FlutterPlatform::Macos, release, version),
        BuildPlatform::FlutterWindows => build_flutter(FlutterPlatform::Windows, release, version),
        BuildPlatform::FlutterLinux => build_flutter(FlutterPlatform::Linux, release, version),
    }
}

/// Package info for manifest.json
#[derive(serde::Serialize)]
struct PackageInfo {
    name: String,
    version: String,
    filename: String,
    size: u64,
    sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    platform: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    architectures: Option<Vec<String>>,
}

/// Manifest containing all packages
#[derive(serde::Serialize)]
struct Manifest {
    version: String,
    created_at: String,
    packages: Vec<PackageInfo>,
}

/// Calculate SHA256 checksum of a file
fn calculate_sha256(path: &PathBuf) -> Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file for checksum: {:?}", path))?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let hash = hasher.finalize();
    Ok(hash.iter().map(|b| format!("{:02x}", b)).collect())
}

/// Simple SHA256 implementation (no external dependencies)
struct Sha256 {
    state: [u32; 8],
    buffer: Vec<u8>,
    total_len: u64,
}

impl Sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            buffer: Vec::new(),
            total_len: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        self.total_len += data.len() as u64;

        while self.buffer.len() >= 64 {
            let chunk: [u8; 64] = self.buffer[..64].try_into().unwrap();
            self.process_block(&chunk);
            self.buffer.drain(..64);
        }
    }

    fn process_block(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];

        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }

        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;

        #[allow(clippy::needless_range_loop)]
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(Self::K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len * 8;

        // Append padding
        self.buffer.push(0x80);
        while (self.buffer.len() % 64) != 56 {
            self.buffer.push(0);
        }

        // Append length
        self.buffer.extend_from_slice(&bit_len.to_be_bytes());

        // Process remaining blocks
        while self.buffer.len() >= 64 {
            let chunk: [u8; 64] = self.buffer[..64].try_into().unwrap();
            self.process_block(&chunk);
            self.buffer.drain(..64);
        }

        let mut result = [0u8; 32];
        for (i, &val) in self.state.iter().enumerate() {
            result[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
        }
        result
    }
}

/// Create a zip archive from a directory
fn create_zip(source_dir: &Path, output_path: &Path) -> Result<()> {
    // Use the zip command (available on macOS, Linux, and most CI environments)
    let status = Command::new("zip")
        .arg("-r")
        .arg("-q")
        .arg(output_path)
        .arg(".")
        .current_dir(source_dir)
        .status()
        .context("Failed to run zip command")?;

    if !status.success() {
        anyhow::bail!("zip command failed");
    }

    Ok(())
}

/// Package build artifacts for distribution
fn package_artifacts(
    version: &str,
    output_dir: &Path,
    skip_apple: bool,
    skip_android: bool,
    skip_flutter: bool,
) -> Result<()> {
    println!("Packaging artifacts (version {})...", version);
    println!();

    // Create output directory
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", output_dir))?;

    let mut packages: Vec<PackageInfo> = Vec::new();

    // Package XCFramework (Apple)
    if !skip_apple {
        if let Some(pkg) = package_xcframework(version, output_dir)? {
            packages.push(pkg);
        }
    } else {
        println!("Skipping Apple artifacts (--skip-apple)");
    }

    // Package Android .so files
    if !skip_android {
        if let Some(pkg) = package_android(version, output_dir)? {
            packages.push(pkg);
        }
    } else {
        println!("Skipping Android artifacts (--skip-android)");
    }

    // Package Flutter plugin
    if !skip_flutter {
        if let Some(pkg) = package_flutter(version, output_dir)? {
            packages.push(pkg);
        }
    } else {
        println!("Skipping Flutter artifacts (--skip-flutter)");
    }

    if packages.is_empty() {
        println!();
        println!("No artifacts were packaged.");
        println!("Run build commands first to generate artifacts:");
        println!("  cargo xtask build-xcframework");
        println!("  cargo xtask build-android");
        println!("  cargo xtask build-flutter --platform <platform>");
        return Ok(());
    }

    // Generate checksums file
    println!();
    println!("Generating checksums...");
    let checksums_path = output_dir.join("checksums.sha256");
    let mut checksums_content = String::new();
    for pkg in &packages {
        checksums_content.push_str(&format!("{}  {}\n", pkg.sha256, pkg.filename));
    }
    std::fs::write(&checksums_path, &checksums_content)
        .context("Failed to write checksums file")?;
    println!("  ✓ {}", checksums_path.display());

    // Generate manifest.json
    println!();
    println!("Generating manifest.json...");
    let manifest = Manifest {
        version: version.to_string(),
        created_at: chrono_now(),
        packages,
    };
    let manifest_path = output_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .context("Failed to serialize manifest")?;
    std::fs::write(&manifest_path, &manifest_json)
        .context("Failed to write manifest.json")?;
    println!("  ✓ {}", manifest_path.display());

    // Print summary
    println!();
    println!("✓ Packaging complete!");
    println!();
    println!("Output directory: {}", output_dir.display());
    println!();
    println!("Packages:");
    for pkg in &manifest.packages {
        let size_kb = pkg.size / 1024;
        let size_str = if size_kb > 1024 {
            format!("{:.1} MB", size_kb as f64 / 1024.0)
        } else {
            format!("{} KB", size_kb)
        };
        println!("  • {} ({})", pkg.filename, size_str);
    }
    println!();
    println!("Files:");
    println!("  • manifest.json");
    println!("  • checksums.sha256");

    Ok(())
}

/// Get current timestamp in ISO 8601 format (simplified, no external dependencies)
fn chrono_now() -> String {
    // Use the date command to get ISO 8601 timestamp
    if let Ok(output) = Command::new("date")
        .arg("-u")
        .arg("+%Y-%m-%dT%H:%M:%SZ")
        .output()
    {
        if output.status.success() {
            return String::from_utf8_lossy(&output.stdout).trim().to_string();
        }
    }
    // Fallback
    "unknown".to_string()
}

/// Package XCFramework as a .zip file
fn package_xcframework(version: &str, output_dir: &Path) -> Result<Option<PackageInfo>> {
    let xcframework_dir = PathBuf::from("bindings/apple/XCFrameworks");
    let xcframework_path = xcframework_dir.join("XybridFFI.xcframework");

    if !xcframework_path.exists() {
        println!("Skipping XCFramework - not found at {:?}", xcframework_path);
        println!("  Run 'cargo xtask build-xcframework' first");
        return Ok(None);
    }

    println!("Packaging XCFramework...");

    let filename = format!("XybridFFI-{}.xcframework.zip", version);
    let output_path = output_dir.join(&filename);

    // Remove existing file
    if output_path.exists() {
        std::fs::remove_file(&output_path)?;
    }

    // Create zip from XCFramework directory
    let status = Command::new("zip")
        .arg("-r")
        .arg("-q")
        .arg(&output_path)
        .arg("XybridFFI.xcframework")
        .current_dir(&xcframework_dir)
        .status()
        .context("Failed to run zip command")?;

    if !status.success() {
        anyhow::bail!("Failed to create XCFramework zip");
    }

    let size = std::fs::metadata(&output_path)?.len();
    let sha256 = calculate_sha256(&output_path)?;

    println!("  ✓ {} ({} bytes, sha256: {}...)", filename, size, &sha256[..16]);

    Ok(Some(PackageInfo {
        name: "XybridFFI.xcframework".to_string(),
        version: version.to_string(),
        filename,
        size,
        sha256,
        platform: Some("apple".to_string()),
        architectures: Some(vec![
            "ios-arm64".to_string(),
            "ios-arm64_x86_64-simulator".to_string(),
            "macos-arm64_x86_64".to_string(),
        ]),
    }))
}

/// Package Android .so files as a .zip file
fn package_android(version: &str, output_dir: &Path) -> Result<Option<PackageInfo>> {
    let libs_dir = PathBuf::from("bindings/kotlin/libs");

    if !libs_dir.exists() {
        println!("Skipping Android libs - not found at {:?}", libs_dir);
        println!("  Run 'cargo xtask build-android' first");
        return Ok(None);
    }

    // Check for at least one ABI directory with .so files
    let abis = ["arm64-v8a", "armeabi-v7a", "x86_64"];
    let mut found_abis = Vec::new();

    for abi in &abis {
        let so_path = libs_dir.join(abi).join("libxybrid_uniffi.so");
        if so_path.exists() {
            found_abis.push(abi.to_string());
        }
    }

    if found_abis.is_empty() {
        println!("Skipping Android libs - no .so files found");
        println!("  Run 'cargo xtask build-android' first");
        return Ok(None);
    }

    println!("Packaging Android libs...");

    let filename = format!("xybrid-android-{}.zip", version);
    let output_path = output_dir.join(&filename);

    // Remove existing file
    if output_path.exists() {
        std::fs::remove_file(&output_path)?;
    }

    // Create a temporary directory structure for zipping
    let temp_dir = output_dir.join(".android-package-temp");
    if temp_dir.exists() {
        std::fs::remove_dir_all(&temp_dir)?;
    }
    std::fs::create_dir_all(&temp_dir)?;

    // Copy .so files to temp directory
    for abi in &found_abis {
        let src_dir = libs_dir.join(abi);
        let dst_dir = temp_dir.join(abi);
        std::fs::create_dir_all(&dst_dir)?;

        let so_src = src_dir.join("libxybrid_uniffi.so");
        let so_dst = dst_dir.join("libxybrid_uniffi.so");
        std::fs::copy(&so_src, &so_dst)?;
    }

    // Create zip
    create_zip(&temp_dir, &output_path)?;

    // Cleanup temp directory
    std::fs::remove_dir_all(&temp_dir)?;

    let size = std::fs::metadata(&output_path)?.len();
    let sha256 = calculate_sha256(&output_path)?;

    println!("  ✓ {} ({} bytes, sha256: {}...)", filename, size, &sha256[..16]);

    Ok(Some(PackageInfo {
        name: "xybrid-android".to_string(),
        version: version.to_string(),
        filename,
        size,
        sha256,
        platform: Some("android".to_string()),
        architectures: Some(found_abis),
    }))
}

/// Package Flutter plugin as a tarball
fn package_flutter(version: &str, output_dir: &Path) -> Result<Option<PackageInfo>> {
    let flutter_dir = PathBuf::from("bindings/flutter");

    if !flutter_dir.exists() {
        println!("Skipping Flutter plugin - not found at {:?}", flutter_dir);
        return Ok(None);
    }

    // Check for pubspec.yaml to verify it's a Flutter plugin
    let pubspec_path = flutter_dir.join("pubspec.yaml");
    if !pubspec_path.exists() {
        println!("Skipping Flutter plugin - pubspec.yaml not found");
        return Ok(None);
    }

    println!("Packaging Flutter plugin...");

    let filename = format!("xybrid-flutter-{}.tar.gz", version);
    let output_path = output_dir.join(&filename);

    // Remove existing file
    if output_path.exists() {
        std::fs::remove_file(&output_path)?;
    }

    // Create tarball, excluding build artifacts and generated code
    let parent = flutter_dir.parent().unwrap_or(&flutter_dir);

    let status = Command::new("tar")
        .arg("-czf")
        .arg(&output_path)
        .arg("--exclude=.dart_tool")
        .arg("--exclude=build")
        .arg("--exclude=.packages")
        .arg("--exclude=pubspec.lock")
        .arg("--exclude=lib/src/rust")  // Exclude FRB-generated code
        .arg("--exclude=rust/target")   // Exclude Rust build artifacts
        .arg("-C")
        .arg(parent)
        .arg("flutter")
        .status()
        .context("Failed to run tar command")?;

    if !status.success() {
        anyhow::bail!("Failed to create Flutter plugin tarball");
    }

    let size = std::fs::metadata(&output_path)?.len();
    let sha256 = calculate_sha256(&output_path)?;

    println!("  ✓ {} ({} bytes, sha256: {}...)", filename, size, &sha256[..16]);

    Ok(Some(PackageInfo {
        name: "xybrid-flutter".to_string(),
        version: version.to_string(),
        filename,
        size,
        sha256,
        platform: Some("flutter".to_string()),
        architectures: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_preset_for_target_macos() {
        assert_eq!(
            platform_preset_for_target("aarch64-apple-darwin"),
            "platform-macos"
        );
        assert_eq!(
            platform_preset_for_target("x86_64-apple-darwin"),
            "platform-macos"
        );
    }

    #[test]
    fn test_platform_preset_for_target_ios() {
        assert_eq!(
            platform_preset_for_target("aarch64-apple-ios"),
            "platform-ios"
        );
        assert_eq!(
            platform_preset_for_target("aarch64-apple-ios-sim"),
            "platform-ios"
        );
        assert_eq!(
            platform_preset_for_target("x86_64-apple-ios"),
            "platform-ios"
        );
    }

    #[test]
    fn test_platform_preset_for_target_android() {
        assert_eq!(
            platform_preset_for_target("aarch64-linux-android"),
            "platform-android"
        );
        assert_eq!(
            platform_preset_for_target("armv7-linux-androideabi"),
            "platform-android"
        );
        assert_eq!(
            platform_preset_for_target("x86_64-linux-android"),
            "platform-android"
        );
    }

    #[test]
    fn test_platform_preset_for_target_desktop() {
        assert_eq!(
            platform_preset_for_target("x86_64-unknown-linux-gnu"),
            "platform-desktop"
        );
        assert_eq!(
            platform_preset_for_target("x86_64-pc-windows-msvc"),
            "platform-desktop"
        );
    }

    #[test]
    fn test_platform_preset_for_target_unknown_falls_back_to_desktop() {
        // Unknown targets should fall back to platform-desktop
        assert_eq!(
            platform_preset_for_target("some-unknown-target"),
            "platform-desktop"
        );
        assert_eq!(
            platform_preset_for_target("riscv64gc-unknown-linux-gnu"),
            "platform-desktop"
        );
    }

    #[test]
    fn test_host_platform_preset() {
        // The host preset should be a valid preset string
        let preset = host_platform_preset();
        assert!(
            preset == "platform-macos"
                || preset == "platform-desktop"
                || preset == "platform-ios"
                || preset == "platform-android"
        );

        // On macOS, it should specifically return platform-macos
        #[cfg(target_os = "macos")]
        assert_eq!(preset, "platform-macos");

        // On Windows, it should return platform-desktop
        #[cfg(target_os = "windows")]
        assert_eq!(preset, "platform-desktop");

        // On Linux, it should return platform-desktop
        #[cfg(target_os = "linux")]
        assert_eq!(preset, "platform-desktop");
    }

    #[test]
    fn test_is_ios_target() {
        // iOS targets
        assert!(is_ios_target("aarch64-apple-ios"));
        assert!(is_ios_target("aarch64-apple-ios-sim"));
        assert!(is_ios_target("x86_64-apple-ios"));

        // Non-iOS targets
        assert!(!is_ios_target("aarch64-apple-darwin"));
        assert!(!is_ios_target("x86_64-apple-darwin"));
        assert!(!is_ios_target("aarch64-linux-android"));
        assert!(!is_ios_target("x86_64-unknown-linux-gnu"));
        assert!(!is_ios_target("x86_64-pc-windows-msvc"));
    }

    #[test]
    fn test_resolve_ort_lib_location_returns_none_for_non_ios_targets() {
        // Non-iOS targets should always return None
        assert!(resolve_ort_lib_location("aarch64-apple-darwin").is_none());
        assert!(resolve_ort_lib_location("x86_64-apple-darwin").is_none());
        assert!(resolve_ort_lib_location("aarch64-linux-android").is_none());
        assert!(resolve_ort_lib_location("x86_64-unknown-linux-gnu").is_none());
        assert!(resolve_ort_lib_location("x86_64-pc-windows-msvc").is_none());
    }

    #[test]
    fn test_resolve_ort_lib_location_for_ios_target() {
        // For iOS targets, the function checks for the library on disk.
        // In test environment where vendor/ort-ios exists, it should return Some.
        // If vendor doesn't exist and no env var is set, it returns None.
        let result = resolve_ort_lib_location("aarch64-apple-ios");

        // The result depends on whether the vendor directory exists in the test environment.
        // We can only verify that the function runs without error and returns the correct type.
        // If vendor/ort-ios/onnxruntime.xcframework/ios-arm64/libonnxruntime.a exists, it's Some.
        // Otherwise, it's None.
        let vendor_lib = PathBuf::from("vendor/ort-ios/onnxruntime.xcframework/ios-arm64/libonnxruntime.a");
        if vendor_lib.exists() {
            assert!(result.is_some());
            assert!(result.unwrap().ends_with("ios-arm64"));
        }
        // If the file doesn't exist and no ORT_LIB_LOCATION is set, result can be None
    }
}
