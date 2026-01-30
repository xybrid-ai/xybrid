mod setup_env;

use clap::{Parser, Subcommand, ValueEnum};
use anyhow::{Context, Result};
use std::process::Command;
use std::path::PathBuf;

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
    },

    /// Build the xybrid-ffi library (C ABI for Unity/C++)
    BuildFfi {
        /// Target triple (e.g., aarch64-apple-darwin, x86_64-unknown-linux-gnu)
        #[arg(long)]
        target: Option<String>,

        /// Build in release mode
        #[arg(long)]
        release: bool,
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
}

#[derive(Clone, Copy, ValueEnum)]
enum BindingsLanguage {
    Swift,
    Kotlin,
    All,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::SetupTestEnv { registry } => {
            setup_env::run(registry)?;
        }
        Commands::BuildUniffi { target, release } => {
            build_uniffi(target, release)?;
        }
        Commands::BuildFfi { target, release } => {
            build_ffi(target, release)?;
        }
        Commands::GenerateBindings { language, out_dir } => {
            generate_bindings(language, out_dir)?;
        }
    }

    Ok(())
}

/// Build the xybrid-uniffi library
fn build_uniffi(target: Option<String>, release: bool) -> Result<()> {
    println!("Building xybrid-uniffi...");

    let mut cmd = Command::new("cargo");
    cmd.arg("build")
        .arg("-p")
        .arg("xybrid-uniffi");

    if release {
        cmd.arg("--release");
    }

    if let Some(ref t) = target {
        cmd.arg("--target").arg(t);
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
fn build_ffi(target: Option<String>, release: bool) -> Result<()> {
    println!("Building xybrid-ffi...");

    let mut cmd = Command::new("cargo");
    cmd.arg("build")
        .arg("-p")
        .arg("xybrid-ffi");

    if release {
        cmd.arg("--release");
    }

    if let Some(ref t) = target {
        cmd.arg("--target").arg(t);
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

    Ok(())
}

/// Generate Swift/Kotlin bindings using uniffi-bindgen
fn generate_bindings(language: BindingsLanguage, out_dir: Option<PathBuf>) -> Result<()> {
    // First, ensure the library is built
    println!("Building xybrid-uniffi (release)...");
    build_uniffi(None, true)?;

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
        let default_out = PathBuf::from(format!("bindings/{}", lang));
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

        println!("✓ {} bindings generated to {:?}", lang, output);
    }

    Ok(())
}
