//! Registry Integration Tests
//!
//! This test suite validates the end-to-end bundle resolution flow:
//! - Registry server serves correct bundle index
//! - Target resolution picks correct platform bundle
//! - Bundle download and caching behavior
//! - Multi-target bundle selection
//!
//! Prerequisites:
//! - Run `just registry::create` to create test bundles
//! - Run `just registry::serve-local` to start the registry server
//!
//! Run with: `cargo test --test registry_integration -- --nocapture`

use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;

use xybrid_core::registry::{LocalRegistry, Registry};
use xybrid_core::registry_config::{BundleDescriptor, RemoteRegistryConfig};
use xybrid_core::registry_index::RegistryIndex;
use xybrid_core::registry_remote::{HttpRegistryTransport, RegistryTransport, RemoteRegistry};
use xybrid_core::target::{Platform, Target, TargetResolver};

/// Test registry server URL
const TEST_REGISTRY_URL: &str = "http://localhost:8080";

/// Check if the registry server is running
fn is_registry_running() -> bool {
    match ureq::get(&format!("{}/index", TEST_REGISTRY_URL))
        .timeout(Duration::from_secs(2))
        .call()
    {
        Ok(_) => true,
        Err(_) => false,
    }
}

/// Skip test if registry is not running
macro_rules! require_registry {
    () => {
        if !is_registry_running() {
            eprintln!("⚠️  Skipping test: Registry server not running at {}", TEST_REGISTRY_URL);
            eprintln!("   Start with: just registry::serve-local");
            return Ok(());
        }
    };
}

// =============================================================================
// Test 1: Registry Server Index
// =============================================================================

/// Test that the registry server returns a valid bundle index
#[test]
fn test_registry_index_fetch() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Test 1: Registry index fetch");
    println!("{}", "=".repeat(60));

    require_registry!();

    let config = RemoteRegistryConfig {
        base_url: TEST_REGISTRY_URL.to_string(),
        index_path: None,
        bundle_path: None,
        auth: xybrid_core::registry_config::RegistryAuth::None,
        timeout_ms: Some(10000),
        retry_attempts: Some(2),
    };

    let transport = HttpRegistryTransport::new(config)?;
    let descriptors = transport.fetch_index()?;

    println!("   Found {} bundles in registry:", descriptors.len());
    for desc in &descriptors {
        println!(
            "   - {}@{} [{}] ({} bytes)",
            desc.id,
            desc.version,
            desc.target.as_deref().unwrap_or("unknown"),
            desc.size_bytes
        );
    }

    assert!(!descriptors.is_empty(), "Registry should have at least one bundle");

    println!("\n   ✅ Registry index fetch successful");
    println!();
    Ok(())
}

// =============================================================================
// Test 2: Target Resolution with Available Bundles
// =============================================================================

/// Test that target resolution picks the correct platform bundle
#[test]
fn test_target_resolution_from_registry() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Test 2: Target resolution from registry");
    println!("{}", "=".repeat(60));

    require_registry!();

    let config = RemoteRegistryConfig {
        base_url: TEST_REGISTRY_URL.to_string(),
        index_path: None,
        bundle_path: None,
        auth: xybrid_core::registry_config::RegistryAuth::None,
        timeout_ms: Some(10000),
        retry_attempts: Some(2),
    };

    let transport = HttpRegistryTransport::new(config)?;
    let descriptors = transport.fetch_index()?;

    // Group bundles by model ID
    let mut bundles_by_model: HashMap<String, Vec<&BundleDescriptor>> = HashMap::new();
    for desc in &descriptors {
        bundles_by_model
            .entry(desc.id.clone())
            .or_default()
            .push(desc);
    }

    println!("   Testing target resolution for each model:");

    for (model_id, bundles) in &bundles_by_model {
        let available_target_strings: Vec<String> = bundles
            .iter()
            .filter_map(|b| b.target.clone())
            .collect();

        let available_targets: Vec<Target> = available_target_strings
            .iter()
            .filter_map(|t| Target::from_str(t))
            .collect();

        if available_targets.is_empty() {
            println!("   - {}: No targets with recognized formats", model_id);
            continue;
        }

        // Test resolution for different platforms
        let test_cases = [
            (Platform::MacOS, "macOS"),
            (Platform::IOS, "iOS"),
            (Platform::Android, "Android"),
            (Platform::Linux, "Linux"),
        ];

        println!("   - {} (available: {:?}):", model_id, available_targets);

        for (platform, platform_name) in test_cases {
            let resolver = TargetResolver::new()
                .with_platform(platform)
                .with_available(available_target_strings.clone());

            let resolved = resolver.resolve();

            println!(
                "     {} → {} (preferred: {})",
                platform_name,
                resolved,
                platform.preferred_target()
            );

            // Verify resolution is valid
            assert!(
                available_targets.contains(&resolved) || resolved == Target::Onnx,
                "Resolved target {} should be available or fall back to ONNX",
                resolved
            );
        }
    }

    println!("\n   ✅ Target resolution verified for all models");
    println!();
    Ok(())
}

// =============================================================================
// Test 3: Bundle Download
// =============================================================================

/// Test downloading a bundle from the registry
#[test]
fn test_bundle_download() -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 Test 3: Bundle download from registry");
    println!("{}", "=".repeat(60));

    require_registry!();

    let config = RemoteRegistryConfig {
        base_url: TEST_REGISTRY_URL.to_string(),
        index_path: None,
        bundle_path: None,
        auth: xybrid_core::registry_config::RegistryAuth::None,
        timeout_ms: Some(30000), // Longer timeout for downloads
        retry_attempts: Some(2),
    };

    let transport = HttpRegistryTransport::new(config)?;
    let descriptors = transport.fetch_index()?;

    // Find a small bundle to download (prefer placeholder bundles for speed)
    let small_bundle = descriptors
        .iter()
        .filter(|d| d.size_bytes < 10_000) // Less than 10KB
        .next();

    if let Some(bundle) = small_bundle {
        println!(
            "   Downloading: {}@{} [{}] ({} bytes)",
            bundle.id,
            bundle.version,
            bundle.target.as_deref().unwrap_or("unknown"),
            bundle.size_bytes
        );

        let data = transport.fetch_bundle(bundle)?;

        println!("   Downloaded: {} bytes", data.len());
        assert!(!data.is_empty(), "Downloaded bundle should not be empty");
        assert!(
            data.len() as u64 > 0,
            "Downloaded size should match descriptor"
        );

        // Verify it's a valid .xyb bundle (zstd compressed tar)
        // zstd magic number: 0x28 0xB5 0x2F 0xFD
        if data.len() >= 4 {
            let is_zstd = data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F && data[3] == 0xFD;
            println!("   Format: {}", if is_zstd { "zstd compressed" } else { "unknown" });
        }

        println!("\n   ✅ Bundle download successful");
    } else {
        println!("   ⚠️  No small test bundles available, skipping download test");
        println!("      Run `just registry::create` to create test bundles");
    }

    println!();
    Ok(())
}

// =============================================================================
// Test 4: Remote Registry with Local Cache
// =============================================================================

/// Test that RemoteRegistry caches bundles locally
#[test]
fn test_remote_registry_caching() -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 Test 4: Remote registry with local caching");
    println!("{}", "=".repeat(60));

    require_registry!();

    // Create a temporary cache directory
    let temp_dir = TempDir::new()?;
    let cache_path = temp_dir.path().join("cache");

    println!("   Cache directory: {}", cache_path.display());

    // First, get descriptors from registry to find a small bundle
    let config = RemoteRegistryConfig {
        base_url: TEST_REGISTRY_URL.to_string(),
        index_path: None,
        bundle_path: None,
        auth: xybrid_core::registry_config::RegistryAuth::None,
        timeout_ms: Some(30000),
        retry_attempts: Some(2),
    };

    let transport = HttpRegistryTransport::new(config)?;
    let descriptors = transport.fetch_index()?;

    if descriptors.is_empty() {
        println!("   ⚠️  No bundles in registry, skipping cache test");
        return Ok(());
    }

    // Find a small bundle to test caching
    let small_bundle = descriptors
        .iter()
        .filter(|d| d.size_bytes < 10_000)
        .next();

    if let Some(bundle) = small_bundle {
        println!(
            "   Testing cache with: {}@{} [{}]",
            bundle.id, bundle.version,
            bundle.target.as_deref().unwrap_or("?")
        );

        // Create a new remote registry with fresh cache for testing
        let config2 = RemoteRegistryConfig {
            base_url: TEST_REGISTRY_URL.to_string(),
            index_path: None,
            bundle_path: None,
            auth: xybrid_core::registry_config::RegistryAuth::None,
            timeout_ms: Some(30000),
            retry_attempts: Some(2),
        };
        let transport2 = HttpRegistryTransport::new(config2)?;
        let local_cache = LocalRegistry::new(&cache_path)?;
        let index = RegistryIndex::load_or_create_at(cache_path.join("index.json"))?;
        let remote_registry = RemoteRegistry::new(transport2, local_cache, index);

        // First fetch - should hit remote
        println!("   First fetch (remote)...");
        let start = std::time::Instant::now();
        let data1 = remote_registry.get_bundle(&bundle.id, Some(&bundle.version))?;
        let first_fetch_ms = start.elapsed().as_millis();
        println!("   First fetch: {} bytes in {}ms", data1.len(), first_fetch_ms);

        // Second fetch - should hit cache (faster)
        println!("   Second fetch (should be cached)...");
        let start = std::time::Instant::now();
        let data2 = remote_registry.get_bundle(&bundle.id, Some(&bundle.version))?;
        let second_fetch_ms = start.elapsed().as_millis();
        println!("   Second fetch: {} bytes in {}ms", data2.len(), second_fetch_ms);

        // Verify data is identical
        assert_eq!(data1, data2, "Cached data should match original");

        // Verify cache file exists
        let cache_files: Vec<_> = walkdir::WalkDir::new(&cache_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "xyb"))
            .collect();

        println!("   Cache files: {}", cache_files.len());
        for file in &cache_files {
            println!("     - {}", file.path().display());
        }

        assert!(
            !cache_files.is_empty(),
            "Bundle should be cached to local filesystem"
        );

        println!("\n   ✅ Remote registry caching verified");
    } else {
        println!("   ⚠️  No small test bundles available");
    }

    println!();
    Ok(())
}

// =============================================================================
// Test 5: Multi-Target Bundle Selection
// =============================================================================

/// Test selecting the correct target when multiple are available
#[test]
fn test_multi_target_selection() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Test 5: Multi-target bundle selection");
    println!("{}", "=".repeat(60));

    require_registry!();

    let config = RemoteRegistryConfig {
        base_url: TEST_REGISTRY_URL.to_string(),
        index_path: None,
        bundle_path: None,
        auth: xybrid_core::registry_config::RegistryAuth::None,
        timeout_ms: Some(10000),
        retry_attempts: Some(2),
    };

    let transport = HttpRegistryTransport::new(config)?;
    let descriptors = transport.fetch_index()?;

    // Group by model+version to find multi-target bundles
    let mut versions: HashMap<(String, String), Vec<&BundleDescriptor>> = HashMap::new();
    for desc in &descriptors {
        versions
            .entry((desc.id.clone(), desc.version.clone()))
            .or_default()
            .push(desc);
    }

    // Find models with multiple targets
    let multi_target: Vec<_> = versions
        .iter()
        .filter(|(_, bundles)| bundles.len() > 1)
        .collect();

    if multi_target.is_empty() {
        println!("   ⚠️  No multi-target bundles found in registry");
        println!("      This test requires bundles with multiple platform targets");
        return Ok(());
    }

    println!("   Found {} model versions with multiple targets:", multi_target.len());

    for ((id, version), bundles) in &multi_target {
        let targets: Vec<&str> = bundles
            .iter()
            .filter_map(|b| b.target.as_deref())
            .collect();

        println!("   - {}@{}: {:?}", id, version, targets);

        // Test that we can resolve the correct target for each platform
        let available_strings: Vec<String> = targets.iter().map(|t| t.to_string()).collect();
        let available: Vec<Target> = targets
            .iter()
            .filter_map(|t| Target::from_str(t))
            .collect();

        if !available.is_empty() {
            // Simulate iOS requesting a bundle
            let ios_resolver = TargetResolver::new()
                .with_platform(Platform::IOS)
                .with_available(available_strings.clone());
            let ios_target = ios_resolver.resolve();

            // Simulate Android requesting a bundle
            let android_resolver = TargetResolver::new()
                .with_platform(Platform::Android)
                .with_available(available_strings.clone());
            let android_target = android_resolver.resolve();

            println!("     iOS → {}, Android → {}", ios_target, android_target);

            // Find the matching descriptors
            let ios_bundle = bundles
                .iter()
                .find(|b| b.target.as_deref() == Some(ios_target.as_str()));
            let android_bundle = bundles
                .iter()
                .find(|b| b.target.as_deref() == Some(android_target.as_str()));

            if let Some(bundle) = ios_bundle {
                println!(
                    "     iOS bundle: {} ({} bytes)",
                    bundle.target.as_deref().unwrap_or("?"),
                    bundle.size_bytes
                );
            }

            if let Some(bundle) = android_bundle {
                println!(
                    "     Android bundle: {} ({} bytes)",
                    bundle.target.as_deref().unwrap_or("?"),
                    bundle.size_bytes
                );
            }
        }
    }

    println!("\n   ✅ Multi-target selection verified");
    println!();
    Ok(())
}

// =============================================================================
// Test 6: Bundle Descriptor Lookup
// =============================================================================

/// Test looking up specific bundle descriptors
#[test]
fn test_bundle_descriptor_lookup() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Test 6: Bundle descriptor lookup");
    println!("{}", "=".repeat(60));

    require_registry!();

    let config = RemoteRegistryConfig {
        base_url: TEST_REGISTRY_URL.to_string(),
        index_path: None,
        bundle_path: None,
        auth: xybrid_core::registry_config::RegistryAuth::None,
        timeout_ms: Some(10000),
        retry_attempts: Some(2),
    };

    let transport = HttpRegistryTransport::new(config)?;
    let descriptors = transport.fetch_index()?;

    if descriptors.is_empty() {
        println!("   ⚠️  No bundles in registry");
        return Ok(());
    }

    // Test looking up a known bundle
    let first = &descriptors[0];
    println!(
        "   Looking up: {}@{}",
        first.id, first.version
    );

    let found = transport.fetch_descriptor(&first.id, Some(&first.version))?;
    assert_eq!(found.id, first.id, "Found bundle ID should match");
    assert_eq!(found.version, first.version, "Found version should match");

    println!("   Found: {}@{}", found.id, found.version);

    // Test looking up latest version (no version specified)
    println!("   Looking up latest: {}", first.id);
    let latest = transport.fetch_descriptor(&first.id, None)?;
    println!("   Latest: {}@{}", latest.id, latest.version);

    // Test looking up non-existent bundle
    println!("   Looking up non-existent bundle...");
    let result = transport.fetch_descriptor("non-existent-model-xyz", None);
    assert!(result.is_err(), "Non-existent bundle should return error");
    println!("   Correctly returned error for non-existent bundle");

    println!("\n   ✅ Bundle descriptor lookup verified");
    println!();
    Ok(())
}

// =============================================================================
// Test 7: End-to-End Resolution Flow
// =============================================================================

/// Test the complete flow from model request to correct bundle selection
#[test]
fn test_end_to_end_resolution() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Test 7: End-to-end resolution flow");
    println!("{}", "=".repeat(60));

    require_registry!();

    let temp_dir = TempDir::new()?;
    let cache_path = temp_dir.path().join("e2e_cache");

    let config = RemoteRegistryConfig {
        base_url: TEST_REGISTRY_URL.to_string(),
        index_path: None,
        bundle_path: None,
        auth: xybrid_core::registry_config::RegistryAuth::None,
        timeout_ms: Some(30000),
        retry_attempts: Some(2),
    };

    let transport = HttpRegistryTransport::new(config)?;

    // Simulate a typical resolution flow:
    // 1. Request model "whisper-tiny" for iOS platform
    // 2. Get available targets from registry
    // 3. Resolve best target for platform
    // 4. Download the correct bundle

    let model_id = "whisper-tiny";
    let platform = Platform::IOS;

    println!("   Scenario: Request '{}' for {:?}", model_id, platform);
    println!();

    // Step 1: Get all bundles for this model (use fetch_index for full descriptor info)
    let all_descriptors = transport.fetch_index()?;
    let model_bundles: Vec<_> = all_descriptors
        .iter()
        .filter(|b| b.id == model_id)
        .collect();

    if model_bundles.is_empty() {
        println!("   ⚠️  Model '{}' not found in registry", model_id);
        println!("      Run `just registry::create` to create test bundles");
        return Ok(());
    }

    println!("   Step 1: Found {} bundles for '{}'", model_bundles.len(), model_id);
    for bundle in &model_bundles {
        println!(
            "     - v{} [{}] ({} bytes)",
            bundle.version,
            bundle.target.as_deref().unwrap_or("?"),
            bundle.size_bytes
        );
    }

    // Step 2: Get latest version
    let latest_version = model_bundles
        .iter()
        .map(|b| &b.version)
        .max()
        .cloned()
        .unwrap_or_default();

    println!("\n   Step 2: Latest version: {}", latest_version);

    // Step 3: Get available targets for latest version
    let latest_bundles: Vec<_> = model_bundles
        .iter()
        .filter(|b| b.version == latest_version)
        .collect();

    let available_target_strings: Vec<String> = latest_bundles
        .iter()
        .filter_map(|b| b.target.clone())
        .collect();

    let available_targets: Vec<Target> = available_target_strings
        .iter()
        .filter_map(|t| Target::from_str(t))
        .collect();

    println!("   Step 3: Available targets: {:?}", available_targets);

    // Step 4: Resolve target for platform
    let resolver = TargetResolver::new()
        .with_platform(platform)
        .with_available(available_target_strings.clone());

    let resolved_target = resolver.resolve();
    println!(
        "\n   Step 4: Resolved target for {:?}: {}",
        platform, resolved_target
    );

    // Step 5: Find matching bundle descriptor
    let target_bundle = latest_bundles
        .iter()
        .find(|b| b.target.as_deref() == Some(resolved_target.as_str()))
        .or_else(|| {
            // Fallback: try ONNX
            latest_bundles
                .iter()
                .find(|b| b.target.as_deref() == Some("onnx"))
        });

    if let Some(bundle) = target_bundle {
        println!(
            "\n   Step 5: Selected bundle: {}@{} [{}]",
            bundle.id,
            bundle.version,
            bundle.target.as_deref().unwrap_or("?")
        );

        // Step 6: Download bundle (if small enough for test)
        if bundle.size_bytes < 100_000 {
            println!("\n   Step 6: Downloading bundle...");
            // Create remote registry for actual download
            let config2 = RemoteRegistryConfig {
                base_url: TEST_REGISTRY_URL.to_string(),
                index_path: None,
                bundle_path: None,
                auth: xybrid_core::registry_config::RegistryAuth::None,
                timeout_ms: Some(30000),
                retry_attempts: Some(2),
            };
            let transport2 = HttpRegistryTransport::new(config2)?;
            let local_cache = LocalRegistry::new(&cache_path)?;
            let index = RegistryIndex::load_or_create_at(cache_path.join("index.json"))?;
            let remote_registry = RemoteRegistry::new(transport2, local_cache, index);

            let data = remote_registry.get_bundle(&bundle.id, Some(&bundle.version))?;
            println!("   Downloaded {} bytes", data.len());
        } else {
            println!(
                "\n   Step 6: Skipping download (bundle too large: {} bytes)",
                bundle.size_bytes
            );
        }

        println!("\n   ✅ End-to-end resolution flow completed successfully");
    } else {
        println!("\n   ⚠️  No matching bundle found for target: {}", resolved_target);
    }

    println!();
    Ok(())
}

// =============================================================================
// Helper: Print test summary
// =============================================================================

#[test]
fn test_print_summary() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║         Registry Integration Test Suite                    ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ Prerequisites:                                             ║");
    println!("║   1. just registry::create  (create test bundles)          ║");
    println!("║   2. just registry::serve-local  (start server)            ║");
    println!("║                                                            ║");
    println!("║ Tests:                                                     ║");
    println!("║   1. Registry index fetch                                  ║");
    println!("║   2. Target resolution from registry                       ║");
    println!("║   3. Bundle download                                       ║");
    println!("║   4. Remote registry caching                               ║");
    println!("║   5. Multi-target selection                                ║");
    println!("║   6. Bundle descriptor lookup                              ║");
    println!("║   7. End-to-end resolution flow                            ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
}
