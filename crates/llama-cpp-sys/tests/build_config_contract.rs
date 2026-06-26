use std::fs;
use std::path::PathBuf;

fn workspace_file(path: &str) -> String {
    let mut absolute = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    absolute.push("../..");
    absolute.push(path);
    fs::read_to_string(&absolute)
        .unwrap_or_else(|err| panic!("failed to read {}: {}", absolute.display(), err))
}

fn block_after<'a>(source: &'a str, marker: &str) -> &'a str {
    let start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing marker: {marker}"));
    let tail = &source[start..];
    let open = tail
        .find('{')
        .unwrap_or_else(|| panic!("missing block for marker: {marker}"));
    let mut depth = 0usize;

    for (offset, ch) in tail[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &tail[open + 1..open + offset];
                }
            }
            _ => {}
        }
    }

    panic!("unterminated block for marker: {marker}");
}

#[test]
fn portable_simd_baseline_disables_native_and_pins_x86_64_v3_simd() {
    let build_rs = workspace_file("crates/llama-cpp-sys/build.rs");
    let baseline = block_after(&build_rs, "fn configure_x86_64_simd_baseline(");

    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_NATIVE\", \"OFF\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_SSE42\", \"ON\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_AVX\", \"ON\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_AVX2\", \"ON\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_FMA\", \"ON\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_F16C\", \"ON\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_BMI2\", \"ON\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_AVX_VNNI\", \"OFF\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_AVX512\", \"OFF\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_AVX512_VBMI\", \"OFF\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_AVX512_VNNI\", \"OFF\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_AVX512_BF16\", \"OFF\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_AMX_TILE\", \"OFF\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_AMX_INT8\", \"OFF\")"));
    assert!(baseline.contains("define_from_env_or(cmake_config, \"GGML_AMX_BF16\", \"OFF\")"));
}

#[test]
fn desktop_x86_64_targets_share_portable_simd_baseline() {
    let build_rs = workspace_file("crates/llama-cpp-sys/build.rs");
    let linux_branch = block_after(&build_rs, "else if ctx.target_os == \"linux\"");
    let windows_branch = block_after(&build_rs, "else if ctx.target_os == \"windows\"");

    assert!(linux_branch.contains("configure_x86_64_simd_baseline(&mut cmake_config)"));
    assert!(windows_branch.contains("configure_x86_64_simd_baseline(&mut cmake_config)"));
    assert!(windows_branch.contains("cmake_config.profile(\"Release\")"));
}

#[test]
fn ggml_cmake_defines_are_env_overridable() {
    let build_rs = workspace_file("crates/llama-cpp-sys/build.rs");

    assert!(build_rs.contains("fn define_from_env_or("));
    assert!(build_rs.contains("println!(\"cargo:rerun-if-env-changed={name}\")"));
}
