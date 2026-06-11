use std::fs;
use std::path::PathBuf;

fn workspace_file(path: &str) -> String {
    let mut absolute = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    absolute.push("../..");
    absolute.push(path);
    fs::read_to_string(&absolute)
        .unwrap_or_else(|err| panic!("failed to read {}: {}", absolute.display(), err))
}

#[test]
fn linux_build_disables_native_and_pins_x86_64_v3_simd() {
    let build_rs = workspace_file("crates/llama-cpp-sys/build.rs");

    assert!(build_rs.contains("define_from_env_or(&mut cmake_config, \"GGML_NATIVE\", \"OFF\")"));
    assert!(build_rs.contains("define_from_env_or(&mut cmake_config, \"GGML_AVX2\", \"ON\")"));
    assert!(build_rs.contains("define_from_env_or(&mut cmake_config, \"GGML_FMA\", \"ON\")"));
    assert!(build_rs.contains("define_from_env_or(&mut cmake_config, \"GGML_F16C\", \"ON\")"));
}

#[test]
fn ggml_cmake_defines_are_env_overridable() {
    let build_rs = workspace_file("crates/llama-cpp-sys/build.rs");

    assert!(build_rs.contains("fn define_from_env_or("));
    assert!(build_rs.contains("println!(\"cargo:rerun-if-env-changed={name}\")"));
}
