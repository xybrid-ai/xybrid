// Ensures wire-format-specific legacy cache-token names don't leak
// outside the one place that legitimately speaks them (the gateway
// parser, which reads them off DeepSeek's response wire). Anything
// else is a canonical-field violation.
//
// Forbidden names are rebuilt at runtime from small shards so the test
// source itself is not an offender when scanned. The scan is rooted at
// `src/` only — this file sits in `tests/`, physically outside the
// scan path, so it cannot self-flag.

#[test]
fn legacy_cache_names_are_contained() {
    // Rebuild at runtime. Individual shards are benign — only the
    // concatenated forms are forbidden.
    let forbidden = [
        format!("prompt{}cache{}hit{}tokens", "_", "_", "_"),
        format!("prompt{}cache{}miss{}tokens", "_", "_", "_"),
        format!("cache{}hit{}tokens", "_", "_"),
        format!("cache{}miss{}tokens", "_", "_"),
    ];

    let scan_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders = Vec::new();
    for entry in walkdir::WalkDir::new(&scan_root) {
        let e = match entry {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !e.file_type().is_file() {
            continue;
        }
        let path = e.path();
        if path.extension().and_then(|o| o.to_str()) != Some("rs") {
            continue;
        }
        // The gateway parser is the ONLY place that speaks DeepSeek's
        // wire names; everywhere else should use canonical names.
        // `ends_with` matches the full final path component so this
        // allow doesn't leak to unrelated client.rs files nested
        // elsewhere in the tree.
        let allow = path.ends_with("cloud/client.rs");
        if allow {
            continue;
        }
        let body = std::fs::read_to_string(path).unwrap_or_default();
        for name in &forbidden {
            if body.contains(name) {
                offenders.push(format!("{} contains {}", path.display(), name));
            }
        }
    }
    assert!(offenders.is_empty(), "{:#?}", offenders);
}
