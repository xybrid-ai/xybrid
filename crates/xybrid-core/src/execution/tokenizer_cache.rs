//! Per-executor cache of loaded HuggingFace tokenizers.
//!
//! The `Tokenize` preprocessing step (BERT-style text models) and the
//! `WhisperDecode` postprocessing step both need a parsed `tokenizer.json`.
//! Parsing one takes milliseconds (~8 ms for MiniLM, ~28 ms for Whisper-tiny
//! on an M4 Max), orders of magnitude more than encoding or decoding a
//! request, so reloading it on every run dominated both steps.
//!
//! [`TokenizerCache`] lives on the `TemplateExecutor`, which the SDK keeps for
//! as long as a model is loaded, so a tokenizer is parsed once per loaded
//! model. Like the TTS session cache, an entry is reused only while the file's
//! `(len, modified)` is readable and unchanged: a `tokenizer.json` replaced in
//! place is reloaded rather than served stale.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokenizers::Tokenizer;

use super::path::{file_identity, FileIdentity};

/// Loaded tokenizers keyed by `tokenizer.json` path.
#[derive(Default)]
pub(crate) struct TokenizerCache {
    entries: HashMap<PathBuf, CachedTokenizer>,
}

struct CachedTokenizer {
    identity: FileIdentity,
    tokenizer: Arc<Tokenizer>,
}

impl TokenizerCache {
    /// Returns the tokenizer at `path`, parsing it on a miss or when the file changed.
    ///
    /// A failed load is not cached, so the next call retries.
    ///
    /// # Errors
    ///
    /// Returns the `tokenizers` error when the file cannot be read or parsed.
    pub(crate) fn get_or_load(&mut self, path: &Path) -> tokenizers::Result<Arc<Tokenizer>> {
        let identity = file_identity(path);
        if let (Some(identity), Some(cached)) = (identity, self.entries.get(path)) {
            if cached.identity == identity {
                return Ok(Arc::clone(&cached.tokenizer));
            }
        }

        let tokenizer = Arc::new(Tokenizer::from_file(path)?);
        match identity {
            Some(identity) => {
                let entry = CachedTokenizer {
                    identity,
                    tokenizer: Arc::clone(&tokenizer),
                };
                self.entries.insert(path.to_path_buf(), entry);
            }
            // Unverifiable identity: serve this load, but never trust it later.
            None => {
                self.entries.remove(path);
            }
        }
        Ok(tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;

    /// Writes a word-level `tokenizer.json` whose vocab is `words`, ids in order.
    fn write_tokenizer(path: &Path, words: &[&str]) {
        let vocab = words
            .iter()
            .enumerate()
            .map(|(id, word)| (word.to_string(), id as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token(words[0].to_string())
            .build()
            .expect("word-level model builds");
        Tokenizer::new(model)
            .save(path, false)
            .expect("tokenizer.json is written");
    }

    #[test]
    fn second_load_of_unchanged_file_reuses_the_tokenizer() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("tokenizer.json");
        write_tokenizer(&path, &["[UNK]", "hello"]);
        let mut cache = TokenizerCache::default();

        let first = cache.get_or_load(&path).expect("first load");
        let second = cache.get_or_load(&path).expect("second load");

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn file_replaced_in_place_is_reloaded() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("tokenizer.json");
        write_tokenizer(&path, &["[UNK]", "hello"]);
        let mut cache = TokenizerCache::default();
        let before = cache.get_or_load(&path).expect("first load");

        // A longer vocab changes the file length, so the identity differs even
        // on filesystems with coarse modification times.
        write_tokenizer(&path, &["[UNK]", "hello", "world"]);
        let after = cache.get_or_load(&path).expect("reload");

        assert!(!Arc::ptr_eq(&before, &after));
        assert_eq!(after.token_to_id("world"), Some(2));
        assert_eq!(before.token_to_id("world"), None);
    }

    #[test]
    fn each_path_gets_its_own_entry() {
        let dir = tempfile::tempdir().expect("temp dir");
        let a = dir.path().join("a.json");
        let b = dir.path().join("b.json");
        write_tokenizer(&a, &["[UNK]", "alpha"]);
        write_tokenizer(&b, &["[UNK]", "beta"]);
        let mut cache = TokenizerCache::default();

        let from_a = cache.get_or_load(&a).expect("load a");
        let from_b = cache.get_or_load(&b).expect("load b");

        assert_eq!(from_a.token_to_id("alpha"), Some(1));
        assert_eq!(from_b.token_to_id("beta"), Some(1));
        assert!(Arc::ptr_eq(
            &from_a,
            &cache.get_or_load(&a).expect("reuse a")
        ));
    }

    #[test]
    fn missing_file_is_an_error_and_is_not_cached() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("tokenizer.json");
        let mut cache = TokenizerCache::default();

        assert!(cache.get_or_load(&path).is_err());
        assert!(cache.entries.is_empty());

        // Once the file appears, the next call loads it.
        write_tokenizer(&path, &["[UNK]", "hello"]);
        assert_eq!(
            cache
                .get_or_load(&path)
                .expect("load after the file appears")
                .token_to_id("hello"),
            Some(1)
        );
    }
}
