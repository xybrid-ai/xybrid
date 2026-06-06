//! Splits text into TTS chunks via sentence and center-break heuristics.
//!
//! Long input is first split at sentence boundaries; any sentence that still
//! exceeds the chunk limit is recursively split at the natural break point
//! nearest its center, with priority (1) comma, (2) break word, (3) whitespace.
//! A post-pass migrates trailing break words to the start of the next chunk for
//! more natural prosody. All functions are pure — no executor state, no I/O.

/// Break words used as secondary split points for center-break chunking.
const BREAK_WORDS: &[&str] = &[
    "and", "or", "but", "because", "if", "however", "which", "when", "where", "while", "although",
    "since", "unless", "after", "before", "that",
];

/// Split text into chunks at sentence boundaries for TTS.
///
/// Uses a center-break algorithm for oversized sentences: splits at the
/// natural break point nearest to the center of the text, with priority
/// (1) comma, (2) break word, (3) whitespace. Recursive splitting handles
/// chunks that remain too long (max depth 3).
///
/// A post-pass migrates trailing break words from the end of one chunk
/// to the start of the next chunk for more natural prosody.
pub(crate) fn chunk_text_for_tts(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    // Split into sentences (keep delimiter)
    let sentences: Vec<&str> = text.split_inclusive(['.', '!', '?']).collect();

    for sentence in sentences {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }

        if sentence.len() > max_chars {
            // Flush current chunk first
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk = String::new();
            }

            // Center-break split with recursive depth
            let mut sub_chunks = Vec::new();
            center_break_split(sentence, max_chars, 0, &mut sub_chunks);

            // Add all sub-chunks except the last to output, keep last as current
            if let Some(last) = sub_chunks.pop() {
                for sc in sub_chunks {
                    if !sc.is_empty() {
                        chunks.push(sc);
                    }
                }
                current_chunk = last;
            }
        } else if current_chunk.len() + sentence.len() + 1 > max_chars {
            // Current chunk would exceed limit, start new chunk
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
            }
            current_chunk = sentence.to_string();
        } else {
            // Add to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
        }
    }

    // Don't forget the last chunk
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    // Post-pass: migrate trailing break words
    migrate_trailing_break_words(&mut chunks);

    chunks
}

/// Recursively split text at the natural break point nearest to center.
///
/// Priority: (1) comma nearest center, (2) break word nearest center,
/// (3) whitespace nearest center.
fn center_break_split(text: &str, max_chars: usize, depth: usize, out: &mut Vec<String>) {
    const MAX_DEPTH: usize = 3;

    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }

    if trimmed.len() <= max_chars || depth >= MAX_DEPTH {
        out.push(trimmed.to_string());
        return;
    }

    let center = trimmed.len() / 2;

    // Priority 1: comma nearest center
    if let Some(pos) = find_nearest(trimmed, center, |i, _| {
        trimmed.as_bytes().get(i) == Some(&b',')
    }) {
        let left = trimmed[..=pos].trim();
        let right = trimmed[pos + 1..].trim();
        center_break_split(left, max_chars, depth + 1, out);
        center_break_split(right, max_chars, depth + 1, out);
        return;
    }

    // Priority 2: break word nearest center (match at word boundary)
    if let Some((word_start, word_len)) = find_nearest_break_word(trimmed, center) {
        let left = trimmed[..word_start].trim();
        let right = trimmed[word_start + word_len..].trim();
        // Include the break word with the right chunk (post-pass may move it)
        let break_word = &trimmed[word_start..word_start + word_len];
        center_break_split(left, max_chars, depth + 1, out);
        let right_with_word = format!("{} {}", break_word, right);
        center_break_split(right_with_word.trim(), max_chars, depth + 1, out);
        return;
    }

    // Priority 3: whitespace nearest center
    if let Some(pos) = find_nearest(trimmed, center, |i, _| {
        trimmed
            .as_bytes()
            .get(i)
            .is_some_and(|b| b.is_ascii_whitespace())
    }) {
        let left = trimmed[..pos].trim();
        let right = trimmed[pos + 1..].trim();
        center_break_split(left, max_chars, depth + 1, out);
        center_break_split(right, max_chars, depth + 1, out);
        return;
    }

    // No split point found — push as-is
    out.push(trimmed.to_string());
}

/// Find the position nearest to `center` where the predicate matches.
/// Searches outward from center in both directions simultaneously.
fn find_nearest<F>(text: &str, center: usize, pred: F) -> Option<usize>
where
    F: Fn(usize, char) -> bool,
{
    let len = text.len();
    for offset in 0..len {
        // Check right of center
        let right = center + offset;
        if right < len {
            if let Some(ch) = text[right..].chars().next() {
                if pred(right, ch) {
                    return Some(right);
                }
            }
        }
        // Check left of center
        if offset > 0 && offset <= center {
            let left = center - offset;
            if let Some(ch) = text[left..].chars().next() {
                if pred(left, ch) {
                    return Some(left);
                }
            }
        }
    }
    None
}

/// Find the break word nearest to center, returning (start_byte, word_byte_len).
fn find_nearest_break_word(text: &str, center: usize) -> Option<(usize, usize)> {
    let lower = text.to_lowercase();
    let mut best: Option<(usize, usize, usize)> = None; // (start, len, distance)

    for word in BREAK_WORDS {
        let pattern = format!(" {} ", word);
        let mut search_start = 0;
        while let Some(pos) = lower[search_start..].find(&pattern) {
            let abs_pos = search_start + pos + 1; // +1 to skip leading space
            let dist = abs_pos.abs_diff(center);
            if best.is_none() || dist < best.unwrap().2 {
                best = Some((abs_pos, word.len(), dist));
            }
            search_start = search_start + pos + 1;
        }
    }

    best.map(|(start, len, _)| (start, len))
}

/// Post-pass: if a chunk ends with a break word, move it to the start of the next chunk.
fn migrate_trailing_break_words(chunks: &mut [String]) {
    let mut i = 0;
    while i + 1 < chunks.len() {
        let ends_with_break = BREAK_WORDS.iter().any(|w| {
            let chunk = &chunks[i];
            let lower = chunk.to_lowercase();
            lower.ends_with(&format!(" {}", w)) || lower == *w
        });

        if ends_with_break {
            // Find the break word at the end
            let chunk = chunks[i].clone();
            if let Some(last_space) = chunk.rfind(' ') {
                let word = &chunk[last_space + 1..];
                let lower_word = word.to_lowercase();
                if BREAK_WORDS.contains(&lower_word.as_str()) {
                    chunks[i] = chunk[..last_space].trim().to_string();
                    chunks[i + 1] = format!("{} {}", word, chunks[i + 1]);
                }
            }
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_short_input_unchanged() {
        // (a) text under 350 chars returns single chunk
        let text = "Hello world, this is a short sentence that is well under the limit.";
        let chunks = chunk_text_for_tts(text, 350);
        assert_eq!(chunks, vec![text]);
    }

    #[test]
    fn test_chunk_text_exactly_at_limit() {
        let text = "A".repeat(350);
        let chunks = chunk_text_for_tts(&text, 350);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 350);
    }

    #[test]
    fn test_chunk_text_splits_at_sentence_boundaries() {
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = chunk_text_for_tts(text, 20);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "First sentence.");
        assert_eq!(chunks[1], "Second sentence.");
        assert_eq!(chunks[2], "Third sentence.");
    }

    #[test]
    fn test_chunk_text_combines_short_sentences() {
        let text = "Hi. Hello. Hey there.";
        let chunks = chunk_text_for_tts(text, 50);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hi. Hello. Hey there.");
    }

    #[test]
    fn test_chunk_text_handles_exclamation_and_question() {
        let text = "What? Really! Yes.";
        let chunks = chunk_text_for_tts(text, 10);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "What?");
        assert_eq!(chunks[1], "Really!");
        assert_eq!(chunks[2], "Yes.");
    }

    #[test]
    fn test_chunk_text_center_break_comma() {
        // (b) 350+ char text with comma splits at comma nearest center
        // Build a long sentence with a comma near the center
        let left = "The quick brown fox jumped over the lazy dog and ran across the wide green meadow towards the old wooden fence in the distance while the birds sang their morning songs above the tall oak trees lining the path";
        let right = " and then it stopped to rest under the shade of a willow tree by the river where the water flowed gently over smooth stones and the fish swam lazily in the warm afternoon sun as clouds drifted slowly overhead";
        let text = format!("{},{}", left, right);
        assert!(
            text.len() > 350,
            "Test text must exceed 350 chars, got {}",
            text.len()
        );

        let chunks = chunk_text_for_tts(&text, 350);
        assert!(chunks.len() >= 2, "Should split into at least 2 chunks");
        // The first chunk should end with a comma (split at comma nearest center)
        assert!(
            chunks[0].ends_with(','),
            "First chunk should end at comma, got: '{}'",
            chunks[0]
        );
    }

    #[test]
    fn test_chunk_text_center_break_word() {
        // (c) 350+ char text with break word and no comma splits at break word nearest center
        let text = "The quick brown fox jumped over the lazy dog running across the wide green meadow towards the old wooden fence in the distance while birds sang their morning songs above the tall oak trees however the gentle breeze carried the sweet scent of wildflowers across the rolling hills and through the valleys where deer grazed peacefully in the golden light of the setting sun painting everything in warm hues";
        assert!(
            text.len() > 350,
            "Test text must exceed 350 chars, got {}",
            text.len()
        );
        // Verify no commas in the text
        assert!(!text.contains(','), "Test text should have no commas");

        let chunks = chunk_text_for_tts(text, 350);
        assert!(chunks.len() >= 2, "Should split into at least 2 chunks");
        // After post-pass migration, chunks[1] should start with the break word "however"
        let second_lower = chunks[1].to_lowercase();
        let starts_with_break = BREAK_WORDS.iter().any(|w| second_lower.starts_with(w));
        assert!(
            starts_with_break,
            "Second chunk should start with a break word after post-pass, got: '{}'",
            &chunks[1][..chunks[1].len().min(40)]
        );
    }

    #[test]
    fn test_chunk_text_center_break_whitespace() {
        // (d) 350+ char text with only whitespace splits at space nearest center
        // No commas, no break words — only plain words
        let text = "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll mmmm nnnn oooo pppp qqqq rrrr ssss tttt uuuu vvvv wwww xxxx yyyy zzzz aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll mmmm nnnn oooo pppp qqqq rrrr ssss tttt uuuu vvvv wwww xxxx yyyy zzzz aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll mmmm nnnn oooo pppp qqqq rrrr ssss tttt uuuu";
        assert!(
            text.len() > 350,
            "Test text must exceed 350 chars, got {}",
            text.len()
        );
        assert!(!text.contains(','), "No commas");

        let chunks = chunk_text_for_tts(text, 350);
        assert!(chunks.len() >= 2, "Should split into at least 2 chunks");
        // Each chunk should be trimmed and non-empty
        for chunk in &chunks {
            assert!(!chunk.is_empty(), "Chunks should not be empty");
            assert_eq!(chunk.as_str(), chunk.trim(), "Chunks should be trimmed");
        }
    }

    #[test]
    fn test_chunk_text_multi_sentence_long_first() {
        // (e) multi-sentence text where first sentence >350 chars gets center-break split
        //     while short second sentence stays intact
        let long_sentence = "The magnificent cathedral stood tall against the stormy sky its ancient stone walls bearing witness to centuries of history while gargoyles perched on every corner watched over the bustling city below where merchants sold their wares in the cobblestone market square filled with the aroma of freshly baked bread and exotic spices brought by traders from distant lands across vast oceans and treacherous mountain passes.";
        let short_sentence = " A bird sang nearby.";
        let text = format!("{}{}", long_sentence, short_sentence);
        assert!(
            long_sentence.len() > 350,
            "First sentence must exceed 350 chars"
        );

        let chunks = chunk_text_for_tts(&text, 350);
        assert!(chunks.len() >= 2, "Should split into at least 2 chunks");
        // The last chunk should contain the short sentence
        let last = chunks.last().unwrap();
        assert!(
            last.contains("A bird sang nearby"),
            "Short sentence should be intact in last chunk, got: '{}'",
            last
        );
    }

    #[test]
    fn test_chunk_text_empty_input() {
        let chunks = chunk_text_for_tts("", 350);
        assert!(chunks.is_empty() || chunks == vec![""]);
    }

    #[test]
    fn test_chunk_text_whitespace_only() {
        let chunks = chunk_text_for_tts("   ", 350);
        assert!(chunks.is_empty() || chunks.iter().all(|c| c.trim().is_empty()));
    }

    #[test]
    fn test_chunk_text_preserves_content() {
        let text =
            "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
        let chunks = chunk_text_for_tts(text, 50);
        let rejoined: String = chunks.join(" ");
        assert!(rejoined.contains("quick"));
        assert!(rejoined.contains("fox"));
        assert!(rejoined.contains("liquor"));
    }

    #[test]
    fn test_chunk_text_post_pass_break_word_migration() {
        // Verify the post-pass migration: if chunk ends with break word, it moves to next chunk
        // Use a shorter max to make behavior predictable
        let text =
            "The fox ran fast and the dog chased it quickly through the woods and over the hill.";
        let chunks = chunk_text_for_tts(text, 30);
        // No chunk should end with a standalone break word (post-pass moves them)
        for (i, chunk) in chunks.iter().enumerate() {
            if i + 1 < chunks.len() {
                let lower = chunk.to_lowercase();
                for w in BREAK_WORDS {
                    assert!(
                        !lower.ends_with(&format!(" {}", w)),
                        "Chunk {} should not end with break word '{}': '{}'",
                        i,
                        w,
                        chunk
                    );
                }
            }
        }
    }
}
