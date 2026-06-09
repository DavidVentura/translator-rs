//! Lightweight sentence splitter — replaces slimt's PCRE2-backed Splitter.
//!
//! Splits on terminal punctuation (`.`, `!`, `?`, including ellipses) when
//! followed by whitespace and a Unicode uppercase letter — optionally with an
//! opening delimiter (`"`, `“`, `(`, `¿`, `«`, …) in front of that letter, so
//! a new sentence that opens with quoted dialogue still splits (`bed. “In
//! judging…”`). This matches ssplit-cpp's regex-fallback heuristic (the path it
//! takes when no abbreviation prefix file is loaded), which works well on
//! modern web / ebook prose. Edge cases like "Mr. Smith. He went..." will
//! mis-split; we'd close those by adding a curated abbreviation list later if a
//! real document hits regressions. The split layer used to live in slimt and
//! pulled PCRE2 into the build (~1 MB binary, 13 MB source); moving it here
//! removes both.
//!
//! The splitter returns `&str` slices borrowed from the input. Callers
//! recover each sentence's byte offset via pointer arithmetic
//! (`slice.as_ptr() - input.as_ptr()`); the alignment-stitching code in
//! `BergamotEngine` uses that to map per-sentence alignments back into
//! the original input's coordinate space.

use std::sync::OnceLock;

use regex::Regex;

fn boundary_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // One or more of `.`, `!`, `?` (handles ellipses and `!?` mixes),
        // optionally followed by *closing* delimiters that belong to the
        // sentence just ended (`wake!”`, `(done.)`), then whitespace, then the
        // start of the next sentence: any run of *opening* delimiters
        // (straight/curly quotes, parens, brackets, Spanish `¿`/`¡`,
        // guillemets) followed by a Unicode uppercase letter (covers accented
        // Spanish/French/etc. and full-width CJK Latin).
        //
        // The trailing uppercase letter is the load-bearing guard against
        // false splits on the overloaded `.` (decimals `3.14`, `e.g.`,
        // lowercase parentheticals `(or so I thought)`). The delimiter runs
        // only let quotes/brackets sit around the boundary, so dialogue like
        // `tomahawk! “Queequeg!” At length…` splits while `it. (or so…)` does
        // not. Group 1 ends after the closing delimiters (they stay with the
        // current sentence); group 2 starts at the first opening delimiter, so
        // it joins the next sentence.
        Regex::new(r#"([.!?]+["'”’)\]»]*)\s+(["'“‘(\[¿¡«]*\p{Lu})"#)
            .expect("valid sentence-boundary regex")
    })
}

/// Split `text` into sentence-sized substrings, each ending after its
/// terminal punctuation. Inter-sentence whitespace is dropped from the
/// output (the boundary point is where the next sentence's first
/// uppercase letter begins). Returns an empty vec if `text` is empty
/// or whitespace-only; otherwise returns at least one slice.
pub fn split_sentences(text: &str) -> Vec<&str> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut last_end = 0usize;
    for caps in boundary_regex().captures_iter(text) {
        let punct = caps.get(1).expect("group 1 always present");
        let next = caps.get(2).expect("group 2 always present");
        let piece = &text[last_end..punct.end()];
        if !piece.trim().is_empty() {
            out.push(piece);
        }
        last_end = next.start();
    }
    if last_end < text.len() {
        let tail = &text[last_end..];
        if !tail.trim().is_empty() {
            out.push(tail);
        }
    }
    if out.is_empty() {
        out.push(text);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_offset(slice: &str, original: &str) -> usize {
        let base = original.as_ptr() as usize;
        let here = slice.as_ptr() as usize;
        here - base
    }

    #[test]
    fn empty_returns_empty() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("   \n  ").is_empty());
    }

    #[test]
    fn single_sentence_round_trips() {
        let s = "Hello world.";
        assert_eq!(split_sentences(s), vec![s]);
    }

    #[test]
    fn paragraph_with_three_sentences() {
        let p = "The cat eats fish. Hello world. Done.";
        let parts = split_sentences(p);
        assert_eq!(parts.len(), 3);
        // Slices retain their byte offsets within the original string;
        // the alignment-combine logic relies on this.
        assert_eq!(byte_offset(parts[0], p), 0);
        assert_eq!(parts[0], "The cat eats fish.");
        assert_eq!(parts[1], "Hello world.");
        assert_eq!(parts[2], "Done.");
        // The space between sentences is dropped from output.
        assert_eq!(
            byte_offset(parts[1], p),
            "The cat eats fish. ".len(),
            "second sentence must start where the original 'Hello' is"
        );
    }

    #[test]
    fn ellipsis_counts_as_one_boundary() {
        // "..." followed by whitespace + Capital is one boundary, not three.
        let p = "Wait... Then it happened.";
        let parts = split_sentences(p);
        assert_eq!(parts, vec!["Wait...", "Then it happened."]);
    }

    #[test]
    fn mixed_terminal_punctuation() {
        let p = "Really?! No way. Yes.";
        let parts = split_sentences(p);
        assert_eq!(parts, vec!["Really?!", "No way.", "Yes."]);
    }

    #[test]
    fn unicode_uppercase_starts_new_sentence() {
        // Spanish-style accented uppercase. Without `\p{Lu}` matching, the
        // regex would miss boundaries before words starting with Á/É.
        let p = "Hola mundo. Él vino. Última cosa.";
        let parts = split_sentences(p);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[1], "Él vino.");
        assert_eq!(parts[2], "Última cosa.");
    }

    #[test]
    fn lowercase_after_period_does_not_split() {
        // "e.g." inside a sentence shouldn't trigger a split because the
        // following character is lowercase. We don't try to handle the
        // "Mr. Smith" case (capital after) — that's the known limitation.
        let p = "We use abbreviations e.g. like this.";
        assert_eq!(split_sentences(p), vec![p]);
    }

    #[test]
    fn trailing_punctuation_no_capital_keeps_one_sentence() {
        let p = "He said hi. ok";
        // No capital after the period, so no split.
        assert_eq!(split_sentences(p), vec![p]);
    }

    #[test]
    fn unterminated_tail_returned() {
        // Last sentence has no terminal punctuation — still gets emitted.
        let p = "First. Second";
        assert_eq!(split_sentences(p), vec!["First.", "Second"]);
    }

    #[test]
    fn known_limitation_mister_splits_falsely() {
        // Documents the regex-fallback limitation. If we ship an
        // abbreviation list we'd merge these back into one sentence.
        let p = "Mr. Smith arrived.";
        let parts = split_sentences(p);
        assert_eq!(
            parts,
            vec!["Mr.", "Smith arrived."],
            "abbreviation handling not implemented; matches ssplit-cpp's regex fallback"
        );
    }

    #[test]
    fn quoted_dialogue_after_period_splits() {
        // The dialogue case that previously merged: an opening curly quote
        // sits between the period and the capital. The quote joins the new
        // sentence.
        let p = "He toasts for bed. “In judging of that wind,” he said.";
        let parts = split_sentences(p);
        assert_eq!(
            parts,
            vec!["He toasts for bed.", "“In judging of that wind,” he said."]
        );
        assert!(
            parts[1].starts_with('“'),
            "quote belongs to the new sentence"
        );
    }

    #[test]
    fn bang_then_quote_splits() {
        // Splits twice: at the opening quote, and again after the closing
        // quote — the bracketed exclamation is its own sentence.
        let p = "with a tomahawk! “Queequeg!” At length he woke.";
        assert_eq!(
            split_sentences(p),
            vec!["with a tomahawk!", "“Queequeg!”", "At length he woke."]
        );
    }

    #[test]
    fn closing_quote_then_capital_splits() {
        // Terminal punctuation inside a closing quote (`wake!” At…`): the
        // close-quote stays with the sentence that ended.
        let p = "“Queequeg, wake!” At length he stirred.";
        let parts = split_sentences(p);
        assert_eq!(parts, vec!["“Queequeg, wake!”", "At length he stirred."]);
        assert!(
            parts[0].ends_with('”'),
            "closing quote stays with its sentence"
        );
    }

    #[test]
    fn straight_quote_and_paren_openers_split() {
        assert_eq!(
            split_sentences("Done. \"Next one starts here.\""),
            vec!["Done.", "\"Next one starts here.\""]
        );
        assert_eq!(
            split_sentences("He left. (See the footnote.)"),
            vec!["He left.", "(See the footnote.)"]
        );
    }

    #[test]
    fn spanish_inverted_opener_splits() {
        // `¿`/`¡` precede the capital in Spanish; both should still split.
        assert_eq!(
            split_sentences("Dijo algo. ¿Vienes conmigo?"),
            vec!["Dijo algo.", "¿Vienes conmigo?"]
        );
    }

    #[test]
    fn lowercase_opener_does_not_split() {
        // The reason the boundary is "openers* THEN capital" rather than
        // "capital OR punctuation": a delimiter followed by lowercase is a
        // mid-sentence aside, not a new sentence. A plain `\p{Punct}` branch
        // would wrongly split all of these.
        assert_eq!(
            split_sentences("I saw it. (or so I thought) and moved on."),
            vec!["I saw it. (or so I thought) and moved on."]
        );
        assert_eq!(
            split_sentences("She paused. \"and then nothing\""),
            vec!["She paused. \"and then nothing\""]
        );
    }
}
