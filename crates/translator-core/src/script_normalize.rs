//! Post-OCR repair for Cyrillic/Latin script confusion.
//!
//! PP-OCR's `cyrillic` and `eslav` recognizers can decode an ambiguous glyph as
//! the Latin twin (e.g. `A` instead of `А`, `O` instead of `О`) even when the
//! surrounding word is clearly Cyrillic. The leaked Latin codepoint breaks
//! downstream language detection and translation — "ОПAСНО" with a Latin `A`
//! looks like a junk mixed-script token instead of Russian "ОПАСНО".
//!
//! The reverse happens too: a Latin word read by the same recognizers can come
//! out with a Cyrillic twin inside it ("Onа" with a Cyrillic `а`), which breaks
//! translation of that word the same way.
//!
//! Each word is assigned the script of the letters that have no twin in the
//! other script (`ж`, `и` are Cyrillic; `t`, `h` are Latin), and its twins are
//! rewritten into that script. A word made only of twins ("My", "ТОР") reads
//! the same in either script, so it follows the nearest assigned word around it.
//!
//! Intentionally mixed-script words ("WD-зал", "VIP-зал") survive: they carry
//! twin-less letters from both scripts, so they are left untouched.

/// Latin → Cyrillic visual twins that PP-OCR routinely confuses on Cyrillic
/// strips. Kept short on purpose; this is not the full Unicode confusables
/// table.
const LATIN_TO_CYRILLIC: &[(char, char)] = &[
    ('A', 'А'),
    ('B', 'В'),
    ('C', 'С'),
    ('E', 'Е'),
    ('H', 'Н'),
    ('I', 'І'),
    ('J', 'Ј'),
    ('K', 'К'),
    ('M', 'М'),
    ('O', 'О'),
    ('P', 'Р'),
    ('S', 'Ѕ'),
    ('T', 'Т'),
    ('X', 'Х'),
    ('Y', 'У'),
    ('a', 'а'),
    ('c', 'с'),
    ('e', 'е'),
    ('i', 'і'),
    ('j', 'ј'),
    ('o', 'о'),
    ('p', 'р'),
    ('s', 'ѕ'),
    ('x', 'х'),
    ('y', 'у'),
];

fn latin_to_cyrillic(ch: char) -> Option<char> {
    LATIN_TO_CYRILLIC
        .iter()
        .find(|(latin, _)| *latin == ch)
        .map(|(_, cyrl)| *cyrl)
}

fn cyrillic_to_latin(ch: char) -> Option<char> {
    LATIN_TO_CYRILLIC
        .iter()
        .find(|(_, cyrl)| *cyrl == ch)
        .map(|(latin, _)| *latin)
}

fn is_cyrillic_letter(ch: char) -> bool {
    matches!(ch as u32, 0x0400..=0x04FF | 0x0500..=0x052F)
}

fn is_latin_letter(ch: char) -> bool {
    ch.is_ascii_alphabetic() || (ch.is_alphabetic() && matches!(ch as u32, 0x00C0..=0x024F))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WordScript {
    Latin,
    Cyrillic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WordKind {
    Script(WordScript),
    /// Only letters with a twin in the other script.
    Ambiguous,
    /// Twin-less letters of both scripts: written mixed on purpose.
    Mixed,
    NoLetters,
}

fn word_kind(word: &str) -> WordKind {
    let mut latin_only = false;
    let mut cyrillic_only = false;
    let mut twins = false;
    for ch in word.chars() {
        if is_cyrillic_letter(ch) {
            match cyrillic_to_latin(ch) {
                Some(_) => twins = true,
                None => cyrillic_only = true,
            }
        } else if is_latin_letter(ch) {
            match latin_to_cyrillic(ch) {
                Some(_) => twins = true,
                None => latin_only = true,
            }
        }
    }
    match (latin_only, cyrillic_only, twins) {
        (true, true, _) => WordKind::Mixed,
        (true, false, _) => WordKind::Script(WordScript::Latin),
        (false, true, _) => WordKind::Script(WordScript::Cyrillic),
        (false, false, true) => WordKind::Ambiguous,
        (false, false, false) => WordKind::NoLetters,
    }
}

/// Whether clause punctuation separates word `i` from word `i + 1`, so neither
/// lends its script to the other ("Tense & Cases: Они" keeps "Cases" with "Tense").
fn clause_break_after(words: &[&str], i: usize) -> bool {
    let ends_clause = words[i]
        .chars()
        .next_back()
        .is_some_and(|c| matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '»' | '”'));
    let opens_clause = words[i + 1]
        .chars()
        .next()
        .is_some_and(|c| matches!(c, '(' | '[' | '«' | '“'));
    ends_clause || opens_clause
}

/// The script of the nearest word in the same clause that has one; `None` when
/// the two sides disagree at the same distance or no word has a script.
fn script_from_neighbours(words: &[&str], kinds: &[WordKind], index: usize) -> Option<WordScript> {
    let mut left = None;
    for i in (0..index).rev() {
        if clause_break_after(words, i) {
            break;
        }
        if let WordKind::Script(script) = kinds[i] {
            left = Some((i, script));
            break;
        }
    }
    let mut right = None;
    for (i, kind) in kinds.iter().enumerate().skip(index + 1) {
        if clause_break_after(words, i - 1) {
            break;
        }
        if let WordKind::Script(script) = kind {
            right = Some((i, *script));
            break;
        }
    }
    match (left, right) {
        (None, None) => None,
        (Some((_, script)), None) | (None, Some((_, script))) => Some(script),
        (Some((li, ls)), Some((ri, rs))) => {
            if ls == rs {
                return Some(ls);
            }
            match (index - li).cmp(&(ri - index)) {
                std::cmp::Ordering::Less => Some(ls),
                std::cmp::Ordering::Greater => Some(rs),
                std::cmp::Ordering::Equal => None,
            }
        }
    }
}

fn rewrite_word(word: &str, script: WordScript, out: &mut String) {
    for ch in word.chars() {
        let twin = match script {
            WordScript::Cyrillic if is_latin_letter(ch) => latin_to_cyrillic(ch),
            WordScript::Latin if is_cyrillic_letter(ch) => cyrillic_to_latin(ch),
            _ => None,
        };
        out.push(twin.unwrap_or(ch));
    }
}

/// Rewrite Latin/Cyrillic twin letters so every word is written in one script.
///
/// See module docs for the heuristic. Whitespace, punctuation, digits and any
/// non-letter characters pass through unchanged. The output preserves the
/// original whitespace layout exactly.
pub fn repair_latin_cyrillic_confusables(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let kinds: Vec<WordKind> = words.iter().map(|w| word_kind(w)).collect();
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    for (index, word) in words.iter().enumerate() {
        let at = rest
            .find(word)
            .expect("split_whitespace word is in the text");
        out.push_str(&rest[..at]);
        rest = &rest[at + word.len()..];
        let script = match kinds[index] {
            WordKind::Script(script) => Some(script),
            WordKind::Ambiguous => script_from_neighbours(&words, &kinds, index),
            WordKind::Mixed | WordKind::NoLetters => None,
        };
        match script {
            Some(script) => rewrite_word(word, script, &mut out),
            None => out.push_str(word),
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repairs_single_latin_letter_in_cyrillic_word() {
        // Latin A and Latin BOT inside Cyrillic words.
        assert_eq!(
            repair_latin_cyrillic_confusables("ОПAСНО ЗА ЖИBOTA"),
            "ОПАСНО ЗА ЖИВОТА"
        );
    }

    #[test]
    fn leaves_pure_cyrillic_untouched() {
        assert_eq!(
            repair_latin_cyrillic_confusables("ВНИМАНИЕ ВИСОКО НАПРЕЖЕНИЕ"),
            "ВНИМАНИЕ ВИСОКО НАПРЕЖЕНИЕ"
        );
    }

    #[test]
    fn leaves_pure_latin_untouched() {
        assert_eq!(
            repair_latin_cyrillic_confusables("HELLO WORLD"),
            "HELLO WORLD"
        );
    }

    #[test]
    fn preserves_intentional_mixed_script_when_latin_run_has_non_confusable() {
        // "WD" is not a confusable run (W has no Cyrillic twin), so the
        // mixed-script word is left as-is.
        assert_eq!(repair_latin_cyrillic_confusables("WD-зал"), "WD-зал");
    }

    #[test]
    fn does_not_touch_lone_latin_words_next_to_cyrillic_words() {
        assert_eq!(
            repair_latin_cyrillic_confusables("купи iPhone сегодня"),
            "купи iPhone сегодня"
        );
    }

    #[test]
    fn preserves_whitespace_layout() {
        assert_eq!(
            repair_latin_cyrillic_confusables("  ОПAСНО\tЗА\nЖИBOTA  "),
            "  ОПАСНО\tЗА\nЖИВОТА  "
        );
    }

    #[test]
    fn handles_lowercase_confusables() {
        // "опacно" with Latin 'a' and 'c'.
        assert_eq!(repair_latin_cyrillic_confusables("опacно"), "опасно");
    }

    #[test]
    fn leaves_punctuation_alone() {
        assert_eq!(repair_latin_cyrillic_confusables("«ОПAСНО!»"), "«ОПАСНО!»");
    }

    #[test]
    fn repairs_cyrillic_twins_inside_latin_words() {
        assert_eq!(
            repair_latin_cyrillic_confusables("(Onа pуot vodu)"),
            "(Ona pyot vodu)"
        );
    }

    #[test]
    fn repairs_a_latin_word_inside_cyrillic_text() {
        // "Cases" with Cyrillic С, а, ѕ, е, ѕ next to Russian words.
        assert_eq!(
            repair_latin_cyrillic_confusables("Past Tense & Саѕеѕ: Они сказали"),
            "Past Tense & Cases: Они сказали"
        );
    }

    #[test]
    fn twin_only_word_follows_its_neighbours() {
        // "ТОР" has only twins; it takes the script of the Russian around it, and
        // "My" (Cyrillic М and у) takes the Latin of the transliteration around it.
        assert_eq!(
            repair_latin_cyrillic_confusables("в TOP десять"),
            "в ТОР десять"
        );
        assert_eq!(
            repair_latin_cyrillic_confusables("Му pishem pisma"),
            "My pishem pisma"
        );
    }

    #[test]
    fn an_opening_bracket_ties_a_twin_only_word_to_what_follows() {
        assert_eq!(
            repair_latin_cyrillic_confusables("Мы пишем письма (Му pishem pisma)"),
            "Мы пишем письма (My pishem pisma)"
        );
    }

    #[test]
    fn twin_only_word_between_disagreeing_neighbours_is_left_alone() {
        assert_eq!(
            repair_latin_cyrillic_confusables("письма Му pishem"),
            "письма Му pishem"
        );
    }
}
