use icu_experimental::transliterate::Transliterator;
use icu_locale_core::Locale;
use unicode_script::{Script, UnicodeScript};

use crate::api::{LanguageCode, ScriptCode};

fn make_transliterator(source_script: &str) -> Option<Transliterator> {
    let locale_str = format!("und-Latn-t-und-{}", source_script.to_lowercase());
    let locale: Locale = locale_str.parse().ok()?;
    Transliterator::try_new(&locale).ok()
}

fn transliterate(text: &str, source_script: &ScriptCode) -> Option<String> {
    match source_script.as_str() {
        "Jpan" => {
            let kana = make_transliterator("Kana")?;
            let hira = make_transliterator("Hira")?;
            let result = kana.transliterate(text.to_string());
            Some(hira.transliterate(result))
        }
        _ => {
            let t = make_transliterator(source_script.as_str())?;
            Some(t.transliterate(text.to_string()))
        }
    }
}

/// ICU source script subtag that romanizes a given run, or `None` for runs we
/// leave untouched (Latin, punctuation, marks). Scripts ICU has no transform
/// for still fall back to pass-through via `transliterate` returning `None`.
fn romanizable_source_code(script: Script) -> Option<String> {
    use crate::script::Script as S;
    match S::from(script) {
        S::Latin | S::Common | S::Inherited | S::Other => None,
        // Both kana share the Japanese transform (katakana then hiragana).
        S::Hiragana | S::Katakana => Some("Jpan".to_owned()),
        other => Some(other.iso15924().to_owned()),
    }
}

/// Punctuation, whitespace, digits and combining marks carry no script of their
/// own; they belong to whichever run surrounds them.
fn is_neutral(script: Script) -> bool {
    matches!(script, Script::Common | Script::Inherited)
}

/// Assign every char a concrete script, letting neutral chars inherit from the
/// run they sit in (preceding run first, leading neutrals from the following
/// one).
fn resolve_char_scripts(text: &str) -> Vec<(char, Script)> {
    let mut chars: Vec<(char, Script)> = text.chars().map(|ch| (ch, ch.script())).collect();

    let mut last_concrete: Option<Script> = None;
    for (_, script) in chars.iter_mut() {
        if is_neutral(*script) {
            if let Some(prev) = last_concrete {
                *script = prev;
            }
        } else {
            last_concrete = Some(*script);
        }
    }

    if let Some(first_concrete) = chars.iter().map(|(_, s)| *s).find(|s| !is_neutral(*s)) {
        for (_, script) in chars.iter_mut() {
            if !is_neutral(*script) {
                break;
            }
            *script = first_concrete;
        }
    }

    chars
}

/// Romanize the non-Latin runs of a mixed-script string, leaving Latin text,
/// punctuation and whitespace as they are. Used when the requested output is a
/// Latin-script voice but the text carries foreign names ("Your line 'Сливница
/// - Летище София' is arriving" → "Your line 'Slivnitsa - Letishte Sofiya' is
/// arriving").
pub fn transliterate_mixed_to_latin(text: &str) -> String {
    let resolved = resolve_char_scripts(text);

    let mut out = String::with_capacity(text.len());
    let mut run = String::new();
    let mut run_script: Option<Script> = None;

    for (ch, script) in resolved {
        if run_script != Some(script) {
            flush_run(&mut out, &run, run_script);
            run.clear();
            run_script = Some(script);
        }
        run.push(ch);
    }
    flush_run(&mut out, &run, run_script);

    out
}

fn flush_run(out: &mut String, run: &str, run_script: Option<Script>) {
    if run.is_empty() {
        return;
    }
    let romanized = run_script
        .and_then(romanizable_source_code)
        .and_then(|source| transliterate(run, &ScriptCode::from(source)));
    match romanized {
        Some(romanized) => out.push_str(&romanized),
        None => out.push_str(run),
    }
}

fn transliterate_with_policy(
    text: &str,
    language_code: &LanguageCode,
    source_script: &ScriptCode,
    target_script: &ScriptCode,
    japanese_preprocessed: Option<&str>,
) -> Option<String> {
    if source_script == target_script {
        return None;
    }

    let input = match language_code.as_str() {
        "ja" => japanese_preprocessed.unwrap_or(text),
        _ => text,
    };

    transliterate(input, source_script)
}

pub fn transliterate_with_policy_for_language(
    text: &str,
    language_code: &LanguageCode,
    source_script: &ScriptCode,
    target_script: &ScriptCode,
    japanese_dict_path: Option<&str>,
    japanese_spaced: bool,
) -> Option<String> {
    let normalized = text.trim();
    if normalized.is_empty() || normalized.is_ascii() {
        return None;
    }

    let japanese_preprocessed = if language_code.as_str() == "ja" {
        preprocess_japanese(normalized, japanese_dict_path, japanese_spaced)
    } else {
        None
    };

    transliterate_with_policy(
        normalized,
        language_code,
        source_script,
        target_script,
        japanese_preprocessed.as_deref(),
    )
}

#[cfg(feature = "mucab")]
fn preprocess_japanese(
    text: &str,
    dict_path: Option<&str>,
    japanese_spaced: bool,
) -> Option<String> {
    let dict_path = dict_path?;
    if dict_path.is_empty() {
        return None;
    }
    crate::mucab::transliterate_with_path(dict_path, text, japanese_spaced).ok()
}

#[cfg(not(feature = "mucab"))]
fn preprocess_japanese(
    _text: &str,
    _dict_path: Option<&str>,
    _japanese_spaced: bool,
) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translit(script: &str, text: &str) -> String {
        transliterate(text, &ScriptCode::from(script)).unwrap()
    }

    #[test]
    fn test_cyrillic() {
        assert_eq!(translit("Cyrl", "Привет мир"), "Privet mir");
    }

    #[test]
    fn test_arabic() {
        assert_eq!(translit("Arab", "مرحبا"), "mrḥbạ");
    }

    #[test]
    fn test_greek() {
        assert_eq!(translit("Grek", "Αθήνα"), "Athḗna");
    }

    #[test]
    fn test_devanagari() {
        assert_eq!(translit("Deva", "नमस्ते"), "namastē");
    }

    #[test]
    fn test_hangul() {
        assert_eq!(translit("Hang", "안녕하세요"), "annyeonghaseyo");
    }

    #[test]
    fn test_hebrew() {
        assert_eq!(translit("Hebr", "שלום"), "şlwm");
    }

    #[test]
    fn test_bengali() {
        assert_eq!(translit("Beng", "নমস্কার"), "namaskāra");
    }

    #[test]
    fn test_tamil() {
        assert_eq!(translit("Taml", "வணக்கம்"), "vaṇakkam");
    }

    #[test]
    fn test_telugu() {
        assert_eq!(translit("Telu", "నమస్కారం"), "namaskāraṁ");
    }

    #[test]
    fn test_han_simplified() {
        assert_eq!(translit("Hans", "你好世界"), "nǐ hǎo shì jiè");
    }

    #[test]
    fn test_han_traditional() {
        assert_eq!(translit("Hant", "你好世界"), "nǐ hǎo shì jiè");
    }

    #[test]
    fn test_japanese_hiragana() {
        assert_eq!(translit("Jpan", "こんにちは"), "kon'nichiha");
    }

    #[test]
    fn test_japanese_katakana() {
        assert_eq!(translit("Jpan", "カタカナ"), "katakana");
    }

    #[test]
    fn test_japanese_mixed_kana() {
        let result = translit("Jpan", "ひらがなカタカナ");
        assert!(result.contains("hiragana"));
        assert!(result.contains("katakana"));
    }

    #[test]
    fn test_jpan_preserves_kanji() {
        // After mucab, some kanji may remain unconverted.
        // Verify they pass through unchanged.
        assert_eq!(translit("Jpan", "東京 の ひと"), "東京 no hito");
    }

    #[test]
    fn test_jpan_simulated_mucab_output() {
        // mucab converts kanji→hiragana and adds spaces.
        // Simulate: "東京タワー" → mucab → "とうきょう タワー"
        // Then ICU should produce: "toukyou tawā"
        assert_eq!(translit("Jpan", "とうきょう タワー"), "toukyou tawā");
    }

    #[test]
    fn test_mixed_cyrillic_in_latin_sentence() {
        assert_eq!(
            transliterate_mixed_to_latin("Your line 'Сливница - Летище София' is arriving"),
            "Your line 'Slivnica - Letiŝe Sofiâ' is arriving"
        );
    }

    #[test]
    fn test_mixed_pure_latin_unchanged() {
        assert_eq!(
            transliterate_mixed_to_latin("Just a plain ASCII sentence."),
            "Just a plain ASCII sentence."
        );
    }

    #[test]
    fn test_mixed_pure_cyrillic() {
        assert_eq!(transliterate_mixed_to_latin("Привет мир"), "Privet mir");
    }

    #[test]
    fn test_mixed_multiple_scripts() {
        assert_eq!(
            transliterate_mixed_to_latin("hello Привет and Αθήνα done"),
            "hello Privet and Athḗna done"
        );
    }

    #[test]
    fn test_latin_is_none() {
        assert!(transliterate("Hello", &ScriptCode::from("Latn")).is_none());
    }

    #[test]
    fn test_policy_skips_same_script() {
        assert!(
            transliterate_with_policy(
                "Hello",
                &LanguageCode::from("en"),
                &ScriptCode::from("Latn"),
                &ScriptCode::from("Latn"),
                None
            )
            .is_none()
        );
    }

    #[test]
    fn test_policy_uses_japanese_preprocessed_text() {
        assert_eq!(
            transliterate_with_policy(
                "東京タワー",
                &LanguageCode::from("ja"),
                &ScriptCode::from("Jpan"),
                &ScriptCode::from("Latn"),
                Some("とうきょう タワー")
            )
            .unwrap(),
            "toukyou tawā"
        );
    }
}
