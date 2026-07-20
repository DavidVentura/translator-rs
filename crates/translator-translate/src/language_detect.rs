use cld2::{Format, Hints, Reliable, detect_language_ext};
use translator_core::api::{LanguageCode, ScriptedLanguage};
use translator_core::script::Script;

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct DetectionResult {
    pub language: String,
    pub is_reliable: bool,
    pub confidence: i32,
}

pub fn detect_language(text: &str, hint: Option<&LanguageCode>) -> Option<DetectionResult> {
    let hints = Hints {
        content_language: hint.map(LanguageCode::as_str),
        ..Default::default()
    };
    let detected = detect_language_ext(text, Format::Text, &hints);
    let language = detected.language?.0.to_string();
    let is_reliable = detected.reliability == Reliable;
    let confidence = detected
        .scores
        .first()
        .map(|score| score.percent as i32)
        .unwrap_or(0);

    Some(DetectionResult {
        language,
        is_reliable,
        confidence,
    })
}

pub fn detect_language_robust_code(
    text: &str,
    hint: Option<&LanguageCode>,
    available_languages: &[ScriptedLanguage],
) -> Option<LanguageCode> {
    if text.trim().is_empty() {
        return None;
    }

    // cld2 confuses same-script languages on short input (सुप्रभात scores as
    // Sanskrit, धन्यवाद as Marathi, नमस्ते as nothing), and hinting only echoes
    // whatever hint it is given, so it can't disambiguate. The text's own script
    // narrows the candidates to the supported languages that use it; when exactly
    // one does, that is the answer regardless of what cld2 thinks.
    let candidates: Vec<&LanguageCode> = match Script::dominant(text) {
        None => available_languages.iter().map(|lang| &lang.code).collect(),
        Some(script) => {
            let same_script: Vec<&LanguageCode> = available_languages
                .iter()
                .filter(|lang| lang.script == script)
                .map(|lang| &lang.code)
                .collect();
            match same_script.len() {
                0 => return None,
                1 => return Some(same_script[0].clone()),
                _ => same_script,
            }
        }
    };

    if let Some(detected) = detect_language(text, hint) {
        if detected.is_reliable
            && candidates
                .iter()
                .any(|code| code.as_str() == detected.language)
        {
            return Some(LanguageCode::from(detected.language));
        }
    }

    // cld2 gave nothing usable, so all that is left is asking which candidates it
    // will rubber-stamp when forced. Short input in a language family gets several
    // (`hej hur mar du` echoes both `da` and `sv`), and taking the first would make
    // catalog order the tiebreaker. Only a lone echo is evidence of anything.
    let echoed: Vec<&LanguageCode> = candidates
        .into_iter()
        .filter(|code| {
            matches!(
                detect_language(text, Some(*code)),
                Some(detected) if detected.is_reliable && detected.language == code.as_str()
            )
        })
        .collect();

    match echoed.as_slice() {
        [only] => Some((*only).clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Routes through the same ISO 15924 parse production uses, so the
    /// composite subtags (`Jpan`, `Hans`) resolve here exactly as they do from
    /// a real catalog.
    fn languages(list: &[&str]) -> Vec<ScriptedLanguage> {
        list.iter()
            .map(|code| ScriptedLanguage {
                code: LanguageCode::from(*code),
                script: Script::from_iso15924(match *code {
                    "hi" => "Deva",
                    "bn" => "Beng",
                    "el" => "Grek",
                    "he" => "Hebr",
                    "th" => "Thai",
                    "ko" => "Hang",
                    "ja" => "Jpan",
                    "zh" => "Hans",
                    "ru" | "uk" => "Cyrl",
                    _ => "Latn",
                })
                .expect("catalog script parses"),
            })
            .collect()
    }

    fn detect(text: &str, available: &[&str]) -> Option<String> {
        detect_language_robust_code(text, None, &languages(available))
            .map(|c| c.as_str().to_string())
    }

    fn detect_hinted(text: &str, hint: &str, available: &[&str]) -> Option<String> {
        let hint = LanguageCode::from(hint);
        detect_language_robust_code(text, Some(&hint), &languages(available))
            .map(|c| c.as_str().to_string())
    }

    #[test]
    fn single_script_language_wins_over_cld2_guess() {
        // cld2 alone scores these as Sanskrit / Marathi / nothing; Devanagari maps
        // to only `hi` among the supported languages, so the script decides.
        let available = ["en", "es", "hi", "fr"];
        assert_eq!(detect("सुप्रभात", &available).as_deref(), Some("hi"));
        assert_eq!(detect("धन्यवाद", &available).as_deref(), Some("hi"));
        assert_eq!(detect("नमस्ते", &available).as_deref(), Some("hi"));
    }

    #[test]
    fn other_single_script_languages() {
        let available = ["en", "hi", "bn", "el", "he", "th", "ko"];
        assert_eq!(detect("ধন্যবাদ", &available).as_deref(), Some("bn"));
        assert_eq!(detect("ευχαριστώ", &available).as_deref(), Some("el"));
        assert_eq!(detect("תודה", &available).as_deref(), Some("he"));
        assert_eq!(detect("ขอบคุณ", &available).as_deref(), Some("th"));
        assert_eq!(detect("고맙습니다", &available).as_deref(), Some("ko"));
    }

    #[test]
    fn japanese_kana_beats_han_collision() {
        // Kanji outnumber kana here, but any kana marks it Japanese rather than
        // colliding with Chinese on the shared Han script.
        let available = ["en", "ja", "zh"];
        assert_eq!(
            detect("日本語を話します", &available).as_deref(),
            Some("ja")
        );
    }

    #[test]
    fn unsupported_script_returns_none() {
        // Devanagari text but `hi` is not installed: nothing to route it to.
        assert_eq!(detect("सुप्रभात", &["en", "es"]), None);
    }

    #[test]
    fn multi_language_script_defers_to_cld2() {
        // Latin and Cyrillic each cover several supported languages, so cld2's
        // ranking still decides within the script.
        let available = ["en", "es", "fr", "de", "ru", "uk"];
        assert_eq!(
            detect("The quick brown fox jumps over the lazy dog", &available).as_deref(),
            Some("en")
        );
        assert_eq!(
            detect("Съешь же ещё этих мягких французских булочек", &available).as_deref(),
            Some("ru")
        );
    }

    #[test]
    fn partial_word_does_not_resolve_to_a_family_member() {
        // Typing "hello how are you?" passes through "hello ho", which cld2 declines
        // to classify unhinted but rubber-stamps as both `da` and `no` when forced.
        let available = ["en", "da", "no", "sv", "de"];
        assert_eq!(detect("hello ho", &available), None);
        assert_eq!(detect_hinted("hello ho", "en", &available), None);
        assert_eq!(
            detect("hello how are you?", &available).as_deref(),
            Some("en")
        );
    }

    #[test]
    fn hint_competes_instead_of_being_skipped() {
        // A lone echo still wins, and the caller's own hint is allowed to be it.
        let available = ["en", "da", "no", "sv", "de"];
        assert_eq!(
            detect_hinted("hei hvordan", "da", &available).as_deref(),
            Some("da")
        );
    }
}
