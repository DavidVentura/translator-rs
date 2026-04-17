use crate::api::LanguageCode;
use cld2::{Format, Hints, Reliable, detect_language_ext};

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
    available_language_codes: &[LanguageCode],
) -> Option<LanguageCode> {
    if text.trim().is_empty() {
        return None;
    }

    if let Some(detected) = detect_language(text, hint) {
        if detected.is_reliable
            && available_language_codes
                .iter()
                .any(|code| code.as_str() == detected.language)
        {
            return Some(LanguageCode::from(detected.language));
        }
    }

    for code in available_language_codes {
        if hint == Some(code) {
            continue;
        }
        let Some(detected) = detect_language(text, Some(code)) else {
            continue;
        };
        if detected.is_reliable && detected.language == code.as_str() {
            return Some(code.clone());
        }
    }

    None
}
