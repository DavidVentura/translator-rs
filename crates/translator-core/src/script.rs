//! Script identification used by the font provider and the image renderer.
//!
//! This is the library's own enum — independent from `unicode-script`'s
//! `Script`, because `font_provider` is reachable from non-image-render builds
//! (the PDF path also wants to query a font per script). The `image-render`
//! feature provides the conversion to/from `unicode_script::Script`.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
pub enum Script {
    Latin,
    Cyrillic,
    Greek,
    Armenian,
    Hebrew,
    Arabic,
    Devanagari,
    Bengali,
    Gurmukhi,
    Gujarati,
    Oriya,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    Sinhala,
    Thai,
    Lao,
    Tibetan,
    Myanmar,
    Georgian,
    Ethiopic,
    Khmer,
    Han,
    Hiragana,
    Katakana,
    Hangul,
    /// Punctuation, digits, symbols. Inherits from neighbours during
    /// itemization.
    Common,
    /// Combining marks. Inherits from the base codepoint they attach to.
    Inherited,
    /// Anything we haven't enumerated. Treated as a hard boundary.
    Other,
}

impl Script {
    /// True for scripts whose default direction is right-to-left.
    pub fn is_rtl(self) -> bool {
        matches!(self, Script::Arabic | Script::Hebrew)
    }

    /// ISO 15924 code (matches BCP-47 `Script` subtag), or `None` for the
    /// itemization categories that name no writing system: punctuation and
    /// digits, combining marks, and anything unenumerated.
    pub fn iso15924(self) -> Option<&'static str> {
        let code = match self {
            Script::Latin => "Latn",
            Script::Cyrillic => "Cyrl",
            Script::Greek => "Grek",
            Script::Armenian => "Armn",
            Script::Hebrew => "Hebr",
            Script::Arabic => "Arab",
            Script::Devanagari => "Deva",
            Script::Bengali => "Beng",
            Script::Gurmukhi => "Guru",
            Script::Gujarati => "Gujr",
            Script::Oriya => "Orya",
            Script::Tamil => "Taml",
            Script::Telugu => "Telu",
            Script::Kannada => "Knda",
            Script::Malayalam => "Mlym",
            Script::Sinhala => "Sinh",
            Script::Thai => "Thai",
            Script::Lao => "Laoo",
            Script::Tibetan => "Tibt",
            Script::Myanmar => "Mymr",
            Script::Georgian => "Geor",
            Script::Ethiopic => "Ethi",
            Script::Khmer => "Khmr",
            Script::Han => "Hani",
            Script::Hiragana => "Hira",
            Script::Katakana => "Kana",
            Script::Hangul => "Hang",
            Script::Common | Script::Inherited | Script::Other => return None,
        };
        Some(code)
    }

    /// Inverse of [`Script::iso15924`], plus the composite subtags the catalog
    /// uses. Returns `None` for anything unrecognized so callers decide what an
    /// unknown script means; guessing a default here is what made a missing
    /// language silently render as Latin.
    pub fn from_iso15924(code: &str) -> Option<Self> {
        let script = match code {
            "Latn" => Script::Latin,
            "Cyrl" => Script::Cyrillic,
            "Grek" => Script::Greek,
            "Armn" => Script::Armenian,
            "Hebr" => Script::Hebrew,
            "Arab" => Script::Arabic,
            "Deva" => Script::Devanagari,
            "Beng" => Script::Bengali,
            "Guru" => Script::Gurmukhi,
            "Gujr" => Script::Gujarati,
            "Orya" => Script::Oriya,
            "Taml" => Script::Tamil,
            "Telu" => Script::Telugu,
            "Knda" => Script::Kannada,
            "Mlym" => Script::Malayalam,
            "Sinh" => Script::Sinhala,
            "Thai" => Script::Thai,
            "Laoo" => Script::Lao,
            "Tibt" => Script::Tibetan,
            "Mymr" => Script::Myanmar,
            "Geor" => Script::Georgian,
            "Ethi" => Script::Ethiopic,
            "Khmr" => Script::Khmer,
            "Hani" => Script::Han,
            "Hira" => Script::Hiragana,
            "Kana" => Script::Katakana,
            "Hang" => Script::Hangul,
            // Composite subtags name a writing system rather than one script.
            // Han is the shared inventory behind both Chinese variants, and kana
            // is what [`Script::dominant`] keys Japanese on, so Jpan resolves to
            // Hiragana for consistency with detection.
            "Hans" | "Hant" => Script::Han,
            "Jpan" => Script::Hiragana,
            "Kore" => Script::Hangul,
            _ => return None,
        };
        Some(script)
    }
}

#[cfg(feature = "script-from-unicode")]
impl Script {
    pub fn of_char(ch: char) -> Self {
        use unicode_script::UnicodeScript;
        Script::from(ch.script())
    }

    /// The script that best identifies the text, aligned with the buckets
    /// [`Script::from_iso15924`] produces so a detected script can be matched
    /// against the catalog's supported languages.
    ///
    /// Kana is a definitive Japanese marker, so any Hiragana/Katakana wins outright —
    /// otherwise a kanji-heavy Japanese sentence would count as Han and collide with
    /// Chinese. Han without kana stays Han (Chinese vs. traditional is cld2's job).
    /// Otherwise it is the most frequent concrete script; `None` when the text carries
    /// no concrete-script characters (digits, punctuation, whitespace only).
    pub fn dominant(text: &str) -> Option<Self> {
        let mut counts: std::collections::HashMap<Script, usize> = std::collections::HashMap::new();
        for ch in text.chars() {
            let script = Script::of_char(ch);
            if matches!(script, Script::Hiragana | Script::Katakana) {
                return Some(Script::Hiragana);
            }
            if matches!(script, Script::Common | Script::Inherited | Script::Other) {
                continue;
            }
            *counts.entry(script).or_default() += 1;
        }
        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(script, _)| script)
    }
}

#[cfg(feature = "script-from-unicode")]
impl From<unicode_script::Script> for Script {
    fn from(s: unicode_script::Script) -> Self {
        use unicode_script::Script as U;
        match s {
            U::Latin => Script::Latin,
            U::Cyrillic => Script::Cyrillic,
            U::Greek => Script::Greek,
            U::Armenian => Script::Armenian,
            U::Hebrew => Script::Hebrew,
            U::Arabic => Script::Arabic,
            U::Devanagari => Script::Devanagari,
            U::Bengali => Script::Bengali,
            U::Gurmukhi => Script::Gurmukhi,
            U::Gujarati => Script::Gujarati,
            U::Oriya => Script::Oriya,
            U::Tamil => Script::Tamil,
            U::Telugu => Script::Telugu,
            U::Kannada => Script::Kannada,
            U::Malayalam => Script::Malayalam,
            U::Sinhala => Script::Sinhala,
            U::Thai => Script::Thai,
            U::Lao => Script::Lao,
            U::Tibetan => Script::Tibetan,
            U::Myanmar => Script::Myanmar,
            U::Georgian => Script::Georgian,
            U::Ethiopic => Script::Ethiopic,
            U::Khmer => Script::Khmer,
            U::Han => Script::Han,
            U::Hiragana => Script::Hiragana,
            U::Katakana => Script::Katakana,
            U::Hangul => Script::Hangul,
            U::Common => Script::Common,
            U::Inherited => Script::Inherited,
            _ => Script::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Script;

    /// Every variant that names a real writing system. The three itemization
    /// categories are deliberately absent: they have no ISO 15924 subtag.
    const WRITING_SYSTEMS: &[Script] = &[
        Script::Latin,
        Script::Cyrillic,
        Script::Greek,
        Script::Armenian,
        Script::Hebrew,
        Script::Arabic,
        Script::Devanagari,
        Script::Bengali,
        Script::Gurmukhi,
        Script::Gujarati,
        Script::Oriya,
        Script::Tamil,
        Script::Telugu,
        Script::Kannada,
        Script::Malayalam,
        Script::Sinhala,
        Script::Thai,
        Script::Lao,
        Script::Tibetan,
        Script::Myanmar,
        Script::Georgian,
        Script::Ethiopic,
        Script::Khmer,
        Script::Han,
        Script::Hiragana,
        Script::Katakana,
        Script::Hangul,
    ];

    #[test]
    fn iso15924_round_trips() {
        for &script in WRITING_SYSTEMS {
            let code = script.iso15924().expect("writing system has a subtag");
            assert_eq!(Script::from_iso15924(code), Some(script));
        }
    }

    #[test]
    fn itemization_categories_have_no_subtag() {
        for script in [Script::Common, Script::Inherited, Script::Other] {
            assert_eq!(script.iso15924(), None);
        }
        for code in ["Zyyy", "Zinh", "Zzzz"] {
            assert_eq!(Script::from_iso15924(code), None);
        }
    }

    /// The values the shipped catalog actually carries, so a language whose
    /// script the catalog knows can never fall back to a guess.
    #[test]
    fn parses_every_catalog_script() {
        let catalog_values = [
            "Latn", "Cyrl", "Arab", "Deva", "Beng", "Grek", "Gujr", "Hebr", "Jpan", "Knda", "Hang",
            "Mlym", "Taml", "Telu", "Thai", "Hans", "Hant",
        ];
        for value in catalog_values {
            assert!(
                Script::from_iso15924(value).is_some(),
                "catalog script {value} does not parse"
            );
        }
    }

    #[test]
    fn composite_subtags_resolve_to_their_inventory() {
        assert_eq!(Script::from_iso15924("Hans"), Some(Script::Han));
        assert_eq!(Script::from_iso15924("Hant"), Some(Script::Han));
        assert_eq!(Script::from_iso15924("Jpan"), Some(Script::Hiragana));
        assert_eq!(Script::from_iso15924("Kore"), Some(Script::Hangul));
    }

    #[test]
    fn unknown_script_is_none() {
        assert_eq!(Script::from_iso15924("Cans"), None);
        assert_eq!(Script::from_iso15924(""), None);
    }
}
