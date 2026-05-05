//! Script identification used by the font provider and the image renderer.
//!
//! This is the library's own enum — independent from `unicode-script`'s
//! `Script`, because `font_provider` is reachable from non-image-render builds
//! (the PDF path also wants to query a font per script). The `image-render`
//! feature provides the conversion to/from `unicode_script::Script`.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    /// ISO 15924 code (matches BCP-47 `Script` subtag).
    pub fn iso15924(self) -> &'static str {
        match self {
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
            Script::Common => "Zyyy",
            Script::Inherited => "Zinh",
            Script::Other => "Zzzz",
        }
    }

    /// Best-effort mapping from a BCP-47 language tag to its dominant script.
    /// Used by callers who pass a target language without a script subtag.
    pub fn from_bcp47(tag: &str) -> Self {
        let primary = tag.split(['-', '_']).next().unwrap_or(tag);
        match primary {
            "ar" | "fa" | "ur" | "ps" | "ku" => Script::Arabic,
            "he" | "yi" => Script::Hebrew,
            "ru" | "uk" | "be" | "bg" | "mk" | "sr" | "kk" | "ky" | "tg" | "mn" => Script::Cyrillic,
            "el" => Script::Greek,
            "hy" => Script::Armenian,
            "hi" | "mr" | "ne" | "sa" | "kok" => Script::Devanagari,
            "bn" | "as" => Script::Bengali,
            "pa" => Script::Gurmukhi,
            "gu" => Script::Gujarati,
            "or" => Script::Oriya,
            "ta" => Script::Tamil,
            "te" => Script::Telugu,
            "kn" => Script::Kannada,
            "ml" => Script::Malayalam,
            "si" => Script::Sinhala,
            "th" => Script::Thai,
            "lo" => Script::Lao,
            "bo" | "dz" => Script::Tibetan,
            "my" => Script::Myanmar,
            "ka" => Script::Georgian,
            "am" | "ti" => Script::Ethiopic,
            "km" => Script::Khmer,
            "zh" | "yue" | "wuu" => Script::Han,
            "ja" => Script::Hiragana,
            "ko" => Script::Hangul,
            _ => Script::Latin,
        }
    }
}

#[cfg(feature = "image-render")]
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
