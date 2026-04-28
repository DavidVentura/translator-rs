//! Glue: extract → translate per page.

use crate::api::{LanguageCode, TranslatorError};
use crate::pdf::{PageDims, PdfError};
use crate::pdf_text::extract_text;
use crate::session::TranslatorSession;
use crate::settings::BackgroundMode;
use crate::styled::TranslatedStyledBlock;

#[derive(Debug, Clone)]
pub struct PageTranslationResult {
    pub page_index: usize,
    pub page: PageDims,
    pub blocks: Vec<TranslatedStyledBlock>,
    pub error: Option<String>,
    /// BCP-47 tag of the language the blocks were translated **into**.
    /// The PDF writer hands this to its [`FontProvider`] when picking a
    /// font for the script.
    pub target_language: String,
}

#[derive(Debug)]
pub enum PdfTranslateError {
    Pdf(PdfError),
    Translator(TranslatorError),
}

impl From<PdfError> for PdfTranslateError {
    fn from(value: PdfError) -> Self {
        Self::Pdf(value)
    }
}

impl From<TranslatorError> for PdfTranslateError {
    fn from(value: TranslatorError) -> Self {
        Self::Translator(value)
    }
}

impl std::fmt::Display for PdfTranslateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pdf(err) => write!(f, "{err}"),
            Self::Translator(err) => write!(f, "translator: {err:?}"),
        }
    }
}

impl std::error::Error for PdfTranslateError {}

/// Extract every page's text via mupdf, run each through the existing
/// structured-translation path, and return per-page translated blocks.
pub fn translate_pdf(
    session: &TranslatorSession,
    pdf_bytes: &[u8],
    forced_source_code: Option<&str>,
    target_code: &str,
    available_language_codes: &[LanguageCode],
) -> Result<Vec<PageTranslationResult>, PdfTranslateError> {
    let extracted = extract_text(pdf_bytes)?;
    let mut results = Vec::with_capacity(extracted.len());

    for page in extracted {
        let translated = session.translate_structured_fragments(
            &page.fragments,
            forced_source_code,
            target_code,
            available_language_codes,
            None,
            BackgroundMode::BlackOnWhite,
        )?;

        results.push(PageTranslationResult {
            page_index: page.page_index,
            page: page.page,
            blocks: translated.blocks,
            error: translated.error_message,
            target_language: target_code.to_string(),
        });
    }

    Ok(results)
}
