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
    NoTextFound,
    Cancelled,
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
            Self::NoTextFound => write!(f, "no extractable text found in PDF"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for PdfTranslateError {}

/// Extract every page's text via mupdf, run each through the existing
/// structured-translation path, and return per-page translated blocks.
///
/// To also translate raster image XObjects embedded in the PDF, run
/// [`crate::pdf_image_translate::translate_pdf_images_in_place`] on the
/// input bytes first and pass the result here — image translation lives
/// in the `pdf-image-translate` feature, not here.
pub fn translate_pdf(
    session: &TranslatorSession,
    pdf_bytes: &[u8],
    forced_source_code: Option<&str>,
    target_code: &str,
    available_language_codes: &[LanguageCode],
) -> Result<Vec<PageTranslationResult>, PdfTranslateError> {
    translate_pdf_with_progress(
        session,
        pdf_bytes,
        forced_source_code,
        target_code,
        available_language_codes,
        |_| {},
    )
}

/// Translate every page's text in a single bergamot call (slimt's batcher packs
/// all pages' sentences across the worker pool). Progress is reported per
/// sentence from worker threads via `on_progress` (cheap, non-blocking,
/// thread-safe), mapped onto the page count. Cancellation is requested
/// out-of-band via [`TranslatorSession::cancel_ongoing_work`] and surfaces as
/// [`PdfTranslateError::Cancelled`].
pub fn translate_pdf_with_progress(
    session: &TranslatorSession,
    pdf_bytes: &[u8],
    forced_source_code: Option<&str>,
    target_code: &str,
    available_language_codes: &[LanguageCode],
    on_progress: impl Fn(f32) + Sync,
) -> Result<Vec<PageTranslationResult>, PdfTranslateError> {
    session.begin_document_translation();
    let extracted = extract_text(pdf_bytes)?;
    if extracted.iter().all(|page| page.fragments.is_empty()) {
        return Err(PdfTranslateError::NoTextFound);
    }

    on_progress(0.0);

    let pages_fragments = extracted
        .iter()
        .map(|page| page.fragments.as_slice())
        .collect::<Vec<_>>();
    let report = |done: usize, total: usize| {
        if total > 0 {
            on_progress(done as f32 / total as f32);
        }
    };
    let translated = session
        .translate_structured_fragments_batch_ctx(
            &pages_fragments,
            forced_source_code,
            target_code,
            available_language_codes,
            BackgroundMode::BlackOnWhite,
            &report,
        )
        .map_err(|error| {
            if error.is_cancelled() {
                PdfTranslateError::Cancelled
            } else {
                PdfTranslateError::Translator(error)
            }
        })?;

    let results = extracted
        .into_iter()
        .zip(translated)
        .map(|(page, result)| PageTranslationResult {
            page_index: page.page_index,
            page: page.page,
            blocks: result.blocks,
            error: result.error_message,
            target_language: target_code.to_string(),
        })
        .collect();

    on_progress(1.0);

    Ok(results)
}
