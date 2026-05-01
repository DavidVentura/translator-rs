//! Glue: extract → translate per page.

use crate::api::{LanguageCode, TranslatorError};
use crate::pdf::{PageDims, PdfError};
use crate::pdf_text::extract_text;
use crate::session::TranslatorSession;
use crate::settings::BackgroundMode;
use crate::styled::TranslatedStyledBlock;

pub enum PdfTranslateProgress {
    TranslatingPage { current: usize, total: usize },
}

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
            Self::Cancelled => write!(f, "cancelled"),
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
    translate_pdf_with_progress(
        session,
        pdf_bytes,
        forced_source_code,
        target_code,
        available_language_codes,
        |_| Ok(()),
    )
}

/// Number of pages bundled into a single bergamot call. With slimt's default
/// 4-worker pool, batching ~8 pages per call keeps every worker fed even when
/// individual pages are short — single-page calls couldn't fill the queue.
/// The chunk boundary is also where progress ticks land, so it doubles as the
/// granularity at which the caller's UI updates.
const PAGE_BATCH_SIZE: usize = 8;

pub fn translate_pdf_with_progress(
    session: &TranslatorSession,
    pdf_bytes: &[u8],
    forced_source_code: Option<&str>,
    target_code: &str,
    available_language_codes: &[LanguageCode],
    mut on_progress: impl FnMut(PdfTranslateProgress) -> Result<(), PdfTranslateError>,
) -> Result<Vec<PageTranslationResult>, PdfTranslateError> {
    let extracted = extract_text(pdf_bytes)?;
    let total = extracted.len();
    let mut results = Vec::with_capacity(total);
    on_progress(PdfTranslateProgress::TranslatingPage { current: 0, total })?;

    for chunk in extracted.chunks(PAGE_BATCH_SIZE) {
        let pages_fragments = chunk
            .iter()
            .map(|page| page.fragments.as_slice())
            .collect::<Vec<_>>();
        let translated = session.translate_structured_fragments_batch(
            &pages_fragments,
            forced_source_code,
            target_code,
            available_language_codes,
            BackgroundMode::BlackOnWhite,
        )?;

        for (page, result) in chunk.iter().zip(translated) {
            results.push(PageTranslationResult {
                page_index: page.page_index,
                page: page.page,
                blocks: result.blocks,
                error: result.error_message,
                target_language: target_code.to_string(),
            });
        }
        on_progress(PdfTranslateProgress::TranslatingPage {
            current: results.len(),
            total,
        })?;
    }

    Ok(results)
}
