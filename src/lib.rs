#[cfg(feature = "uniffi")]
uniffi::setup_scaffolding!();

pub mod api;
pub mod bergamot;
pub mod catalog;
#[cfg(feature = "pdf")]
pub mod font_metrics;
#[cfg(feature = "pdf")]
pub mod font_provider;
#[cfg(feature = "html")]
pub mod html_translate;
pub mod language;
pub mod language_detect;
#[cfg(feature = "mucab")]
pub mod mucab;
pub mod ocr;
#[cfg(feature = "tesseract")]
mod ocr_runtime;
#[cfg(feature = "odt")]
pub mod odt;
#[cfg(feature = "pdf")]
pub mod pdf;
#[cfg(feature = "pdf")]
mod pdf_content;
#[cfg(feature = "pdf")]
pub mod pdf_font_embed;
#[cfg(feature = "pdf")]
mod pdf_overlay;
#[cfg(feature = "pdf")]
mod pdf_resources;
#[cfg(feature = "pdf")]
mod pdf_surgery;
#[cfg(feature = "pdf")]
pub mod pdf_text;
#[cfg(feature = "pdf")]
pub mod pdf_translate;
#[cfg(feature = "pdf")]
pub mod pdf_write;
mod routing;
pub mod session;
pub mod settings;
#[cfg(feature = "tts")]
mod speech;
mod styled;
#[cfg(feature = "dictionary")]
pub mod tarkka;
#[cfg(feature = "tesseract")]
pub mod tesseract;
mod translate;
#[cfg(feature = "transliterate")]
pub mod transliterate;
pub mod tts;

pub use api::{DictionaryCode, LanguageCode, ScriptCode, TranslatorError, TranslatorErrorKind};
pub use catalog::{
    CatalogSnapshot, DeletePlan, DictionaryInfo, DownloadPlan, DownloadTask, FsPackInstallChecker,
    LanguageAvailabilityRow, LanguageCatalog, LanguageOverview, TtsVoicePickerRegion,
    language_rows_in_snapshot, parse_and_validate_catalog,
};
pub use language_detect::DetectionResult;
pub use ocr::{OverlayColors, PreparedImageOverlay, ReadingOrder, Rect, sample_overlay_colors};
pub use routing::MixedTextTranslationResult;
pub use session::{Feature, TranslatorSession};
pub use settings::BackgroundMode;
pub use styled::{
    OverlayScreenshot, StructuredTranslationResult, StyledFragment as StructuredStyledFragment,
};
pub use translate::{TokenAlignment, TranslationWithAlignment};
pub use tts::{PcmAudio, SpeechChunk, TtsVoiceOption};
