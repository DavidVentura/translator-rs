#[cfg(feature = "uniffi")]
uniffi::setup_scaffolding!();

pub mod api;
pub mod bergamot;
pub mod catalog;
pub mod language;
pub mod language_detect;
#[cfg(feature = "mucab")]
pub mod mucab;
pub mod ocr;
#[cfg(feature = "tesseract")]
mod ocr_runtime;
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
    OverlayScreenshot, StructuredTranslationResult,
    StyledFragment as StructuredStyledFragment,
};
pub use tts::{PcmAudio, SpeechChunk, TtsVoiceOption};
