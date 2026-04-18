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

pub use api::{
    DictionaryCode, LanguageCode, ScriptCode, TranslatorError, TranslatorErrorKind, VoiceName,
};
pub use bergamot::BergamotEngine;
pub use catalog::{
    AssetFileV2, AssetPackMetadataV2, CatalogSnapshot, CatalogSourcesV2, DeletePlan,
    DictionaryInfo, DownloadPlan, DownloadTask, FsPackInstallChecker, LangAvailability,
    LanguageAvailabilityRow, LanguageCatalog, LanguageFeature, LanguageOverview,
    LanguageTtsRegionV2, LanguageTtsV2, PackInstallChecker, PackInstallStatus, PackKind,
    PackRecord, ResolvedTtsVoiceFiles, TtsVoiceOverview, TtsVoicePackInfo, TtsVoicePickerRegion,
    TtsVoiceRegionOverview, build_catalog_snapshot, build_language_overview, can_translate,
    language_rows_in_snapshot, parse_and_validate_catalog, parse_language_catalog,
    plan_delete_dictionary, plan_delete_language, plan_delete_superseded_tts, plan_delete_tts,
    plan_dictionary_download, plan_language_download, plan_tts_download, resolve_tts_voice_files,
    select_best_catalog,
};
pub use language::Language;
pub use language_detect::{DetectionResult, detect_language};
pub use ocr::{
    DetectedWord, OverlayColors, OverlayLayoutHints, OverlayLayoutMode, PreparedImageOverlay,
    PreparedTextBlock, PreparedTextLine, ReadingOrder, Rect, TextBlock, TextLine,
    build_text_blocks, prepare_overlay_image, sample_overlay_colors,
};
pub use routing::{MixedTextTranslationResult, NothingReason, TextTranslation};
pub use session::{Feature, TranslatorSession, parse_selected_catalog};
pub use settings::{AppSettings, BackgroundMode, DEFAULT_CATALOG_INDEX_URL};
pub use styled::{
    OverlayScreenshot, StructuredTranslationResult, StyleSpan as StructuredStyleSpan,
    StyledFragment as StructuredStyledFragment, TextStyle, TranslatedStyledBlock,
    TranslationSegment,
};
#[cfg(feature = "tesseract")]
pub use tesseract::{PageSegMode, TesseractWrapper};
pub use translate::{TokenAlignment, TranslatedText, TranslationWithAlignment, Translator};
pub use tts::{
    PcmAudio, PhonemeChunk, SpeechChunk, SpeechChunkBoundary, TtsVoiceOption, plan_speech_chunks,
};
