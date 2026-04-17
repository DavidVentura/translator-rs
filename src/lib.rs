#[cfg(feature = "uniffi")]
uniffi::setup_scaffolding!();

pub mod bergamot;
pub mod catalog;
pub mod language;
pub mod language_detect;
#[cfg(feature = "mucab")]
pub mod mucab;
pub mod ocr;
#[cfg(feature = "tesseract")]
mod ocr_runtime;
pub mod routing;
pub mod settings;
#[cfg(feature = "tts")]
mod speech;
pub mod styled;
#[cfg(feature = "dictionary")]
pub mod tarkka;
#[cfg(feature = "tesseract")]
pub mod tesseract;
pub mod translate;
#[cfg(feature = "transliterate")]
pub mod transliterate;
pub mod tts;

pub use bergamot::BergamotEngine;
pub use catalog::{
    AssetFileV2, AssetPackMetadataV2, CatalogSnapshot, CatalogSourcesV2, DeletePlan,
    DictionaryInfo, DownloadPlan, DownloadTask, LangAvailability, LanguageAvailabilityRow,
    LanguageCatalog, LanguageFeature, LanguageTtsRegionV2, LanguageTtsV2, PackInstallChecker,
    PackInstallStatus, PackKind, PackRecord, PackResolver, ResolvedTtsVoiceFiles, TtsVoicePackInfo,
    TtsVoicePickerRegion, build_catalog_snapshot, can_swap_languages_installed, can_translate,
    can_translate_in_snapshot, can_translate_with_checker, compute_language_availability,
    has_translation_direction_installed, installed_tts_pack_id_for_language, is_pack_installed,
    language_rows_in_snapshot, parse_and_validate_catalog, parse_language_catalog,
    plan_delete_dictionary, plan_delete_dictionary_in_snapshot, plan_delete_language,
    plan_delete_language_in_snapshot, plan_delete_superseded_tts,
    plan_delete_superseded_tts_in_snapshot, plan_delete_tts, plan_delete_tts_in_snapshot,
    plan_dictionary_download, plan_dictionary_download_in_snapshot, plan_language_download,
    plan_language_download_in_snapshot, plan_tts_download, plan_tts_download_in_snapshot,
    resolve_tts_voice_files, resolve_tts_voice_files_in_snapshot, select_best_catalog,
};
pub use language::Language;
pub use language_detect::{DetectionResult, detect_language};
pub use ocr::{
    DetectedWord, ImageTranslationOutcome, OverlayColors, OverlayLayoutHints, OverlayLayoutMode,
    PreparedImageOverlay, PreparedTextBlock, PreparedTextLine, ReadingOrder, Rect, TextBlock,
    TextLine, build_text_blocks, prepare_overlay_image, sample_overlay_colors,
};
#[cfg(feature = "tesseract")]
pub use ocr_runtime::translate_image_rgba_in_snapshot;
pub use routing::{
    MixedTextTranslationResult, NothingReason, TextTranslation, detect_language_robust_code,
    translate_mixed_texts_in_snapshot,
};
pub use settings::{AppSettings, BackgroundMode, DEFAULT_CATALOG_INDEX_URL};
#[cfg(feature = "tts")]
pub use speech::{
    available_tts_voices_in_snapshot, clear_cached_model, plan_speech_chunks_for_text_in_snapshot,
    synthesize_pcm_in_snapshot, warm_tts_model_in_snapshot,
};
pub use styled::{
    OverlayScreenshot, StructuredTranslationResult, StyleSpan as StructuredStyleSpan,
    StyledFragment as StructuredStyledFragment, TextStyle, TranslatedStyledBlock,
    TranslationSegment, translate_structured_fragments_in_snapshot,
};
#[cfg(feature = "dictionary")]
pub use tarkka::{close_dictionary, lookup_dictionary};
#[cfg(feature = "tesseract")]
pub use tesseract::{PageSegMode, TesseractWrapper};
pub use translate::{
    TokenAlignment, TranslatedText, TranslationWithAlignment, translate_texts_in_snapshot,
    translate_texts_with_alignment_in_snapshot,
};
#[cfg(feature = "transliterate")]
pub use transliterate::{
    transliterate, transliterate_with_policy, transliterate_with_policy_for_language,
};
pub use tts::{
    PcmAudio, PhonemeChunk, SpeechChunk, SpeechChunkBoundary, TtsVoiceOption, plan_speech_chunks,
};
