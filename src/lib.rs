#[cfg(feature = "uniffi")]
uniffi::setup_scaffolding!();

pub use translator_core::api;
pub mod bergamot;
pub use translator_core::catalog;
#[cfg(feature = "planar-tracker")]
pub mod coarse_tracker;
#[cfg(any(feature = "ppocr", feature = "planar-tracker"))]
pub mod color_matting;
pub use translator_core::coords;
#[cfg(feature = "doc-align")]
pub mod doc_align;
#[cfg(feature = "doc-align")]
pub mod doc_align_refine;
#[cfg(any(feature = "pdf", feature = "image-render"))]
pub mod font_metrics;
#[cfg(any(feature = "pdf", feature = "image-render"))]
pub mod font_provider;
#[cfg(feature = "gpu")]
pub mod gl_renderer;
//#[cfg(any(feature = "ppocr", feature = "planar-tracker"))]
#[cfg(feature = "dom-translate")]
mod dom_translate;
#[cfg(feature = "epub")]
pub mod epub;
pub use translator_core::homography;
pub use translator_core::homography_ekf;
#[cfg(feature = "html")]
pub mod html_translate;
#[cfg(feature = "image-render")]
pub mod image_render;
#[cfg(feature = "doc-align")]
mod inference;
#[cfg(feature = "planar-tracker")]
pub mod klt;
pub use translator_core::language;
pub mod language_detect;
#[cfg(any(feature = "ppocr", feature = "planar-tracker"))]
pub mod live_frame;
#[cfg(feature = "gpu")]
pub mod live_gpu_tick;
#[cfg(feature = "planar-tracker")]
pub mod live_screen;
#[cfg(feature = "planar-tracker")]
pub mod live_session;
#[cfg(feature = "planar-tracker")]
pub mod live_tracker_pipeline;
#[cfg(feature = "planar-tracker")]
pub mod live_worker;
#[cfg(feature = "ppocr")]
mod mnn_inference;
#[cfg(feature = "mucab")]
pub use translator_mucab::mucab;
pub mod ocr;
#[cfg(feature = "ppocr")]
mod ocr_runtime;
#[cfg(feature = "odt")]
pub mod odt;
#[cfg(feature = "pdf")]
pub mod pdf;
#[cfg(feature = "pdf")]
mod pdf_content;
#[cfg(feature = "pdf")]
pub mod pdf_font_embed;
#[cfg(feature = "pdf-image-translate")]
pub mod pdf_image_translate;
#[cfg(feature = "pdf")]
mod pdf_overlay;
#[cfg(feature = "pdf")]
mod pdf_resources;
#[cfg(feature = "pdf")]
mod pdf_surgery;
#[cfg(feature = "pdf")]
pub mod pdf_text;
#[cfg(feature = "pdf-image-translate")]
mod pdf_text_overlay;
#[cfg(feature = "pdf")]
pub mod pdf_translate;
#[cfg(feature = "pdf")]
pub mod pdf_write;
#[cfg(feature = "planar-tracker")]
pub mod planar_engine;
#[cfg(feature = "planar-tracker")]
pub mod planar_tracker;
#[cfg(feature = "ppocr")]
pub mod ppocr;
mod routing;
pub mod screen_monitor;
#[cfg(feature = "gpu")]
pub mod screen_monitor_gpu;
pub use translator_core::script;
pub use translator_core::script_normalize;
mod sentence_split;
pub mod session;
pub use translator_core::settings;
#[cfg(feature = "tts")]
mod speech;
mod styled;
pub mod surface_map;
#[cfg(feature = "dictionary")]
pub use translator_dictionary::tarkka;
pub mod text_metrics;
#[cfg(any(feature = "pdf", feature = "image-render"))]
pub mod text_runs;
#[cfg(any(feature = "pdf", feature = "image-render"))]
pub mod text_shape;
mod translate;
#[cfg(feature = "transliterate")]
pub mod transliterate;
pub use translator_core::tts;
pub mod txt;

pub use api::{DictionaryCode, LanguageCode, ScriptCode, TranslatorError, TranslatorErrorKind};
pub use catalog::{
    CatalogSnapshot, DeletePlan, DictionaryInfo, DownloadPlan, DownloadTask, FileRole,
    FsPackInstallChecker, InstalledTtsPack, LanguageAvailabilityRow, LanguageCatalog,
    LanguageOverview, OcrEngine, OcrPack, PpocrScript, TtsSpeakerEntry, TtsVoicePickerRegion,
    available_ocr_engines_for_language, installed_ocr_engines_for_language,
    installed_tts_voice_picker_regions, language_rows_in_snapshot, parse_and_validate_catalog,
    plan_delete_superseded_files, plan_ocr_engine_download, plan_ocr_engine_downloads,
    plan_ocr_engine_upgrades,
};
pub use language_detect::DetectionResult;
pub use ocr::{
    DetectedTextBox, OcrSourceSelection, OrientedRect, OverlayColors, PreparedImageOverlay,
    ReadingOrder, RecognizedTextLine, Rect, sample_overlay_colors,
};
pub use routing::MixedTextTranslationResult;
pub use session::{Feature, TranslatorSession};
pub use settings::BackgroundMode;
pub use styled::{
    OverlayScreenshot, StructuredTranslationResult, StyledFragment as StructuredStyledFragment,
};
pub use translate::{TokenAlignment, TranslationWithAlignment};
pub use tts::{PcmAudio, SpeechChunk, TtsVoiceOption};
