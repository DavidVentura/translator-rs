#[cfg(feature = "uniffi")]
uniffi::setup_scaffolding!();

pub mod api;
pub mod bergamot;
pub mod catalog;
#[cfg(feature = "planar-tracker")]
pub mod coarse_tracker;
#[cfg(any(feature = "ppocr", feature = "planar-tracker"))]
pub mod color_matting;
#[cfg(any(feature = "ppocr", feature = "planar-tracker"))]
pub mod coords;
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
#[cfg(any(feature = "ppocr", feature = "planar-tracker"))]
pub mod homography;
#[cfg(any(feature = "ppocr", feature = "planar-tracker"))]
pub mod homography_ekf;
#[cfg(feature = "html")]
pub mod html_translate;
#[cfg(feature = "image-render")]
pub mod image_render;
#[cfg(feature = "doc-align")]
mod inference;
#[cfg(feature = "planar-tracker")]
pub mod klt;
pub mod language;
pub mod language_detect;
#[cfg(feature = "planar-tracker")]
pub mod live_compositor;
#[cfg(any(feature = "ppocr", feature = "planar-tracker"))]
pub mod live_frame;
#[cfg(feature = "planar-tracker")]
pub mod live_session;
#[cfg(feature = "planar-tracker")]
pub mod live_tracker_pipeline;
#[cfg(feature = "ppocr")]
mod mnn_inference;
#[cfg(feature = "mucab")]
pub mod mucab;
pub mod ocr;
#[cfg(any(feature = "tesseract", feature = "ppocr"))]
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
#[cfg(any(feature = "pdf", feature = "image-render"))]
pub mod script;
#[cfg(feature = "ppocr")]
mod script_normalize;
mod sentence_split;
pub mod session;
pub mod settings;
#[cfg(feature = "tts")]
mod speech;
mod styled;
#[cfg(feature = "planar-tracker")]
pub mod surface_map;
#[cfg(feature = "dictionary")]
pub mod tarkka;
#[cfg(feature = "tesseract")]
pub mod tesseract;
#[cfg(feature = "image-render")]
pub mod text_runs;
mod translate;
#[cfg(feature = "transliterate")]
pub mod transliterate;
pub mod tts;

pub use api::{DictionaryCode, LanguageCode, ScriptCode, TranslatorError, TranslatorErrorKind};
pub use catalog::{
    CatalogSnapshot, DeletePlan, DictionaryInfo, DownloadPlan, DownloadTask, FsPackInstallChecker,
    InstalledTtsPack, LanguageAvailabilityRow, LanguageCatalog, LanguageOverview, OcrEngine,
    OcrPack, PpocrScript, TtsSpeakerEntry, TtsVoicePickerRegion,
    available_ocr_engines_for_language, installed_ocr_engines_for_language,
    installed_tts_voice_picker_regions, language_rows_in_snapshot, parse_and_validate_catalog,
    plan_ocr_engine_download, plan_ocr_engine_downloads,
};
pub use language_detect::DetectionResult;
pub use ocr::{
    DetectedTextBox, OcrSourceSelection, OverlayColors, PreparedImageOverlay, ReadingOrder,
    RecognizedTextLine, Rect, sample_overlay_colors,
};
pub use routing::MixedTextTranslationResult;
pub use session::{Feature, TranslatorSession};
pub use settings::{BackgroundMode, PreferredOcrEngine};
pub use styled::{
    OverlayScreenshot, StructuredTranslationResult, StyledFragment as StructuredStyledFragment,
};
pub use translate::{TokenAlignment, TranslationWithAlignment};
pub use tts::{PcmAudio, SpeechChunk, TtsVoiceOption};
