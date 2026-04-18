#[path = "catalog_model.rs"]
mod model;
#[path = "catalog_planner.rs"]
mod planner;
#[path = "catalog_wire.rs"]
mod wire;

pub use model::{
    AssetFileV2, AssetPackMetadataV2, CatalogSourcesV2, DeletePlan, DictionaryInfo, DownloadPlan,
    DownloadTask, LangAvailability, LanguageCatalog, LanguageFeature, LanguageTtsRegionV2,
    LanguageTtsV2, PackKind, PackRecord, ResolvedTtsVoiceFiles, TtsVoicePackInfo,
    TtsVoicePickerRegion,
};
pub use planner::{
    CatalogSnapshot, FsPackInstallChecker, LanguageAvailabilityRow, PackInstallChecker,
    PackInstallStatus, build_catalog_snapshot, can_translate, language_rows_in_snapshot,
    plan_delete_dictionary, plan_delete_language, plan_delete_superseded_tts, plan_delete_tts,
    plan_dictionary_download, plan_language_download, plan_tts_download, resolve_tts_voice_files,
};
pub use wire::{parse_and_validate_catalog, parse_language_catalog, select_best_catalog};

#[cfg(test)]
#[path = "catalog_tests.rs"]
mod tests;
