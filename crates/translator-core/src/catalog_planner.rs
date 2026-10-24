use std::collections::{BTreeSet, HashMap, HashSet};

use crate::api::LanguageCode;
use crate::language::Language;
use crate::tts::voice_display_name;

use super::model::{
    AssetFileV2, DeletePlan, DownloadPlan, DownloadTask, FileRequirement, FileRole, InstallMarker,
    InstalledTtsPack, LangAvailability, LanguageCatalog, LanguageFeature, MigrationAction,
    MigrationJob, OcrPack, PackKind, PackRecord, ResolvedTtsVoiceFiles, TaskInstall, TtsPack,
    TtsSpeakerEntry, TtsVoicePackInfo, TtsVoicePickerRegion, ZipEntryTarget, ZipExtractionPlan,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackInstallStatus {
    pub pack_id: String,
    pub installed: bool,
    /// At least one of the pack's own files is on disk. Together with
    /// `!installed` this distinguishes a broken partial install (repairable)
    /// from a pack that was simply never downloaded.
    pub any_file_present: bool,
    pub missing_files: Vec<AssetFileV2>,
    pub missing_dependency_ids: Vec<String>,
    /// Improvement downloads that don't gate `installed`: higher-priority role
    /// alternatives than the best file on disk, plus optional-role files that
    /// aren't present at all (a newly-added optional model on an already-
    /// installed pack). Surfaced to the user as an available upgrade.
    pub upgrade_files: Vec<AssetFileV2>,
    /// Role alternatives on disk that are outranked by a better file also on
    /// disk, safe to delete.
    pub superseded_files: Vec<AssetFileV2>,
    /// The highest-priority present file of each role — the pack's resolved
    /// on-disk configuration. Runtime resolution (translation model/vocab, TTS
    /// voice model/aux) reads these so it never picks a catalog alternative that
    /// is not actually installed.
    pub active_files: Vec<AssetFileV2>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MissingPackFile {
    pack_id: String,
    file: AssetFileV2,
}

pub trait PackInstallChecker {
    fn file_exists(&self, install_path: &str) -> bool;

    fn install_marker_exists(&self, marker_path: &str, expected_version: i32) -> bool;
}

pub struct FsPackInstallChecker {
    base_dir: std::path::PathBuf,
}

impl FsPackInstallChecker {
    pub fn new(base_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    fn resolve(&self, relative_path: &str) -> std::path::PathBuf {
        self.base_dir.join(relative_path)
    }
}

impl PackInstallChecker for FsPackInstallChecker {
    fn file_exists(&self, install_path: &str) -> bool {
        self.resolve(install_path).exists()
    }

    fn install_marker_exists(&self, marker_path: &str, expected_version: i32) -> bool {
        let marker_file = self.resolve(marker_path);
        if !marker_file.exists() {
            return false;
        }

        let Ok(contents) = std::fs::read_to_string(marker_file) else {
            return false;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&contents) else {
            return false;
        };
        json.get("version")
            .and_then(serde_json::Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            == Some(expected_version)
    }
}

/// Select the on-device ONNX→MNN conversions for the current install: only
/// entries whose source `.onnx` is actually present. An entry whose `.mnn`
/// already exists alongside is cleanup-only (delete the stray `.onnx`), so the
/// step is idempotent across re-runs and partial completions.
pub fn plan_migrations(
    catalog: &LanguageCatalog,
    checker: &impl PackInstallChecker,
) -> Vec<MigrationJob> {
    catalog
        .migrations
        .iter()
        .filter(|entry| checker.file_exists(&entry.onnx))
        .map(|entry| {
            let action = if checker.file_exists(&entry.mnn) {
                MigrationAction::CleanupOnly
            } else {
                MigrationAction::Convert
            };
            MigrationJob {
                entry: entry.clone(),
                action,
            }
        })
        .collect()
}

pub(crate) struct PackResolver<'a, C> {
    catalog: &'a LanguageCatalog,
    install_checker: &'a C,
    status_cache: HashMap<String, PackInstallStatus>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogSnapshot {
    pub catalog: LanguageCatalog,
    pub base_dir: String,
    pub pack_statuses: HashMap<String, PackInstallStatus>,
    pub availability_by_code: HashMap<String, LangAvailability>,
}

impl CatalogSnapshot {
    pub fn can_translate(&self, from_code: &LanguageCode, to_code: &LanguageCode) -> bool {
        self::can_translate(self, from_code, to_code)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct LanguageAvailabilityRow {
    pub language: Language,
    pub availability: LangAvailability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct LanguageOverview {
    pub language: Language,
    pub availability: LangAvailability,
    pub core_size_bytes: u64,
    pub core_installed: bool,
    pub dictionary_size_bytes: u64,
    pub dictionary_installed: bool,
    pub tts_size_bytes: u64,
    pub tts_voice_regions: Vec<TtsVoiceRegionOverview>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct TtsVoiceRegionOverview {
    pub code: String,
    pub display_name: String,
    pub voices: Vec<TtsVoiceOverview>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct TtsVoiceOverview {
    pub pack_info: TtsVoicePackInfo,
    pub installed: bool,
}

impl<'a, C> PackResolver<'a, C>
where
    C: PackInstallChecker,
{
    pub(crate) fn new(catalog: &'a LanguageCatalog, install_checker: &'a C) -> Self {
        Self {
            catalog,
            install_checker,
            status_cache: HashMap::new(),
        }
    }

    pub(crate) fn status(&mut self, pack_id: &str) -> Option<PackInstallStatus> {
        if let Some(status) = self.status_cache.get(pack_id) {
            return Some(status.clone());
        }

        let pack = self.catalog.pack(pack_id)?;
        let checker = self.install_checker;
        let file_present = |file: &AssetFileV2| {
            installed_at(checker, &file.install_path, file.install_marker.as_ref())
        };

        let missing_dependency_ids = pack
            .depends_on
            .iter()
            .filter(|dep_id| self.status(dep_id).is_none_or(|status| !status.installed))
            .cloned()
            .collect::<Vec<_>>();

        let mut missing_files = Vec::new();
        let mut upgrade_files = Vec::new();
        let mut superseded_files = Vec::new();
        let mut active_files = Vec::new();
        let mut any_file_present = false;
        let mut role_groups: Vec<(&FileRole, Vec<&AssetFileV2>)> = Vec::new();
        for file in &pack.files {
            let role = &file.role;
            match role_groups.iter_mut().find(|(r, _)| *r == role) {
                Some((_, group)) => group.push(file),
                None => role_groups.push((role, vec![file])),
            }
        }
        for (_, mut group) in role_groups {
            group.sort_by_key(|file| std::cmp::Reverse(file.priority));
            let present = group
                .iter()
                .filter(|file| file_present(file))
                .copied()
                .collect::<Vec<_>>();
            any_file_present |= !present.is_empty();
            if present.is_empty()
                && let Some(stand_in) = replaced_stand_in(checker, &group, &missing_dependency_ids)
            {
                // An install from an older catalog keeps loading the file this
                // one replaces, so it doesn't break and needs no download; the
                // replacement is offered as an upgrade.
                any_file_present = true;
                upgrade_files.push(group[0].clone());
                active_files.push(stand_in);
                continue;
            }
            let Some(best_present) = present.first() else {
                // No file of this role on disk. A required role gates the pack;
                // an optional one is offered as an improvement upgrade instead so
                // it doesn't mark an installed language as missing.
                match group[0].requirement {
                    FileRequirement::Required => missing_files.push(group[0].clone()),
                    FileRequirement::Optional => upgrade_files.push(group[0].clone()),
                }
                continue;
            };
            // The disk-resolved file for this role: what runtime resolution
            // (translation/TTS) must load, so an install still on an older
            // alternative uses the file it actually has.
            active_files.push((*best_present).clone());
            if group[0].install_path != best_present.install_path {
                upgrade_files.push(group[0].clone());
            }
            superseded_files.extend(present[1..].iter().map(|file| (*file).clone()));
        }

        let status = PackInstallStatus {
            pack_id: pack_id.to_string(),
            installed: missing_files.is_empty() && missing_dependency_ids.is_empty(),
            any_file_present,
            missing_files,
            missing_dependency_ids,
            upgrade_files,
            superseded_files,
            active_files,
        };
        self.status_cache
            .insert(pack_id.to_string(), status.clone());
        Some(status)
    }

    pub(crate) fn is_installed(&mut self, pack_id: &str) -> bool {
        self.status(pack_id).is_some_and(|status| status.installed)
    }
}

pub fn build_catalog_snapshot<C>(
    catalog: LanguageCatalog,
    base_dir: String,
    install_checker: &C,
) -> CatalogSnapshot
where
    C: PackInstallChecker,
{
    let mut resolver = PackResolver::new(&catalog, install_checker);
    let pack_ids = catalog.packs.keys().cloned().collect::<Vec<_>>();
    for pack_id in &pack_ids {
        let _ = resolver.status(pack_id);
    }

    let availability_by_code = compute_language_availability(&catalog, &mut resolver)
        .into_iter()
        .map(|(language, availability)| (language.code, availability))
        .collect::<HashMap<_, _>>();

    let pack_statuses = pack_ids
        .into_iter()
        .filter_map(|pack_id| resolver.status(&pack_id).map(|status| (pack_id, status)))
        .collect::<HashMap<_, _>>();

    CatalogSnapshot {
        catalog,
        base_dir,
        pack_statuses,
        availability_by_code,
    }
}

pub fn language_rows_in_snapshot(snapshot: &CatalogSnapshot) -> Vec<LanguageAvailabilityRow> {
    snapshot
        .catalog
        .language_list()
        .into_iter()
        .map(|language| {
            let availability = snapshot
                .availability_by_code
                .get(&language.code)
                .copied()
                .unwrap_or_default();
            LanguageAvailabilityRow {
                language,
                availability,
            }
        })
        .collect()
}

pub fn build_language_overview(snapshot: &CatalogSnapshot) -> Vec<LanguageOverview> {
    let catalog = &snapshot.catalog;
    language_rows_in_snapshot(snapshot)
        .into_iter()
        .map(|row| {
            let language = row.language;
            let language_code = LanguageCode::from(language.code.as_str());

            let core_size_bytes = catalog.translation_size_bytes_for_language(&language_code);
            let dictionary_size_bytes = catalog
                .dictionary_info_for(&language)
                .map(|info| info.size)
                .unwrap_or(0);
            let tts_size_bytes = catalog.tts_size_bytes_for_language(&language_code);

            let core_installed = core_size_bytes > 0
                && plan_language_download(snapshot, &language_code)
                    .tasks
                    .is_empty();
            let dictionary_installed = dictionary_size_bytes > 0
                && plan_dictionary_download(snapshot, &language_code)
                    .is_some_and(|plan| plan.tasks.is_empty());

            let tts_voice_regions = tts_voice_picker_regions(snapshot, &language_code)
                .into_iter()
                .map(|region| TtsVoiceRegionOverview {
                    code: region.code,
                    display_name: region.display_name,
                    voices: region
                        .voices
                        .into_iter()
                        .map(|pack_info| {
                            let installed = plan_tts_download(
                                snapshot,
                                &language_code,
                                Some(pack_info.pack_id.as_str()),
                            )
                            .is_some_and(|plan| plan.tasks.is_empty());
                            TtsVoiceOverview {
                                pack_info,
                                installed,
                            }
                        })
                        .collect(),
                })
                .collect();

            LanguageOverview {
                language,
                availability: row.availability,
                core_size_bytes,
                core_installed,
                dictionary_size_bytes,
                dictionary_installed,
                tts_size_bytes,
                tts_voice_regions,
            }
        })
        .collect()
}

fn pack_installed_in_snapshot(snapshot: &CatalogSnapshot, pack_id: &str) -> bool {
    snapshot
        .pack_statuses
        .get(pack_id)
        .is_some_and(|status| status.installed)
}

pub fn installed_ocr_engines_for_language(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> Vec<String> {
    snapshot
        .catalog
        .ocr_engines_for_language(language_code)
        .into_iter()
        .filter(|(_, pack_id)| pack_installed_in_snapshot(snapshot, pack_id))
        .map(|(engine, _)| engine)
        .collect()
}

/// Whether the shared OCR foundation is usable at all: the ppocr detector pack
/// is installed. Recognizers are per-language, but without the detector every
/// OCR path fails outright — callers use this to refuse to start (with a
/// user-visible message) instead of silently rendering nothing.
pub fn ocr_engine_ready(snapshot: &CatalogSnapshot) -> bool {
    snapshot.catalog.packs.iter().any(|(pack_id, pack)| {
        matches!(&pack.kind, PackKind::Ocr(OcrPack::PpocrDetector))
            && pack_installed_in_snapshot(snapshot, pack_id)
    })
}

pub fn available_ocr_engines_for_language(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> Vec<String> {
    snapshot
        .catalog
        .ocr_engines_for_language(language_code)
        .into_iter()
        .map(|(engine, _)| engine)
        .collect()
}

pub fn plan_ocr_engine_download(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
    engine: &str,
) -> Option<DownloadPlan> {
    let pack_id = snapshot
        .catalog
        .ocr_pack_id_for_engine(language_code, engine)?;
    Some(download_plan(
        snapshot,
        install_files_in_snapshot(snapshot, [pack_id.as_str()]),
    ))
}

pub fn plan_ocr_engine_downloads(
    snapshot: &CatalogSnapshot,
    language_codes: &[LanguageCode],
    engine: &str,
) -> DownloadPlan {
    let pack_ids = language_codes
        .iter()
        .filter_map(|language_code| {
            snapshot
                .catalog
                .ocr_pack_id_for_engine(language_code, engine)
        })
        .collect::<Vec<_>>();
    download_plan(
        snapshot,
        install_files_in_snapshot(snapshot, pack_ids.iter().map(String::as_str)),
    )
}

// A fresh install fetches the best file of every role, optional ones included:
// the highest-priority optional surfaces as an upgrade_file when nothing of that
// role is on disk, so folding those into the download plan means a fresh install
// lands the best+latest optional and never self-prompts for it afterwards.
// Required version upgrades of files already on disk stay out of the plan — they
// remain a deliberate, user-initiated upgrade rather than a silent re-download.
fn install_files_in_snapshot<'a, I>(
    snapshot: &'a CatalogSnapshot,
    pack_ids: I,
) -> Vec<MissingPackFile>
where
    I: IntoIterator<Item = &'a str>,
{
    status_files_in_snapshot(snapshot, pack_ids, |status| {
        status
            .missing_files
            .iter()
            .chain(
                status
                    .upgrade_files
                    .iter()
                    .filter(|file| matches!(file.requirement, FileRequirement::Optional)),
            )
            .collect()
    })
}

fn status_files_in_snapshot<'a, I>(
    snapshot: &'a CatalogSnapshot,
    pack_ids: I,
    select: impl for<'s> Fn(&'s PackInstallStatus) -> Vec<&'s AssetFileV2>,
) -> Vec<MissingPackFile>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut selected = Vec::new();
    let mut seen_install_paths = HashSet::new();

    for pack_id in snapshot.catalog.dependency_closure(pack_ids) {
        let Some(pack) = snapshot.catalog.pack(&pack_id) else {
            continue;
        };
        let Some(status) = snapshot.pack_statuses.get(&pack_id) else {
            continue;
        };
        for file in select(status) {
            if seen_install_paths.insert(file.install_path.clone()) {
                selected.push(MissingPackFile {
                    pack_id: pack.id.clone(),
                    file: file.clone(),
                });
            }
        }
    }

    selected
}

pub fn plan_ocr_engine_upgrades(
    snapshot: &CatalogSnapshot,
    language_codes: &[LanguageCode],
    engine: &str,
) -> DownloadPlan {
    let pack_ids = language_codes
        .iter()
        .filter_map(|language_code| {
            snapshot
                .catalog
                .ocr_pack_id_for_engine(language_code, engine)
        })
        .collect::<Vec<_>>();
    upgrade_plan(snapshot, &pack_ids)
}

fn translation_pack_ids_for_languages(
    catalog: &LanguageCatalog,
    language_codes: &[LanguageCode],
) -> Vec<String> {
    let mut ids = Vec::new();
    for code in language_codes {
        for id in [
            catalog.translation_pack_id("en", code.as_str()),
            catalog.translation_pack_id(code.as_str(), "en"),
        ]
        .into_iter()
        .flatten()
        {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

/// Non-English language codes whose installed translation pack has a
/// higher-priority variant available on top of what is on disk — i.e. a newer
/// model. Drives the "Upgrades for: ..." card. A pair still on its old model
/// keeps working; this only lists the ones where an upgrade exists.
pub fn translation_upgrade_language_codes(snapshot: &CatalogSnapshot) -> Vec<LanguageCode> {
    let mut codes: Vec<LanguageCode> = Vec::new();
    for (pack_id, status) in &snapshot.pack_statuses {
        if status.upgrade_files.is_empty() {
            continue;
        }
        let Some(pack) = snapshot.catalog.pack(pack_id) else {
            continue;
        };
        let PackKind::Translation(translation) = &pack.kind else {
            continue;
        };
        let code = if translation.from == "en" {
            &translation.to
        } else {
            &translation.from
        };
        let code = LanguageCode::new(code.clone());
        if !codes.contains(&code) {
            codes.push(code);
        }
    }
    codes.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    codes
}

/// The download that upgrades the given languages' translation packs to their
/// highest-priority variant. Mirrors plan_ocr_engine_upgrades for translation.
pub fn plan_translation_upgrades(
    snapshot: &CatalogSnapshot,
    language_codes: &[LanguageCode],
) -> DownloadPlan {
    let pack_ids = translation_pack_ids_for_languages(&snapshot.catalog, language_codes);
    upgrade_plan(snapshot, &pack_ids)
}

fn installed_tts_packs(snapshot: &CatalogSnapshot) -> impl Iterator<Item = (&String, &TtsPack)> {
    snapshot
        .pack_statuses
        .iter()
        .filter(|(_, status)| status.installed)
        .filter_map(|(pack_id, _)| match &snapshot.catalog.pack(pack_id)?.kind {
            PackKind::Tts(tts) => Some((pack_id, tts)),
            _ => None,
        })
}

pub fn tts_upgrade_language_codes(snapshot: &CatalogSnapshot) -> Vec<LanguageCode> {
    let mut codes = installed_tts_packs(snapshot)
        .filter(|(pack_id, _)| {
            snapshot
                .catalog
                .dependency_closure([pack_id.as_str()])
                .iter()
                .filter_map(|id| snapshot.pack_statuses.get(id))
                .any(|status| !status.upgrade_files.is_empty())
        })
        .map(|(_, tts)| LanguageCode::new(tts.language.clone()))
        .collect::<Vec<_>>();
    codes.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    codes.dedup();
    codes
}

pub fn plan_tts_upgrades(
    snapshot: &CatalogSnapshot,
    language_codes: &[LanguageCode],
) -> DownloadPlan {
    let pack_ids = installed_tts_packs(snapshot)
        .filter(|(_, tts)| {
            language_codes
                .iter()
                .any(|code| code.as_str() == tts.language)
        })
        .map(|(pack_id, _)| pack_id.clone())
        .collect::<Vec<_>>();
    upgrade_plan(snapshot, &pack_ids)
}

// Downloads every upgrade file of the packs' closure, then deletes what those
// files outrank: the on-disk alternatives of each upgraded role plus anything
// already superseded in those packs. A path another (non-upgraded) pack still
// resolves to stays, since deleting it would break that pack.
fn upgrade_plan(snapshot: &CatalogSnapshot, pack_ids: &[String]) -> DownloadPlan {
    let closure = snapshot
        .catalog
        .dependency_closure(pack_ids.iter().map(String::as_str))
        .into_iter()
        .collect::<HashSet<_>>();
    let files = status_files_in_snapshot(snapshot, pack_ids.iter().map(String::as_str), |status| {
        status.upgrade_files.iter().collect()
    });
    let in_use_elsewhere = snapshot
        .pack_statuses
        .iter()
        .filter(|(pack_id, _)| !closure.contains(*pack_id))
        .flat_map(|(_, status)| &status.active_files)
        .map(|file| file.install_path.as_str())
        .collect::<HashSet<_>>();
    let mut file_paths = closure
        .iter()
        .filter_map(|pack_id| snapshot.pack_statuses.get(pack_id))
        .flat_map(|status| {
            let upgraded_roles = status
                .upgrade_files
                .iter()
                .map(|file| &file.role)
                .collect::<HashSet<_>>();
            status
                .active_files
                .iter()
                .filter(move |file| upgraded_roles.contains(&file.role))
                .chain(&status.superseded_files)
        })
        .map(|file| file.install_path.clone())
        .filter(|path| !in_use_elsewhere.contains(path.as_str()))
        .collect::<Vec<_>>();
    file_paths.sort();
    file_paths.dedup();
    DownloadPlan {
        delete_after: DeletePlan {
            file_paths,
            directory_paths: Vec::new(),
        },
        ..download_plan(snapshot, files)
    }
}

/// Downloads that restore packs the user already has to a working state: packs
/// with some files on disk but missing required ones (a partial download, or a
/// catalog change that orphaned the on-disk variant), plus whatever their
/// dependency chain is missing. Packs with no files at all are not included —
/// those were never installed, and this app never downloads unrequested content.
/// The exception is a language whose translation is installed: its core packs
/// were requested along with it, so any that are absent (e.g. installs from
/// before the language gained PP-OCR packs, issue #341) are restored too.
pub fn plan_repair(snapshot: &CatalogSnapshot) -> DownloadPlan {
    let broken_pack_ids = snapshot
        .pack_statuses
        .values()
        .filter(|status| !status.installed && status.any_file_present)
        .map(|status| status.pack_id.clone());
    let incomplete_language_pack_ids = snapshot
        .catalog
        .languages
        .keys()
        .filter(|code| {
            snapshot
                .catalog
                .root_pack_ids_for_feature(
                    &LanguageCode::from(code.as_str()),
                    LanguageFeature::Translation,
                )
                .iter()
                .any(|pack_id| pack_installed_in_snapshot(snapshot, pack_id))
        })
        .flat_map(|code| snapshot.catalog.core_pack_ids_for_language(code));
    let root_pack_ids = broken_pack_ids
        .chain(incomplete_language_pack_ids)
        .collect::<BTreeSet<_>>();
    download_plan(
        snapshot,
        status_files_in_snapshot(
            snapshot,
            root_pack_ids.iter().map(String::as_str),
            |status| status.missing_files.iter().collect(),
        ),
    )
}

pub fn plan_delete_superseded_files(snapshot: &CatalogSnapshot) -> DeletePlan {
    let mut file_paths = snapshot
        .pack_statuses
        .values()
        .flat_map(|status| &status.superseded_files)
        .map(|file| file.install_path.clone())
        .collect::<Vec<_>>();
    file_paths.sort();
    file_paths.dedup();
    DeletePlan {
        file_paths,
        directory_paths: Vec::new(),
    }
}

pub fn can_translate(
    snapshot: &CatalogSnapshot,
    from_code: &LanguageCode,
    to_code: &LanguageCode,
) -> bool {
    if from_code == to_code {
        return true;
    }

    let direction_installed = |from: &str, to: &str| {
        snapshot
            .catalog
            .translation_pack_id(from, to)
            .as_deref()
            .is_some_and(|pack_id| pack_installed_in_snapshot(snapshot, pack_id))
    };

    if from_code.as_str() == "en" {
        return direction_installed("en", to_code.as_str());
    }
    if to_code.as_str() == "en" {
        return direction_installed(from_code.as_str(), "en");
    }

    direction_installed(from_code.as_str(), "en") && direction_installed("en", to_code.as_str())
}

pub(crate) fn installed_tts_pack_id_for_language<C>(
    catalog: &LanguageCatalog,
    language_code: &LanguageCode,
    resolver: &mut PackResolver<'_, C>,
) -> Option<String>
where
    C: PackInstallChecker,
{
    catalog
        .tts_pack_ids_for_language(language_code)
        .into_iter()
        .find(|pack_id| resolver.is_installed(pack_id))
}

pub(crate) fn compute_language_availability<C>(
    catalog: &LanguageCatalog,
    resolver: &mut PackResolver<'_, C>,
) -> HashMap<Language, LangAvailability>
where
    C: PackInstallChecker,
{
    let mut rows = HashMap::new();

    for (code, info) in &catalog.languages {
        let language = info.language.clone();
        let any_ocr_installed = info
            .resources
            .ocr_packs
            .iter()
            .any(|(_, pack_id)| resolver.is_installed(pack_id));
        let dictionary_pack_id = catalog.dictionary_pack_id_for_language(code);
        let availability = if language.is_english() {
            LangAvailability {
                has_from_english: true,
                has_to_english: true,
                ocr_files: any_ocr_installed,
                dictionary_files: dictionary_pack_id
                    .as_deref()
                    .is_some_and(|pack_id| resolver.is_installed(pack_id)),
                tts_files: installed_tts_pack_id_for_language(
                    catalog,
                    &LanguageCode::from(code.as_str()),
                    resolver,
                )
                .is_some(),
            }
        } else {
            let from_pack_id = catalog.translation_pack_id("en", code);
            let to_pack_id = catalog.translation_pack_id(code, "en");
            LangAvailability {
                has_from_english: from_pack_id
                    .as_deref()
                    .is_some_and(|pack_id| resolver.is_installed(pack_id)),
                has_to_english: to_pack_id
                    .as_deref()
                    .is_some_and(|pack_id| resolver.is_installed(pack_id)),
                ocr_files: any_ocr_installed,
                dictionary_files: dictionary_pack_id
                    .as_deref()
                    .is_some_and(|pack_id| resolver.is_installed(pack_id)),
                tts_files: installed_tts_pack_id_for_language(
                    catalog,
                    &LanguageCode::from(code.as_str()),
                    resolver,
                )
                .is_some(),
            }
        };
        rows.insert(language, availability);
    }

    rows
}

pub fn resolve_tts_voice_files(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> Option<ResolvedTtsVoiceFiles> {
    resolve_tts_voice_files_for_pack(snapshot, language_code, None)
}

pub fn resolve_tts_voice_files_for_pack(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
    pack_id: Option<&str>,
) -> Option<ResolvedTtsVoiceFiles> {
    let voice_pack_id = match pack_id {
        Some(requested) if pack_installed_in_snapshot(snapshot, requested) => requested.to_string(),
        _ => snapshot
            .catalog
            .tts_pack_ids_for_language(language_code)
            .into_iter()
            .find(|pack_id| pack_installed_in_snapshot(snapshot, pack_id))?,
    };
    let voice_pack = snapshot.catalog.pack(&voice_pack_id)?;
    let PackKind::Tts(tts) = &voice_pack.kind else {
        return None;
    };
    let engine = tts.engine.clone().unwrap_or_else(|| "piper".to_string());

    // The model and its aux file may live in the voice pack or in a shared
    // dependency (kokoro core/voices, cotovia lexicon), so resolve across the
    // closure's on-disk files.
    let active_files = snapshot
        .catalog
        .dependency_closure([voice_pack_id.as_str()])
        .into_iter()
        .filter_map(|pack_id| snapshot.pack_statuses.get(&pack_id))
        .flat_map(|status| status.active_files.iter())
        .collect::<Vec<_>>();
    let by_role = |role: &FileRole| {
        active_files
            .iter()
            .copied()
            .filter(|file| &file.role == role)
            .max_by_key(|file| file.priority)
    };

    let model_asset = by_role(&FileRole::Model)?;
    let aux_asset = by_role(&tts.aux_role)?;
    let vocoder_asset = by_role(&FileRole::new("vocoder"));
    Some(ResolvedTtsVoiceFiles {
        engine,
        model_install_path: model_asset.install_path.clone(),
        aux_install_path: aux_asset.install_path.clone(),
        vocoder_install_path: vocoder_asset.map(|file| file.install_path.clone()),
        language_code: language_code.as_str().to_string(),
        locale: tts.locale.clone(),
        voice: tts.voice.clone(),
        speaker_id: tts.default_speaker_id,
    })
}

fn download_task_for(pack: &PackRecord, file: &AssetFileV2) -> DownloadTask {
    let gzipped = file
        .source_path
        .as_deref()
        .unwrap_or(file.url.as_str())
        .ends_with(".gz");
    let install = match &file.archive {
        Some(archive) => TaskInstall::Zip {
            extract_to: archive.extract_to.clone(),
            delete_after_extract: archive.delete_after_extract,
            marker: file.install_marker.clone(),
        },
        None if gzipped && matches!(&pack.kind, PackKind::Translation(_)) => TaskInstall::Gunzip,
        None => TaskInstall::Raw,
    };
    DownloadTask {
        pack_id: pack.id.clone(),
        install_path: file.install_path.clone(),
        url: file.url.clone(),
        size_bytes: file.size_bytes,
        install,
    }
}

fn download_plan(snapshot: &CatalogSnapshot, files: Vec<MissingPackFile>) -> DownloadPlan {
    let tasks = files
        .into_iter()
        .filter_map(|item| {
            let pack = snapshot.catalog.pack(&item.pack_id)?;
            Some(download_task_for(pack, &item.file))
        })
        .collect::<Vec<_>>();
    DownloadPlan {
        total_size: tasks.iter().map(|task| task.size_bytes).sum(),
        tasks,
        delete_after: DeletePlan::default(),
    }
}

// A replaced file can be shared by sibling packs (one voices file for several
// voices), so its presence alone does not say which of them were installed.
// The catalog it came from counted such a pack installed once its dependencies
// were, so only then does the replaced file stand in for this pack.
fn replaced_stand_in(
    checker: &impl PackInstallChecker,
    group: &[&AssetFileV2],
    missing_dependency_ids: &[String],
) -> Option<AssetFileV2> {
    if !missing_dependency_ids.is_empty() {
        return None;
    }
    group.iter().find_map(|file| {
        let path = file
            .replaces
            .iter()
            .find(|path| checker.file_exists(path))?;
        let name = path
            .rsplit('/')
            .next()
            .expect("rsplit yields at least one piece");
        Some(AssetFileV2 {
            name: name.to_string(),
            install_path: path.clone(),
            replaces: Vec::new(),
            ..(*file).clone()
        })
    })
}

fn installed_at(
    checker: &impl PackInstallChecker,
    install_path: &str,
    marker: Option<&InstallMarker>,
) -> bool {
    match marker {
        Some(marker) => checker.install_marker_exists(&marker.path, marker.version),
        None => checker.file_exists(install_path),
    }
}

// Plans are computed before any transfer starts, so a task can be satisfied by
// the time it runs (another download shared the file); the downloader re-checks
// right before fetching.
pub fn download_task_installed(checker: &impl PackInstallChecker, task: &DownloadTask) -> bool {
    let marker = match &task.install {
        TaskInstall::Zip { marker, .. } => marker.as_ref(),
        TaskInstall::Raw | TaskInstall::Gunzip => None,
    };
    installed_at(checker, &task.install_path, marker)
}

// Archives are built either with their install root as the top-level directory
// or with its contents at the top level; entries are re-rooted under the
// marker's directory so both layouts land in the same place.
pub fn plan_zip_extraction(
    extract_to: &str,
    marker: Option<&InstallMarker>,
    entry_names: &[String],
) -> Result<ZipExtractionPlan, String> {
    let install_root = marker
        .and_then(|marker| std::path::Path::new(&marker.path).parent())
        .and_then(|parent| parent.file_name())
        .map(|name| name.to_string_lossy().to_string());
    let mut entries = Vec::new();
    let mut replace_paths = BTreeSet::new();
    for entry_name in entry_names {
        let trimmed = entry_name.trim_start_matches('/');
        let trimmed = trimmed.strip_prefix("./").unwrap_or(trimmed);
        let parts = trimmed
            .split('/')
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        if parts.iter().any(|part| *part == "..") {
            return Err(format!(
                "zip entry {entry_name} escapes the extraction root"
            ));
        }
        if parts.is_empty() {
            continue;
        }
        let parts = match install_root.as_deref() {
            Some(root) if parts[0] != root => std::iter::once(root).chain(parts).collect(),
            _ => parts,
        };
        replace_paths.insert(join_install_path(extract_to, &parts[..parts.len().min(2)]));
        entries.push(ZipEntryTarget {
            entry_name: entry_name.clone(),
            install_path: join_install_path(extract_to, &parts),
            is_directory: entry_name.ends_with('/'),
        });
    }
    Ok(ZipExtractionPlan {
        entries,
        replace_paths: replace_paths.into_iter().collect(),
    })
}

fn join_install_path(base: &str, parts: &[&str]) -> String {
    std::iter::once(base.trim_end_matches('/'))
        .chain(parts.iter().copied())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("/")
}

pub fn plan_language_download(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> DownloadPlan {
    let root_pack_ids = snapshot
        .catalog
        .core_pack_ids_for_language(language_code.as_str());
    download_plan(
        snapshot,
        install_files_in_snapshot(snapshot, root_pack_ids.iter().map(String::as_str)),
    )
}

// Support kinds the app needs as soon as any language is installed; installing
// a language brings them along when they are missing.
const CORE_SUPPORT_KINDS: [&str; 1] = ["doc_detect"];

pub fn plan_language_install(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> DownloadPlan {
    let core_pack_ids = snapshot
        .catalog
        .core_pack_ids_for_language(language_code.as_str());
    let support_pack_ids = CORE_SUPPORT_KINDS
        .iter()
        .flat_map(|kind| snapshot.catalog.support_pack_ids_by_kind(kind));
    let root_pack_ids = core_pack_ids
        .into_iter()
        .chain(support_pack_ids)
        .collect::<BTreeSet<_>>();
    download_plan(
        snapshot,
        install_files_in_snapshot(snapshot, root_pack_ids.iter().map(String::as_str)),
    )
}

pub fn plan_dictionary_download(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> Option<DownloadPlan> {
    let pack_id = snapshot
        .catalog
        .dictionary_pack_id_for_language(language_code.as_str())?;
    Some(download_plan(
        snapshot,
        install_files_in_snapshot(snapshot, [pack_id.as_str()]),
    ))
}

pub fn installed_tts_voices_for_language(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> Vec<InstalledTtsPack> {
    snapshot
        .catalog
        .tts_pack_ids_for_language(language_code)
        .into_iter()
        .filter(|pack_id| pack_installed_in_snapshot(snapshot, pack_id))
        .filter_map(|pack_id| {
            let pack = snapshot.catalog.pack(&pack_id)?;
            let PackKind::Tts(tts) = &pack.kind else {
                return None;
            };
            let voice = tts.voice.clone().unwrap_or_else(|| pack.id.clone());
            let display_name = voice_display_name(&voice);
            let voices = if tts.speakers.is_empty() {
                vec![TtsSpeakerEntry {
                    name: voice,
                    display_name: display_name.clone(),
                    speaker_id: tts.default_speaker_id.unwrap_or(0),
                }]
            } else {
                // The engine resolves a requested voice by the model's own speaker names.
                (0..)
                    .zip(&tts.speakers)
                    .map(|(speaker_id, name)| TtsSpeakerEntry {
                        name: name.clone(),
                        display_name: voice_display_name(name),
                        speaker_id,
                    })
                    .collect()
            };
            Some(InstalledTtsPack {
                pack_id,
                display_name,
                voices,
            })
        })
        .collect()
}

pub fn tts_voice_picker_regions(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> Vec<TtsVoicePickerRegion> {
    snapshot
        .catalog
        .ordered_tts_regions_for_language(language_code)
        .into_iter()
        .map(|(code, region)| TtsVoicePickerRegion {
            code,
            display_name: region.display_name,
            voices: region
                .voices
                .into_iter()
                .filter_map(|pack_id| tts_voice_pack_info(snapshot, &pack_id))
                .collect(),
        })
        .filter(|region| !region.voices.is_empty())
        .collect()
}

fn tts_voice_pack_info(snapshot: &CatalogSnapshot, pack_id: &str) -> Option<TtsVoicePackInfo> {
    let pack = snapshot.catalog.pack(pack_id)?;
    let PackKind::Tts(tts) = &pack.kind else {
        return None;
    };
    Some(TtsVoicePackInfo {
        pack_id: pack.id.clone(),
        display_name: tts.voice.clone().unwrap_or_else(|| pack.id.clone()),
        quality: tts.quality.clone(),
        size_bytes: snapshot.catalog.pack_size_bytes(pack_id),
        download_size_bytes: download_plan(
            snapshot,
            install_files_in_snapshot(snapshot, [pack_id]),
        )
        .total_size,
        speaker_count: u32::try_from(tts.speakers.len().max(1))
            .expect("a voice pack lists fewer than 2^32 speakers"),
        sample_url: tts.sample_url.clone(),
    })
}

pub fn installed_tts_voice_picker_regions(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> Vec<TtsVoicePickerRegion> {
    tts_voice_picker_regions(snapshot, language_code)
        .into_iter()
        .filter_map(|region| {
            let installed_voices: Vec<_> = region
                .voices
                .into_iter()
                .filter(|pack_info| pack_installed_in_snapshot(snapshot, &pack_info.pack_id))
                .collect();
            if installed_voices.is_empty() {
                None
            } else {
                Some(TtsVoicePickerRegion {
                    code: region.code,
                    display_name: region.display_name,
                    voices: installed_voices,
                })
            }
        })
        .collect()
}

pub fn plan_tts_download(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
    selected_pack_id: Option<&str>,
) -> Option<DownloadPlan> {
    let selected_pack_id = match selected_pack_id {
        Some(pack_id)
            if snapshot
                .catalog
                .tts_pack_ids_for_language(language_code)
                .iter()
                .any(|candidate| candidate == pack_id) =>
        {
            pack_id.to_string()
        }
        Some(_) => return None,
        None => snapshot
            .catalog
            .default_tts_pack_id_for_language(language_code)?,
    };
    Some(download_plan(
        snapshot,
        install_files_in_snapshot(snapshot, [selected_pack_id.as_str()]),
    ))
}

pub fn plan_support_download_by_kind(
    snapshot: &CatalogSnapshot,
    support_kind: &str,
) -> Option<DownloadPlan> {
    let pack_ids = snapshot.catalog.support_pack_ids_by_kind(support_kind);
    if pack_ids.is_empty() {
        return None;
    }
    Some(download_plan(
        snapshot,
        install_files_in_snapshot(snapshot, pack_ids.iter().map(String::as_str)),
    ))
}

fn delete_plan_for_pack_ids<'a, I>(snapshot: &CatalogSnapshot, pack_ids: I) -> DeletePlan
where
    I: IntoIterator<Item = &'a str>,
{
    let pack_ids = pack_ids.into_iter().collect::<HashSet<_>>();
    // A path can be loaded by a pack that stays (a replaced file shared by sibling
    // packs), and deleting it would break that pack.
    let in_use_elsewhere = snapshot
        .pack_statuses
        .iter()
        .filter(|(pack_id, _)| !pack_ids.contains(pack_id.as_str()))
        .flat_map(|(_, status)| &status.active_files)
        .map(|file| file.install_path.as_str())
        .collect::<HashSet<_>>();
    let mut file_paths = Vec::new();
    let mut file_seen = HashSet::new();
    let mut directory_paths = Vec::new();
    let mut directory_seen = HashSet::new();

    for pack_id in &pack_ids {
        let Some(pack) = snapshot.catalog.pack(pack_id) else {
            continue;
        };
        for file in &pack.files {
            for path in std::iter::once(&file.install_path).chain(&file.replaces) {
                if !in_use_elsewhere.contains(path.as_str()) && file_seen.insert(path.clone()) {
                    file_paths.push(path.clone());
                }
            }
            if file.archive.is_some()
                && let Some(marker) = &file.install_marker
                && let Some(parent) = std::path::Path::new(&marker.path).parent()
            {
                let path = parent.to_string_lossy().to_string();
                if !path.is_empty() && directory_seen.insert(path.clone()) {
                    directory_paths.push(path);
                }
            }
        }
    }

    DeletePlan {
        file_paths,
        directory_paths,
    }
}

fn delete_pack_ids<'a, I>(
    catalog: &LanguageCatalog,
    target_root_packs: I,
    keep_root_packs: HashSet<String>,
) -> HashSet<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let target = catalog
        .dependency_closure(target_root_packs)
        .into_iter()
        .collect::<HashSet<_>>();
    let keep = catalog
        .dependency_closure(keep_root_packs.iter().map(String::as_str))
        .into_iter()
        .collect::<HashSet<_>>();
    target.difference(&keep).cloned().collect()
}

pub fn plan_delete_dictionary(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> DeletePlan {
    let Some(target_pack) = snapshot
        .catalog
        .dictionary_pack_id_for_language(language_code.as_str())
    else {
        return DeletePlan::default();
    };
    let keep_root_packs = snapshot
        .catalog
        .languages
        .keys()
        .filter(|code| code.as_str() != language_code.as_str())
        .filter_map(|code| snapshot.catalog.dictionary_pack_id_for_language(code))
        .filter(|pack_id| pack_id != &target_pack && pack_installed_in_snapshot(snapshot, pack_id))
        .collect::<HashSet<_>>();
    let delete_pack_ids =
        delete_pack_ids(&snapshot.catalog, [target_pack.as_str()], keep_root_packs);
    delete_plan_for_pack_ids(snapshot, delete_pack_ids.iter().map(String::as_str))
}

pub fn plan_delete_language(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> DeletePlan {
    let target_root_packs = snapshot
        .catalog
        .core_pack_ids_for_language(language_code.as_str());
    let keep_root_packs = snapshot
        .catalog
        .languages
        .keys()
        .filter(|code| code.as_str() != language_code.as_str())
        .flat_map(|code| snapshot.catalog.core_pack_ids_for_language(code))
        .filter(|pack_id| pack_installed_in_snapshot(snapshot, pack_id))
        .collect::<HashSet<_>>();
    let delete_pack_ids = delete_pack_ids(
        &snapshot.catalog,
        target_root_packs.iter().map(String::as_str),
        keep_root_packs,
    );
    delete_plan_for_pack_ids(snapshot, delete_pack_ids.iter().map(String::as_str))
}

pub fn plan_delete_tts(snapshot: &CatalogSnapshot, language_code: &LanguageCode) -> DeletePlan {
    let target_root_packs = snapshot
        .catalog
        .tts_pack_ids_for_language(language_code)
        .into_iter()
        .filter(|pack_id| pack_installed_in_snapshot(snapshot, pack_id))
        .collect::<HashSet<_>>();
    if target_root_packs.is_empty() {
        return DeletePlan::default();
    }
    let keep_root_packs = snapshot
        .catalog
        .languages
        .keys()
        .filter(|code| code.as_str() != language_code.as_str())
        .flat_map(|code| {
            snapshot
                .catalog
                .tts_pack_ids_for_language(&LanguageCode::from(code.as_str()))
        })
        .filter(|pack_id| pack_installed_in_snapshot(snapshot, pack_id))
        .collect::<HashSet<_>>();
    let delete_pack_ids = delete_pack_ids(
        &snapshot.catalog,
        target_root_packs.iter().map(String::as_str),
        keep_root_packs,
    );
    delete_plan_for_pack_ids(snapshot, delete_pack_ids.iter().map(String::as_str))
}

pub fn plan_delete_support_by_kind(snapshot: &CatalogSnapshot, support_kind: &str) -> DeletePlan {
    let pack_ids = snapshot
        .catalog
        .support_pack_ids_by_kind(support_kind)
        .into_iter()
        .collect::<HashSet<_>>();
    delete_plan_for_pack_ids(snapshot, pack_ids.iter().map(String::as_str))
}

pub fn plan_delete_tts_pack(snapshot: &CatalogSnapshot, pack_id: &str) -> DeletePlan {
    if !pack_installed_in_snapshot(snapshot, pack_id) {
        return DeletePlan::default();
    }
    let keep_root_packs = snapshot
        .catalog
        .languages
        .keys()
        .flat_map(|code| {
            snapshot
                .catalog
                .tts_pack_ids_for_language(&LanguageCode::from(code.as_str()))
        })
        .filter(|other| other != pack_id && pack_installed_in_snapshot(snapshot, other))
        .collect::<HashSet<_>>();
    let delete_pack_ids = delete_pack_ids(&snapshot.catalog, [pack_id], keep_root_packs);
    delete_plan_for_pack_ids(snapshot, delete_pack_ids.iter().map(String::as_str))
}

pub fn plan_delete_superseded_tts(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
    selected_pack_id: &str,
) -> DeletePlan {
    let installed_language_packs = snapshot
        .catalog
        .tts_pack_ids_for_language(language_code)
        .into_iter()
        .filter(|pack_id| pack_installed_in_snapshot(snapshot, pack_id))
        .collect::<HashSet<_>>();
    let superseded_root_packs = snapshot
        .catalog
        .tts_pack_ids_for_language(language_code)
        .into_iter()
        .filter(|pack_id| pack_id != selected_pack_id && installed_language_packs.contains(pack_id))
        .collect::<HashSet<_>>();
    if superseded_root_packs.is_empty() {
        return DeletePlan::default();
    }
    let mut keep_root_packs = HashSet::new();
    if pack_installed_in_snapshot(snapshot, selected_pack_id) {
        keep_root_packs.insert(selected_pack_id.to_string());
    }
    keep_root_packs.extend(
        snapshot
            .catalog
            .languages
            .keys()
            .filter(|code| code.as_str() != language_code.as_str())
            .flat_map(|code| {
                snapshot
                    .catalog
                    .tts_pack_ids_for_language(&LanguageCode::from(code.as_str()))
            })
            .filter(|pack_id| pack_installed_in_snapshot(snapshot, pack_id)),
    );
    let delete_pack_ids = delete_pack_ids(
        &snapshot.catalog,
        superseded_root_packs.iter().map(String::as_str),
        keep_root_packs,
    );
    delete_plan_for_pack_ids(snapshot, delete_pack_ids.iter().map(String::as_str))
}
