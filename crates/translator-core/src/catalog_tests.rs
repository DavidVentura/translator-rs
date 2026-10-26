use std::collections::HashMap;

use crate::api::{DictionaryCode, LanguageCode};
use crate::catalog::{
    AssetFileV2, AssetPackMetadataV2, CatalogSourcesV2, DownloadTask, FileRequirement, FileRole,
    InstallMarker, LangAvailability, LanguageCatalog, LanguageFeature, LanguageTtsRegionV2,
    LanguageTtsV2, PackInstallChecker, PackRecord, TaskInstall, build_catalog_snapshot,
    download_task_installed, plan_delete_superseded_files, plan_language_download,
    plan_language_install,
};
use crate::catalog::{plan_ocr_engine_downloads, plan_ocr_engine_upgrades, plan_repair};
use crate::catalog::{plan_translation_upgrades, translation_upgrade_language_codes};
use crate::language::Language;
use crate::tts::TtsEngine;

use super::model::{
    DictionaryPack, LanguageInfo, LanguageResources, OcrPack, PackKind, PpocrScript, SupportPack,
    TranslationPack, TtsPack,
};

struct FakeInstallChecker {
    files: std::collections::HashSet<String>,
    markers: std::collections::HashSet<(String, i32)>,
}

impl FakeInstallChecker {
    fn with_files(paths: &[&str]) -> Self {
        Self {
            files: paths.iter().map(|path| path.to_string()).collect(),
            markers: std::collections::HashSet::new(),
        }
    }
}

impl PackInstallChecker for FakeInstallChecker {
    fn file_exists(&self, install_path: &str) -> bool {
        self.files.contains(install_path)
    }

    fn install_marker_exists(&self, marker_path: &str, expected_version: i32) -> bool {
        self.markers
            .contains(&(marker_path.to_string(), expected_version))
    }
}

fn asset_file(name: &str, install_path: &str, size_bytes: u64) -> AssetFileV2 {
    AssetFileV2 {
        name: name.to_string(),
        size_bytes,
        install_path: install_path.to_string(),
        url: format!("https://example.invalid/{name}"),
        source_path: None,
        archive: None,
        // These fixtures exercise install/status/download planning, not runtime
        // role resolution, so each file just needs a distinct role to stay its
        // own singleton group (the filename guarantees that).
        role: FileRole::new(name),
        priority: 0,
        requirement: FileRequirement::Required,
        replaces: Vec::new(),
    }
}

fn role_file(
    name: &str,
    install_path: &str,
    size_bytes: u64,
    role: FileRole,
    priority: i32,
) -> AssetFileV2 {
    AssetFileV2 {
        role,
        priority,
        ..asset_file(name, install_path, size_bytes)
    }
}

fn optional_role_file(
    name: &str,
    install_path: &str,
    size_bytes: u64,
    role: FileRole,
    priority: i32,
) -> AssetFileV2 {
    AssetFileV2 {
        requirement: FileRequirement::Optional,
        ..role_file(name, install_path, size_bytes, role, priority)
    }
}

fn pack_record(
    id: &str,
    kind: PackKind,
    files: Vec<AssetFileV2>,
    depends_on: Vec<&str>,
) -> PackRecord {
    PackRecord {
        id: id.to_string(),
        files,
        depends_on: depends_on.into_iter().map(ToString::to_string).collect(),
        kind,
    }
}

fn translation_pack(
    id: &str,
    from: &str,
    to: &str,
    files: Vec<AssetFileV2>,
    depends_on: Vec<&str>,
) -> PackRecord {
    pack_record(
        id,
        PackKind::Translation(TranslationPack {
            from: from.to_string(),
            to: to.to_string(),
            experimental: false,
        }),
        files,
        depends_on,
    )
}

fn ocr_pack(id: &str, file: AssetFileV2) -> PackRecord {
    pack_record(
        id,
        PackKind::Ocr(OcrPack::PpocrRecognizer {
            script: PpocrScript::Latin,
        }),
        vec![file],
        vec![],
    )
}

fn ppocr_recognizer_pack(
    id: &str,
    script: PpocrScript,
    file: AssetFileV2,
    depends_on: Vec<&str>,
) -> PackRecord {
    pack_record(
        id,
        PackKind::Ocr(OcrPack::PpocrRecognizer { script }),
        vec![file],
        depends_on,
    )
}

fn dictionary_pack(
    id: &str,
    language: &str,
    dictionary_code: &str,
    file: AssetFileV2,
    metadata: Option<AssetPackMetadataV2>,
) -> PackRecord {
    pack_record(
        id,
        PackKind::Dictionary(DictionaryPack {
            language: Some(language.to_string()),
            dictionary_code: dictionary_code.to_string(),
            metadata,
        }),
        vec![file],
        vec![],
    )
}

fn support_pack(id: &str, language: Option<&str>, file: AssetFileV2) -> PackRecord {
    pack_record(
        id,
        PackKind::Support(SupportPack {
            language: language.map(ToString::to_string),
            languages: vec![],
            aliases: vec![],
            kind: None,
            metadata: None,
        }),
        vec![file],
        vec![],
    )
}

fn tts_pack(
    id: &str,
    language: &str,
    region: &str,
    default_speaker_id: i32,
    files: Vec<AssetFileV2>,
) -> PackRecord {
    pack_record(
        id,
        PackKind::Tts(TtsPack {
            language: language.to_string(),
            engine: TtsEngine::Piper,
            aux_role: FileRole::new("sidecar"),
            locale: None,
            region: Some(region.to_string()),
            voice: None,
            quality: None,
            speakers: Vec::new(),
            default_speaker_id: Some(default_speaker_id),
            sample_url: None,
        }),
        files,
        vec![],
    )
}

fn language_info(
    language: Language,
    translation_root_packs: Vec<&str>,
    ocr_pack_id: &str,
    dictionary_pack_id: Option<&str>,
    support_root_packs: Vec<&str>,
    tts_regions: Vec<(&str, &str, Vec<&str>)>,
    default_region: Option<&str>,
) -> LanguageInfo {
    LanguageInfo {
        language,
        resources: LanguageResources {
            translation_root_packs: translation_root_packs
                .into_iter()
                .map(ToString::to_string)
                .collect(),
            ocr_packs: vec![("ppocr".to_string(), ocr_pack_id.to_string())],
            dictionary_pack_id: dictionary_pack_id.map(ToString::to_string),
            support_root_packs: support_root_packs
                .into_iter()
                .map(ToString::to_string)
                .collect(),
        },
        tts: Some(LanguageTtsV2 {
            default_region: default_region.map(ToString::to_string),
            sample_text: None,
            regions: tts_regions
                .into_iter()
                .map(|(code, display_name, voices)| {
                    (
                        code.to_string(),
                        LanguageTtsRegionV2 {
                            display_name: display_name.to_string(),
                            voices: voices.into_iter().map(ToString::to_string).collect(),
                        },
                    )
                })
                .collect(),
        }),
    }
}

fn base_catalog() -> LanguageCatalog {
    let english = Language {
        code: "en".to_string(),
        display_name: "English".to_string(),
        short_display_name: "English".to_string(),
        script: crate::script::Script::Latin,
        writing_system: crate::script::WritingSystem::Single(crate::script::Script::Latin),
        dictionary_code: "en".to_string(),
    };
    let spanish = Language {
        code: "es".to_string(),
        display_name: "Spanish".to_string(),
        short_display_name: "Spanish".to_string(),
        script: crate::script::Script::Latin,
        writing_system: crate::script::WritingSystem::Single(crate::script::Script::Latin),
        dictionary_code: "es".to_string(),
    };

    let languages = HashMap::from([
        (
            "en".to_string(),
            language_info(
                english,
                vec![],
                "ocr-en",
                Some("dict-en"),
                vec![],
                vec![("us", "United States", vec!["tts-en-us"])],
                Some("us"),
            ),
        ),
        (
            "es".to_string(),
            language_info(
                spanish,
                vec!["translate-es-en", "translate-en-es"],
                "ocr-es",
                Some("dict-es"),
                vec!["support-es"],
                vec![("es", "Spain", vec!["tts-es"])],
                Some("es"),
            ),
        ),
    ]);

    let packs = HashMap::from([
        (
            "ocr-en".to_string(),
            ocr_pack(
                "ocr-en",
                asset_file("rec_latin_en.mnn", "ppocr/rec_latin_en.mnn", 10),
            ),
        ),
        (
            "ocr-es".to_string(),
            ocr_pack(
                "ocr-es",
                asset_file("rec_latin_es.mnn", "ppocr/rec_latin_es.mnn", 11),
            ),
        ),
        (
            "dict-en".to_string(),
            dictionary_pack(
                "dict-en",
                "en",
                "en",
                asset_file("en.dict", "dictionaries/en.dict", 12),
                Some(AssetPackMetadataV2 {
                    date: Some(123),
                    type_name: Some("wiktionary".to_string()),
                    word_count: Some(456),
                }),
            ),
        ),
        (
            "dict-es".to_string(),
            dictionary_pack(
                "dict-es",
                "es",
                "es",
                asset_file("es.dict", "dictionaries/es.dict", 13),
                None,
            ),
        ),
        (
            "support-es".to_string(),
            support_pack(
                "support-es",
                Some("es"),
                asset_file("mucab.bin", "bin/mucab.bin", 14),
            ),
        ),
        (
            "translate-en-es".to_string(),
            translation_pack(
                "translate-en-es",
                "en",
                "es",
                vec![
                    asset_file("model.enes.bin", "bin/model.enes.bin", 20),
                    asset_file("lex.enen.bin", "bin/lex.enen.bin", 21),
                    asset_file("vocab.aa.spm", "bin/vocab.aa.spm", 22),
                    asset_file("vocab.zz.spm", "bin/vocab.zz.spm", 23),
                ],
                vec!["shared-translate"],
            ),
        ),
        (
            "translate-es-en".to_string(),
            translation_pack(
                "translate-es-en",
                "es",
                "en",
                vec![
                    asset_file("model.esen.bin", "bin/model.esen.bin", 24),
                    asset_file("lex.esen.bin", "bin/lex.esen.bin", 25),
                    asset_file("vocab.bb.spm", "bin/vocab.bb.spm", 26),
                    asset_file("vocab.yy.spm", "bin/vocab.yy.spm", 27),
                ],
                vec!["shared-translate"],
            ),
        ),
        (
            "shared-translate".to_string(),
            support_pack(
                "shared-translate",
                None,
                asset_file("shared.bin", "bin/shared.bin", 28),
            ),
        ),
        (
            "tts-en-us".to_string(),
            tts_pack(
                "tts-en-us",
                "en",
                "us",
                7,
                vec![
                    asset_file("voice.onnx", "tts/voice.onnx", 31),
                    asset_file("voice.onnx.json", "tts/voice.onnx.json", 32),
                ],
            ),
        ),
        (
            "tts-es".to_string(),
            tts_pack(
                "tts-es",
                "es",
                "es",
                8,
                vec![
                    asset_file("voice-es.onnx", "tts/voice-es.onnx", 33),
                    asset_file("voice-es.onnx.json", "tts/voice-es.onnx.json", 34),
                ],
            ),
        ),
    ]);

    LanguageCatalog {
        format_version: 2,
        generated_at: 1,
        dictionary_version: 1,
        sources: CatalogSourcesV2 {
            language_index_version: 1,
            language_index_updated_at: 1,
            dictionary_index_version: 1,
            dictionary_index_updated_at: 1,
        },
        languages,
        packs,
        translation_pack_ids: HashMap::from([
            (
                ("en".to_string(), "es".to_string()),
                "translate-en-es".to_string(),
            ),
            (
                ("es".to_string(), "en".to_string()),
                "translate-es-en".to_string(),
            ),
        ]),
        dictionary_pack_ids_by_code: HashMap::from([
            ("en".to_string(), "dict-en".to_string()),
            ("es".to_string(), "dict-es".to_string()),
        ]),
        root_pack_ids_by_language_feature: HashMap::from([
            (("en".to_string(), LanguageFeature::Translation), Vec::new()),
            (
                ("en".to_string(), LanguageFeature::Ocr),
                vec!["ocr-en".to_string()],
            ),
            (
                ("en".to_string(), LanguageFeature::Dictionary),
                vec!["dict-en".to_string()],
            ),
            (("en".to_string(), LanguageFeature::Support), Vec::new()),
            (
                ("en".to_string(), LanguageFeature::Tts),
                vec!["tts-en-us".to_string()],
            ),
            (
                ("es".to_string(), LanguageFeature::Translation),
                vec!["translate-es-en".to_string(), "translate-en-es".to_string()],
            ),
            (
                ("es".to_string(), LanguageFeature::Ocr),
                vec!["ocr-es".to_string()],
            ),
            (
                ("es".to_string(), LanguageFeature::Dictionary),
                vec!["dict-es".to_string()],
            ),
            (
                ("es".to_string(), LanguageFeature::Support),
                vec!["support-es".to_string()],
            ),
            (
                ("es".to_string(), LanguageFeature::Tts),
                vec!["tts-es".to_string()],
            ),
        ]),
        migrations: Vec::new(),
    }
}

#[test]
fn computes_dependency_closure_and_pack_size_without_double_counting() {
    let catalog = base_catalog();

    let closure = catalog.dependency_closure(["translate-en-es"]);
    assert_eq!(
        closure,
        vec![
            "translate-en-es".to_string(),
            "shared-translate".to_string()
        ]
    );
    assert_eq!(
        catalog.pack_size_bytes("translate-en-es"),
        20 + 21 + 22 + 23 + 28
    );
    assert_eq!(
        catalog.translation_size_bytes_for_language(&LanguageCode::from("es")),
        (20 + 21 + 22 + 23 + 28) + (24 + 25 + 26 + 27 + 28)
    );
}

#[test]
fn resolves_missing_pack_files_through_install_boundary() {
    let catalog = base_catalog();
    let checker = FakeInstallChecker::with_files(&["bin/model.enes.bin", "bin/shared.bin"]);
    let snapshot = build_catalog_snapshot(catalog, "/base".to_string(), &checker);
    let plan = plan_language_download(&snapshot, &LanguageCode::from("es"));
    let missing_paths = plan
        .tasks
        .into_iter()
        .filter(|task| task.pack_id == "translate-en-es")
        .map(|task| task.install_path)
        .collect::<Vec<_>>();

    assert_eq!(
        missing_paths,
        vec![
            "bin/lex.enen.bin".to_string(),
            "bin/vocab.aa.spm".to_string(),
            "bin/vocab.zz.spm".to_string(),
        ]
    );
}

#[test]
fn batch_ocr_download_plan_dedupes_shared_dependencies() {
    let mut catalog = base_catalog();
    catalog.packs.insert(
        "ocr-ppocr-detector".to_string(),
        pack_record(
            "ocr-ppocr-detector",
            PackKind::Ocr(OcrPack::PpocrDetector),
            vec![asset_file(
                "det.mnn",
                "ppocr/PP-OCRv5/PP-OCRv5_mobile_det.mnn",
                100,
            )],
            vec![],
        ),
    );
    catalog.packs.insert(
        "ocr-ppocr-latin".to_string(),
        ppocr_recognizer_pack(
            "ocr-ppocr-latin",
            PpocrScript::Latin,
            asset_file("latin.mnn", "ppocr/PP-OCRv5/latin.mnn", 10),
            vec!["ocr-ppocr-detector"],
        ),
    );
    catalog.packs.insert(
        "ocr-ppocr-cj".to_string(),
        ppocr_recognizer_pack(
            "ocr-ppocr-cj",
            PpocrScript::Cj,
            asset_file("cj.mnn", "ppocr/PP-OCRv5/cj.mnn", 20),
            vec!["ocr-ppocr-detector"],
        ),
    );
    catalog.languages.get_mut("en").unwrap().resources.ocr_packs =
        vec![("ppocr".to_string(), "ocr-ppocr-latin".to_string())];
    catalog.languages.get_mut("es").unwrap().resources.ocr_packs =
        vec![("ppocr".to_string(), "ocr-ppocr-cj".to_string())];

    let checker = FakeInstallChecker::with_files(&[]);
    let snapshot = build_catalog_snapshot(catalog, "/base".to_string(), &checker);
    let plan = plan_ocr_engine_downloads(
        &snapshot,
        &[LanguageCode::from("en"), LanguageCode::from("es")],
        "ppocr",
    );
    let mut paths = plan
        .tasks
        .iter()
        .map(|task| task.install_path.as_str())
        .collect::<Vec<_>>();
    paths.sort_unstable();

    assert_eq!(plan.total_size, 130);
    assert_eq!(
        paths,
        vec![
            "ppocr/PP-OCRv5/PP-OCRv5_mobile_det.mnn",
            "ppocr/PP-OCRv5/cj.mnn",
            "ppocr/PP-OCRv5/latin.mnn",
        ]
    );
}

fn catalog_with_detector_alternatives() -> LanguageCatalog {
    let mut catalog = base_catalog();
    catalog.packs.insert(
        "ocr-ppocr-detector".to_string(),
        pack_record(
            "ocr-ppocr-detector",
            PackKind::Ocr(OcrPack::PpocrDetector),
            vec![
                role_file(
                    "det_old.mnn",
                    "ppocr/det_old.mnn",
                    100,
                    FileRole::Detector,
                    0,
                ),
                role_file(
                    "det_new.mnn",
                    "ppocr/det_new.mnn",
                    40,
                    FileRole::Detector,
                    1,
                ),
                role_file(
                    "pulc.mnn",
                    "ppocr/pulc.mnn",
                    10,
                    FileRole::ScriptClassifier,
                    0,
                ),
            ],
            vec![],
        ),
    );
    catalog.packs.insert(
        "ocr-ppocr-latin".to_string(),
        ppocr_recognizer_pack(
            "ocr-ppocr-latin",
            PpocrScript::Latin,
            role_file("latin.mnn", "ppocr/latin.mnn", 10, FileRole::Recognizer, 0),
            vec!["ocr-ppocr-detector"],
        ),
    );
    catalog.languages.get_mut("en").unwrap().resources.ocr_packs =
        vec![("ppocr".to_string(), "ocr-ppocr-latin".to_string())];
    catalog
}

#[test]
fn lower_priority_alternative_keeps_pack_installed_and_offers_upgrade() {
    let catalog = catalog_with_detector_alternatives();
    let checker =
        FakeInstallChecker::with_files(&["ppocr/det_old.mnn", "ppocr/pulc.mnn", "ppocr/latin.mnn"]);
    let snapshot = build_catalog_snapshot(catalog, "/base".to_string(), &checker);

    assert!(snapshot.pack_statuses["ocr-ppocr-detector"].installed);
    assert!(snapshot.pack_statuses["ocr-ppocr-latin"].installed);

    let en = [LanguageCode::from("en")];
    let downloads = plan_ocr_engine_downloads(&snapshot, &en, "ppocr");
    assert!(downloads.tasks.is_empty());

    let upgrades = plan_ocr_engine_upgrades(&snapshot, &en, "ppocr");
    let paths = upgrades
        .tasks
        .iter()
        .map(|task| task.install_path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(paths, vec!["ppocr/det_new.mnn"]);
    assert_eq!(upgrades.total_size, 40);
}

fn catalog_with_translation_priority() -> LanguageCatalog {
    // A translation pack with a legacy p1 model at the flat path (what an old
    // install has on disk) and a p2 model namespaced under bin/en-tl/v2 (the new
    // upgrade). Mirrors the en->tl opusmt(p1) / hymt2(p2) shape.
    let mut catalog = base_catalog();
    catalog.packs.insert(
        "translate-en-tl".to_string(),
        translation_pack(
            "translate-en-tl",
            "en",
            "tl",
            vec![
                role_file(
                    "model.entl.bin",
                    "bin/model.entl.bin",
                    100,
                    FileRole::Model,
                    1,
                ),
                role_file(
                    "model.entl.bin",
                    "bin/en-tl/v2/model.entl.bin",
                    40,
                    FileRole::Model,
                    2,
                ),
                role_file("lex.entl.bin", "bin/lex.entl.bin", 10, FileRole::Lex, 1),
                role_file(
                    "lex.entl.bin",
                    "bin/en-tl/v2/lex.entl.bin",
                    8,
                    FileRole::Lex,
                    2,
                ),
                role_file(
                    "vocab.entl.spm",
                    "bin/vocab.entl.spm",
                    5,
                    FileRole::Vocab,
                    1,
                ),
                role_file(
                    "vocab.entl.spm",
                    "bin/en-tl/v2/vocab.entl.spm",
                    4,
                    FileRole::Vocab,
                    2,
                ),
            ],
            vec![],
        ),
    );
    catalog.translation_pack_ids.insert(
        ("en".to_string(), "tl".to_string()),
        "translate-en-tl".to_string(),
    );
    catalog
}

#[test]
fn translation_upgrade_offered_when_lower_priority_variant_installed() {
    let catalog = catalog_with_translation_priority();
    // Only the p1 (flat) files on disk — the shipped older model.
    let checker = FakeInstallChecker::with_files(&[
        "bin/model.entl.bin",
        "bin/lex.entl.bin",
        "bin/vocab.entl.spm",
    ]);
    let snapshot = build_catalog_snapshot(catalog, "/base".to_string(), &checker);

    assert!(snapshot.pack_statuses["translate-en-tl"].installed);
    assert_eq!(
        translation_upgrade_language_codes(&snapshot),
        vec![LanguageCode::from("tl")]
    );

    let plan = plan_translation_upgrades(&snapshot, &[LanguageCode::from("tl")]);
    let mut paths = plan
        .tasks
        .iter()
        .map(|task| task.install_path.as_str())
        .collect::<Vec<_>>();
    paths.sort();
    assert_eq!(
        paths,
        vec![
            "bin/en-tl/v2/lex.entl.bin",
            "bin/en-tl/v2/model.entl.bin",
            "bin/en-tl/v2/vocab.entl.spm",
        ]
    );
    assert_eq!(plan.total_size, 40 + 8 + 4);
}

#[test]
fn no_translation_upgrade_when_best_variant_already_installed() {
    let catalog = catalog_with_translation_priority();
    let checker = FakeInstallChecker::with_files(&[
        "bin/en-tl/v2/model.entl.bin",
        "bin/en-tl/v2/lex.entl.bin",
        "bin/en-tl/v2/vocab.entl.spm",
    ]);
    let snapshot = build_catalog_snapshot(catalog, "/base".to_string(), &checker);

    assert!(translation_upgrade_language_codes(&snapshot).is_empty());
    assert!(
        plan_translation_upgrades(&snapshot, &[LanguageCode::from("tl")])
            .tasks
            .is_empty()
    );
}

#[test]
fn repair_plan_restores_partially_present_packs() {
    let catalog = catalog_with_detector_alternatives();
    // No detector model on disk (catalog change orphaned it / partial download)
    // but the pack's classifier survives; the recognizer itself is complete and
    // broken only through its dependency.
    let checker = FakeInstallChecker::with_files(&["ppocr/pulc.mnn", "ppocr/latin.mnn"]);
    let snapshot = build_catalog_snapshot(catalog, "/base".to_string(), &checker);

    let detector_status = &snapshot.pack_statuses["ocr-ppocr-detector"];
    assert!(!detector_status.installed);
    assert!(detector_status.any_file_present);
    assert!(!snapshot.pack_statuses["ocr-ppocr-latin"].installed);

    let plan = plan_repair(&snapshot);
    let paths = plan
        .tasks
        .iter()
        .map(|task| task.install_path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(paths, vec!["ppocr/det_new.mnn"]);
    assert_eq!(plan.total_size, 40);
}

#[test]
fn repair_plan_restores_absent_core_packs_of_installed_language() {
    // Translation installed by a version that predates the language's OCR pack:
    // none of the OCR files were ever on disk, yet the language is in use.
    let checker = FakeInstallChecker::with_files(&[
        "bin/model.enes.bin",
        "bin/lex.enen.bin",
        "bin/vocab.aa.spm",
        "bin/vocab.zz.spm",
        "bin/model.esen.bin",
        "bin/lex.esen.bin",
        "bin/vocab.bb.spm",
        "bin/vocab.yy.spm",
        "bin/shared.bin",
        "bin/mucab.bin",
    ]);
    let snapshot = build_catalog_snapshot(base_catalog(), "/base".to_string(), &checker);
    assert!(!snapshot.pack_statuses["ocr-es"].any_file_present);

    let plan = plan_repair(&snapshot);
    let paths = plan
        .tasks
        .iter()
        .map(|task| task.install_path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(paths, vec!["ppocr/rec_latin_es.mnn"]);
    assert_eq!(plan.total_size, 11);
}

#[test]
fn repair_plan_ignores_never_installed_packs() {
    let snapshot = build_catalog_snapshot(
        catalog_with_detector_alternatives(),
        "/base".to_string(),
        &FakeInstallChecker::with_files(&[]),
    );
    assert!(plan_repair(&snapshot).tasks.is_empty());
}

fn catalog_with_optional_ink() -> LanguageCatalog {
    let mut catalog = catalog_with_detector_alternatives();
    let detector = catalog.packs.get_mut("ocr-ppocr-detector").unwrap();
    detector.files.push(optional_role_file(
        "ink.mnn",
        "ppocr/ink.mnn",
        50,
        FileRole::Ink,
        0,
    ));
    catalog
}

#[test]
fn missing_optional_file_keeps_pack_installed_and_offers_it_as_upgrade() {
    let catalog = catalog_with_optional_ink();
    // Everything present except the optional ink model.
    let checker =
        FakeInstallChecker::with_files(&["ppocr/det_new.mnn", "ppocr/pulc.mnn", "ppocr/latin.mnn"]);
    let snapshot = build_catalog_snapshot(catalog, "/base".to_string(), &checker);

    let detector_status = &snapshot.pack_statuses["ocr-ppocr-detector"];
    assert!(detector_status.installed);
    assert!(detector_status.missing_files.is_empty());

    // A (re)download now pulls the best optional up front, so the user isn't
    // immediately prompted to fetch it right after installing the language.
    let en = [LanguageCode::from("en")];
    let downloads = plan_ocr_engine_downloads(&snapshot, &en, "ppocr");
    let download_paths = downloads
        .tasks
        .iter()
        .map(|task| task.install_path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(download_paths, vec!["ppocr/ink.mnn"]);

    // It is still offered as an upgrade for installs that predate it on disk.
    let upgrades = plan_ocr_engine_upgrades(&snapshot, &en, "ppocr");
    let paths = upgrades
        .tasks
        .iter()
        .map(|task| task.install_path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(paths, vec!["ppocr/ink.mnn"]);
}

#[test]
fn fresh_install_pulls_best_optional_and_then_offers_no_upgrade() {
    let en = [LanguageCode::from("en")];

    let empty = FakeInstallChecker::with_files(&[]);
    let snapshot = build_catalog_snapshot(catalog_with_optional_ink(), "/base".to_string(), &empty);
    let download = plan_ocr_engine_downloads(&snapshot, &en, "ppocr");
    let mut paths = download
        .tasks
        .iter()
        .map(|task| task.install_path.as_str())
        .collect::<Vec<_>>();
    paths.sort_unstable();
    assert_eq!(
        paths,
        vec![
            "ppocr/det_new.mnn",
            "ppocr/ink.mnn",
            "ppocr/latin.mnn",
            "ppocr/pulc.mnn"
        ]
    );

    // After that full plan lands on disk there is nothing left to prompt for.
    let installed = FakeInstallChecker::with_files(&[
        "ppocr/det_new.mnn",
        "ppocr/ink.mnn",
        "ppocr/latin.mnn",
        "ppocr/pulc.mnn",
    ]);
    let snapshot =
        build_catalog_snapshot(catalog_with_optional_ink(), "/base".to_string(), &installed);
    assert!(
        plan_ocr_engine_upgrades(&snapshot, &en, "ppocr")
            .tasks
            .is_empty()
    );
}

#[test]
fn fresh_install_plan_downloads_only_best_alternative() {
    let catalog = catalog_with_detector_alternatives();
    let checker = FakeInstallChecker::with_files(&[]);
    let snapshot = build_catalog_snapshot(catalog, "/base".to_string(), &checker);

    let en = [LanguageCode::from("en")];
    let plan = plan_ocr_engine_downloads(&snapshot, &en, "ppocr");
    let mut paths = plan
        .tasks
        .iter()
        .map(|task| task.install_path.as_str())
        .collect::<Vec<_>>();
    paths.sort_unstable();

    assert_eq!(
        paths,
        vec!["ppocr/det_new.mnn", "ppocr/latin.mnn", "ppocr/pulc.mnn"]
    );
    assert!(
        plan_ocr_engine_upgrades(&snapshot, &en, "ppocr")
            .tasks
            .is_empty()
    );
}

#[test]
fn outranked_alternative_on_disk_is_superseded() {
    let catalog = catalog_with_detector_alternatives();
    let checker = FakeInstallChecker::with_files(&[
        "ppocr/det_old.mnn",
        "ppocr/det_new.mnn",
        "ppocr/pulc.mnn",
        "ppocr/latin.mnn",
    ]);
    let snapshot = build_catalog_snapshot(catalog, "/base".to_string(), &checker);

    let en = [LanguageCode::from("en")];
    assert!(
        plan_ocr_engine_upgrades(&snapshot, &en, "ppocr")
            .tasks
            .is_empty()
    );
    let cleanup = plan_delete_superseded_files(&snapshot);
    assert_eq!(cleanup.file_paths, vec!["ppocr/det_old.mnn".to_string()]);
}

#[test]
fn computes_language_availability_from_pack_install_state() {
    let catalog = base_catalog();
    let checker = FakeInstallChecker::with_files(&[
        "ppocr/rec_latin_en.mnn",
        "dictionaries/en.dict",
        "tts/voice.onnx",
        "tts/voice.onnx.json",
        "bin/model.enes.bin",
        "bin/lex.enen.bin",
        "bin/vocab.aa.spm",
        "bin/vocab.zz.spm",
        "bin/model.esen.bin",
        "bin/lex.esen.bin",
        "bin/vocab.bb.spm",
        "bin/vocab.yy.spm",
        "bin/shared.bin",
        "ppocr/rec_latin_es.mnn",
        "dictionaries/es.dict",
        "bin/mucab.bin",
        "tts/voice-es.onnx",
        "tts/voice-es.onnx.json",
    ]);
    let snapshot = build_catalog_snapshot(catalog, "/base".to_string(), &checker);

    assert_eq!(
        snapshot.availability_by_code.get("es"),
        Some(&LangAvailability {
            has_from_english: true,
            has_to_english: true,
            ocr_files: true,
            dictionary_files: true,
            tts_files: true,
        })
    );
    assert_eq!(
        snapshot.availability_by_code.get("en"),
        Some(&LangAvailability {
            has_from_english: true,
            has_to_english: true,
            ocr_files: true,
            dictionary_files: true,
            tts_files: true,
        })
    );
}

#[test]
fn builds_languages_and_dictionary_info_from_catalog() {
    let catalog = base_catalog();
    let spanish = catalog.language_by_code(&LanguageCode::from("es")).unwrap();
    let dictionary_info = catalog
        .dictionary_info(&DictionaryCode::from("en"))
        .unwrap();

    assert_eq!(spanish.display_name, "Spanish");
    assert_eq!(dictionary_info.filename, "en.dict");
    assert_eq!(dictionary_info.type_name, "wiktionary");
    assert_eq!(dictionary_info.word_count, 456);
}

#[test]
fn returns_support_files_by_kind() {
    let mut catalog = base_catalog();
    catalog.packs.insert(
        "support-adblock-easylist".to_string(),
        pack_record(
            "support-adblock-easylist",
            PackKind::Support(SupportPack {
                language: None,
                languages: vec![],
                aliases: vec![],
                kind: Some("adblock".to_string()),
                metadata: None,
            }),
            vec![asset_file("easylist.txt", "adblock/easylist.txt", 100)],
            vec!["support-adblock-shared"],
        ),
    );
    catalog.packs.insert(
        "support-adblock-shared".to_string(),
        pack_record(
            "support-adblock-shared",
            PackKind::Support(SupportPack {
                language: None,
                languages: vec![],
                aliases: vec![],
                kind: Some("adblock".to_string()),
                metadata: None,
            }),
            vec![asset_file("shared.txt", "adblock/shared.txt", 101)],
            vec![],
        ),
    );
    catalog.packs.insert(
        "support-other".to_string(),
        pack_record(
            "support-other",
            PackKind::Support(SupportPack {
                language: None,
                languages: vec![],
                aliases: vec![],
                kind: Some("other".to_string()),
                metadata: None,
            }),
            vec![asset_file("other.txt", "support/other.txt", 102)],
            vec![],
        ),
    );

    let files = catalog.support_files_by_kind("adblock");
    let install_paths = files
        .iter()
        .map(|file| file.install_path.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        install_paths,
        vec!["adblock/easylist.txt", "adblock/shared.txt"]
    );
}

#[test]
fn parses_bundled_catalog_asset() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let asset_path = manifest_dir
        .parent()
        .and_then(|parent| parent.parent())
        .map(|parent| {
            parent.join("AndroidStudioProjects/Translator/app/src/main/assets/index_v4.json")
        })
        .expect("repo layout should have a parent");
    let Ok(json) = std::fs::read_to_string(asset_path) else {
        return;
    };
    let catalog =
        crate::catalog::parse_and_validate_catalog(&json).expect("bundled catalog should parse");
    assert!(!catalog.language_list().is_empty());
    assert!(
        catalog
            .language_by_code(&LanguageCode::from("en"))
            .is_some()
    );
}

#[test]
fn parses_v6_migrations_from_asset() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let asset_path = manifest_dir
        .parent()
        .and_then(|parent| parent.parent())
        .map(|parent| {
            parent.join("AndroidStudioProjects/Translator/app/src/main/assets/index_v7.json")
        })
        .expect("repo layout should have a parent");
    let Ok(json) = std::fs::read_to_string(asset_path) else {
        return;
    };
    let catalog =
        crate::catalog::parse_and_validate_catalog(&json).expect("v6 catalog should parse");
    assert!(
        !catalog.migrations.is_empty(),
        "v6 asset should carry migrations"
    );
    assert!(
        catalog
            .migrations
            .iter()
            .any(|entry| entry.feature == "doc_detect"),
        "doc_detect migration should be present"
    );
    assert!(
        catalog
            .migrations
            .iter()
            .all(|entry| entry.onnx.ends_with(".onnx") && entry.mnn.ends_with(".mnn")),
    );
}

#[test]
fn selects_best_catalog_using_headers_only() {
    let bundled = r#"{"formatVersion":6,"generatedAt":1}"#;
    let disk = r#"{"formatVersion":6,"generatedAt":2}"#;

    let selected = crate::catalog::select_best_catalog(bundled, Some(disk))
        .expect("header-only catalogs should still compare");

    assert_eq!(selected, disk);
}

fn catalog_json_with_migrations(migrations: &str) -> String {
    format!(
        r#"{{
          "formatVersion":6,"generatedAt":1,"dictionaryVersion":1,
          "sources":{{"languageIndexVersion":1,"languageIndexUpdatedAt":1,
                      "dictionaryIndexVersion":1,"dictionaryIndexUpdatedAt":1}},
          "languages":{{}},"packs":{{}},
          "migrations":{migrations}
        }}"#
    )
}

#[test]
fn parses_migrations_section() {
    let json = catalog_json_with_migrations(
        r#"[
          {"onnx":"ocr/docrec/docaligner_lcnet050.onnx","mnn":"ocr/docrec/docaligner_lcnet050.mnn",
           "quantBits":8,"onnxBytes":4911217,"mnnBytes":1760380,"feature":"doc_detect"},
          {"onnx":"bin/piper/ar/v.onnx","mnn":"bin/piper/ar/v.mnn",
           "quantBits":8,"onnxBytes":100,"mnnBytes":40,"feature":"tts"}
        ]"#,
    );
    let catalog = crate::catalog::parse_and_validate_catalog(&json).expect("v6 catalog parses");
    assert_eq!(catalog.migrations.len(), 2);
    assert_eq!(catalog.migrations[0].feature, "doc_detect");
    assert_eq!(
        catalog.migrations[0].mnn,
        "ocr/docrec/docaligner_lcnet050.mnn"
    );
    assert_eq!(catalog.migrations[1].onnx_bytes, 100);
    assert_eq!(catalog.migrations[1].quant_bits, 8);
}

#[test]
fn catalog_without_migrations_is_empty() {
    let json = r#"{
      "formatVersion":6,"generatedAt":1,"dictionaryVersion":1,
      "sources":{"languageIndexVersion":1,"languageIndexUpdatedAt":1,
                 "dictionaryIndexVersion":1,"dictionaryIndexUpdatedAt":1},
      "languages":{},"packs":{}
    }"#;
    let catalog = crate::catalog::parse_and_validate_catalog(json).expect("v4 catalog parses");
    assert!(catalog.migrations.is_empty());
}

#[test]
fn plan_migrations_classifies_by_disk_state() {
    use crate::catalog::MigrationAction;

    let json = catalog_json_with_migrations(
        r#"[
          {"onnx":"a.onnx","mnn":"a.mnn","quantBits":8,"onnxBytes":10,"mnnBytes":4,"feature":"tts"},
          {"onnx":"b.onnx","mnn":"b.mnn","quantBits":8,"onnxBytes":10,"mnnBytes":4,"feature":"tts"},
          {"onnx":"c.onnx","mnn":"c.mnn","quantBits":8,"onnxBytes":10,"mnnBytes":4,"feature":"tts"}
        ]"#,
    );
    let catalog = crate::catalog::parse_and_validate_catalog(&json).expect("v6 catalog parses");
    // a: onnx present, mnn absent -> Convert
    // b: onnx present, mnn already present -> CleanupOnly
    // c: onnx absent -> not in the plan
    let checker = FakeInstallChecker::with_files(&["a.onnx", "b.onnx", "b.mnn"]);

    let jobs = crate::catalog::plan_migrations(&catalog, &checker);
    assert_eq!(jobs.len(), 2);

    let job_a = jobs
        .iter()
        .find(|job| job.entry.onnx == "a.onnx")
        .expect("a planned");
    assert_eq!(job_a.action, MigrationAction::Convert);

    let job_b = jobs
        .iter()
        .find(|job| job.entry.onnx == "b.onnx")
        .expect("b planned");
    assert_eq!(job_b.action, MigrationAction::CleanupOnly);

    assert!(jobs.iter().all(|job| job.entry.onnx != "c.onnx"));
}

/// The `georgian` recognizer must resolve from the shipping index: `ka` maps to the pack,
/// the pack parses as a PPOCR recognizer, and the script comes back LTR. Catches a slug
/// typo or a missing `from_slug` arm, which otherwise surface only as a silent
/// "no ppocr pack for language ka" at runtime.
#[test]
fn georgian_recognizer_resolves_for_ka() {
    let index = std::path::Path::new("/home/david/AndroidStudioProjects/bucket/index_v7.json");
    if !index.exists() {
        return; // the bucket checkout is not present in every environment
    }
    let raw = std::fs::read_to_string(index).expect("read index");
    let catalog = crate::catalog::parse_and_validate_catalog(&raw).expect("parse index");
    let code = crate::api::LanguageCode::new("ka".to_string());
    let pack_id = catalog
        .ocr_pack_id_for_engine(&code, "ppocr")
        .expect("ka should have a ppocr pack");
    let pack = catalog.pack(&pack_id).expect("pack present");
    match &pack.kind {
        crate::catalog::PackKind::Ocr(crate::catalog::OcrPack::PpocrRecognizer { script }) => {
            assert_eq!(script.as_slug(), "georgian");
            assert!(!script.is_rtl(), "Georgian is left-to-right");
        }
        other => panic!("ka resolved to the wrong pack kind: {other:?}"),
    }
}

#[test]
fn translation_upgrade_deletes_the_alternatives_it_outranks() {
    let checker = FakeInstallChecker::with_files(&[
        "bin/model.entl.bin",
        "bin/lex.entl.bin",
        "bin/vocab.entl.spm",
    ]);
    let snapshot = build_catalog_snapshot(
        catalog_with_translation_priority(),
        "/base".to_string(),
        &checker,
    );

    let plan = plan_translation_upgrades(&snapshot, &[LanguageCode::from("tl")]);

    assert_eq!(
        plan.delete_after.file_paths,
        vec![
            "bin/lex.entl.bin".to_string(),
            "bin/model.entl.bin".to_string(),
            "bin/vocab.entl.spm".to_string(),
        ]
    );
    assert!(plan.delete_after.directory_paths.is_empty());
}

#[test]
fn translation_upgrade_keeps_files_another_pack_still_resolves_to() {
    let mut catalog = catalog_with_translation_priority();
    catalog.packs.insert(
        "translate-tl-en".to_string(),
        translation_pack(
            "translate-tl-en",
            "tl",
            "en",
            vec![
                role_file(
                    "model.tlen.bin",
                    "bin/model.tlen.bin",
                    100,
                    FileRole::Model,
                    1,
                ),
                role_file(
                    "vocab.entl.spm",
                    "bin/vocab.entl.spm",
                    5,
                    FileRole::Vocab,
                    1,
                ),
            ],
            vec![],
        ),
    );
    let checker = FakeInstallChecker::with_files(&[
        "bin/model.entl.bin",
        "bin/lex.entl.bin",
        "bin/vocab.entl.spm",
        "bin/model.tlen.bin",
    ]);
    let snapshot = build_catalog_snapshot(catalog, "/base".to_string(), &checker);

    let plan = plan_translation_upgrades(&snapshot, &[LanguageCode::from("tl")]);

    assert_eq!(
        plan.delete_after.file_paths,
        vec![
            "bin/lex.entl.bin".to_string(),
            "bin/model.entl.bin".to_string(),
        ]
    );
}

#[test]
fn ocr_upgrade_deletes_the_outranked_detector_only() {
    let checker =
        FakeInstallChecker::with_files(&["ppocr/det_old.mnn", "ppocr/pulc.mnn", "ppocr/latin.mnn"]);
    let snapshot = build_catalog_snapshot(
        catalog_with_detector_alternatives(),
        "/base".to_string(),
        &checker,
    );

    let plan = plan_ocr_engine_upgrades(&snapshot, &[LanguageCode::from("en")], "ppocr");

    assert_eq!(
        plan.delete_after.file_paths,
        vec!["ppocr/det_old.mnn".to_string()]
    );
}

#[test]
fn install_plans_never_delete() {
    let snapshot = build_catalog_snapshot(
        catalog_with_detector_alternatives(),
        "/base".to_string(),
        &FakeInstallChecker::with_files(&["ppocr/det_old.mnn"]),
    );

    let plan = plan_language_install(&snapshot, &LanguageCode::from("en"));

    assert!(!plan.tasks.is_empty());
    assert_eq!(plan.delete_after, crate::catalog::DeletePlan::default());
}

fn catalog_with_doc_detect_support() -> LanguageCatalog {
    let mut catalog = base_catalog();
    catalog.packs.insert(
        "support-doc-detect".to_string(),
        pack_record(
            "support-doc-detect",
            PackKind::Support(SupportPack {
                language: None,
                languages: vec![],
                aliases: vec![],
                kind: Some("doc_detect".to_string()),
                metadata: None,
            }),
            vec![asset_file("docaligner.mnn", "ocr/docrec/docaligner.mnn", 7)],
            vec![],
        ),
    );
    catalog
}

#[test]
fn language_install_brings_missing_core_support() {
    let snapshot = build_catalog_snapshot(
        catalog_with_doc_detect_support(),
        "/base".to_string(),
        &FakeInstallChecker::with_files(&[]),
    );
    let es = LanguageCode::from("es");

    let install = plan_language_install(&snapshot, &es);
    let language_only = plan_language_download(&snapshot, &es);

    let install_paths = install
        .tasks
        .iter()
        .map(|task| task.install_path.as_str())
        .collect::<std::collections::HashSet<_>>();
    assert!(install_paths.contains("ocr/docrec/docaligner.mnn"));
    assert_eq!(install.tasks.len(), language_only.tasks.len() + 1);
    assert_eq!(install.total_size, language_only.total_size + 7);
}

#[test]
fn language_install_skips_core_support_already_on_disk() {
    let snapshot = build_catalog_snapshot(
        catalog_with_doc_detect_support(),
        "/base".to_string(),
        &FakeInstallChecker::with_files(&["ocr/docrec/docaligner.mnn"]),
    );
    let es = LanguageCode::from("es");
    let paths = |plan: crate::catalog::DownloadPlan| {
        plan.tasks
            .into_iter()
            .map(|task| task.install_path)
            .collect::<std::collections::BTreeSet<_>>()
    };

    assert_eq!(
        paths(plan_language_install(&snapshot, &es)),
        paths(plan_language_download(&snapshot, &es))
    );
}

fn zip_task(marker: InstallMarker) -> DownloadTask {
    DownloadTask {
        pack_id: "support-adblock".to_string(),
        install_path: "adblock/lists.zip".to_string(),
        url: "https://example.invalid/lists.zip".to_string(),
        size_bytes: 10,
        install: TaskInstall::Zip {
            archive: crate::catalog::ZipArchive {
                extract_to: "adblock".to_string(),
                root_dir: "lists".to_string(),
                delete_after_extract: true,
                marker,
            },
        },
    }
}

#[test]
fn zip_task_counts_as_installed_by_its_marker_not_its_archive() {
    let marker = InstallMarker {
        path: "adblock/lists/.installed".to_string(),
        version: 3,
    };
    let task = zip_task(marker);
    let archive_only = FakeInstallChecker::with_files(&["adblock/lists.zip"]);
    let mut marker_present = FakeInstallChecker::with_files(&[]);
    marker_present
        .markers
        .insert(("adblock/lists/.installed".to_string(), 3));
    let mut stale_marker = FakeInstallChecker::with_files(&[]);
    stale_marker
        .markers
        .insert(("adblock/lists/.installed".to_string(), 2));

    assert!(!download_task_installed(&archive_only, &task));
    assert!(download_task_installed(&marker_present, &task));
    assert!(!download_task_installed(&stale_marker, &task));
}

#[test]
fn raw_task_counts_as_installed_by_its_file() {
    let task = DownloadTask {
        install: TaskInstall::Raw,
        install_path: "bin/model.bin".to_string(),
        ..zip_task(InstallMarker {
            path: "adblock/lists/.installed".to_string(),
            version: 1,
        })
    };

    assert!(download_task_installed(
        &FakeInstallChecker::with_files(&["bin/model.bin"]),
        &task
    ));
    assert!(!download_task_installed(
        &FakeInstallChecker::with_files(&[]),
        &task
    ));
}

fn catalog_json_with_file(file: &str) -> String {
    format!(
        r#"{{
          "formatVersion":6,"generatedAt":1,"dictionaryVersion":1,
          "sources":{{"languageIndexVersion":1,"languageIndexUpdatedAt":1,
                      "dictionaryIndexVersion":1,"dictionaryIndexUpdatedAt":1}},
          "languages":{{}},
          "packs":{{"support-x":{{"feature":"support","kind":"adblock","files":[{file}]}}}}
        }}"#
    )
}

#[test]
fn parses_zip_files_into_typed_archive_and_marker() {
    let json = catalog_json_with_file(
        r#"{"name":"l.zip","sizeBytes":1,"installPath":"adblock/l.zip","url":"u",
            "archiveFormat":"zip","extractTo":"adblock","deleteAfterExtract":true,
            "installMarkerPath":"adblock/l/.installed","installMarkerVersion":2,
            "role":"lists","priority":0}"#,
    );
    let catalog = crate::catalog::parse_and_validate_catalog(&json).expect("catalog parses");

    let file = &catalog.packs["support-x"].files[0];
    assert_eq!(
        file.archive,
        Some(crate::catalog::ZipArchive {
            extract_to: "adblock".to_string(),
            root_dir: "l".to_string(),
            delete_after_extract: true,
            marker: InstallMarker {
                path: "adblock/l/.installed".to_string(),
                version: 2,
            },
        })
    );
}

#[test]
fn zip_extracting_into_the_install_root_keeps_its_root_dir() {
    let json = catalog_json_with_file(
        r#"{"name":"l.zip","sizeBytes":1,"installPath":"adblock/l.zip","url":"u",
            "archiveFormat":"zip","extractTo":".","deleteAfterExtract":true,
            "installMarkerPath":"adblock/.install-info.json","installMarkerVersion":1,
            "role":"lists","priority":0}"#,
    );
    let catalog = crate::catalog::parse_and_validate_catalog(&json).expect("catalog parses");

    let archive = catalog.packs["support-x"].files[0]
        .archive
        .clone()
        .expect("zip file");
    assert_eq!(archive.extract_to, "");
    assert_eq!(archive.root_dir, "adblock");
    assert_eq!(archive.install_dir(), "adblock");
}

#[test]
fn drops_packs_whose_archive_and_marker_disagree() {
    for file in [
        // A zip without a marker could never read as installed once its archive is deleted.
        r#"{"name":"l.zip","sizeBytes":1,"installPath":"a/l.zip","url":"u",
            "archiveFormat":"zip","extractTo":"a","role":"lists","priority":0}"#,
        r#"{"name":"l.txt","sizeBytes":1,"installPath":"a/l.txt","url":"u",
            "installMarkerPath":"a/l/.installed","installMarkerVersion":1,
            "role":"lists","priority":0}"#,
        // The marker has to sit in the archive's root directory, one level under extractTo.
        r#"{"name":"l.zip","sizeBytes":1,"installPath":"a/l.zip","url":"u",
            "archiveFormat":"zip","extractTo":"a",
            "installMarkerPath":"a/.installed","installMarkerVersion":1,
            "role":"lists","priority":0}"#,
        r#"{"name":"l.zip","sizeBytes":1,"installPath":"a/l.zip","url":"u",
            "archiveFormat":"zip","extractTo":"a",
            "installMarkerPath":"a/l/nested/.installed","installMarkerVersion":1,
            "role":"lists","priority":0}"#,
        r#"{"name":"l.zip","sizeBytes":1,"installPath":"a/l.zip","url":"u",
            "archiveFormat":"zip","extractTo":"a",
            "installMarkerPath":"b/l/.installed","installMarkerVersion":1,
            "role":"lists","priority":0}"#,
        r#"{"name":"l.zip","sizeBytes":1,"installPath":"a/l.zip","url":"u",
            "archiveFormat":"zip","extractTo":"../a",
            "installMarkerPath":"../a/l/.installed","installMarkerVersion":1,
            "role":"lists","priority":0}"#,
    ] {
        let catalog = crate::catalog::parse_and_validate_catalog(&catalog_json_with_file(file))
            .expect("catalog parses");
        assert!(!catalog.packs.contains_key("support-x"), "{file}");
    }
}

#[test]
fn drops_packs_whose_files_this_build_cannot_install() {
    for file in [
        r#"{"name":"l.tar","sizeBytes":1,"installPath":"a/l.tar","url":"u",
            "archiveFormat":"tar","extractTo":"a","role":"lists","priority":0}"#,
        r#"{"name":"l.zip","sizeBytes":1,"installPath":"a/l.zip","url":"u",
            "archiveFormat":"zip","role":"lists","priority":0}"#,
        r#"{"name":"l.zip","sizeBytes":1,"installPath":"a/l.zip","url":"u",
            "installMarkerPath":"a/.installed","role":"lists","priority":0}"#,
    ] {
        let catalog = crate::catalog::parse_and_validate_catalog(&catalog_json_with_file(file))
            .expect("catalog parses");
        assert!(!catalog.packs.contains_key("support-x"), "{file}");
    }
}

// ja and ko existed in the previous catalog and read every voice from one shared
// voices file; es is new. Each now ships its own voice file, and the Kokoro core
// carries a newer model next to the previous one.
const LEGACY_VOICES: &str = "bin/kokoro/voices-v1.0.bin";
const OLD_MODEL: &str = "bin/kokoro_mnn/kokoro.mnn";
const OLD_WEIGHT: &str = "bin/kokoro_mnn/kokoro.mnn.weight";
const NEW_MODEL: &str = "bin/kokoro_mnn/v2/kokoro.mnn";
const NEW_WEIGHT: &str = "bin/kokoro_mnn/v2/kokoro.mnn.weight";
const JA_VOICE: &str = "bin/kokoro_mnn/ja/ja_JP/jf_alpha/high/jf_alpha.bin";
const KO_VOICE: &str = "bin/kokoro_mnn/ko/ko_KR/jf_alpha/high/jf_alpha.bin";
const ES_VOICE: &str = "bin/kokoro_mnn/es/es_ES/ef_dora/high/ef_dora.bin";
const ESPEAK: &str = "bin/espeak-ng-data/phondata";

fn kokoro_language(code: &str, region: &str) -> String {
    format!(
        r#""{code}":{{"meta":{{"code":"{code}","name":"{code}","shortName":"{code}","script":"Latn"}},
          "assets":{{"translate":[],"ocr":{{}}}},
          "tts":{{"defaultRegion":"{region}","regions":{{"{region}":{{"displayName":"{region}",
            "voices":["tts-kokoro-{code}"]}}}}}}}}"#
    )
}

fn kokoro_voice_pack(
    code: &str,
    path: &str,
    replaces: &str,
    voice: &str,
    speakers: &str,
) -> String {
    format!(
        r#""tts-kokoro-{code}":{{"feature":"tts","engine":"kokoro_mnn","auxRole":"voices","language":"{code}",
          "voice":"{voice}","speakers":[{speakers}],
          "files":[{{"name":"v.bin","sizeBytes":261390,"installPath":"{path}","url":"u","role":"voices",
                     "priority":0,"replaces":[{replaces}]}}],
          "dependsOn":["dict-{code}","kokoro-core"]}},
        "dict-{code}":{{"feature":"support","kind":"tts-espeak-dict","files":[{{"name":"{code}_dict",
          "sizeBytes":50000,"installPath":"bin/espeak-ng-data/{code}_dict","url":"u","role":"dict","priority":0}}],
          "dependsOn":["espeak"]}}"#
    )
}

fn kokoro_catalog() -> LanguageCatalog {
    let legacy = format!("\"{LEGACY_VOICES}\"");
    let json = format!(
        r#"{{"formatVersion":6,"generatedAt":1,"dictionaryVersion":1,
          "sources":{{"languageIndexVersion":1,"languageIndexUpdatedAt":1,
                      "dictionaryIndexVersion":1,"dictionaryIndexUpdatedAt":1}},
          "languages":{{{},{},{}}},
          "packs":{{{},{},{},
            "kokoro-core":{{"feature":"support","kind":"tts-kokoro-mnn-core","files":[
              {{"name":"kokoro.mnn","sizeBytes":967400,"installPath":"{NEW_MODEL}","url":"u","role":"model","priority":1}},
              {{"name":"kokoro.mnn.weight","sizeBytes":70179368,"installPath":"{NEW_WEIGHT}","url":"u","role":"modelWeight","priority":1}},
              {{"name":"kokoro.mnn","sizeBytes":1028464,"installPath":"{OLD_MODEL}","url":"u","role":"model","priority":0}},
              {{"name":"kokoro.mnn.weight","sizeBytes":86949555,"installPath":"{OLD_WEIGHT}","url":"u","role":"modelWeight","priority":0}}],
              "dependsOn":[]}},
            "espeak":{{"feature":"support","kind":"tts-espeak-core","files":[{{"name":"phondata","sizeBytes":1,
              "installPath":"{ESPEAK}","url":"u","role":"espeakCore","priority":0}}],"dependsOn":[]}}
          }}}}"#,
        kokoro_language("ja", "JP"),
        kokoro_language("ko", "KR"),
        kokoro_language("es", "ES"),
        kokoro_voice_pack("ja", JA_VOICE, &legacy, "jf_alpha", ""),
        kokoro_voice_pack("ko", KO_VOICE, &legacy, "jf_alpha", ""),
        kokoro_voice_pack(
            "es",
            ES_VOICE,
            "",
            "ef_dora",
            r#""ef_dora","em_alex","em_santa""#
        ),
    );
    crate::catalog::parse_and_validate_catalog(&json).expect("kokoro catalog parses")
}

fn kokoro_snapshot(files: &[&str]) -> crate::catalog::CatalogSnapshot {
    build_catalog_snapshot(
        kokoro_catalog(),
        "/base".to_string(),
        &FakeInstallChecker::with_files(files),
    )
}

// A device that installed Japanese Kokoro from the previous catalog.
const JA_INSTALL: &[&str] = &[
    ESPEAK,
    "bin/espeak-ng-data/ja_dict",
    OLD_MODEL,
    OLD_WEIGHT,
    LEGACY_VOICES,
];

fn task_paths(plan: &crate::catalog::DownloadPlan) -> Vec<&str> {
    let mut paths = plan
        .tasks
        .iter()
        .map(|task| task.install_path.as_str())
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn sorted(paths: &[String]) -> Vec<&str> {
    let mut paths = paths.iter().map(String::as_str).collect::<Vec<_>>();
    paths.sort();
    paths
}

#[test]
fn previous_kokoro_install_keeps_loading_its_replaced_files() {
    let snapshot = kokoro_snapshot(JA_INSTALL);

    let resolved = crate::catalog::resolve_tts_voice_files_for_pack(
        &snapshot,
        &LanguageCode::from("ja"),
        Some("tts-kokoro-ja"),
    )
    .expect("ja still resolves");
    assert_eq!(resolved.model_install_path, OLD_MODEL);
    assert_eq!(resolved.aux_install_path, LEGACY_VOICES);
    assert!(plan_repair(&snapshot).tasks.is_empty());
}

#[test]
fn replaced_file_does_not_install_a_sibling_whose_dependencies_are_missing() {
    let snapshot = kokoro_snapshot(JA_INSTALL);

    let ko = &snapshot.pack_statuses["tts-kokoro-ko"];
    assert!(!ko.installed);
    assert!(!ko.any_file_present);
    assert!(plan_repair(&snapshot).tasks.is_empty());
}

#[test]
fn previous_kokoro_install_is_offered_its_voice_file_and_the_new_model() {
    let snapshot = kokoro_snapshot(JA_INSTALL);

    assert_eq!(
        crate::catalog::tts_upgrade_language_codes(&snapshot),
        vec![LanguageCode::from("ja")]
    );
    let plan = crate::catalog::plan_tts_upgrades(&snapshot, &[LanguageCode::from("ja")]);
    assert_eq!(task_paths(&plan), vec![JA_VOICE, NEW_MODEL, NEW_WEIGHT]);
    assert_eq!(
        sorted(&plan.delete_after.file_paths),
        vec![LEGACY_VOICES, OLD_MODEL, OLD_WEIGHT]
    );
}

#[test]
fn upgrading_one_voice_keeps_the_replaced_file_a_sibling_still_loads() {
    let mut files = JA_INSTALL.to_vec();
    files.push("bin/espeak-ng-data/ko_dict");
    let snapshot = kokoro_snapshot(&files);

    let plan = crate::catalog::plan_tts_upgrades(&snapshot, &[LanguageCode::from("ja")]);
    assert_eq!(task_paths(&plan), vec![JA_VOICE, NEW_MODEL, NEW_WEIGHT]);
    assert_eq!(
        sorted(&plan.delete_after.file_paths),
        vec![OLD_MODEL, OLD_WEIGHT]
    );

    let both = crate::catalog::plan_tts_upgrades(
        &snapshot,
        &[LanguageCode::from("ja"), LanguageCode::from("ko")],
    );
    assert_eq!(
        sorted(&both.delete_after.file_paths),
        vec![LEGACY_VOICES, OLD_MODEL, OLD_WEIGHT]
    );
}

#[test]
fn adding_a_kokoro_language_reuses_the_installed_model() {
    let snapshot = kokoro_snapshot(JA_INSTALL);

    let plan = crate::catalog::plan_tts_download(&snapshot, &LanguageCode::from("es"), None)
        .expect("es has a voice");
    assert_eq!(
        task_paths(&plan),
        vec!["bin/espeak-ng-data/es_dict", ES_VOICE]
    );

    let mut files = JA_INSTALL.to_vec();
    files.extend(["bin/espeak-ng-data/es_dict", ES_VOICE]);
    let snapshot = kokoro_snapshot(&files);
    let resolved = crate::catalog::resolve_tts_voice_files_for_pack(
        &snapshot,
        &LanguageCode::from("es"),
        Some("tts-kokoro-es"),
    )
    .expect("es resolves");
    assert_eq!(resolved.model_install_path, OLD_MODEL);
    assert_eq!(resolved.aux_install_path, ES_VOICE);
}

#[test]
fn fresh_kokoro_install_downloads_only_the_current_files() {
    let snapshot = kokoro_snapshot(&[]);

    let plan = crate::catalog::plan_tts_download(&snapshot, &LanguageCode::from("ja"), None)
        .expect("ja has a voice");
    assert_eq!(
        task_paths(&plan),
        vec![
            "bin/espeak-ng-data/ja_dict",
            ESPEAK,
            JA_VOICE,
            NEW_MODEL,
            NEW_WEIGHT
        ]
    );
}

#[test]
fn deleting_a_voice_keeps_the_replaced_file_a_sibling_still_loads() {
    let mut files = JA_INSTALL.to_vec();
    files.push("bin/espeak-ng-data/ko_dict");
    let snapshot = kokoro_snapshot(&files);

    let plan = crate::catalog::plan_delete_tts_pack(&snapshot, "tts-kokoro-ja");
    assert_eq!(
        sorted(&plan.file_paths),
        vec!["bin/espeak-ng-data/ja_dict", JA_VOICE]
    );

    let snapshot = kokoro_snapshot(JA_INSTALL);
    let plan = crate::catalog::plan_delete_tts_pack(&snapshot, "tts-kokoro-ja");
    assert!(plan.file_paths.iter().any(|path| path == LEGACY_VOICES));
    assert!(plan.file_paths.iter().any(|path| path == OLD_MODEL));
}

#[test]
fn voice_size_counts_only_the_preferred_kokoro_model() {
    let snapshot = kokoro_snapshot(&[]);

    assert_eq!(
        snapshot.catalog.pack_size_bytes("tts-kokoro-es"),
        261390 + 50000 + 1 + 967400 + 70179368
    );
}

#[test]
fn voice_download_size_leaves_out_files_another_kokoro_voice_brought() {
    let snapshot = kokoro_snapshot(&[
        ESPEAK,
        "bin/espeak-ng-data/ja_dict",
        NEW_MODEL,
        NEW_WEIGHT,
        JA_VOICE,
    ]);

    let regions = crate::catalog::tts_voice_picker_regions(&snapshot, &LanguageCode::from("es"));
    let voice = &regions[0].voices[0];
    assert_eq!(voice.download_size_bytes, 261390 + 50000);
    assert_eq!(voice.size_bytes, 261390 + 50000 + 1 + 967400 + 70179368);

    let regions = crate::catalog::tts_voice_picker_regions(&snapshot, &LanguageCode::from("ja"));
    assert_eq!(regions[0].voices[0].download_size_bytes, 0);
}

#[test]
fn installed_multi_speaker_voice_offers_its_speakers_by_name() {
    let snapshot = kokoro_snapshot(&[
        ESPEAK,
        "bin/espeak-ng-data/es_dict",
        NEW_MODEL,
        NEW_WEIGHT,
        ES_VOICE,
    ]);

    let packs =
        crate::catalog::installed_tts_voices_for_language(&snapshot, &LanguageCode::from("es"));
    assert_eq!(packs[0].display_name, "Kokoro");
    let speakers = packs[0]
        .voices
        .iter()
        .map(|voice| {
            (
                voice.name.as_str(),
                voice.display_name.as_str(),
                voice.speaker_id,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        speakers,
        [
            ("ef_dora", "Dora", 0),
            ("em_alex", "Alex", 1),
            ("em_santa", "Santa", 2)
        ]
    );
}

#[test]
fn voice_picker_counts_each_pack_speakers() {
    let snapshot = kokoro_snapshot(&[]);

    let es = crate::catalog::tts_voice_picker_regions(&snapshot, &LanguageCode::from("es"));
    assert_eq!(es[0].voices[0].speaker_count, 3);
    assert_eq!(es[0].voices[0].voice_name, "ef_dora");
    assert_eq!(es[0].voices[0].display_name, "Kokoro");
    let ja = crate::catalog::tts_voice_picker_regions(&snapshot, &LanguageCode::from("ja"));
    assert_eq!(ja[0].voices[0].speaker_count, 1);
}
