use std::collections::HashMap;

use crate::api::{DictionaryCode, LanguageCode};
use crate::catalog::{
    AssetFileV2, AssetPackMetadataV2, CatalogSourcesV2, FileRequirement, FileRole,
    LangAvailability, LanguageCatalog, LanguageFeature, LanguageTtsRegionV2, LanguageTtsV2,
    PackInstallChecker, PackRecord, build_catalog_snapshot, plan_delete_superseded_files,
    plan_language_download,
};
use crate::catalog::{plan_ocr_engine_downloads, plan_ocr_engine_upgrades};
use crate::language::Language;
use crate::translate::resolve_translation_plan_in_snapshot;

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
        archive_format: None,
        extract_to: None,
        delete_after_extract: false,
        install_marker_path: None,
        install_marker_version: None,
        role: None,
        priority: 0,
        requirement: FileRequirement::Required,
    }
}

fn role_file(
    name: &str,
    install_path: &str,
    size_bytes: u64,
    role: &str,
    priority: i32,
) -> AssetFileV2 {
    AssetFileV2 {
        role: Some(FileRole::new(role)),
        priority,
        ..asset_file(name, install_path, size_bytes)
    }
}

fn optional_role_file(
    name: &str,
    install_path: &str,
    size_bytes: u64,
    role: &str,
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
            engine: Some("piper".to_string()),
            locale: None,
            region: Some(region.to_string()),
            voice: None,
            quality: None,
            num_speakers: None,
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
        script: "Latn".to_string(),
        dictionary_code: "en".to_string(),
    };
    let spanish = Language {
        code: "es".to_string(),
        display_name: "Spanish".to_string(),
        short_display_name: "Spanish".to_string(),
        script: "Latn".to_string(),
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
                    FileRole::DETECTOR,
                    0,
                ),
                role_file(
                    "det_new.mnn",
                    "ppocr/det_new.mnn",
                    40,
                    FileRole::DETECTOR,
                    1,
                ),
                role_file(
                    "pulc.mnn",
                    "ppocr/pulc.mnn",
                    10,
                    FileRole::SCRIPT_CLASSIFIER,
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
            role_file("latin.mnn", "ppocr/latin.mnn", 10, FileRole::RECOGNIZER, 0),
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

fn catalog_with_optional_ink() -> LanguageCatalog {
    let mut catalog = catalog_with_detector_alternatives();
    let detector = catalog.packs.get_mut("ocr-ppocr-detector").unwrap();
    detector.files.push(optional_role_file(
        "ink.mnn",
        "ppocr/ink.mnn",
        50,
        FileRole::INK,
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

    let en = [LanguageCode::from("en")];
    let downloads = plan_ocr_engine_downloads(&snapshot, &en, "ppocr");
    assert!(downloads.tasks.is_empty());

    let upgrades = plan_ocr_engine_upgrades(&snapshot, &en, "ppocr");
    let paths = upgrades
        .tasks
        .iter()
        .map(|task| task.install_path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(paths, vec!["ppocr/ink.mnn"]);
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
fn selects_best_catalog_using_headers_only() {
    let bundled = r#"{"formatVersion":3,"generatedAt":1}"#;
    let disk = r#"{"formatVersion":3,"generatedAt":2}"#;

    let selected = crate::catalog::select_best_catalog(bundled, Some(disk))
        .expect("header-only catalogs should still compare");

    assert_eq!(selected, disk);
}

#[test]
fn resolves_direct_translation_plan_from_installed_catalog() {
    let catalog = base_catalog();
    let checker = FakeInstallChecker::with_files(&[
        "bin/model.enes.bin",
        "bin/lex.enen.bin",
        "bin/vocab.aa.spm",
        "bin/vocab.zz.spm",
        "bin/shared.bin",
    ]);
    let snapshot = build_catalog_snapshot(
        catalog,
        "/data/user/0/dev.davidv.translator/files".to_string(),
        &checker,
    );

    let plan = resolve_translation_plan_in_snapshot(&snapshot, "en", "es")
        .expect("direct plan should resolve");

    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].cache_key, "en-es");
    let paths = &plan.steps[0].paths;
    assert_eq!(
        paths.model.to_str().unwrap(),
        "/data/user/0/dev.davidv.translator/files/bin/model.enes.bin"
    );
    assert_eq!(
        paths.vocabulary.to_str().unwrap(),
        "/data/user/0/dev.davidv.translator/files/bin/vocab.aa.spm"
    );
}

#[test]
fn resolves_pivot_translation_plan_from_installed_catalog() {
    let mut catalog = base_catalog();
    let french = Language {
        code: "fr".to_string(),
        display_name: "French".to_string(),
        short_display_name: "French".to_string(),
        script: "Latn".to_string(),
        dictionary_code: "fr".to_string(),
    };
    catalog.languages.insert(
        "fr".to_string(),
        language_info(
            french,
            vec!["translate-fr-en", "translate-en-fr"],
            "ocr-en",
            None,
            vec![],
            vec![],
            None,
        ),
    );
    catalog.packs.insert(
        "translate-en-fr".to_string(),
        translation_pack(
            "translate-en-fr",
            "en",
            "fr",
            vec![
                asset_file("model.enfr.bin", "bin/model.enfr.bin", 40),
                asset_file("lex.enfr.bin", "bin/lex.enfr.bin", 41),
                asset_file("vocab.cc.spm", "bin/vocab.cc.spm", 42),
                asset_file("vocab.dd.spm", "bin/vocab.dd.spm", 43),
            ],
            vec!["shared-translate"],
        ),
    );
    catalog.packs.insert(
        "translate-fr-en".to_string(),
        translation_pack(
            "translate-fr-en",
            "fr",
            "en",
            vec![
                asset_file("model.fren.bin", "bin/model.fren.bin", 44),
                asset_file("lex.fren.bin", "bin/lex.fren.bin", 45),
                asset_file("vocab.ee.spm", "bin/vocab.ee.spm", 46),
                asset_file("vocab.ff.spm", "bin/vocab.ff.spm", 47),
            ],
            vec!["shared-translate"],
        ),
    );
    catalog.translation_pack_ids.insert(
        ("en".to_string(), "fr".to_string()),
        "translate-en-fr".to_string(),
    );
    catalog.translation_pack_ids.insert(
        ("fr".to_string(), "en".to_string()),
        "translate-fr-en".to_string(),
    );
    catalog.root_pack_ids_by_language_feature.insert(
        ("fr".to_string(), LanguageFeature::Translation),
        vec!["translate-fr-en".to_string(), "translate-en-fr".to_string()],
    );

    let checker = FakeInstallChecker::with_files(&[
        "bin/model.esen.bin",
        "bin/lex.esen.bin",
        "bin/vocab.bb.spm",
        "bin/vocab.yy.spm",
        "bin/model.enfr.bin",
        "bin/lex.enfr.bin",
        "bin/vocab.cc.spm",
        "bin/vocab.dd.spm",
        "bin/shared.bin",
    ]);
    let snapshot = build_catalog_snapshot(catalog, "/tmp/base".to_string(), &checker);

    let plan = resolve_translation_plan_in_snapshot(&snapshot, "es", "fr")
        .expect("pivot plan should resolve");

    assert_eq!(plan.steps.len(), 2);
    assert_eq!(plan.steps.len(), 2);
    assert_eq!(plan.steps[0].cache_key, "es-en");
    assert_eq!(plan.steps[1].cache_key, "en-fr");
}

#[test]
fn refuses_translation_plan_when_required_direction_is_missing() {
    let catalog = base_catalog();
    let checker = FakeInstallChecker::with_files(&["bin/model.esen.bin"]);
    let snapshot = build_catalog_snapshot(catalog, "/tmp/base".to_string(), &checker);

    assert!(resolve_translation_plan_in_snapshot(&snapshot, "es", "en").is_none());
}
