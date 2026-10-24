//! Drives one `TranslatorSession` over a real install: translation and speech
//! from two threads at once, voices from engines that do and don't phonemize
//! with eSpeak, and a voice deletion while the session stays open.
//!
//! Needs `TRANSLATOR_TEST_INDEX` (a catalog JSON) and `TRANSLATOR_TEST_BUCKET`
//! (a local mirror of the download bucket, with `espeak-ng-data` unzipped next
//! to its zip), plus the es/en translation models installed under
//! `~/.local/share/dev.davidv.translator`. Skipped when any is missing.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

use translator::TranslatorSession;
use translator::catalog::{AssetFileV2, FsPackInstallChecker, LanguageCatalog};

const KOKORO_ES: &str = "tts-kokoro_mnn-es-es-ef-dora-kokoro-mnn-v1.0";
const PIPER_ES: &str = "tts-piper-es-es-davefx-medium";
const MMS_AZ: &str = "tts-mms-az-az-north-latin-mms";
const PACKS: [&str; 5] = [
    "translate-es-en",
    "translate-en-es",
    KOKORO_ES,
    PIPER_ES,
    MMS_AZ,
];

struct Sources {
    index: PathBuf,
    bucket: PathBuf,
    desktop_install: PathBuf,
}

fn sources() -> Option<Sources> {
    let index = PathBuf::from(std::env::var_os("TRANSLATOR_TEST_INDEX")?);
    let bucket = PathBuf::from(std::env::var_os("TRANSLATOR_TEST_BUCKET")?);
    let desktop_install =
        PathBuf::from(std::env::var_os("HOME")?).join(".local/share/dev.davidv.translator");
    (index.is_file() && bucket.is_dir() && desktop_install.is_dir()).then_some(Sources {
        index,
        bucket,
        desktop_install,
    })
}

fn bucket_path(bucket: &Path, url: &str) -> PathBuf {
    let path = url
        .split_once("://")
        .and_then(|(_, rest)| rest.split_once('/'))
        .map(|(_, path)| path)
        .expect("catalog urls are absolute");
    bucket.join(path)
}

fn link(source: &Path, target: &Path) {
    if target.exists() {
        return;
    }
    fs::create_dir_all(target.parent().expect("install paths have a parent"))
        .expect("create install directory");
    std::os::unix::fs::symlink(source, target).expect("link install file");
}

/// Lays out one file as the app's installer would leave it. Zips are taken
/// pre-extracted from the bucket, and gzipped translation models from the
/// desktop install.
fn install_file(sources: &Sources, base: &Path, file: &AssetFileV2) {
    if let Some(marker) = &file.install_marker {
        let extracted = bucket_path(&sources.bucket, &file.url).with_extension("");
        let marker_path = base.join(&marker.path);
        let target_dir = marker_path
            .parent()
            .expect("markers sit in their directory");
        for entry in fs::read_dir(&extracted).expect("read extracted archive") {
            let entry = entry.expect("archive entry");
            link(&entry.path(), &target_dir.join(entry.file_name()));
        }
        fs::write(&marker_path, format!("{{\"version\":{}}}", marker.version))
            .expect("write install marker");
        return;
    }
    let from_bucket = bucket_path(&sources.bucket, &file.url);
    let from_desktop = sources.desktop_install.join(&file.install_path);
    let source = if from_bucket.extension().is_some_and(|ext| ext == "gz") {
        from_desktop
    } else {
        from_bucket
    };
    if source.exists() {
        link(&source, &base.join(&file.install_path));
    }
}

fn install(sources: &Sources, catalog: &LanguageCatalog, base: &Path) {
    let mut pending: Vec<String> = PACKS.iter().map(|id| id.to_string()).collect();
    let mut seen = HashSet::new();
    while let Some(pack_id) = pending.pop() {
        if !seen.insert(pack_id.clone()) {
            continue;
        }
        let pack = catalog
            .pack(&pack_id)
            .unwrap_or_else(|| panic!("catalog has no pack `{pack_id}`"));
        // Markers first, so a file landing inside an extracted archive finds its directory.
        let mut files = pack.files.iter().collect::<Vec<_>>();
        files.sort_by_key(|file| file.install_marker.is_none());
        for file in files {
            install_file(sources, base, file);
        }
        pending.extend(pack.depends_on.iter().cloned());
    }
}

fn speak(session: &TranslatorSession, language: &str, pack_id: &str, voice: Option<&str>) -> usize {
    let text = match language {
        "az" => "Bu gün hava çox gözəldir.",
        _ => "Hola, hoy hace muy buen tiempo en Madrid.",
    };
    let chunks = session
        .plan_speech_chunks(
            language,
            text,
            Some(pack_id),
            translator::tts::UrlsAndHashtags::Read,
        )
        .unwrap_or_else(|err| panic!("plan speech for `{pack_id}`: {err:?}"));
    assert!(!chunks.is_empty(), "`{pack_id}` planned no chunks");
    chunks
        .iter()
        .map(|chunk| {
            session
                .synthesize_pcm(
                    language,
                    &chunk.content,
                    1.0,
                    voice,
                    chunk.is_phonemes,
                    Some(pack_id),
                )
                .unwrap_or_else(|err| panic!("synthesize `{pack_id}`: {err:?}"))
                .pcm_samples
                .len()
        })
        .sum()
}

#[test]
fn one_session_translates_and_speaks_from_several_threads() {
    let Some(sources) = sources() else {
        eprintln!(
            "TRANSLATOR_TEST_INDEX / TRANSLATOR_TEST_BUCKET or the desktop install missing; skipping"
        );
        return;
    };
    let index = fs::read_to_string(&sources.index).expect("read catalog index");
    let catalog =
        translator::catalog::parse_language_catalog(&index).expect("catalog index parses");
    let base = std::env::temp_dir().join(format!("session_smoke-{}", std::process::id()));
    install(&sources, &catalog, &base);

    let checker = FsPackInstallChecker::new(base.to_str().expect("utf-8 temp dir"));
    let session = Arc::new(
        TranslatorSession::open(&index, None, base.display().to_string(), &checker)
            .expect("session opens"),
    );

    let translator = {
        let session = Arc::clone(&session);
        thread::spawn(move || {
            (0..3)
                .map(|_| {
                    session
                        .translate_text("es", "en", "Hoy hace muy buen tiempo en Madrid.")
                        .expect("translate es→en")
                })
                .collect::<Vec<_>>()
        })
    };

    let kokoro_dora = speak(&session, "es", KOKORO_ES, Some("ef_dora"));
    let kokoro_alex = speak(&session, "es", KOKORO_ES, Some("em_alex"));
    let piper = speak(&session, "es", PIPER_ES, None);
    let mms = speak(&session, "az", MMS_AZ, None);
    // Back to an eSpeak voice after one that never opened it.
    let kokoro_again = speak(&session, "es", KOKORO_ES, Some("ef_dora"));

    for samples in [kokoro_dora, kokoro_alex, piper, mms] {
        assert!(samples > 0);
    }
    assert_eq!(kokoro_again, kokoro_dora);
    assert_ne!(
        kokoro_alex, kokoro_dora,
        "em_alex must not fall back to ef_dora"
    );

    for translation in translator.join().expect("translation thread") {
        assert!(
            translation.to_lowercase().contains("weather"),
            "{translation}"
        );
    }

    session.prepare_delete_tts_pack(PIPER_ES);
    assert!(speak(&session, "es", KOKORO_ES, Some("ef_dora")) > 0);

    fs::remove_dir_all(&base).expect("remove the temporary install");
}
