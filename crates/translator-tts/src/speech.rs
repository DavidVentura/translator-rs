use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::Instant;

use piper_rs::{
    Backend, BoundaryAfter, CoquiVitsModel, CotoviaVitsModel, Espeak, GlowTtsHifiganModel,
    KokoroLanguage, KokoroMnnModel, MmsModel, PhonemeChunk as PiperPhonemeChunk, PiperModel,
    SherpaVitsModel,
};
use translator_core::api::{LanguageCode, TranslatorError, VoiceName};
use translator_core::catalog::{CatalogSnapshot, ResolvedTtsVoiceFiles};
use translator_core::script::Script;
use translator_core::tts::{
    BoundaryPauses, PcmAudio, PhonemeChunk, SpeechChunk, SpeechChunkBoundary, TtsEngine,
    TtsVoiceOption, UrlsAndHashtags, plan_speech_chunks, voice_display_name,
};

fn log_debug(message: impl AsRef<str>) {
    log::debug!("{}", message.as_ref());
}

const ESPEAK_OPEN: &str = "eSpeak is open for every engine that phonemizes with it";

fn log_error(message: impl AsRef<str>) {
    log::error!("{}", message.as_ref());
}

pub struct SpeechCache {
    model: Option<CachedSpeechModel>,
    espeak: Option<OpenEspeak>,
}

struct OpenEspeak {
    data_dir: PathBuf,
    espeak: Espeak,
}

impl SpeechCache {
    pub fn new() -> Self {
        Self {
            model: None,
            espeak: None,
        }
    }

    pub fn clear(&mut self) {
        self.model = None;
    }
}

impl Default for SpeechCache {
    fn default() -> Self {
        Self::new()
    }
}

fn audio_duration_ms(sample_count: usize, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        return 0;
    }
    ((sample_count as u64) * 1000) / u64::from(sample_rate)
}

enum SpeechModel {
    Piper(PiperModel),
    KokoroMnn(KokoroMnnModel),
    Mms(MmsModel),
    CoquiVits(CoquiVitsModel),
    CotoviaVits(CotoviaVitsModel),
    SherpaVits(SherpaVitsModel),
    GlowTtsHifigan(GlowTtsHifiganModel),
}

/// Every file a voice loads from, resolved out of the catalog. They travel as
/// one value because each layer below needs the whole set, and which files a
/// voice has depends on its engine.
#[derive(Clone, PartialEq, Eq)]
struct SpeechModelPaths {
    engine: TtsEngine,
    model: String,
    aux: String,
    /// Second network, for engines that split the acoustic model from the vocoder.
    vocoder: Option<String>,
    language_code: String,
    locale: Option<String>,
    voice: Option<String>,
    support_data_root: Option<String>,
}

impl SpeechModelPaths {
    fn support_root(&self) -> &str {
        self.support_data_root.as_deref().unwrap_or_default()
    }
}

struct CachedSpeechModel {
    paths: SpeechModelPaths,
    voices: Vec<(String, i64)>,
    default_voice: Option<(String, i64)>,
    model: SpeechModel,
}

struct ResolvedSpeechAssets {
    paths: SpeechModelPaths,
    script: Script,
    speaker_id: Option<i64>,
}

fn log_timing(step: &str, started_at: Instant) {
    log_debug(format!(
        "{step} took {} ms",
        started_at.elapsed().as_millis()
    ));
}

fn summarize_phoneme_chunk_sizes(chunks: &[PiperPhonemeChunk]) -> String {
    const MAX_CHUNKS_TO_LOG: usize = 6;

    let preview = chunks
        .iter()
        .take(MAX_CHUNKS_TO_LOG)
        .map(|chunk| chunk.phonemes.chars().count().to_string())
        .collect::<Vec<_>>()
        .join(", ");

    if chunks.len() > MAX_CHUNKS_TO_LOG {
        format!("{preview}, ...")
    } else {
        preview
    }
}

fn available_voices(model: &SpeechModel) -> Vec<(String, i64)> {
    let mut voices = match model {
        SpeechModel::Piper(model) => model
            .voices()
            .map(|voices| {
                voices
                    .iter()
                    .map(|(name, id)| (name.clone(), *id))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        // One voices file holds every Kokoro language; a pack only offers the
        // voices trained on its own.
        SpeechModel::KokoroMnn(model) => {
            let prefix = model.language().voice_prefix();
            model
                .voices()
                .map(|voices| {
                    voices
                        .iter()
                        .filter(|(name, _)| name.starts_with(prefix))
                        .map(|(name, id)| (name.clone(), *id))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        }
        SpeechModel::Mms(model) => model
            .voices()
            .map(|voices| {
                voices
                    .iter()
                    .map(|(name, id)| (name.clone(), *id))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        SpeechModel::CoquiVits(model) => model
            .voices()
            .map(|voices| {
                voices
                    .iter()
                    .map(|(name, id)| (name.clone(), *id))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        SpeechModel::CotoviaVits(model) => model
            .voices()
            .map(|voices| {
                voices
                    .iter()
                    .map(|(name, id)| (name.clone(), *id))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        SpeechModel::GlowTtsHifigan(model) => model
            .voices()
            .map(|voices| {
                voices
                    .iter()
                    .map(|(name, id)| (name.clone(), *id))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        SpeechModel::SherpaVits(model) => model
            .voices()
            .map(|voices| {
                voices
                    .iter()
                    .map(|(name, id)| (name.clone(), *id))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
    };
    voices.sort_by(|left, right| left.0.cmp(&right.0));
    voices
}

fn visible_voices(voices: &[(String, i64)]) -> Vec<TtsVoiceOption> {
    voices
        .iter()
        .map(|(name, speaker_id)| TtsVoiceOption {
            display_name: voice_display_name(name),
            name: name.clone(),
            speaker_id: *speaker_id,
        })
        .collect()
}

/// Kokoro packs name their default voice; every other engine's default comes
/// from the pack's speaker id.
fn select_default_voice(
    model: &SpeechModel,
    paths: &SpeechModelPaths,
    voices: &[(String, i64)],
) -> Result<Option<(String, i64)>, String> {
    let SpeechModel::KokoroMnn(_) = model else {
        return Ok(None);
    };
    let voice = paths
        .voice
        .as_deref()
        .ok_or_else(|| "Kokoro voice pack does not name its voice".to_string())?;
    voices
        .iter()
        .find(|(name, _)| name == voice)
        .cloned()
        .map(Some)
        .ok_or_else(|| {
            format!("Kokoro voice `{voice}` is not in the voices file for this language")
        })
}

fn resolve_requested_voice(
    cached_model: &CachedSpeechModel,
    requested_voice_name: Option<&str>,
    speaker_id: Option<i64>,
) -> Option<(String, i64)> {
    if let Some(requested_voice_name) = requested_voice_name.filter(|value| !value.is_empty()) {
        if let Some((name, id)) = cached_model
            .voices
            .iter()
            .find(|(name, _)| name == requested_voice_name)
        {
            return Some((name.clone(), *id));
        }
        log_error(format!(
            "requested voice `{requested_voice_name}` was not found for engine={} language={}; falling back",
            cached_model.paths.engine, cached_model.paths.language_code
        ));
    }

    if let Some(speaker_id) = speaker_id {
        if let Some((name, id)) = cached_model.voices.iter().find(|(_, id)| *id == speaker_id) {
            return Some((name.clone(), *id));
        }
    }

    cached_model
        .default_voice
        .clone()
        .or_else(|| speaker_id.map(|id| (format!("speaker_{id}"), id)))
}

fn clamp_speech_speed(speech_speed: f32) -> f32 {
    speech_speed.clamp(0.5, 2.0)
}

fn support_data_root(snapshot: &CatalogSnapshot, _files: &ResolvedTtsVoiceFiles) -> Option<String> {
    let data_dir = Path::new(&snapshot.base_dir).join("bin");
    data_dir
        .join("espeak-ng-data")
        .is_dir()
        .then(|| data_dir.display().to_string())
}

fn absolute_install_path(snapshot: &CatalogSnapshot, relative_path: &str) -> String {
    Path::new(&snapshot.base_dir)
        .join(relative_path)
        .display()
        .to_string()
}

fn resolve_speech_assets(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> Option<ResolvedSpeechAssets> {
    resolve_speech_assets_for_pack(snapshot, language_code, None)
}

fn resolve_speech_assets_for_pack(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
    pack_id: Option<&str>,
) -> Option<ResolvedSpeechAssets> {
    let files = translator_core::catalog::resolve_tts_voice_files_for_pack(
        snapshot,
        language_code,
        pack_id,
    )?;
    let support_data_root = support_data_root(snapshot, &files);
    let model_path = absolute_install_path(snapshot, &files.model_install_path);
    if !Path::new(&model_path).exists() {
        return None;
    }
    let aux_path = absolute_install_path(snapshot, &files.aux_install_path);
    if !Path::new(&aux_path).exists() {
        return None;
    }
    let vocoder = match &files.vocoder_install_path {
        Some(relative) => {
            let path = absolute_install_path(snapshot, relative);
            if !Path::new(&path).exists() {
                return None;
            }
            Some(path)
        }
        None => None,
    };
    Some(ResolvedSpeechAssets {
        paths: SpeechModelPaths {
            engine: files.engine,
            model: model_path,
            aux: aux_path,
            vocoder,
            language_code: files.language_code,
            locale: files.locale,
            voice: files.voice,
            support_data_root,
        },
        script: snapshot.catalog.script_for(language_code)?,
        speaker_id: files.speaker_id.map(i64::from),
    })
}

fn piper_length_scale_for_speed(speech_speed: f32) -> f32 {
    1.0 / clamp_speech_speed(speech_speed)
}

/// When the TTS voice writes a Latin script, romanize any foreign-script runs
/// in the text so the voice pronounces them instead of skipping them. A voice
/// renders exactly one script, so anything outside it is dropped at
/// tokenization; voices that natively write the source script keep the text
/// intact.
#[cfg(feature = "transliterate")]
fn romanize_foreign_runs_for_voice(text: &str, script: Script) -> String {
    if script != Script::Latin {
        return text.to_owned();
    }
    translator_transliterate::transliterate::transliterate_mixed_to_latin(text)
}

#[cfg(not(feature = "transliterate"))]
fn romanize_foreign_runs_for_voice(text: &str, _script: Script) -> String {
    text.to_owned()
}

fn missing_tts_asset(language_code: &LanguageCode) -> TranslatorError {
    TranslatorError::missing_asset(format!(
        "TTS voice not installed for {}",
        language_code.as_str()
    ))
}

pub fn available_tts_voices_in_snapshot(
    snapshot: &CatalogSnapshot,
    cache: &mut SpeechCache,
    language_code: &LanguageCode,
) -> Result<Vec<TtsVoiceOption>, TranslatorError> {
    let assets = resolve_speech_assets(snapshot, language_code)
        .ok_or_else(|| missing_tts_asset(language_code))?;
    list_voices(cache, &assets.paths).map_err(TranslatorError::tts)
}

pub fn warm_tts_model_in_snapshot(
    snapshot: &CatalogSnapshot,
    cache: &mut SpeechCache,
    language_code: &LanguageCode,
) -> Result<(), TranslatorError> {
    available_tts_voices_in_snapshot(snapshot, cache, language_code).map(|_| ())
}

pub fn plan_speech_chunks_for_text_in_snapshot(
    snapshot: &CatalogSnapshot,
    cache: &mut SpeechCache,
    language_code: &LanguageCode,
    text: &str,
    pack_id: Option<&str>,
    urls_and_hashtags: UrlsAndHashtags,
) -> Result<Vec<SpeechChunk>, TranslatorError> {
    let assets = resolve_speech_assets_for_pack(snapshot, language_code, pack_id)
        .ok_or_else(|| missing_tts_asset(language_code))?;
    plan_speech_chunks_for_text(cache, &assets.paths, assets.script, text, urls_and_hashtags)
        .map_err(TranslatorError::tts)
}

pub fn synthesize_pcm_in_snapshot(
    snapshot: &CatalogSnapshot,
    cache: &mut SpeechCache,
    language_code: &LanguageCode,
    text: &str,
    speech_speed: f32,
    voice_name: Option<&VoiceName>,
    is_phonemes: bool,
    pack_id: Option<&str>,
) -> Result<PcmAudio, TranslatorError> {
    let assets = resolve_speech_assets_for_pack(snapshot, language_code, pack_id)
        .ok_or_else(|| missing_tts_asset(language_code))?;
    synthesize_pcm(
        cache,
        &assets.paths,
        assets.script,
        text,
        speech_speed,
        voice_name.map(VoiceName::as_str),
        assets.speaker_id,
        is_phonemes,
    )
    .map_err(TranslatorError::tts)
}

fn japanese_dict_path(support_data_root: &str) -> Result<PathBuf, String> {
    if support_data_root.is_empty() {
        return Err(
            "Japanese speech needs the mucab dictionary, but no support data is installed"
                .to_owned(),
        );
    }

    let candidate = Path::new(support_data_root).join("mucab.bin");
    if !candidate.exists() {
        return Err(format!(
            "Japanese speech needs the mucab dictionary at {}, which is not installed",
            candidate.display()
        ));
    }
    Ok(candidate)
}

fn load_speech_model(paths: &SpeechModelPaths) -> Result<SpeechModel, String> {
    let SpeechModelPaths {
        engine,
        model: model_path,
        aux: aux_path,
        vocoder,
        language_code,
        ..
    } = paths;
    if aux_path.is_empty() {
        return Err(format!("Missing auxiliary path for TTS engine `{engine}`"));
    }

    match engine {
        TtsEngine::KokoroMnn => {
            let locale = paths
                .locale
                .as_deref()
                .ok_or_else(|| "Kokoro voice pack has no locale".to_string())?;
            let language = KokoroLanguage::from_locale(locale).map_err(|err| format!("{err}"))?;
            let mut model =
                KokoroMnnModel::new(Path::new(model_path), Path::new(aux_path), language)
                    .map_err(|err| format!("Failed to load Kokoro MNN voice: {err}"))?;
            if language == KokoroLanguage::Japanese {
                let dict_path = japanese_dict_path(paths.support_root())?;
                model
                    .load_japanese_dict(dict_path.to_string_lossy().as_ref())
                    .map_err(|err| format!("Failed to load Japanese dictionary: {err}"))?;
            }
            Ok(SpeechModel::KokoroMnn(model))
        }
        TtsEngine::Mms => MmsModel::new(
            Path::new(model_path),
            Path::new(aux_path),
            language_code,
            &Backend::Cpu,
        )
        .map(SpeechModel::Mms)
        .map_err(|err| format!("Failed to load MMS voice: {err}")),
        TtsEngine::CoquiVits => CoquiVitsModel::new(
            Path::new(model_path),
            Path::new(aux_path),
            language_code,
            &Backend::Cpu,
        )
        .map(SpeechModel::CoquiVits)
        .map_err(|err| format!("Failed to load Coqui VITS voice: {err}")),
        TtsEngine::CotoviaVits => {
            CotoviaVitsModel::new(Path::new(model_path), Path::new(aux_path), &Backend::Cpu)
                .map(SpeechModel::CotoviaVits)
                .map_err(|err| format!("Failed to load Cotovia VITS voice: {err}"))
        }
        TtsEngine::GlowTtsHifigan => {
            let vocoder = vocoder
                .as_ref()
                .ok_or_else(|| format!("Missing vocoder path for TTS engine `{engine}`"))?;
            GlowTtsHifiganModel::new(
                Path::new(model_path),
                Path::new(vocoder),
                Path::new(aux_path),
                &Backend::Cpu,
            )
            .map(SpeechModel::GlowTtsHifigan)
            .map_err(|err| format!("Failed to load GlowTTS + HiFiGAN voice: {err}"))
        }
        TtsEngine::SherpaVits => {
            SherpaVitsModel::new(Path::new(model_path), Path::new(aux_path), &Backend::Cpu)
                .map(SpeechModel::SherpaVits)
                .map_err(|err| format!("Failed to load Sherpa VITS voice: {err}"))
        }
        TtsEngine::Mimic3 => {
            PiperModel::from_mimic3(Path::new(model_path), Path::new(aux_path), &Backend::Cpu)
                .map(SpeechModel::Piper)
                .map_err(|err| format!("Failed to load Mimic3 voice: {err}"))
        }
        TtsEngine::Piper => {
            PiperModel::new(Path::new(model_path), Path::new(aux_path), &Backend::Cpu)
                .map(SpeechModel::Piper)
                .map_err(|err| format!("Failed to load Piper voice: {err}"))
        }
    }
}

fn with_cached_model<T>(
    slot: &mut Option<CachedSpeechModel>,
    paths: &SpeechModelPaths,
    f: impl FnOnce(&mut CachedSpeechModel) -> Result<T, String>,
) -> Result<T, String> {
    let load_started_at = Instant::now();
    let engine = paths.engine;
    let language_code = paths.language_code.as_str();

    let cache_hit = slot
        .as_ref()
        .map(|cached| &cached.paths == paths)
        .unwrap_or(false);

    if !cache_hit {
        eprintln!(
            "tts.model: cache_miss engine={} language={} model={} aux={} support_root={}",
            engine,
            language_code,
            paths.model,
            paths.aux,
            paths.support_root()
        );
        let model_load_started_at = Instant::now();
        eprintln!(
            "tts.model: load.start engine={} language={}",
            engine, language_code
        );
        let model = load_speech_model(paths)?;
        eprintln!(
            "tts.model: load.done engine={} language={} took_ms={}",
            engine,
            language_code,
            model_load_started_at.elapsed().as_millis()
        );
        let voices = available_voices(&model);
        let default_voice = select_default_voice(&model, paths, &voices)?;
        if voices.is_empty() {
            eprintln!(
                "tts.model: voices.loaded engine={} language={} count=0",
                engine, language_code
            );
        } else if let Some((name, id)) = default_voice.as_ref() {
            eprintln!(
                "tts.model: voices.loaded engine={} language={} count={} default={}={}",
                engine,
                language_code,
                voices.len(),
                name,
                id
            );
        } else {
            eprintln!(
                "tts.model: voices.loaded engine={} language={} count={} default=<none>",
                engine,
                language_code,
                voices.len()
            );
        }
        *slot = Some(CachedSpeechModel {
            paths: paths.clone(),
            voices,
            default_voice,
            model,
        });

        eprintln!(
            "tts.model: ready engine={} language={} total_wait_ms={}",
            engine,
            language_code,
            load_started_at.elapsed().as_millis()
        );
    }

    let cached = slot
        .as_mut()
        .ok_or_else(|| "TTS model cache was unexpectedly empty".to_owned())?;
    f(cached)
}

fn espeak_data_dir(support_data_root: &str) -> PathBuf {
    let data_root = Path::new(support_data_root);
    let required = ["phondata", "phonindex", "phontab", "intonations"];
    let direct_layout_ok = required.iter().all(|name| data_root.join(name).exists());
    let nested_layout_ok = required
        .iter()
        .all(|name| data_root.join("espeak-ng-data").join(name).exists());

    log_debug(format!(
        "eSpeak data probe root={support_data_root} direct_layout_ok={direct_layout_ok} nested_layout_ok={nested_layout_ok}"
    ));

    // piper-rs's espeak expects the PARENT directory of `espeak-ng-data/`;
    // if the caller passed us the parent directly, use it, otherwise strip.
    if nested_layout_ok {
        data_root.to_path_buf()
    } else {
        data_root
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| data_root.to_path_buf())
    }
}

fn open_espeak<'a>(
    slot: &'a mut Option<OpenEspeak>,
    paths: &SpeechModelPaths,
) -> Result<Option<&'a mut Espeak>, String> {
    if !paths.engine.phonemizes_with_espeak() {
        return Ok(None);
    }
    let support_data_root = paths.support_data_root.as_deref().ok_or_else(|| {
        format!(
            "{} voices need eSpeak data, which is not installed",
            paths.engine
        )
    })?;
    let data_dir = espeak_data_dir(support_data_root);
    // eSpeak reads its dictionaries from the directory it was opened with, and
    // only one handle may be open, so a voice from elsewhere closes it first.
    if slot.as_ref().is_some_and(|open| open.data_dir != data_dir) {
        *slot = None;
    }
    if slot.is_none() {
        let espeak = Espeak::open(&data_dir).map_err(|err| {
            format!(
                "Failed to open eSpeak with data directory `{}`: {err}",
                data_dir.display()
            )
        })?;
        log_debug(format!("Opened eSpeak at {}", data_dir.display()));
        *slot = Some(OpenEspeak { data_dir, espeak });
    }
    Ok(slot.as_mut().map(|open| &mut open.espeak))
}

fn phonemize(
    model: &mut SpeechModel,
    espeak: Option<&mut Espeak>,
    text: &str,
) -> Result<String, String> {
    match model {
        SpeechModel::Piper(model) => model
            .phonemize(espeak.expect(ESPEAK_OPEN), text)
            .map_err(|err| format!("Speech synthesis failed: {err}")),
        SpeechModel::KokoroMnn(model) => model
            .phonemize(espeak.expect(ESPEAK_OPEN), text)
            .map_err(|err| format!("Speech synthesis failed: {err}")),
        SpeechModel::Mms(model) => model
            .phonemize(text)
            .map_err(|err| format!("Speech synthesis failed: {err}")),
        SpeechModel::CoquiVits(model) => model
            .phonemize(text)
            .map_err(|err| format!("Speech synthesis failed: {err}")),
        SpeechModel::CotoviaVits(model) => model
            .phonemize(text)
            .map_err(|err| format!("Speech synthesis failed: {err}")),
        SpeechModel::SherpaVits(model) => model
            .phonemize(text)
            .map_err(|err| format!("Speech synthesis failed: {err}")),
        SpeechModel::GlowTtsHifigan(model) => model
            .phonemize(text)
            .map_err(|err| format!("Speech synthesis failed: {err}")),
    }
}

/// `content` is phonemes for every engine but Cotovia, which reads its ids
/// from the text itself.
fn synthesize(
    cached_model: &mut CachedSpeechModel,
    content: &str,
    speech_speed: f32,
    voice_name: Option<&str>,
    speaker_id: Option<i64>,
) -> Result<(Vec<f32>, u32), String> {
    let selected_voice = resolve_requested_voice(cached_model, voice_name, speaker_id);
    let effective_speaker_id = selected_voice.as_ref().map(|(_, id)| *id);
    let clamped_speech_speed = clamp_speech_speed(speech_speed);
    if let Some((name, id)) = selected_voice.as_ref() {
        log_debug(format!(
            "using voice {name}={id} for engine={} language={} speech_speed={} ({} available)",
            cached_model.paths.engine,
            cached_model.paths.language_code,
            clamped_speech_speed,
            cached_model.voices.len()
        ));
    }
    let speed = Some(clamped_speech_speed);
    match &mut cached_model.model {
        SpeechModel::Piper(model) => model.synthesize_phonemes_with_options(
            content,
            effective_speaker_id,
            Some(piper_length_scale_for_speed(clamped_speech_speed)),
        ),
        SpeechModel::KokoroMnn(model) => {
            model.synthesize_phonemes(content, effective_speaker_id, speed)
        }
        SpeechModel::Mms(model) => model.synthesize_phonemes(content, effective_speaker_id, speed),
        SpeechModel::CoquiVits(model) => {
            model.synthesize_phonemes(content, effective_speaker_id, speed)
        }
        SpeechModel::SherpaVits(model) => {
            model.synthesize_phonemes(content, effective_speaker_id, speed)
        }
        SpeechModel::GlowTtsHifigan(model) => {
            model.synthesize_phonemes(content, effective_speaker_id, speed)
        }
        SpeechModel::CotoviaVits(model) => model.synthesize(content, effective_speaker_id, speed),
    }
    .map_err(|err| format!("Speech synthesis failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
fn synthesize_pcm(
    cache: &mut SpeechCache,
    paths: &SpeechModelPaths,
    script: Script,
    text: &str,
    speech_speed: f32,
    voice_name: Option<&str>,
    speaker_id: Option<i64>,
    is_phonemes: bool,
) -> Result<PcmAudio, String> {
    if text.trim().is_empty() {
        return Err("Text is empty".to_owned());
    }

    let total_started_at = Instant::now();
    let espeak = open_espeak(&mut cache.espeak, paths)?;

    let romanized = (!is_phonemes).then(|| romanize_foreign_runs_for_voice(text, script));
    let text = romanized.as_deref().unwrap_or(text);

    log_debug(format!(
        "Synthesizing speech with engine={} model={}",
        paths.engine, paths.model
    ));
    let (samples, sample_rate) = with_cached_model(&mut cache.model, paths, |cached_model| {
        let reads_text = matches!(cached_model.model, SpeechModel::CotoviaVits(_));
        let content = if is_phonemes || reads_text {
            log_debug(format!(
                "synthesizing direct chunk with {} char(s)",
                text.chars().count()
            ));
            Cow::Borrowed(text)
        } else {
            let phonemize_started_at = Instant::now();
            let phonemes = phonemize(&mut cached_model.model, espeak, text)?;
            log_debug(format!(
                "phonemize produced 1 chunk, {} phoneme char(s)",
                phonemes.chars().count(),
            ));
            log_timing("phonemize", phonemize_started_at);
            Cow::Owned(phonemes)
        };

        let synth_started_at = Instant::now();
        let result = synthesize(cached_model, &content, speech_speed, voice_name, speaker_id)?;
        log_timing("infer", synth_started_at);
        Ok(result)
    })?;

    let convert_started_at = Instant::now();
    let pcm_samples: Vec<i16> = samples
        .into_iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)
        .collect();
    let duration_ms = audio_duration_ms(pcm_samples.len(), sample_rate);
    log_debug(format!(
        "convert_to_pcm produced {} sample(s) at {} Hz (~{} ms audio)",
        pcm_samples.len(),
        sample_rate,
        duration_ms,
    ));
    log_timing("convert_to_pcm", convert_started_at);
    log_timing("synthesize_total", total_started_at);

    Ok(PcmAudio {
        sample_rate: sample_rate as i32,
        pcm_samples,
    })
}

fn list_voices(
    cache: &mut SpeechCache,
    paths: &SpeechModelPaths,
) -> Result<Vec<TtsVoiceOption>, String> {
    with_cached_model(&mut cache.model, paths, |cached_model| {
        Ok(visible_voices(&cached_model.voices))
    })
}

fn phonemize_chunks(
    cache: &mut SpeechCache,
    paths: &SpeechModelPaths,
    text: &str,
) -> Result<Vec<PiperPhonemeChunk>, String> {
    if text.trim().is_empty() {
        return Err("Text is empty".to_owned());
    }

    let espeak = open_espeak(&mut cache.espeak, paths)?;

    log_debug(format!(
        "Phonemizing text with engine={} model={}",
        paths.engine, paths.model
    ));

    with_cached_model(&mut cache.model, paths, |cached_model| {
        let phonemize_started_at = Instant::now();
        let phonemes = phonemize(&mut cached_model.model, espeak, text)?;
        let phoneme_chunks = vec![PiperPhonemeChunk {
            phonemes,
            boundary_after: BoundaryAfter::Paragraph,
        }];
        log_debug(format!(
            "phonemize produced {} chunk(s), {} phoneme char(s), chunk sizes [{}]",
            phoneme_chunks.len(),
            phoneme_chunks[0].phonemes.chars().count(),
            summarize_phoneme_chunk_sizes(&phoneme_chunks),
        ));
        log_timing("phonemize", phonemize_started_at);
        Ok(phoneme_chunks)
    })
}

fn to_phoneme_chunk(chunk: PiperPhonemeChunk) -> PhonemeChunk {
    PhonemeChunk {
        content: chunk.phonemes,
        boundary_after: match chunk.boundary_after {
            BoundaryAfter::None => SpeechChunkBoundary::None,
            BoundaryAfter::Sentence => SpeechChunkBoundary::Sentence,
            BoundaryAfter::Paragraph => SpeechChunkBoundary::Paragraph,
        },
    }
}

fn is_url(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("www.")
}

fn strip_urls_and_hashtags(text: &str) -> String {
    text.split_whitespace()
        .filter(|word| !is_url(word) && !word.starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ")
}

fn plan_speech_chunks_for_text(
    cache: &mut SpeechCache,
    paths: &SpeechModelPaths,
    script: Script,
    text: &str,
    urls_and_hashtags: UrlsAndHashtags,
) -> Result<Vec<SpeechChunk>, String> {
    let filtered = match urls_and_hashtags {
        UrlsAndHashtags::Read => Cow::Borrowed(text),
        UrlsAndHashtags::Skip => Cow::Owned(strip_urls_and_hashtags(text)),
    };
    let text = romanize_foreign_runs_for_voice(&filtered, script);
    plan_speech_chunks(
        &text,
        BoundaryPauses::for_engine(paths.engine),
        |chunk_text| {
            phonemize_chunks(cache, paths, chunk_text)
                .map(|chunks| chunks.into_iter().map(to_phoneme_chunk).collect::<Vec<_>>())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::strip_urls_and_hashtags;

    #[test]
    fn strips_urls_and_hashtags_keeping_surrounding_words() {
        let input = "Check https://example.com/path now #cool and www.foo.org too";
        assert_eq!(strip_urls_and_hashtags(input), "Check now and too");
    }

    #[test]
    fn keeps_plain_text_untouched() {
        assert_eq!(
            strip_urls_and_hashtags("just plain words"),
            "just plain words"
        );
    }

    #[test]
    fn keeps_hash_only_when_not_leading() {
        assert_eq!(strip_urls_and_hashtags("C# is fine"), "C# is fine");
    }
}
