use translator_core::api::{LanguageCode, TranslatorError, VoiceName};
use translator_core::catalog::CatalogSnapshot;
use translator_core::tts::{PcmAudio, SpeechChunk, TtsVoiceOption, UrlsAndHashtags};

use crate::speech::{
    SpeechCache, available_tts_voices_in_snapshot, plan_speech_chunks_for_text_in_snapshot,
    synthesize_pcm_in_snapshot, warm_tts_model_in_snapshot,
};

pub struct TtsEngine {
    cache: SpeechCache,
}

impl Default for TtsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TtsEngine {
    pub fn new() -> Self {
        Self {
            cache: SpeechCache::new(),
        }
    }

    pub fn available_voices(
        &mut self,
        snap: &CatalogSnapshot,
        language: &LanguageCode,
    ) -> Result<Vec<TtsVoiceOption>, TranslatorError> {
        available_tts_voices_in_snapshot(snap, &mut self.cache, language)
    }

    pub fn warm_model(
        &mut self,
        snap: &CatalogSnapshot,
        language: &LanguageCode,
    ) -> Result<(), TranslatorError> {
        warm_tts_model_in_snapshot(snap, &mut self.cache, language)
    }

    pub fn plan_speech_chunks(
        &mut self,
        snap: &CatalogSnapshot,
        language: &LanguageCode,
        text: &str,
        pack_id: Option<&str>,
        urls_and_hashtags: UrlsAndHashtags,
    ) -> Result<Vec<SpeechChunk>, TranslatorError> {
        plan_speech_chunks_for_text_in_snapshot(
            snap,
            &mut self.cache,
            language,
            text,
            pack_id,
            urls_and_hashtags,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_pcm(
        &mut self,
        snap: &CatalogSnapshot,
        language: &LanguageCode,
        text: &str,
        speech_speed: f32,
        voice_name: Option<&VoiceName>,
        is_phonemes: bool,
        pack_id: Option<&str>,
    ) -> Result<PcmAudio, TranslatorError> {
        synthesize_pcm_in_snapshot(
            snap,
            &mut self.cache,
            language,
            text,
            speech_speed,
            voice_name,
            is_phonemes,
            pack_id,
        )
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
}
