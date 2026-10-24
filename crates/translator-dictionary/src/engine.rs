use translator_core::api::{LanguageCode, TranslatorError};
use translator_core::catalog::CatalogSnapshot;

use crate::tarkka::{
    DictionaryCache, WordWithTaggedEntries, close_dictionary_in_snapshot,
    lookup_dictionary_in_snapshot,
};

pub struct DictionaryEngine {
    cache: DictionaryCache,
}

impl Default for DictionaryEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DictionaryEngine {
    pub fn new() -> Self {
        Self {
            cache: DictionaryCache::new(),
        }
    }

    pub fn lookup(
        &mut self,
        snap: &CatalogSnapshot,
        language: &LanguageCode,
        word: &str,
    ) -> Result<Option<WordWithTaggedEntries>, TranslatorError> {
        lookup_dictionary_in_snapshot(snap, &mut self.cache, language, word)
    }

    pub fn close(&mut self, snap: &CatalogSnapshot, language: &LanguageCode) {
        close_dictionary_in_snapshot(snap, &mut self.cache, language);
    }
}
