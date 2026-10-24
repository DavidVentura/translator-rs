use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use translator_core::api::{DictionaryCode, LanguageCode, TranslatorError};
use translator_core::catalog::CatalogSnapshot;

pub use tarkka::WordWithTaggedEntries;
use tarkka::reader::DictionaryReader;

pub struct DictionaryCache {
    readers: HashMap<String, DictionaryReader<'static, File>>,
}

impl DictionaryCache {
    pub fn new() -> Self {
        Self {
            readers: HashMap::new(),
        }
    }

    pub fn close(&mut self, path: &str) {
        self.readers.remove(path);
    }

    fn lookup(&mut self, path: &str, word: &str) -> Result<Option<WordWithTaggedEntries>, String> {
        if !self.readers.contains_key(path) {
            let file = File::open(path)
                .map_err(|err| format!("failed to open dictionary file {path}: {err}"))?;
            let reader = DictionaryReader::open(file)
                .map_err(|err| format!("failed to open dictionary reader {path}: {err}"))?;
            eprintln!(
                "Dict version {}, timestamp {:?}",
                reader.version(),
                reader.created_at(),
            );
            self.readers.insert(path.to_string(), reader);
        }
        let reader = self
            .readers
            .get_mut(path)
            .ok_or_else(|| "dictionary reader missing after initialization".to_string())?;
        reader
            .lookup(word)
            .map_err(|err| format!("dictionary lookup failed: {err}"))
    }
}

impl Default for DictionaryCache {
    fn default() -> Self {
        Self::new()
    }
}

fn dictionary_path(base_dir: &str, dictionary_code: &str) -> Option<String> {
    let path = Path::new(base_dir)
        .join("dictionaries")
        .join(format!("{dictionary_code}.dict"));
    path.exists().then(|| path.to_string_lossy().into_owned())
}

pub fn lookup_dictionary_for_code(
    base_dir: &str,
    cache: &mut DictionaryCache,
    dictionary_code: &DictionaryCode,
    word: &str,
) -> Result<Option<WordWithTaggedEntries>, TranslatorError> {
    let keys = lookup_keys(word);
    if keys.is_empty() {
        return Ok(None);
    }

    let path = dictionary_path(base_dir, dictionary_code.as_str()).ok_or_else(|| {
        TranslatorError::missing_asset(format!(
            "dictionary not installed: {}",
            dictionary_code.as_str()
        ))
    })?;

    for key in &keys {
        if let Some(found) = cache
            .lookup(&path, key)
            .map_err(TranslatorError::dictionary)?
        {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

// Words picked off an image keep the punctuation OCR read around them (`"Stel`, `in."`), but
// entries like `etc.` keep theirs too, so the form as given is tried before the stripped one.
fn lookup_keys(word: &str) -> Vec<String> {
    let trimmed = word.trim();
    let stripped = trimmed.trim_matches(|c: char| !c.is_alphanumeric());
    let mut keys: Vec<String> = Vec::new();
    for form in [trimmed, stripped] {
        for key in [form.to_string(), form.to_lowercase()] {
            if !key.is_empty() && !keys.contains(&key) {
                keys.push(key);
            }
        }
    }
    keys
}

fn dictionary_path_for_language(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> Option<String> {
    let language = snapshot.catalog.language_by_code(language_code)?;
    dictionary_path(&snapshot.base_dir, &language.dictionary_code)
}

pub fn lookup_dictionary_in_snapshot(
    snapshot: &CatalogSnapshot,
    cache: &mut DictionaryCache,
    language_code: &LanguageCode,
    word: &str,
) -> Result<Option<WordWithTaggedEntries>, TranslatorError> {
    let language = snapshot
        .catalog
        .language_by_code(language_code)
        .ok_or_else(|| {
            TranslatorError::dictionary(format!(
                "unknown dictionary language: {}",
                language_code.as_str()
            ))
        })?;
    lookup_dictionary_for_code(
        &snapshot.base_dir,
        cache,
        &DictionaryCode::from(language.dictionary_code.clone()),
        word,
    )
}

pub fn close_dictionary_in_snapshot(
    snapshot: &CatalogSnapshot,
    cache: &mut DictionaryCache,
    language_code: &LanguageCode,
) {
    if let Some(path) = dictionary_path_for_language(snapshot, language_code) {
        cache.close(&path);
    }
}

#[cfg(test)]
mod tests {
    use super::lookup_keys;

    #[test]
    fn plain_word_is_looked_up_as_given_then_lowercased() {
        assert_eq!(lookup_keys(" Huis "), vec!["Huis", "huis"]);
        assert_eq!(lookup_keys("huis"), vec!["huis"]);
    }

    #[test]
    fn surrounding_punctuation_is_stripped_after_the_exact_form() {
        assert_eq!(lookup_keys("in.\""), vec!["in.\"", "in"]);
        assert_eq!(
            lookup_keys("\"Stel"),
            vec!["\"Stel", "\"stel", "Stel", "stel"]
        );
        assert_eq!(lookup_keys("etc."), vec!["etc.", "etc"]);
    }

    #[test]
    fn inner_punctuation_is_kept() {
        assert_eq!(lookup_keys("l'eau,"), vec!["l'eau,", "l'eau"]);
    }

    #[test]
    fn punctuation_only_or_blank_has_nothing_to_look_up() {
        assert_eq!(lookup_keys("  "), Vec::<String>::new());
        assert_eq!(lookup_keys("—"), vec!["—"]);
    }
}
