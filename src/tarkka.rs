use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use crate::CatalogSnapshot;
use crate::api::{DictionaryCode, DictionaryLookupOutcome, LanguageCode, TranslatorError};

pub use tarkka::WordWithTaggedEntries;
use tarkka::reader::DictionaryReader;

static DICTIONARY_READERS: OnceLock<Mutex<HashMap<String, DictionaryReader<File>>>> =
    OnceLock::new();

fn with_reader_cache<T, F>(f: F) -> Result<T, String>
where
    F: FnOnce(&mut HashMap<String, DictionaryReader<File>>) -> Result<T, String>,
{
    let mut readers = DICTIONARY_READERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| "dictionary cache mutex poisoned".to_string())?;
    f(&mut readers)
}

fn lookup_dictionary(path: &str, word: &str) -> Result<Option<WordWithTaggedEntries>, String> {
    with_reader_cache(|readers| {
        if !readers.contains_key(path) {
            let file = File::open(path)
                .map_err(|err| format!("failed to open dictionary file {path}: {err}"))?;
            let reader = DictionaryReader::open(file)
                .map_err(|err| format!("failed to open dictionary reader {path}: {err}"))?;
            eprintln!(
                "Dict version {}, timestamp {:?}",
                reader.version(),
                reader.created_at(),
            );
            readers.insert(path.to_string(), reader);
        }
        let reader = readers
            .get_mut(path)
            .ok_or_else(|| "dictionary reader missing after initialization".to_string())?;
        reader
            .lookup(word)
            .map_err(|err| format!("dictionary lookup failed: {err}"))
    })
}

fn close_dictionary(path: &str) -> Result<(), String> {
    with_reader_cache(|readers| {
        readers.remove(path);
        Ok(())
    })
}

fn dictionary_path(base_dir: &str, dictionary_code: &str) -> Option<String> {
    let path = Path::new(base_dir)
        .join("dictionaries")
        .join(format!("{dictionary_code}.dict"));
    path.exists().then(|| path.to_string_lossy().into_owned())
}

pub fn lookup_dictionary_for_code(
    base_dir: &str,
    dictionary_code: &DictionaryCode,
    word: &str,
) -> Result<DictionaryLookupOutcome, TranslatorError> {
    let normalized = word.trim();
    if normalized.is_empty() {
        return Ok(DictionaryLookupOutcome::NotFound);
    }

    let Some(path) = dictionary_path(base_dir, dictionary_code.as_str()) else {
        return Ok(DictionaryLookupOutcome::MissingDictionary);
    };

    let lowered = normalized.to_lowercase();
    match lookup_dictionary(&path, normalized) {
        Ok(Some(word_data)) => Ok(DictionaryLookupOutcome::Found(word_data)),
        Ok(None) if lowered != normalized => lookup_dictionary(&path, &lowered)
            .map(|result| {
                result
                    .map(DictionaryLookupOutcome::Found)
                    .unwrap_or(DictionaryLookupOutcome::NotFound)
            })
            .map_err(TranslatorError::dictionary),
        Ok(None) => Ok(DictionaryLookupOutcome::NotFound),
        Err(err) => Err(TranslatorError::dictionary(err)),
    }
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
    language_code: &LanguageCode,
    word: &str,
) -> Result<DictionaryLookupOutcome, TranslatorError> {
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
        &DictionaryCode::from(language.dictionary_code.clone()),
        word,
    )
}

pub fn close_dictionary_in_snapshot(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> Result<(), TranslatorError> {
    let Some(path) = dictionary_path_for_language(snapshot, language_code) else {
        return Ok(());
    };
    close_dictionary(&path).map_err(TranslatorError::dictionary)
}
