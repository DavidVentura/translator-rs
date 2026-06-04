//! EPUB translation.
//!
//! An EPUB is a ZIP package whose chapters are XHTML documents. This module
//! repackages the archive unchanged except for those documents: each is parsed
//! as XML (xml5ever, so namespaces, self-closing tags and entities round-trip),
//! run through the shared `dom_translate` core — which preserves element
//! structure, attributes and order and only rewrites text leaves — then
//! re-serialised as XML. The `mimetype` entry stays first and stored, as the OCF
//! container spec requires.
//!
//! Translatable entries are selected by extension: XHTML content documents
//! (`.xhtml`/`.html`/`.htm`, which also covers the EPUB3 nav document) and the
//! EPUB2 NCX table of contents (`.ncx`, whose `<text>` nav labels carry the
//! chapter titles) get all their text translated; the OPF package document
//! (`.opf`) gets only its allowlisted Dublin Core metadata — the book
//! `<dc:title>` — translated, leaving the identifier, language code, dates and
//! manifest untouched. Other resources pass through unchanged.

use std::fmt;
use std::io::{Cursor, Read, Write};

use markup5ever_rcdom::{RcDom, SerializableHandle};
use xml5ever::driver::{XmlParseOpts, parse_document};
use xml5ever::serialize::{SerializeOpts, TraversalScope, serialize};
use xml5ever::tendril::TendrilSink;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::TranslationWithAlignment;
use crate::api::{LanguageCode, TranslatorError};
use crate::dom_translate::{Scope, apply_indexed, collect_and_index, collect_named_elements};
use crate::language_detect::detect_language_robust_code;
use crate::session::TranslatorSession;
use crate::translate::identity_char_alignments;

const EPUB_MIMETYPE: &str = "application/epub+zip";
const EPUB_TRANSLATION_BATCH_SIZE: usize = 32;

pub enum EpubTranslateProgress {
    TranslatingBlock { current: usize, total: usize },
}

#[derive(Debug)]
pub enum EpubTranslateError {
    InvalidInput(String),
    Zip(String),
    Io(String),
    Utf8(String),
    Translation(String),
    Cancelled,
}

impl fmt::Display for EpubTranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message)
            | Self::Zip(message)
            | Self::Io(message)
            | Self::Utf8(message)
            | Self::Translation(message) => message.fmt(f),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for EpubTranslateError {}

impl From<zip::result::ZipError> for EpubTranslateError {
    fn from(value: zip::result::ZipError) -> Self {
        Self::Zip(value.to_string())
    }
}

impl From<std::io::Error> for EpubTranslateError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<std::string::FromUtf8Error> for EpubTranslateError {
    fn from(value: std::string::FromUtf8Error) -> Self {
        Self::Utf8(value.to_string())
    }
}

impl From<TranslatorError> for EpubTranslateError {
    fn from(value: TranslatorError) -> Self {
        Self::Translation(value.message)
    }
}

/// Translation abstraction for EPUB rewriting, so tests can inject a
/// deterministic translator while the session-backed implementation uses the
/// same alignment-producing path as the rest of the document pipeline.
pub trait EpubTextTranslator {
    fn translate_texts_with_alignment(
        &mut self,
        texts: &[String],
    ) -> Result<Vec<TranslationWithAlignment>, EpubTranslateError>;
}

pub struct SessionEpubTranslator<'a> {
    session: &'a TranslatorSession,
    forced_source_code: Option<&'a str>,
    target_code: &'a str,
    available_language_codes: &'a [LanguageCode],
}

impl<'a> SessionEpubTranslator<'a> {
    pub fn new(
        session: &'a TranslatorSession,
        forced_source_code: Option<&'a str>,
        target_code: &'a str,
        available_language_codes: &'a [LanguageCode],
    ) -> Self {
        Self {
            session,
            forced_source_code,
            target_code,
            available_language_codes,
        }
    }
}

impl EpubTextTranslator for SessionEpubTranslator<'_> {
    fn translate_texts_with_alignment(
        &mut self,
        texts: &[String],
    ) -> Result<Vec<TranslationWithAlignment>, EpubTranslateError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let target_code = LanguageCode::from(self.target_code);
        let source_code = match self.forced_source_code {
            Some(code) => LanguageCode::from(code),
            None => {
                let combined = texts.join(" ");
                detect_language_robust_code(&combined, None, self.available_language_codes)
                    .ok_or_else(|| {
                        EpubTranslateError::Translation(
                            "could not detect EPUB source language".to_string(),
                        )
                    })?
            }
        };

        if source_code == target_code {
            return Ok(texts
                .iter()
                .map(|text| TranslationWithAlignment {
                    source_text: text.clone(),
                    translated_text: text.clone(),
                    alignments: identity_char_alignments(text),
                })
                .collect());
        }

        let Some(translations) =
            self.session
                .translate_texts_with_alignment(&source_code, &target_code, texts)?
        else {
            return Err(EpubTranslateError::Translation(format!(
                "Language pair {} -> {} not installed",
                source_code.as_str(),
                target_code.as_str()
            )));
        };

        Ok(translations)
    }
}

pub fn translate_epub(
    session: &TranslatorSession,
    epub_bytes: &[u8],
    forced_source_code: Option<&str>,
    target_code: &str,
    available_language_codes: &[LanguageCode],
) -> Result<Vec<u8>, EpubTranslateError> {
    translate_epub_with_progress(
        session,
        epub_bytes,
        forced_source_code,
        target_code,
        available_language_codes,
        |_| Ok(()),
    )
}

pub fn translate_epub_with_progress(
    session: &TranslatorSession,
    epub_bytes: &[u8],
    forced_source_code: Option<&str>,
    target_code: &str,
    available_language_codes: &[LanguageCode],
    on_progress: impl FnMut(EpubTranslateProgress) -> Result<(), EpubTranslateError>,
) -> Result<Vec<u8>, EpubTranslateError> {
    let mut translator = SessionEpubTranslator::new(
        session,
        forced_source_code,
        target_code,
        available_language_codes,
    );
    translate_epub_with_translator_and_progress(epub_bytes, &mut translator, on_progress)
}

pub fn translate_epub_with_translator(
    epub_bytes: &[u8],
    translator: &mut dyn EpubTextTranslator,
) -> Result<Vec<u8>, EpubTranslateError> {
    translate_epub_with_translator_and_progress(epub_bytes, translator, |_| Ok(()))
}

pub fn translate_epub_with_translator_and_progress(
    epub_bytes: &[u8],
    translator: &mut dyn EpubTextTranslator,
    mut on_progress: impl FnMut(EpubTranslateProgress) -> Result<(), EpubTranslateError>,
) -> Result<Vec<u8>, EpubTranslateError> {
    let mut archive = ZipArchive::new(Cursor::new(epub_bytes))?;
    let mut entries = Vec::with_capacity(archive.len());
    let mut mimetype = None;

    for index in 0..archive.len() {
        let mut file = archive.by_index(index)?;
        let name = file.name().to_string();
        let mut data = Vec::new();
        if !file.is_dir() {
            file.read_to_end(&mut data)?;
        }
        if name == "mimetype" {
            mimetype = Some(data.clone());
        }
        entries.push(PackageEntry {
            name,
            data,
            compression: file.compression(),
            modified: file.last_modified(),
            unix_mode: file.unix_mode(),
            is_dir: file.is_dir(),
        });
    }

    match mimetype {
        Some(value) if value == EPUB_MIMETYPE.as_bytes() => {}
        Some(_) => {
            return Err(EpubTranslateError::InvalidInput(
                "EPUB mimetype entry is not application/epub+zip".to_string(),
            ));
        }
        None => {
            return Err(EpubTranslateError::InvalidInput(
                "EPUB package is missing mimetype entry".to_string(),
            ));
        }
    }

    let mut all_texts = Vec::new();
    let mut prepared = Vec::new();
    for (entry_index, entry) in entries.iter().enumerate() {
        if entry.is_dir {
            continue;
        }
        let Some(kind) = entry_kind(&entry.name) else {
            continue;
        };
        let xml = String::from_utf8(entry.data.clone())?;
        let xml_declaration = extract_xml_declaration(&xml);
        let dom = parse_xml_document(&xml);
        let (scopes, translation_idx) = match kind {
            EntryKind::Content => collect_and_index(&dom, &mut all_texts),
            EntryKind::Package => {
                collect_named_elements(&dom, OPF_TRANSLATABLE_METADATA, &mut all_texts)
            }
        };
        prepared.push(PreparedDoc {
            entry_index,
            dom,
            scopes,
            translation_idx,
            xml_declaration,
        });
    }

    let total_blocks = all_texts.len();
    on_progress(EpubTranslateProgress::TranslatingBlock {
        current: 0,
        total: total_blocks,
    })?;

    let mut translations = Vec::with_capacity(total_blocks);
    let mut translated_blocks = 0usize;
    for chunk in all_texts.chunks(EPUB_TRANSLATION_BATCH_SIZE) {
        translations.extend(translator.translate_texts_with_alignment(chunk)?);
        translated_blocks += chunk.len();
        on_progress(EpubTranslateProgress::TranslatingBlock {
            current: translated_blocks,
            total: total_blocks,
        })?;
    }

    for doc in &prepared {
        apply_indexed(&doc.scopes, &doc.translation_idx, &translations);
        let serialized = serialize_xml_document(&doc.dom)?;
        // xml5ever does not re-emit the XML declaration (it isn't a tree node),
        // so restore the original one verbatim when the source had it.
        let output = match &doc.xml_declaration {
            Some(declaration) => format!("{declaration}\n{serialized}"),
            None => serialized,
        };
        entries[doc.entry_index].data = output.into_bytes();
        entries[doc.entry_index].compression = CompressionMethod::Deflated;
    }

    write_epub_package(&entries)
}

struct PreparedDoc {
    entry_index: usize,
    dom: RcDom,
    scopes: Vec<Scope>,
    translation_idx: Vec<Option<usize>>,
    xml_declaration: Option<String>,
}

#[derive(Debug)]
struct PackageEntry {
    name: String,
    data: Vec<u8>,
    compression: CompressionMethod,
    modified: Option<zip::DateTime>,
    unix_mode: Option<u32>,
    is_dir: bool,
}

fn write_epub_package(entries: &[PackageEntry]) -> Result<Vec<u8>, EpubTranslateError> {
    let cursor = Cursor::new(Vec::new());
    let mut writer = ZipWriter::new(cursor);

    let mimetype_options =
        SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    writer.start_file("mimetype", mimetype_options)?;
    writer.write_all(EPUB_MIMETYPE.as_bytes())?;

    for entry in entries.iter().filter(|entry| entry.name != "mimetype") {
        let mut options = SimpleFileOptions::default().compression_method(entry.compression);
        if let Some(modified) = entry.modified {
            options = options.last_modified_time(modified);
        }
        if let Some(mode) = entry.unix_mode {
            options = options.unix_permissions(mode);
        }
        if entry.is_dir {
            writer.add_directory(entry.name.clone(), options)?;
        } else {
            writer.start_file(entry.name.clone(), options)?;
            writer.write_all(&entry.data)?;
        }
    }

    Ok(writer.finish()?.into_inner())
}

/// What an archive entry is, for picking how to translate it.
enum EntryKind {
    /// Prose document — an XHTML chapter, the EPUB3 nav, or the EPUB2 NCX table
    /// of contents — where all text is translated.
    Content,
    /// The OPF package document, where only the allowlisted metadata fields are
    /// translated and the rest (manifest, spine, identifiers) is left alone.
    Package,
}

/// Dublin Core metadata fields safe to translate in the OPF. Only the book
/// title: the identifier, language code, dates and creator/contributor names
/// are codes or proper nouns that the model must not touch.
const OPF_TRANSLATABLE_METADATA: &[&str] = &["title"];

fn entry_kind(name: &str) -> Option<EntryKind> {
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".xhtml")
        || lower.ends_with(".html")
        || lower.ends_with(".htm")
        || lower.ends_with(".ncx")
    {
        Some(EntryKind::Content)
    } else if lower.ends_with(".opf") {
        Some(EntryKind::Package)
    } else {
        None
    }
}

fn extract_xml_declaration(xml: &str) -> Option<String> {
    let trimmed = xml.trim_start();
    if !trimmed.starts_with("<?xml") {
        return None;
    }
    let end = trimmed.find("?>")?;
    Some(trimmed[..end + 2].to_string())
}

fn parse_xml_document(xml: &str) -> RcDom {
    parse_document(RcDom::default(), XmlParseOpts::default()).one(xml)
}

fn serialize_xml_document(dom: &RcDom) -> Result<String, EpubTranslateError> {
    let serializable = SerializableHandle::from(dom.document.clone());
    let mut buf: Vec<u8> = Vec::new();
    serialize(
        &mut buf,
        &serializable,
        SerializeOpts {
            traversal_scope: TraversalScope::ChildrenOnly(None),
        },
    )?;
    Ok(String::from_utf8(buf)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Length-preserving fake: uppercases each scope with an identity alignment,
    /// so the reassembly maps every source char straight back to its leaf.
    struct UppercaseTranslator;

    impl EpubTextTranslator for UppercaseTranslator {
        fn translate_texts_with_alignment(
            &mut self,
            texts: &[String],
        ) -> Result<Vec<TranslationWithAlignment>, EpubTranslateError> {
            Ok(texts
                .iter()
                .map(|text| TranslationWithAlignment {
                    source_text: text.clone(),
                    translated_text: text.to_uppercase(),
                    alignments: identity_char_alignments(text),
                })
                .collect())
        }
    }

    fn build_epub(chapter: &str) -> Vec<u8> {
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        writer
            .start_file(
                "mimetype",
                SimpleFileOptions::default().compression_method(CompressionMethod::Stored),
            )
            .unwrap();
        writer.write_all(EPUB_MIMETYPE.as_bytes()).unwrap();
        let deflated = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        let ncx = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n\
            <ncx xmlns=\"http://www.daisy.org/z3986/2005/ncx/\" version=\"2005-1\">\
            <docTitle><text>my book</text></docTitle>\
            <navMap><navPoint id=\"n1\" playOrder=\"1\"><navLabel><text>chapter one</text></navLabel>\
            <content src=\"chapter1.xhtml\"/></navPoint></navMap></ncx>";
        let opf = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n\
            <package xmlns=\"http://www.idpf.org/2007/opf\" version=\"3.0\" unique-identifier=\"bookid\">\
            <metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\
            <dc:title>my book</dc:title><dc:language>en</dc:language>\
            <dc:identifier id=\"bookid\">urn:uuid:12345</dc:identifier></metadata>\
            <manifest><item id=\"c1\" href=\"chapter1.xhtml\" media-type=\"application/xhtml+xml\"/></manifest>\
            <spine><itemref idref=\"c1\"/></spine></package>";
        for (name, body) in [
            ("META-INF/container.xml", "<container/>"),
            ("OEBPS/content.opf", opf),
            ("OEBPS/chapter1.xhtml", chapter),
            ("OEBPS/toc.ncx", ncx),
        ] {
            writer.start_file(name, deflated).unwrap();
            writer.write_all(body.as_bytes()).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    fn chapter_text(epub: &[u8], name: &str) -> String {
        let mut archive = ZipArchive::new(Cursor::new(epub.to_vec())).unwrap();
        let mut out = String::new();
        archive
            .by_name(name)
            .unwrap()
            .read_to_string(&mut out)
            .unwrap();
        out
    }

    #[test]
    fn translates_text_preserves_structure_and_repackages_mimetype_first() {
        let chapter = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n\
            <html xmlns=\"http://www.w3.org/1999/xhtml\"><head><title>chapter one</title></head>\
            <body><p>hello <em>brave</em> world</p></body></html>";
        let translated =
            translate_epub_with_translator(&build_epub(chapter), &mut UppercaseTranslator)
                .expect("translate epub");

        let mut archive = ZipArchive::new(Cursor::new(translated.clone())).unwrap();
        let first = archive.by_index(0).unwrap();
        assert_eq!(first.name(), "mimetype");
        assert_eq!(first.compression(), CompressionMethod::Stored);
        drop(first);

        let out = chapter_text(&translated, "OEBPS/chapter1.xhtml");
        assert!(out.starts_with("<?xml"), "xml declaration preserved: {out}");
        assert!(out.contains("HELLO"), "block text translated: {out}");
        assert!(out.contains("BRAVE"), "inline text translated: {out}");
        assert!(out.contains("WORLD"));
        assert!(out.contains("CHAPTER ONE"), "head title translated: {out}");
        assert!(out.contains("<em"), "inline element preserved: {out}");
        assert!(out.contains("<p"), "block element preserved: {out}");

        let ncx = chapter_text(&translated, "OEBPS/toc.ncx");
        assert!(ncx.contains("MY BOOK"), "ncx doc title translated: {ncx}");
        assert!(
            ncx.contains("CHAPTER ONE"),
            "ncx nav label translated: {ncx}"
        );
        assert!(ncx.contains("navPoint"), "ncx structure preserved: {ncx}");
        assert!(
            ncx.contains("chapter1.xhtml"),
            "ncx content src preserved: {ncx}"
        );

        let opf = chapter_text(&translated, "OEBPS/content.opf");
        assert!(opf.contains("MY BOOK"), "opf book title translated: {opf}");
        // Identifier and language are not in the allowlist: the uppercase fake
        // would have mangled them to URN:UUID / EN had they been translated.
        assert!(
            opf.contains("urn:uuid:12345"),
            "opf identifier left untouched: {opf}"
        );
        assert!(
            opf.contains(">en<"),
            "opf language code left untouched: {opf}"
        );
        assert!(
            opf.contains("chapter1.xhtml"),
            "opf manifest href preserved: {opf}"
        );
    }

    #[test]
    fn rejects_wrong_mimetype() {
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        writer
            .start_file(
                "mimetype",
                SimpleFileOptions::default().compression_method(CompressionMethod::Stored),
            )
            .unwrap();
        writer.write_all(b"application/zip").unwrap();
        let bytes = writer.finish().unwrap().into_inner();

        let result = translate_epub_with_translator(&bytes, &mut UppercaseTranslator);
        assert!(matches!(result, Err(EpubTranslateError::InvalidInput(_))));
    }
}
