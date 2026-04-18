use std::path::Path;

use crate::catalog::CatalogSnapshot;
use crate::ocr::{
    DetectedWord, PreparedImageOverlay, ReadingOrder, Rect, TextBlock, build_text_blocks,
    prepare_overlay_image,
};
use crate::settings::BackgroundMode;
use crate::tesseract::DetectedWord as TesseractDetectedWord;
use crate::translate::Translator;
use crate::api::{LanguageCode, TranslatorError};
use crate::bergamot::BergamotEngine;
use crate::tesseract::{PageSegMode, TesseractWrapper};

struct OcrEngineState {
    engine: TesseractWrapper,
    language_spec: String,
    reading_order: ReadingOrder,
    tessdata_path: String,
}

pub struct OcrCache {
    state: Option<OcrEngineState>,
}

impl OcrCache {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl Default for OcrCache {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn translate_image_rgba_in_snapshot(
    engine: &mut BergamotEngine,
    ocr_cache: &mut OcrCache,
    snapshot: &CatalogSnapshot,
    rgba_bytes: &[u8],
    width: u32,
    height: u32,
    source_code: &LanguageCode,
    target_code: &LanguageCode,
    min_confidence: u32,
    reading_order: ReadingOrder,
    background_mode: BackgroundMode,
) -> Result<PreparedImageOverlay, TranslatorError> {
    let bytes_per_pixel = 4i32;
    let i_width = width as i32;
    let i_height = height as i32;
    let bytes_per_line = i_width
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| TranslatorError::ocr("image width overflow"))?;

    let page_seg_mode = match reading_order {
        ReadingOrder::LeftToRight => PageSegMode::PsmAutoOsd,
        ReadingOrder::TopToBottomLeftToRight => PageSegMode::PsmSingleBlockVertText,
    };

    let join_without_spaces = source_code.as_str() == "ja";
    let relax_single_char_confidence = reading_order == ReadingOrder::TopToBottomLeftToRight;

    let blocks = with_ocr_engine(
        ocr_cache,
        snapshot,
        source_code.as_str(),
        reading_order,
        |ocr| {
            ocr.set_page_seg_mode(page_seg_mode);
            ocr.set_frame(
                rgba_bytes,
                i_width,
                i_height,
                bytes_per_pixel,
                bytes_per_line,
            )
            .map_err(|err| format!("failed to set OCR frame: {err}"))?;
            let words = ocr
                .get_word_boxes()
                .map_err(|err| format!("failed to read OCR words: {err}"))?;
            let detected_words = words
                .into_iter()
                .map(map_tesseract_word)
                .collect::<Vec<_>>();
            Ok(build_text_blocks(
                &detected_words,
                min_confidence,
                join_without_spaces,
                relax_single_char_confidence,
            ))
        },
    )
    .map_err(TranslatorError::ocr)?;

    let translated_blocks =
        translate_block_texts(engine, snapshot, source_code, target_code, &blocks)?;

    prepare_overlay_image(
        rgba_bytes,
        width,
        height,
        &blocks,
        &translated_blocks,
        background_mode,
        reading_order,
    )
    .map_err(TranslatorError::ocr)
}

fn map_tesseract_word(word: TesseractDetectedWord) -> DetectedWord {
    DetectedWord {
        text: word.text,
        confidence: word.confidence,
        bounding_box: Rect {
            left: word.bounding_rect.left as u32,
            top: word.bounding_rect.top as u32,
            right: word.bounding_rect.right as u32,
            bottom: word.bounding_rect.bottom as u32,
        },
        is_at_beginning_of_para: word.is_at_beginning_of_para,
        end_para: word.end_para,
        end_line: word.end_line,
    }
}

fn translate_block_texts(
    engine: &mut BergamotEngine,
    snapshot: &CatalogSnapshot,
    source_code: &LanguageCode,
    target_code: &LanguageCode,
    blocks: &[TextBlock],
) -> Result<Vec<String>, TranslatorError> {
    let block_texts = blocks
        .iter()
        .map(TextBlock::translation_text)
        .collect::<Vec<_>>();
    let non_empty_indices = block_texts
        .iter()
        .enumerate()
        .filter_map(|(index, text)| (!text.trim().is_empty()).then_some(index))
        .collect::<Vec<_>>();

    if non_empty_indices.is_empty() {
        return Err(TranslatorError::ocr("No text found in image"));
    }

    if source_code == target_code {
        return Ok(block_texts);
    }

    let texts_to_translate = non_empty_indices
        .iter()
        .map(|&index| block_texts[index].clone())
        .collect::<Vec<_>>();
    let translated = Translator::new(engine, snapshot).translate_texts(
        source_code,
        target_code,
        &texts_to_translate,
    )?;

    Ok(merge_translated_block_texts(
        &block_texts,
        &non_empty_indices,
        translated,
    ))
}

fn merge_translated_block_texts(
    block_texts: &[String],
    non_empty_indices: &[usize],
    translated_non_empty: Vec<String>,
) -> Vec<String> {
    let mut translated_blocks = block_texts.to_vec();
    for (index, translated_text) in non_empty_indices
        .iter()
        .copied()
        .zip(translated_non_empty.into_iter())
    {
        translated_blocks[index] = translated_text;
    }
    translated_blocks
}

fn with_ocr_engine<T, F>(
    cache: &mut OcrCache,
    snapshot: &CatalogSnapshot,
    source_code: &str,
    reading_order: ReadingOrder,
    f: F,
) -> Result<T, String>
where
    F: FnOnce(&mut TesseractWrapper) -> Result<T, String>,
{
    let language = snapshot
        .catalog
        .language_by_code(&LanguageCode::from(source_code))
        .ok_or_else(|| format!("unknown source language: {source_code}"))?;
    let tessdata_path = Path::new(&snapshot.base_dir)
        .join("tesseract")
        .join("tessdata");
    let has_japanese_vertical_model =
        source_code == "ja" && tessdata_path.join("jpn_vert.traineddata").exists();
    let language_spec = match (source_code, reading_order, has_japanese_vertical_model) {
        ("ja", ReadingOrder::TopToBottomLeftToRight, true) => "jpn_vert".to_string(),
        _ => format!("{}+eng", language.tess_name),
    };

    let tessdata_path_string = tessdata_path.to_string_lossy().into_owned();
    let needs_reinit = cache.state.as_ref().is_none_or(|state| {
        state.language_spec != language_spec
            || state.reading_order != reading_order
            || state.tessdata_path != tessdata_path_string
    });

    if needs_reinit {
        let engine = TesseractWrapper::new(
            Some(
                tessdata_path
                    .to_str()
                    .ok_or_else(|| "invalid tessdata path".to_string())?,
            ),
            Some(&language_spec),
        )
        .map_err(|err| format!("failed to initialize tesseract: {err}"))?;
        cache.state = Some(OcrEngineState {
            engine,
            language_spec,
            reading_order,
            tessdata_path: tessdata_path_string,
        });
    }

    let state = cache
        .state
        .as_mut()
        .ok_or_else(|| "OCR engine unavailable".to_string())?;
    f(&mut state.engine)
}

#[cfg(test)]
mod tests {
    use super::merge_translated_block_texts;

    #[test]
    fn preserves_blank_blocks_when_merging_translations() {
        let block_texts = vec![
            "hello".to_string(),
            String::new(),
            "world".to_string(),
            "   ".to_string(),
        ];

        let merged = merge_translated_block_texts(
            &block_texts,
            &[0, 2],
            vec!["hola".to_string(), "mundo".to_string()],
        );

        assert_eq!(
            merged,
            vec![
                "hola".to_string(),
                String::new(),
                "mundo".to_string(),
                "   ".to_string(),
            ]
        );
    }
}
