//! Translate raster image XObjects embedded in a PDF.
//!
//! Walks every page's `/Resources /XObject` dict, finds streams whose
//! `/Subtype` is `/Image`, decodes the supported filter combinations
//! (DCTDecode → JPEG; FlateDecode → raw RGB or grayscale), runs the
//! existing OCR + image renderer on the pixels, and writes the rendered
//! result back as a `FlateDecode`-compressed `DeviceRGB` stream — keeping
//! the same object id so existing `cm` / `Do` references in page content
//! streams resolve unchanged.
//!
//! Unsupported filter combinations (JPX, JBIG2, CCITTFax, indexed
//! colorspaces) are skipped: the original image survives untranslated.
//! Likewise images smaller than [`MIN_IMAGE_DIMENSION`] are skipped — they
//! are almost certainly icons or decorative bullets, not text.

use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use log::{debug, info, warn};
use lopdf::{Dictionary, Document, Object, ObjectId, Stream};

use crate::api::{LanguageCode, TranslatorError};
use crate::font_provider::FontProvider;
use crate::image_render::{RenderOptions, render_overlay};
use crate::ocr::ReadingOrder;
use crate::pdf_text::extract_text;
use crate::session::TranslatorSession;
use crate::settings::BackgroundMode;

/// Minimum total pixel area for an image to be considered worth OCR'ing.
/// Roughly equivalent to "bigger than a 50×50 icon". Banner-shaped images
/// (e.g. 313×43 = 13,459 px²) clear this floor; small navigation icons
/// (32×32 = 1,024 px²) don't.
const MIN_IMAGE_AREA_PX: u64 = 2_500;

/// Default OCR confidence floor — matches what the Android caller passes
/// in normal image translation flow.
const DEFAULT_MIN_CONFIDENCE: u32 = 30;

#[derive(Debug)]
pub enum ImageTranslateError {
    Pdf(lopdf::Error),
}

impl std::fmt::Display for ImageTranslateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pdf(e) => write!(f, "pdf: {e}"),
        }
    }
}

impl std::error::Error for ImageTranslateError {}

impl From<lopdf::Error> for ImageTranslateError {
    fn from(value: lopdf::Error) -> Self {
        Self::Pdf(value)
    }
}

/// Run image-XObject translation over a PDF. Returns modified PDF bytes
/// (or the original bytes unchanged if no eligible image was found).
///
/// `source_code` is the BCP-47 of the language present *in the images*. If
/// the caller doesn't have one, pass the page-text source language — it's
/// the same document, the OCR'd images are usually in the same script.
///
/// Per-image errors (decode failure, no text detected, OCR cancel) are
/// swallowed; the image is left untranslated and the function moves on.
pub fn translate_pdf_images_in_place(
    pdf_bytes: &[u8],
    session: &TranslatorSession,
    source_code: &str,
    target_code: &str,
    fonts: &dyn FontProvider,
    is_cancelled: impl Fn() -> bool,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<Vec<u8>, ImageTranslateError> {
    let mut doc = Document::load_mem(pdf_bytes)?;
    let image_ids = collect_image_ids(&doc);
    info!(
        "[pdf_image_translate] found {} candidate image XObject(s)",
        image_ids.len()
    );
    if image_ids.is_empty() {
        return Ok(pdf_bytes.to_vec());
    }

    let total = image_ids.len();
    on_progress(0, total);
    let mut any_translated = false;
    for (idx, image_id) in image_ids.into_iter().enumerate() {
        if is_cancelled() {
            break;
        }
        match try_translate_image(&mut doc, image_id, session, source_code, target_code, fonts) {
            Ok(true) => {
                info!("[pdf_image_translate] {image_id:?} translated");
                any_translated = true;
            }
            Ok(false) => {}
            Err(reason) => {
                debug!("[pdf_image_translate] {image_id:?} skipped: {reason}");
            }
        }
        on_progress(idx + 1, total);
    }

    if !any_translated {
        info!("[pdf_image_translate] no images were modified");
        return Ok(pdf_bytes.to_vec());
    }

    let mut out = Vec::new();
    doc.save_to(&mut out).map_err(lopdf::Error::IO)?;
    Ok(out)
}

/// Walk every page resource dict and collect the ObjectIds of XObjects
/// whose Subtype is /Image. Deduplicated — the same image ref shared
/// across pages is translated once.
fn collect_image_ids(doc: &Document) -> Vec<ObjectId> {
    let mut seen: HashSet<ObjectId> = HashSet::new();
    for (_page_num, page_id) in doc.get_pages() {
        let Ok((inline_dict, inherited_ids)) = doc.get_page_resources(page_id) else {
            continue;
        };
        // get_page_resources returns one optional inline /Resources dict
        // plus any /Resources references inherited via the page tree;
        // both can carry /XObject entries.
        let mut resource_dicts: Vec<&lopdf::Dictionary> = Vec::new();
        if let Some(d) = inline_dict {
            resource_dicts.push(d);
        }
        for id in &inherited_ids {
            if let Ok(d) = doc.get_object(*id).and_then(|o| o.as_dict()) {
                resource_dicts.push(d);
            }
        }
        for res_dict in resource_dicts {
            let Ok(xobjects) = res_dict.get(b"XObject") else {
                continue;
            };
            let xobj_dict = match xobjects {
                Object::Dictionary(d) => Some(d),
                Object::Reference(id) => doc.get_object(*id).ok().and_then(|o| o.as_dict().ok()),
                _ => None,
            };
            let Some(xobj_dict) = xobj_dict else {
                continue;
            };
            for (_name, value) in xobj_dict.iter() {
                let Ok(id) = value.as_reference() else {
                    continue;
                };
                if seen.contains(&id) {
                    continue;
                }
                let Ok(obj) = doc.get_object(id) else {
                    continue;
                };
                let Ok(stream) = obj.as_stream() else {
                    continue;
                };
                if !is_image_stream(stream) {
                    continue;
                }
                seen.insert(id);
            }
        }
    }
    seen.into_iter().collect()
}

fn is_image_stream(stream: &Stream) -> bool {
    let Ok(subtype) = stream.dict.get(b"Subtype") else {
        return false;
    };
    matches!(subtype.as_name().ok(), Some(name) if name == b"Image")
}

#[derive(Debug)]
enum SkipReason {
    TooSmall,
    UnsupportedFilter(String),
    UnsupportedColorSpace(String),
    MissingDims,
    Decode(String),
    Ocr(String),
    Render(String),
    NoTextDetected,
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooSmall => write!(f, "too small"),
            Self::UnsupportedFilter(s) => write!(f, "unsupported filter: {s}"),
            Self::UnsupportedColorSpace(s) => write!(f, "unsupported colorspace: {s}"),
            Self::MissingDims => write!(f, "missing /Width or /Height"),
            Self::Decode(s) => write!(f, "decode failed: {s}"),
            Self::Ocr(s) => write!(f, "ocr/translate failed: {s}"),
            Self::Render(s) => write!(f, "render failed: {s}"),
            Self::NoTextDetected => write!(f, "no text detected"),
        }
    }
}

/// Translate a single image XObject in place. Returns `Ok(true)` when the
/// stream was actually rewritten, `Ok(false)` when no rewrite was needed
/// (no detected text, etc.).
fn try_translate_image(
    doc: &mut Document,
    image_id: ObjectId,
    session: &TranslatorSession,
    source_code: &str,
    target_code: &str,
    fonts: &dyn FontProvider,
) -> Result<bool, SkipReason> {
    let (width, height, rgba) = {
        let stream = doc
            .get_object(image_id)
            .map_err(|e| SkipReason::Decode(e.to_string()))?
            .as_stream()
            .map_err(|e| SkipReason::Decode(e.to_string()))?;
        decode_image_to_rgba(stream, doc)?
    };

    let area = (width as u64) * (height as u64);
    if area < MIN_IMAGE_AREA_PX {
        return Err(SkipReason::TooSmall);
    }

    let prepared = session
        .translate_image_rgba(
            &rgba,
            width,
            height,
            source_code,
            target_code,
            DEFAULT_MIN_CONFIDENCE,
            ReadingOrder::LeftToRight,
            BackgroundMode::AutoDetect,
        )
        .map_err(|e: TranslatorError| {
            // The OCR layer returns "No text found in image" as a normal
            // result for plain photos; treat that as a skip rather than
            // an error.
            if e.message.to_lowercase().contains("no text") {
                SkipReason::NoTextDetected
            } else {
                SkipReason::Ocr(e.message)
            }
        })?;

    if prepared.blocks.is_empty() {
        return Err(SkipReason::NoTextDetected);
    }

    let opts = RenderOptions {
        language: target_code.to_string(),
        ..RenderOptions::default()
    };
    let rendered =
        render_overlay(&prepared, fonts, &opts).map_err(|e| SkipReason::Render(e.to_string()))?;

    // The render output is BGRA byte order (u32 ARGB on little-endian).
    // PDF DeviceRGB streams want plain interleaved RGB. Drop alpha and
    // swap R/B at the same time.
    let rgb = bgra_to_rgb(&rendered);
    let compressed = flate_compress(&rgb);

    let _ = LanguageCode::from(source_code);

    let stream = doc
        .get_object_mut(image_id)
        .map_err(|e| SkipReason::Decode(e.to_string()))?
        .as_stream_mut()
        .map_err(|e| SkipReason::Decode(e.to_string()))?;
    stream.set_content(compressed);
    let dict = &mut stream.dict;
    dict.set("Width", width as i64);
    dict.set("Height", height as i64);
    dict.set("BitsPerComponent", 8i64);
    dict.set("ColorSpace", Object::Name(b"DeviceRGB".to_vec()));
    dict.set("Filter", Object::Name(b"FlateDecode".to_vec()));
    dict.remove(b"DecodeParms");
    dict.remove(b"SMask");
    dict.remove(b"Mask");
    Ok(true)
}

fn decode_image_to_rgba(
    stream: &Stream,
    doc: &Document,
) -> Result<(u32, u32, Vec<u8>), SkipReason> {
    let dict = &stream.dict;
    let width = dict
        .get(b"Width")
        .ok()
        .and_then(|o| o.as_i64().ok())
        .ok_or(SkipReason::MissingDims)? as u32;
    let height = dict
        .get(b"Height")
        .ok()
        .and_then(|o| o.as_i64().ok())
        .ok_or(SkipReason::MissingDims)? as u32;
    if width == 0 || height == 0 {
        return Err(SkipReason::MissingDims);
    }
    let bits_per_component = dict
        .get(b"BitsPerComponent")
        .ok()
        .and_then(|o| o.as_i64().ok())
        .unwrap_or(8) as u8;
    let filter_chain = read_filter_chain(stream);
    let colorspace_kind = read_colorspace(stream, doc);
    debug!(
        "[pdf_image_translate]   {}x{} bpc={} filter={:?} colorspace={:?}",
        width, height, bits_per_component, filter_chain, colorspace_kind
    );

    // DCTDecode = JPEG. lopdf doesn't decode this; feed the raw stream
    // straight into the image crate. JPEG carries its own colorspace; we
    // ignore the PDF /ColorSpace because JPEG decode produces RGB
    // directly.
    if filter_chain.iter().any(|f| f == "DCTDecode") {
        if filter_chain.iter().any(|f| f != "DCTDecode") {
            return Err(SkipReason::UnsupportedFilter(format!("{filter_chain:?}")));
        }
        let img = image::load_from_memory_with_format(&stream.content, image::ImageFormat::Jpeg)
            .map_err(|e| SkipReason::Decode(e.to_string()))?;
        return Ok((img.width(), img.height(), img.to_rgba8().into_raw()));
    }

    // For everything else lopdf can do flate/lzw/ascii85; let it handle
    // the unwrap and we'll deal with the raw pixel bytes.
    let raw = stream
        .decompressed_content()
        .map_err(|e| SkipReason::Decode(e.to_string()))?;

    // Edge case: some PDF producers (non-spec) embed entire PNG/JPEG
    // files inside a FlateDecode stream. If the decompressed bytes carry
    // a PNG or JPEG signature, decode as that file format and we're done.
    if has_png_signature(&raw) || has_jpeg_signature(&raw) {
        let img = image::load_from_memory(&raw).map_err(|e| SkipReason::Decode(e.to_string()))?;
        return Ok((img.width(), img.height(), img.to_rgba8().into_raw()));
    }

    if bits_per_component != 8 {
        return Err(SkipReason::Decode(format!(
            "unsupported BitsPerComponent: {bits_per_component}"
        )));
    }

    match colorspace_kind {
        ColorSpaceKind::Rgb => {
            let needed = (width as usize) * (height as usize) * 3;
            if raw.len() < needed {
                return Err(SkipReason::Decode(format!(
                    "rgb payload {} bytes, need {needed}",
                    raw.len()
                )));
            }
            Ok((width, height, rgb_to_rgba(&raw[..needed])))
        }
        ColorSpaceKind::Gray => {
            let needed = (width as usize) * (height as usize);
            if raw.len() < needed {
                return Err(SkipReason::Decode(format!(
                    "gray payload {} bytes, need {needed}",
                    raw.len()
                )));
            }
            Ok((width, height, gray_to_rgba(&raw[..needed])))
        }
        ColorSpaceKind::Unsupported(name) => Err(SkipReason::UnsupportedColorSpace(name)),
    }
}

fn read_filter_chain(stream: &Stream) -> Vec<String> {
    let Ok(filter) = stream.dict.get(b"Filter") else {
        return Vec::new();
    };
    match filter {
        Object::Name(n) => vec![String::from_utf8_lossy(n).into_owned()],
        Object::Array(items) => items
            .iter()
            .filter_map(|i| i.as_name().ok())
            .map(|n| String::from_utf8_lossy(n).into_owned())
            .collect(),
        _ => Vec::new(),
    }
}

#[derive(Debug)]
enum ColorSpaceKind {
    Rgb,
    Gray,
    Unsupported(String),
}

/// Resolve `/ColorSpace` to a kind we can decode. Handles names directly,
/// arrays whose first element is a colorspace family name, indirect
/// references to either, and the common `[/ICCBased <stream>]` pattern by
/// reading the embedded stream's `/N` (component count).
fn read_colorspace(stream: &Stream, doc: &Document) -> ColorSpaceKind {
    let Ok(cs) = stream.dict.get(b"ColorSpace") else {
        return ColorSpaceKind::Unsupported("<missing>".to_string());
    };
    classify_colorspace(cs, doc)
}

fn classify_colorspace(obj: &Object, doc: &Document) -> ColorSpaceKind {
    match obj {
        Object::Name(n) => match n.as_slice() {
            b"DeviceRGB" | b"CalRGB" | b"RGB" => ColorSpaceKind::Rgb,
            b"DeviceGray" | b"CalGray" | b"G" => ColorSpaceKind::Gray,
            other => ColorSpaceKind::Unsupported(String::from_utf8_lossy(other).into_owned()),
        },
        Object::Reference(id) => match doc.get_object(*id) {
            Ok(resolved) => classify_colorspace(resolved, doc),
            Err(_) => ColorSpaceKind::Unsupported(format!("<bad-ref {id:?}>")),
        },
        Object::Array(items) => {
            let Some(first) = items.first() else {
                return ColorSpaceKind::Unsupported("<empty array>".to_string());
            };
            // Resolve first element if it's itself a reference.
            let family_obj = match first {
                Object::Reference(id) => match doc.get_object(*id) {
                    Ok(resolved) => resolved,
                    Err(_) => return ColorSpaceKind::Unsupported("<bad family ref>".to_string()),
                },
                other => other,
            };
            let Ok(family) = family_obj.as_name() else {
                return ColorSpaceKind::Unsupported("<no family name>".to_string());
            };
            match family {
                b"ICCBased" => {
                    // Look up the params stream's /N to decide RGB vs Gray.
                    let n_components = items
                        .get(1)
                        .and_then(|o| match o {
                            Object::Reference(id) => doc.get_object(*id).ok(),
                            other => Some(other),
                        })
                        .and_then(|o| match o {
                            Object::Stream(s) => {
                                s.dict.get(b"N").ok().and_then(|n| n.as_i64().ok())
                            }
                            _ => None,
                        });
                    match n_components {
                        Some(1) => ColorSpaceKind::Gray,
                        Some(3) => ColorSpaceKind::Rgb,
                        Some(other) => ColorSpaceKind::Unsupported(format!("ICCBased N={other}")),
                        None => ColorSpaceKind::Unsupported("ICCBased <no /N>".to_string()),
                    }
                }
                b"DeviceRGB" | b"CalRGB" => ColorSpaceKind::Rgb,
                b"DeviceGray" | b"CalGray" => ColorSpaceKind::Gray,
                other => ColorSpaceKind::Unsupported(String::from_utf8_lossy(other).into_owned()),
            }
        }
        _ => ColorSpaceKind::Unsupported(format!("{obj:?}")),
    }
}

fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    let pixels = rgb.len() / 3;
    let mut out = Vec::with_capacity(pixels * 4);
    for px in rgb.chunks_exact(3) {
        // The renderer expects 4-byte ARGB-as-bytes: BGRA on
        // little-endian. Swap R/B to keep parity with the existing
        // `crate::ocr` byte order.
        out.push(px[2]); // B
        out.push(px[1]); // G
        out.push(px[0]); // R
        out.push(0xFF);
    }
    out
}

fn gray_to_rgba(gray: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(gray.len() * 4);
    for &v in gray {
        out.push(v);
        out.push(v);
        out.push(v);
        out.push(0xFF);
    }
    out
}

fn bgra_to_rgb(bgra: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bgra.len() / 4 * 3);
    for px in bgra.chunks_exact(4) {
        out.push(px[2]); // R from BGRA[2]
        out.push(px[1]); // G
        out.push(px[0]); // B
    }
    out
}

fn has_png_signature(b: &[u8]) -> bool {
    b.len() >= 8 && b[..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
}

fn has_jpeg_signature(b: &[u8]) -> bool {
    b.len() >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF
}

fn flate_compress(input: &[u8]) -> Vec<u8> {
    use flate2::Compression;
    use flate2::write::ZlibEncoder;
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(input).expect("flate write to vec");
    encoder.finish().expect("flate finish")
}

// ---------------------------------------------------------------------------
// Page-as-raster fallback for PDFs whose body content is outlined vector
// glyphs (CorelDRAW, design-tool exports). Neither text extraction nor
// image-XObject translation finds anything; the only path is to rasterize
// the page, OCR it, and replace the page's content stream.

/// DPI to render PDF pages at when running them through the OCR pipeline.
/// 200 is a balance between OCR accuracy (Tesseract likes ~300) and PDF
/// payload size (200 DPI A4 ≈ 1700×2200 = 3.7M pixels).
const RASTER_PAGE_DPI: f32 = 200.0;

/// Number of worker threads that run mupdf render + OCR + image_render +
/// flate compress in parallel. Tied to the size of `OcrPool` in the
/// session — extra workers would just queue on the OCR mutexes.
const RASTER_WORKERS: usize = 4;

/// Run page-rasterize-and-replace on every page that has no extractable
/// text. Writes the translated raster back as a single page-spanning
/// image XObject whose `Do` operator becomes the page's only content. The
/// original page's vector content is dropped — it gets visually captured
/// in the rendered raster anyway.
///
/// Pages that already have extractable text are left alone for the
/// existing text-translation pipeline.
/// Worker output for one rasterized + translated page. Workers do every
/// CPU-bound step (mupdf render, OCR, render_overlay, flate compress);
/// the main thread only stitches results into the lopdf::Document.
struct RasterizedPage {
    width: u32,
    height: u32,
    page_w_pts: f32,
    page_h_pts: f32,
    bounds_x0: f32,
    bounds_y0: f32,
    /// Flate-compressed RGB bytes ready to drop into a /FlateDecode
    /// /DeviceRGB image XObject.
    compressed_rgb: Vec<u8>,
}

pub fn translate_pdf_pages_as_raster_in_place(
    pdf_bytes: &[u8],
    session: &TranslatorSession,
    source_code: &str,
    target_code: &str,
    fonts: &(dyn FontProvider + Send + Sync),
    is_cancelled: &(dyn Fn() -> bool + Send + Sync),
    mut on_progress: impl FnMut(usize, usize),
) -> Result<Vec<u8>, ImageTranslateError> {
    // Walk text fragments to find pages with nothing extractable. If
    // extraction itself fails we silently no-op — the upstream text
    // translation will surface that error.
    let extracted = match extract_text(pdf_bytes) {
        Ok(e) => e,
        Err(err) => {
            warn!("[pdf_image_translate] extract_text failed: {err}; skipping page-raster pass");
            return Ok(pdf_bytes.to_vec());
        }
    };
    let pages_without_text: HashSet<usize> = extracted
        .iter()
        .filter(|p| p.fragments.is_empty())
        .map(|p| p.page_index)
        .collect();
    if pages_without_text.is_empty() {
        return Ok(pdf_bytes.to_vec());
    }
    info!(
        "[pdf_image_translate] {} page(s) have no extractable text; rasterizing for OCR ({} workers)",
        pages_without_text.len(),
        RASTER_WORKERS
    );

    let mut doc = Document::load_mem(pdf_bytes)?;
    let pages: Vec<(u32, ObjectId)> = doc.get_pages().into_iter().collect();
    // Build a Vec to hand to workers (need indexable, deterministic order).
    // Sorted so the round-robin progress feels left-to-right through the doc.
    let mut pages_to_do: Vec<usize> = pages_without_text.into_iter().collect();
    pages_to_do.sort();
    let total = pages_to_do.len();
    on_progress(0, total);

    // Worker dispatch via shared atomic counter. Each worker fetches the
    // next index, OCRs/renders, and pushes a result on the mpsc channel.
    // Cancellation: workers check is_cancelled before each fetch_add and
    // before send; in-flight OCR can't be interrupted (Tesseract is a
    // blocking C call), so worst case 4 pages finish after cancel.
    let next_page = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel::<(usize, Result<RasterizedPage, SkipReason>)>();

    let mut any_replaced = false;
    let mut processed = 0usize;
    let cancelled_during_collect = std::cell::Cell::new(false);

    thread::scope(|scope| {
        for _ in 0..RASTER_WORKERS {
            let tx = tx.clone();
            let pages_to_do = &pages_to_do;
            let next_page = &next_page;
            scope.spawn(move || {
                // Each worker owns its own mupdf::Document — it isn't Sync
                // and a fresh open over the same bytes is cheap.
                let mupdf_doc = match mupdf::Document::from_bytes(pdf_bytes, "application/pdf") {
                    Ok(d) => d,
                    Err(err) => {
                        warn!("[pdf_image_translate] worker mupdf load failed: {err}");
                        return;
                    }
                };
                loop {
                    if is_cancelled() {
                        break;
                    }
                    let slot = next_page.fetch_add(1, Ordering::Relaxed);
                    if slot >= pages_to_do.len() {
                        break;
                    }
                    let page_index = pages_to_do[slot];
                    if is_cancelled() {
                        break;
                    }
                    let result = ocr_render_encode_page(
                        &mupdf_doc,
                        session,
                        source_code,
                        target_code,
                        fonts,
                        page_index,
                    );
                    if tx.send((page_index, result)).is_err() {
                        // Receiver dropped (main thread aborted); stop.
                        break;
                    }
                }
            });
        }
        drop(tx);

        // Main-thread collector: receive each completed page and install
        // it into the lopdf doc. After cancellation we keep draining so
        // workers can finish and the scope can join cleanly, but we don't
        // apply further results.
        while let Ok((page_index, result)) = rx.recv() {
            if is_cancelled() {
                cancelled_during_collect.set(true);
                continue;
            }
            let page_id = pages.iter().find_map(|(num, id)| {
                if (*num as usize).saturating_sub(1) == page_index {
                    Some(*id)
                } else {
                    None
                }
            });
            let Some(page_id) = page_id else {
                continue;
            };
            match result {
                Ok(r) => match install_rasterized_page(&mut doc, page_id, page_index, r) {
                    Ok(()) => {
                        info!(
                            "[pdf_image_translate] page {} rasterized + translated",
                            page_index
                        );
                        any_replaced = true;
                    }
                    Err(reason) => {
                        debug!("[pdf_image_translate] page {page_index} install failed: {reason}");
                    }
                },
                Err(reason) => {
                    debug!("[pdf_image_translate] page {page_index} skipped: {reason}");
                }
            }
            processed += 1;
            on_progress(processed, total);
        }
    });

    if cancelled_during_collect.get() {
        debug!("[pdf_image_translate] page-raster pass cancelled mid-flight");
    }

    if !any_replaced {
        return Ok(pdf_bytes.to_vec());
    }

    let mut out = Vec::new();
    doc.save_to(&mut out).map_err(lopdf::Error::IO)?;
    Ok(out)
}

/// Worker step: render the page, OCR + translate it, run image_render to
/// produce the translated raster, and flate-compress it. No lopdf state
/// is touched; the caller stitches the result into the document on the
/// main thread.
fn ocr_render_encode_page(
    mupdf_doc: &mupdf::Document,
    session: &TranslatorSession,
    source_code: &str,
    target_code: &str,
    fonts: &(dyn FontProvider + Send + Sync),
    page_index: usize,
) -> Result<RasterizedPage, SkipReason> {
    let page = mupdf_doc
        .load_page(page_index as i32)
        .map_err(|e| SkipReason::Decode(format!("mupdf load_page: {e}")))?;
    let bounds = page
        .bounds()
        .map_err(|e| SkipReason::Decode(format!("page bounds: {e}")))?;
    let page_w_pts = bounds.x1 - bounds.x0;
    let page_h_pts = bounds.y1 - bounds.y0;
    if page_w_pts <= 0.0 || page_h_pts <= 0.0 {
        return Err(SkipReason::MissingDims);
    }

    let scale = RASTER_PAGE_DPI / 72.0;
    let ctm = mupdf::Matrix::new_scale(scale, scale);
    let pixmap = page
        .to_pixmap(&ctm, &mupdf::Colorspace::device_rgb(), true, false)
        .map_err(|e| SkipReason::Decode(format!("to_pixmap: {e}")))?;
    if pixmap.n() != 4 {
        return Err(SkipReason::Decode(format!(
            "expected RGBA pixmap, got n={}",
            pixmap.n()
        )));
    }
    let width = pixmap.width();
    let height = pixmap.height();
    // mupdf hands us RGBA; the OCR + render pipeline operates on BGRA
    // byte order (u32 ARGB on little-endian). Swap R/B going in.
    let mut bgra = pixmap.samples().to_vec();
    swap_r_b(&mut bgra);

    let prepared = session
        .translate_image_rgba(
            &bgra,
            width,
            height,
            source_code,
            target_code,
            DEFAULT_MIN_CONFIDENCE,
            ReadingOrder::LeftToRight,
            BackgroundMode::AutoDetect,
        )
        .map_err(|e: TranslatorError| {
            if e.message.to_lowercase().contains("no text") {
                SkipReason::NoTextDetected
            } else {
                SkipReason::Ocr(e.message)
            }
        })?;
    if prepared.blocks.is_empty() {
        return Err(SkipReason::NoTextDetected);
    }

    let opts = RenderOptions {
        language: target_code.to_string(),
        ..RenderOptions::default()
    };
    let translated_bgra =
        render_overlay(&prepared, fonts, &opts).map_err(|e| SkipReason::Render(e.to_string()))?;

    let rgb = bgra_to_rgb(&translated_bgra);
    let compressed_rgb = flate_compress(&rgb);

    Ok(RasterizedPage {
        width,
        height,
        page_w_pts,
        page_h_pts,
        bounds_x0: bounds.x0,
        bounds_y0: bounds.y0,
        compressed_rgb,
    })
}

/// Main-thread step: build the image XObject + content stream from a
/// completed worker result and graft them into the lopdf::Document at
/// the right page id.
fn install_rasterized_page(
    doc: &mut Document,
    page_id: ObjectId,
    page_index: usize,
    r: RasterizedPage,
) -> Result<(), SkipReason> {
    let mut img_dict = Dictionary::new();
    img_dict.set("Type", Object::Name(b"XObject".to_vec()));
    img_dict.set("Subtype", Object::Name(b"Image".to_vec()));
    img_dict.set("Width", r.width as i64);
    img_dict.set("Height", r.height as i64);
    img_dict.set("ColorSpace", Object::Name(b"DeviceRGB".to_vec()));
    img_dict.set("BitsPerComponent", 8i64);
    img_dict.set("Filter", Object::Name(b"FlateDecode".to_vec()));
    let img_stream = Stream::new(img_dict, r.compressed_rgb);
    let img_id = doc.add_object(Object::Stream(img_stream));

    let content = format!(
        "q\n{w} 0 0 {h} {x} {y} cm\n/TransImg{idx} Do\nQ\n",
        w = r.page_w_pts,
        h = r.page_h_pts,
        x = r.bounds_x0,
        y = r.bounds_y0,
        idx = page_index,
    );
    let content_stream = Stream::new(Dictionary::new(), content.into_bytes());
    let content_id = doc.add_object(Object::Stream(content_stream));

    let resource_name = format!("TransImg{page_index}");
    install_image_in_page(doc, page_id, content_id, img_id, &resource_name)
}

fn install_image_in_page(
    doc: &mut Document,
    page_id: ObjectId,
    content_id: ObjectId,
    image_id: ObjectId,
    resource_name: &str,
) -> Result<(), SkipReason> {
    // /Resources lookup: in lopdf the page Object is a Dictionary; the
    // /Resources entry can be inline or a reference. We mutate the inline
    // dict directly, or — if it's a reference — mutate the referenced
    // dict. If /Resources is missing entirely, install a new inline one.
    let page_obj = doc
        .get_object_mut(page_id)
        .map_err(|e| SkipReason::Decode(e.to_string()))?;
    let page_dict = page_obj
        .as_dict_mut()
        .map_err(|e| SkipReason::Decode(e.to_string()))?;

    page_dict.set("Contents", Object::Reference(content_id));
    let resources_obj = page_dict.get(b"Resources").ok().cloned();

    match resources_obj {
        Some(Object::Reference(res_id)) => {
            // Mutate the referenced resource dict.
            install_xobject_in_resources(
                doc.get_object_mut(res_id)
                    .map_err(|e| SkipReason::Decode(e.to_string()))?
                    .as_dict_mut()
                    .map_err(|e| SkipReason::Decode(e.to_string()))?,
                image_id,
                resource_name,
            );
        }
        _ => {
            // Inline dict (or missing). Re-fetch mutably and rewrite.
            let page_obj = doc
                .get_object_mut(page_id)
                .map_err(|e| SkipReason::Decode(e.to_string()))?;
            let page_dict = page_obj
                .as_dict_mut()
                .map_err(|e| SkipReason::Decode(e.to_string()))?;
            let resources = match page_dict.get_mut(b"Resources") {
                Ok(Object::Dictionary(d)) => d,
                _ => {
                    page_dict.set("Resources", Object::Dictionary(Dictionary::new()));
                    page_dict
                        .get_mut(b"Resources")
                        .ok()
                        .and_then(|o| match o {
                            Object::Dictionary(d) => Some(d),
                            _ => None,
                        })
                        .expect("just inserted")
                }
            };
            install_xobject_in_resources(resources, image_id, resource_name);
        }
    }
    Ok(())
}

fn install_xobject_in_resources(
    resources: &mut Dictionary,
    image_id: ObjectId,
    resource_name: &str,
) {
    let xobjects = match resources.get_mut(b"XObject") {
        Ok(Object::Dictionary(d)) => d,
        _ => {
            resources.set("XObject", Object::Dictionary(Dictionary::new()));
            resources
                .get_mut(b"XObject")
                .ok()
                .and_then(|o| match o {
                    Object::Dictionary(d) => Some(d),
                    _ => None,
                })
                .expect("just inserted")
        }
    };
    xobjects.set(resource_name, Object::Reference(image_id));
}

fn swap_r_b(buf: &mut [u8]) {
    for px in buf.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_rgba_swaps_to_bgra_byte_order() {
        let rgb = vec![10, 20, 30, 40, 50, 60];
        let bgra = rgb_to_rgba(&rgb);
        // Pixel 0: R=10, G=20, B=30 -> bytes [30, 20, 10, 255].
        assert_eq!(&bgra[..4], &[30, 20, 10, 255]);
        assert_eq!(&bgra[4..], &[60, 50, 40, 255]);
    }

    #[test]
    fn bgra_to_rgb_round_trips() {
        let rgb_in = vec![10, 20, 30, 99, 0, 0];
        let bgra = rgb_to_rgba(&rgb_in);
        let rgb_out = bgra_to_rgb(&bgra);
        assert_eq!(rgb_out, rgb_in);
    }

    #[test]
    fn finds_jpeg_image_xobject_in_naur_pdf() {
        let path = "files/1985-naur.pdf";
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("skipping: {path} not present");
            return;
        };
        let doc = Document::load_mem(&bytes).expect("load pdf");
        let ids = collect_image_ids(&doc);
        assert!(
            !ids.is_empty(),
            "expected at least one image XObject in {path}"
        );
        // Decode the first one — naur is a scanned-page JPEG.
        let stream = doc
            .get_object(ids[0])
            .expect("get xobject")
            .as_stream()
            .expect("stream");
        let (w, h, rgba) = decode_image_to_rgba(stream, &doc).expect("decode jpeg");
        assert!(w >= 100 && h >= 100, "naur scan should be big");
        assert_eq!(rgba.len(), (w as usize) * (h as usize) * 4);
    }

    #[test]
    fn decodes_iccbased_jpeg_in_guru_menu() {
        let path = "files/Guru-menu.pdf";
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("skipping: {path} not present");
            return;
        };
        let doc = Document::load_mem(&bytes).expect("load pdf");
        let ids = collect_image_ids(&doc);
        assert!(
            ids.len() >= 1,
            "expected ICCBased JPEG XObjects in {path}, got {}",
            ids.len()
        );
        // Each page is one big JPEG. Decode the first to confirm
        // ICCBased(3) doesn't trip the colorspace check.
        let stream = doc
            .get_object(ids[0])
            .expect("get xobject")
            .as_stream()
            .expect("stream");
        let (w, h, rgba) =
            decode_image_to_rgba(stream, &doc).expect("decode iccbased jpeg in guru menu");
        assert!(w >= 100 && h >= 100, "guru page should be big: {w}x{h}");
        assert_eq!(rgba.len(), (w as usize) * (h as usize) * 4);
    }
}
