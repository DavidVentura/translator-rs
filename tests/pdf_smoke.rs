//! End-to-end smoke test for the PDF rasterize → DocLayout-YOLO pipeline.
//!
//! Skipped (with a clear message) unless every required asset is provided
//! via env vars:
//!
//!   ORT_DYLIB_PATH       absolute path to libonnxruntime.so
//!   DOCLAYOUT_MODEL_PATH absolute path to doclayout-yolo .onnx
//!   PDF_TEST_FILE        absolute path to a sample digital PDF
//!
//! When `PDF_SMOKE_DUMP_DIR` is set, also writes one annotated PNG per page
//! showing the model's detected regions.
//!
//! The test never inspects or prints the textual contents of the PDF — only
//! geometric and layout-class statistics — so it's safe to point at private
//! documents.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use image::{ImageBuffer, Rgba, RgbaImage};

use translator::pdf::{PageTransform, render_pages_for_layout};
use translator::pdf_layout::{Backend, LayoutClass, LayoutRegion, build_session, detect_regions};
use translator::pdf_text::extract_text;
use translator::pdf_translate::translate_pdf;
use translator::pdf_write::write_translated_pdf;
use translator::{FsPackInstallChecker, LanguageCode, StructuredStyledFragment, TranslatorSession};

const BITMAP_SIZE: u32 = 1024;
const DEFAULT_SCORE_THRESHOLD: f32 = 0.25;

fn require_env(name: &str) -> Option<String> {
    match env::var(name) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => {
            eprintln!("[pdf_smoke] skipping: {name} not set");
            None
        }
    }
}

#[test]
fn smoke_render_and_detect() {
    let Some(ort_path) = require_env("ORT_DYLIB_PATH") else {
        return;
    };
    let Some(model_path) = require_env("DOCLAYOUT_MODEL_PATH") else {
        return;
    };
    let Some(pdf_path) = require_env("PDF_TEST_FILE") else {
        return;
    };
    let dump_dir = env::var("PDF_SMOKE_DUMP_DIR").ok().map(PathBuf::from);
    let score_threshold = env::var("PDF_SMOKE_SCORE_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(DEFAULT_SCORE_THRESHOLD);

    // SAFETY: propagating an env var the caller already provided; no other
    // thread races env access during test setup.
    unsafe {
        env::set_var("ORT_DYLIB_PATH", &ort_path);
    }

    let pdf_bytes = fs::read(&pdf_path).expect("read PDF_TEST_FILE");
    let pages = render_pages_for_layout(&pdf_bytes, BITMAP_SIZE).expect("render pages");
    assert!(!pages.is_empty(), "PDF rendered zero pages");

    let mut session = build_session(&model_path, Backend::Cpu).expect("build ORT session");

    if let Some(dir) = &dump_dir {
        fs::create_dir_all(dir).expect("create PDF_SMOKE_DUMP_DIR");
    }

    let mut total_regions = 0usize;
    let mut histogram: BTreeMap<&'static str, usize> = BTreeMap::new();

    for (page_idx, rendered) in pages.iter().enumerate() {
        let regions = detect_regions(
            &mut session,
            &rendered.rgba,
            &rendered.transform,
            score_threshold,
        )
        .expect("detect_regions");

        total_regions += regions.len();
        for region in &regions {
            *histogram.entry(class_name(region.class)).or_default() += 1;
        }

        eprintln!(
            "[pdf_smoke] page {} ({}x{} pts): {} region(s)",
            page_idx,
            rendered.transform.page.width_pts as u32,
            rendered.transform.page.height_pts as u32,
            regions.len()
        );

        if let Some(dir) = &dump_dir {
            let out = dir.join(format!("page-{page_idx:03}.png"));
            dump_annotated(&rendered.rgba, &rendered.transform, &regions, &out);
            eprintln!("[pdf_smoke]   wrote {}", out.display());
        }
    }

    eprintln!("[pdf_smoke] total regions: {total_regions}");
    for (name, count) in &histogram {
        eprintln!("[pdf_smoke]   {name}: {count}");
    }

    assert!(total_regions > 0, "model returned no regions on any page");
}

#[test]
fn smoke_extract_text_fragments() {
    let Some(pdf_path) = require_env("PDF_TEST_FILE") else {
        return;
    };
    let dump_dir = env::var("PDF_SMOKE_DUMP_DIR").ok().map(PathBuf::from);

    let pdf_bytes = fs::read(&pdf_path).expect("read PDF_TEST_FILE");
    let extracted = extract_text(&pdf_bytes).expect("extract_text");

    let mut total_fragments = 0usize;
    let mut total_translation_groups = 0usize;

    if let Some(dir) = &dump_dir {
        fs::create_dir_all(dir).expect("create PDF_SMOKE_DUMP_DIR");
    }

    // Render the same pages so we can overlay fragment bboxes.
    let rendered_pages = render_pages_for_layout(&pdf_bytes, BITMAP_SIZE).expect("render");

    for (page, rendered) in extracted.iter().zip(rendered_pages.iter()) {
        total_fragments += page.fragments.len();
        let groups: std::collections::BTreeSet<_> =
            page.fragments.iter().map(|f| f.translation_group).collect();
        total_translation_groups += groups.len();

        eprintln!(
            "[pdf_smoke] page {} ({}x{} pts): {} fragment(s) in {} group(s)",
            page.page_index,
            page.page.width_pts as u32,
            page.page.height_pts as u32,
            page.fragments.len(),
            groups.len(),
        );

        if let Some(dir) = &dump_dir {
            let out = dir.join(format!("page-{:03}-text.png", page.page_index));
            dump_text_fragments(rendered, &page.fragments, &out);
            eprintln!("[pdf_smoke]   wrote {}", out.display());
        }
    }

    eprintln!("[pdf_smoke] total fragments: {total_fragments}");
    eprintln!("[pdf_smoke] total translation groups: {total_translation_groups}");
    assert!(total_fragments > 0, "extractor returned no fragments");
}

fn dump_text_fragments(
    rendered: &translator::pdf::RenderedPage,
    fragments: &[StructuredStyledFragment],
    out: &Path,
) {
    let size = rendered.transform.bitmap_size;
    let mut img: RgbaImage =
        ImageBuffer::from_raw(size, size, rendered.rgba.to_vec()).expect("rgba into ImageBuffer");

    // Cycle a few hues per translation_group so block boundaries are visible.
    let palette: [Rgba<u8>; 6] = [
        Rgba([0xE0, 0x10, 0x10, 0xFF]),
        Rgba([0x00, 0x80, 0xE0, 0xFF]),
        Rgba([0x00, 0xA0, 0x40, 0xFF]),
        Rgba([0xC0, 0x40, 0xC0, 0xFF]),
        Rgba([0xE0, 0x80, 0x00, 0xFF]),
        Rgba([0x40, 0x40, 0xA0, 0xFF]),
    ];

    for fragment in fragments {
        // Fragment bbox is in PDF points (top-left origin). Convert to image space.
        // Our PageTransform uses PDF user-space (bottom-left origin); fragments are
        // already in mupdf top-left points, so the y-axis is opposite of pdf_bbox_to_image.
        let (x0, y0, x1, y1) = pdf_top_left_to_image(
            &rendered.transform,
            fragment.bounding_box.left as f32,
            fragment.bounding_box.top as f32,
            fragment.bounding_box.right as f32,
            fragment.bounding_box.bottom as f32,
        );
        let color = palette[(fragment.translation_group as usize) % palette.len()];
        draw_rect_outline(
            &mut img, x0 as i32, y0 as i32, x1 as i32, y1 as i32, color, 1,
        );
    }

    img.save(out).expect("save text-fragments PNG");
}

/// mupdf stext bbox is in PDF *points* with **top-left** origin (it's already
/// flipped). Convert that to letterboxed image-pixel coords.
fn pdf_top_left_to_image(
    t: &PageTransform,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
) -> (f32, f32, f32, f32) {
    let x0 = left * t.scale + t.pad_x;
    let x1 = right * t.scale + t.pad_x;
    let y0 = top * t.scale + t.pad_y;
    let y1 = bottom * t.scale + t.pad_y;
    (x0, y0, x1, y1)
}

/// End-to-end: translate the test PDF and write the translated output to disk.
///
/// Requires:
///   PDF_TEST_FILE          source PDF
///   PDF_SMOKE_BUCKET_DIR   path to a translator-bucket directory containing
///                          `index.json` plus the relevant `translation/`
///                          model files
///   PDF_SMOKE_TARGET_LANG  target language code (e.g. "en")
/// Optional:
///   PDF_SMOKE_FORCED_SOURCE_LANG  force source detection
///   PDF_SMOKE_DUMP_DIR     where to write the output PDF
#[test]
fn smoke_translate_and_write_pdf() {
    let Some(pdf_path) = require_env("PDF_TEST_FILE") else {
        return;
    };
    let Some(bucket) = require_env("PDF_SMOKE_BUCKET_DIR") else {
        return;
    };
    let Some(target_lang) = require_env("PDF_SMOKE_TARGET_LANG") else {
        return;
    };
    let forced_source = env::var("PDF_SMOKE_FORCED_SOURCE_LANG").ok();
    let dump_dir = env::var("PDF_SMOKE_DUMP_DIR").ok().map(PathBuf::from);

    let bucket_path = PathBuf::from(&bucket);
    let catalog_path = bucket_path.join("index.json");
    let bundled_json = fs::read_to_string(&catalog_path).expect("read catalog index.json");
    let checker = FsPackInstallChecker::new(&bucket);
    let session = TranslatorSession::open(&bundled_json, None, bucket.clone(), &checker)
        .expect("open TranslatorSession");

    let available_langs: Vec<LanguageCode> = session
        .language_overview()
        .into_iter()
        .map(|row| LanguageCode::new(row.language.code))
        .collect();

    let pdf_bytes = fs::read(&pdf_path).expect("read PDF_TEST_FILE");
    let translations = translate_pdf(
        &session,
        &pdf_bytes,
        forced_source.as_deref(),
        &target_lang,
        &available_langs,
    )
    .expect("translate_pdf");

    let mut total_blocks = 0usize;
    for page in &translations {
        if let Some(err) = &page.error {
            eprintln!("[pdf_smoke] page {} error: {err}", page.page_index);
        }
        total_blocks += page.blocks.len();
        eprintln!(
            "[pdf_smoke] page {}: {} translated block(s)",
            page.page_index,
            page.blocks.len()
        );
    }

    assert!(total_blocks > 0, "no blocks were translated");

    let out_pdf = write_translated_pdf(&pdf_bytes, &translations).expect("write_translated_pdf");
    assert!(!out_pdf.is_empty());
    eprintln!("[pdf_smoke] translated pdf bytes: {}", out_pdf.len());

    if let Some(dir) = dump_dir {
        fs::create_dir_all(&dir).expect("mkdir dump dir");
        let out_path = dir.join("translated.pdf");
        fs::write(&out_path, &out_pdf).expect("write translated.pdf");
        eprintln!("[pdf_smoke] wrote {}", out_path.display());
    }
}

fn dump_annotated(rgba: &[u8], transform: &PageTransform, regions: &[LayoutRegion], out: &Path) {
    let size = transform.bitmap_size;
    let mut img: RgbaImage =
        ImageBuffer::from_raw(size, size, rgba.to_vec()).expect("rgba into ImageBuffer");

    for region in regions {
        let (x0, y0, x1, y1) = transform.pdf_bbox_to_image(&region.bbox);
        // 1px at score 0, ~5px at score 1.
        let thickness = (1.0 + region.score * 4.0).round() as i32;
        draw_rect_outline(
            &mut img,
            x0 as i32,
            y0 as i32,
            x1 as i32,
            y1 as i32,
            class_color(region.class),
            thickness,
        );
    }

    img.save(out).expect("save annotated PNG");
}

fn draw_rect_outline(
    img: &mut RgbaImage,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: Rgba<u8>,
    thickness: i32,
) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let x0 = x0.clamp(0, w - 1);
    let x1 = x1.clamp(0, w - 1);
    let y0 = y0.clamp(0, h - 1);
    let y1 = y1.clamp(0, h - 1);

    for t in 0..thickness {
        for x in x0..=x1 {
            put(img, x, y0 + t, color);
            put(img, x, y1 - t, color);
        }
        for y in y0..=y1 {
            put(img, x0 + t, y, color);
            put(img, x1 - t, y, color);
        }
    }
}

fn put(img: &mut RgbaImage, x: i32, y: i32, color: Rgba<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, color);
    }
}

fn class_color(c: LayoutClass) -> Rgba<u8> {
    match c {
        LayoutClass::Title => Rgba([0xFF, 0x00, 0x00, 0xFF]), // red
        LayoutClass::PlainText => Rgba([0x00, 0xA0, 0xFF, 0xFF]), // blue
        LayoutClass::Abandon => Rgba([0x88, 0x88, 0x88, 0xFF]), // grey
        LayoutClass::Figure => Rgba([0xFF, 0x80, 0x00, 0xFF]), // orange
        LayoutClass::FigureCaption => Rgba([0xFF, 0xC0, 0x40, 0xFF]),
        LayoutClass::Table => Rgba([0x00, 0xC0, 0x40, 0xFF]), // green
        LayoutClass::TableCaption => Rgba([0x40, 0xE0, 0x80, 0xFF]),
        LayoutClass::TableFootnote => Rgba([0x80, 0xC0, 0x60, 0xFF]),
        LayoutClass::IsolateFormula => Rgba([0xC0, 0x40, 0xFF, 0xFF]), // purple
        LayoutClass::FormulaCaption => Rgba([0xE0, 0x80, 0xFF, 0xFF]),
    }
}

fn class_name(c: LayoutClass) -> &'static str {
    match c {
        LayoutClass::Title => "title",
        LayoutClass::PlainText => "plain_text",
        LayoutClass::Abandon => "abandon",
        LayoutClass::Figure => "figure",
        LayoutClass::FigureCaption => "figure_caption",
        LayoutClass::Table => "table",
        LayoutClass::TableCaption => "table_caption",
        LayoutClass::TableFootnote => "table_footnote",
        LayoutClass::IsolateFormula => "isolate_formula",
        LayoutClass::FormulaCaption => "formula_caption",
    }
}
