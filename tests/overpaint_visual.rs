//! Visual integration test: overpaint the text regions of `data/kindle.jpg`
//! and re-render the OCR-extracted source text back into each line using the
//! library's detected foreground color. Output lands at
//! `target/overpaint_visual/kindle.overpaint.png` so the background/foreground
//! detection can be eyeballed on a real photograph.
//!
//! Only runs with `--features tesseract` (and requires system tessdata +
//! DejaVuSans installed).

use std::path::PathBuf;

use ab_glyph::{Font, FontVec, PxScale, ScaleFont};
use image::{ImageReader, RgbaImage};

use translator::ocr::{
    DetectedWord, ReadingOrder, Rect, TextBlock, build_text_blocks, prepare_overlay_image,
};
use translator::settings::BackgroundMode;
use translator::tesseract::TesseractWrapper;

const TESSDATA: &str = "/usr/share/tesseract-ocr/5/tessdata";
const FONT_PATH: &str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
const OUTPUT_DIR: &str = "target/overpaint_visual";

#[test]
fn overpaint_visual_kindle() {
    run_case("data/kindle.jpg", "kindle.overpaint.png");
}

#[test]
fn overpaint_visual_lobsters() {
    run_case("data/lobsters.png", "lobsters.overpaint.png");
}

fn run_case(input: &str, output_file: &str) {
    let decoded = ImageReader::open(input)
        .unwrap_or_else(|err| panic!("open {input}: {err}"))
        .decode()
        .unwrap_or_else(|err| panic!("decode {input}: {err}"))
        .to_rgba8();
    let (width, height) = decoded.dimensions();
    let rgba = decoded.into_raw();

    // The library treats 4-byte pixels as u32 ARGB via `to_ne_bytes` (little-endian
    // → bytes in BGRA order). `image` hands us RGBA byte order, so swap R/B going
    // in and coming out.
    let mut bgra = rgba.clone();
    swap_r_b(&mut bgra);

    let words = run_tesseract(&rgba, width, height);
    let detected = words.into_iter().map(to_ocr_word).collect::<Vec<_>>();
    let blocks = build_text_blocks(&detected, 30, false, false);
    assert!(!blocks.is_empty(), "tesseract returned no text blocks");
    let translated = blocks
        .iter()
        .map(TextBlock::translation_text)
        .collect::<Vec<_>>();

    let prepared = prepare_overlay_image(
        &bgra,
        width,
        height,
        &blocks,
        &translated,
        BackgroundMode::AutoDetect,
        ReadingOrder::LeftToRight,
    )
    .expect("prepare_overlay_image");

    let mut out_bgra = prepared.rgba_bytes.clone();

    let font_bytes = std::fs::read(FONT_PATH).expect("read DejaVuSans.ttf");
    let font = FontVec::try_from_vec(font_bytes).expect("parse font");

    for block in &prepared.blocks {
        for line in &block.lines {
            draw_line(
                &mut out_bgra,
                width,
                height,
                &line.text,
                line.bounding_box,
                line.foreground_argb,
                &font,
            );
        }
    }

    swap_r_b(&mut out_bgra);
    let output_image = RgbaImage::from_raw(width, height, out_bgra).expect("rebuild rgba image");
    let out_dir = PathBuf::from(OUTPUT_DIR);
    std::fs::create_dir_all(&out_dir).expect("create output dir");
    let out_path = out_dir.join(output_file);
    output_image.save(&out_path).expect("save png");
    eprintln!("wrote {}", out_path.display());
}

fn swap_r_b(buf: &mut [u8]) {
    for pixel in buf.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
}

fn run_tesseract(rgba: &[u8], width: u32, height: u32) -> Vec<translator::tesseract::DetectedWord> {
    let mut engine = TesseractWrapper::new(Some(TESSDATA), Some("eng")).expect("init tesseract");
    let bpp = 4i32;
    let bpl = (width as i32) * bpp;
    engine
        .set_frame(rgba, width as i32, height as i32, bpp, bpl)
        .expect("tesseract set_frame");
    engine.get_word_boxes().expect("tesseract word boxes")
}

fn to_ocr_word(word: translator::tesseract::DetectedWord) -> DetectedWord {
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

fn draw_line(
    buf: &mut [u8],
    width: u32,
    height: u32,
    text: &str,
    rect: Rect,
    fg_argb: u32,
    font: &FontVec,
) {
    let rect_height = rect.bottom.saturating_sub(rect.top).max(1) as f32;
    let rect_width = rect.right.saturating_sub(rect.left).max(1) as f32;
    let mut scale_px = rect_height;
    let text_width_at = |s: f32| -> f32 {
        let sf = font.as_scaled(PxScale::from(s));
        text.chars().map(|c| sf.h_advance(font.glyph_id(c))).sum()
    };
    let width_at_height = text_width_at(scale_px);
    if width_at_height > rect_width {
        scale_px *= rect_width / width_at_height;
    }
    let scale = PxScale::from(scale_px);
    let scaled = font.as_scaled(scale);
    let baseline = rect.top as f32 + scaled.ascent();
    let fg_bytes = fg_argb.to_ne_bytes();
    let mut cursor = rect.left as f32;

    for ch in text.chars() {
        let glyph_id = font.glyph_id(ch);
        let glyph = glyph_id.with_scale_and_position(scale, ab_glyph::point(cursor, baseline));
        if let Some(outlined) = font.outline_glyph(glyph) {
            let bounds = outlined.px_bounds();
            outlined.draw(|gx, gy, coverage| {
                let px = bounds.min.x as i32 + gx as i32;
                let py = bounds.min.y as i32 + gy as i32;
                if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                    return;
                }
                let idx = ((py as usize) * (width as usize) + px as usize) * 4;
                let a = coverage.clamp(0.0, 1.0);
                let inv = 1.0 - a;
                for c in 0..3 {
                    let blended = fg_bytes[c] as f32 * a + buf[idx + c] as f32 * inv;
                    buf[idx + c] = blended.round().clamp(0.0, 255.0) as u8;
                }
            });
        }
        cursor += scaled.h_advance(glyph_id);
    }
}
