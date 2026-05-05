//! Visual integration test: drive the library's image renderer end-to-end on
//! a real photograph. The pipeline is:
//!   image -> tesseract OCR -> build text blocks -> prepare overlay (erase
//!   text regions) -> render translated text back via translator::image_render.
//!
//! Output lands at `target/overpaint_visual/<input>.overpaint.png`.
//!
//! Requires `--features tesseract,image-render` and a host font directory at
//! `/usr/share/fonts/truetype/dejavu`.

use std::path::PathBuf;

use image::{ImageReader, RgbaImage};

use translator::font_provider::{FontHandle, FontProvider, FontRequest};
use translator::image_render::{RenderOptions, render_overlay};
use translator::ocr::{
    DetectedWord, ReadingOrder, Rect, TextBlock, build_text_blocks, prepare_overlay_image,
};
use translator::script::Script;
use translator::settings::BackgroundMode;
use translator::tesseract::TesseractWrapper;

const TESSDATA: &str = "/usr/share/tesseract-ocr/5/tessdata";
const DEJAVU_DIR: &str = "/usr/share/fonts/truetype/dejavu";
const NOTO_TT_DIR: &str = "/usr/share/fonts/truetype/noto";
const NOTO_OTF_DIR: &str = "/usr/share/fonts/opentype/noto";
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

    let provider = HostFonts::default();
    let opts = RenderOptions {
        language: "en".to_string(),
        ..RenderOptions::default()
    };
    let mut out_bgra = render_overlay(&prepared, &provider, &opts).expect("render_overlay");

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

#[derive(Default)]
struct HostFonts;

impl FontProvider for HostFonts {
    fn locate(&self, req: &FontRequest) -> Vec<FontHandle> {
        let mut chain: Vec<FontHandle> = Vec::new();

        // Script-specific primary fonts where DejaVu doesn't cover.
        match req.script {
            Script::Devanagari => {
                let p = PathBuf::from(NOTO_TT_DIR).join(if req.bold {
                    "NotoSansDevanagari-Bold.ttf"
                } else {
                    "NotoSansDevanagari-Regular.ttf"
                });
                if p.is_file() {
                    chain.push(FontHandle::from(p));
                }
            }
            Script::Bengali => {
                let p = PathBuf::from(NOTO_TT_DIR).join(if req.bold {
                    "NotoSansBengali-Bold.ttf"
                } else {
                    "NotoSansBengali-Regular.ttf"
                });
                if p.is_file() {
                    chain.push(FontHandle::from(p));
                }
            }
            Script::Han | Script::Hiragana | Script::Katakana | Script::Hangul => {
                let p = PathBuf::from(NOTO_OTF_DIR).join(if req.bold {
                    "NotoSansCJK-Bold.ttc"
                } else {
                    "NotoSansCJK-Regular.ttc"
                });
                if p.is_file() {
                    let ttc_index = match req.script {
                        Script::Han => 2,
                        Script::Hiragana | Script::Katakana => 0,
                        Script::Hangul => 1,
                        _ => 0,
                    };
                    chain.push(FontHandle::new(p, ttc_index));
                }
            }
            _ => {}
        }

        // DejaVu as primary for Latin/Cyrillic/Greek and as a Latin fallback
        // for everything else.
        let dejavu_dir = PathBuf::from(DEJAVU_DIR);
        if dejavu_dir.is_dir() {
            let leaf = match (req.monospace, req.bold, req.italic) {
                (true, true, true) => "DejaVuSansMono-BoldOblique.ttf",
                (true, true, false) => "DejaVuSansMono-Bold.ttf",
                (true, false, true) => "DejaVuSansMono-Oblique.ttf",
                (true, false, false) => "DejaVuSansMono.ttf",
                (false, true, true) => "DejaVuSans-BoldOblique.ttf",
                (false, true, false) => "DejaVuSans-Bold.ttf",
                (false, false, true) => "DejaVuSans-Oblique.ttf",
                (false, false, false) => "DejaVuSans.ttf",
            };
            chain.push(FontHandle::from(dejavu_dir.join(leaf)));
        }
        chain
    }
}
