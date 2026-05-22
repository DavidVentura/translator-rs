//! Full-pipeline single-frame integration test across the four 90° rotations.
//!
//! Renders synthetic horizontal text into a single RGBA frame, then for each
//! rotation in {R0, R90, R180, R270}:
//!
//!   1. Builds an `OrientedImage` from the rotated frame.
//!   2. Drives one tick of `LivePlanarEngine::process_frame` to leave Idle.
//!   3. Runs ppocr `detect_only_image`.
//!   4. Calls `estimate_canonical_quadrant` on the detections.
//!   5. Calls `acquire_now_with_orientation` with the estimated quadrant.
//!   6. Runs `LiveSession::run_post_detect` (which routes through the same
//!      detect→recognize→translate→overlay-upsert path the live app uses).
//!   7. Composites the resulting overlay items over the camera and saves a
//!      PNG to `smoke-out/ppocr-single-frame/<label>.png` for inspection.
//!
//! Assertions:
//!   - Estimator returns the expected quadrant.
//!   - Recognised text contains each source line snippet (case-insensitive).
//!   - Surface map blocks contain the source lines IN THE ORIGINAL READING
//!     ORDER. This catches the block-grouping / sort-direction bug visible
//!     on CW-rotated and 180°-rotated pages.

#![cfg(all(
    feature = "ppocr",
    feature = "planar-tracker",
    feature = "image-render"
))]

use std::path::{Path, PathBuf};

use ab_glyph::{FontArc, PxScale};
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::draw_text_mut;

use translator::coords::Quadrant;
use translator::font_provider::{FontHandle, FontProvider, FontRequest};
use translator::live_compositor::{OverlayItem as CompositeOverlayItem, composite_frame_into};
use translator::live_frame::OrientedImage;
use translator::live_session::{LiveRecognizer, LiveSession, NoopTranslator, PostDetectInput};
use translator::ocr::{
    DetectedTextBox, OcrSourceSelection, OrientedRect, RecognizedTextLine, Rect,
};
use translator::planar_engine::{EngineConfig, LivePlanarEngine};
use translator::ppocr::{PpocrEngine, PpocrProfile, PpocrRecognizerSpec};
use translator::{LanguageCode, PpocrScript};

const MODEL_DIR: &str = "/home/david/AndroidStudioProjects/bucket/ocr/1/PP-OCRv5";
const FONT_PATH: &str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf";
const DUMP_DIR: &str = "smoke-out/ppocr-single-frame";
const IDENTITY_H: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// First word of each source line, used both for assertion and as the
/// expected reading order.
const LINES: &[&str] = &["DESIGNING DATA", "INTENSIVE", "APPLICATIONS"];

#[test]
fn single_frame_overlay_pipeline_in_all_four_rotations() {
    let _ = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("translator=info"),
    )
    .is_test(false)
    .try_init();

    let Some(paths) = ppocr_paths() else {
        eprintln!("PPOCR bucket files missing under {MODEL_DIR}; skipping");
        return;
    };
    let Ok(font_bytes) = std::fs::read(FONT_PATH) else {
        eprintln!("font {FONT_PATH} missing; skipping");
        return;
    };
    let font = FontArc::try_from_vec(font_bytes).expect("parse font");

    std::fs::create_dir_all(DUMP_DIR).expect("create dump dir");

    let ppocr = PpocrEngine::load(
        &paths.det,
        None,
        Some(&paths.textline_ori),
        vec![PpocrRecognizerSpec {
            script: PpocrScript::Latin,
            model_path: paths.rec,
            keys_path: paths.keys,
        }],
        1,
    )
    .expect("load ppocr");
    let recognizer = PpocrLiveRecognizer {
        engine: &ppocr,
        script: PpocrScript::Latin,
    };
    let translator = NoopTranslator;
    let font_provider = HostFonts::new();

    // Slight margin so detect sees clean text away from the frame edge.
    let base = DynamicImage::ImageRgba8(render_text(&font, LINES));
    base.save(Path::new(DUMP_DIR).join("base_r0.png"))
        .expect("save base");

    let cases: [(&str, Quadrant, DynamicImage); 4] = [
        ("r0", Quadrant::R0, base.clone()),
        ("r90", Quadrant::R90, base.rotate90()),
        ("r180", Quadrant::R180, base.rotate180()),
        ("r270", Quadrant::R270, base.rotate270()),
    ];

    let mut failures: Vec<String> = Vec::new();
    for (label, expected, rotated) in &cases {
        if let Err(e) = run_case(
            &ppocr,
            &recognizer,
            &translator,
            &font_provider,
            label,
            *expected,
            rotated,
        ) {
            failures.push(format!("{label}: {e}"));
        }
    }

    if !failures.is_empty() {
        panic!(
            "single-frame pipeline failures (inspect {}):\n  - {}",
            DUMP_DIR,
            failures.join("\n  - "),
        );
    }
}

fn run_case(
    ppocr: &PpocrEngine,
    recognizer: &PpocrLiveRecognizer<'_>,
    translator: &NoopTranslator,
    font_provider: &HostFonts,
    label: &str,
    expected: Quadrant,
    rotated: &DynamicImage,
) -> Result<(), String> {
    rotated
        .save(Path::new(DUMP_DIR).join(format!("rotated_{label}.png")))
        .map_err(|e| format!("save rotated: {e}"))?;

    let rgba = rotated.to_rgba8();
    let (w, h) = rgba.dimensions();
    let crop = Rect {
        left: 0,
        top: 0,
        right: w,
        bottom: h,
    };
    let oriented = OrientedImage::build_with_rgb(&rgba, w, h, 0, crop, 1_000_000_000)
        .map_err(|e| format!("build oriented: {e:?}"))?;

    let mut engine_cfg = EngineConfig::default();
    engine_cfg.stable_required_ns = 0;
    engine_cfg.acquire_cooldown_ns = 0;
    let mut engine = LivePlanarEngine::new(engine_cfg);
    let session = LiveSession::new();

    // One tick to leave Idle; we don't care about the command, then force-acquire.
    let _ = engine.process_frame(&oriented.gray, true, 1_000_000);

    let rgb_det = oriented
        .rgb_det
        .as_ref()
        .ok_or_else(|| "rgb_det missing".to_string())?;
    let raw_boxes = ppocr
        .detect_only_image(rgb_det, PpocrProfile::Still)
        .map_err(|e| format!("detect: {e:?}"))?;
    if raw_boxes.is_empty() {
        return Err(format!(
            "no detections on rotated frame (expected ≥ {})",
            LINES.len()
        ));
    }
    let rgb = oriented
        .rgb
        .as_ref()
        .ok_or_else(|| "rgb missing".to_string())?;
    let detections: Vec<DetectedTextBox> = raw_boxes
        .into_iter()
        .map(|b| scale_detected_box(b, oriented.det_to_full_scale, rgb.width(), rgb.height()))
        .collect();
    eprintln!("[{label}] detected {} boxes", detections.len());

    let gray_display = image::imageops::grayscale(&rgb.to_rgb8());
    let estimated = translator::live_session::estimate_canonical_quadrant(
        ppocr,
        rgb,
        &gray_display,
        &detections,
    );
    eprintln!("[{label}] estimator → {estimated:?} (expected {expected:?})");

    let q = estimated
        .ok_or_else(|| format!("estimator returned None (expected Some({expected:?}))"))?;
    if q != expected {
        return Err(format!("estimator returned {q:?}, expected {expected:?}"));
    }

    // Use the detection rects (in full-display coords) to constrain the
    // anchor regions to text. Pad slightly so feature extraction has room.
    let regions: Vec<(u32, u32, u32, u32)> = detections
        .iter()
        .map(|d| (d.rect.left, d.rect.top, d.rect.right, d.rect.bottom))
        .collect();
    let anchor_id = engine
        .acquire_now_with_orientation(&oriented.gray, &regions, 16, 2_000_000, Some(q))
        .ok_or_else(|| "acquire_now_with_orientation returned None".to_string())?;
    eprintln!("[{label}] acquired anchor {anchor_id} with quadrant {q:?}");

    let from_lang = "en".to_string();
    let to_lang = "en".to_string();
    let available_codes = vec![LanguageCode::from("en")];
    let outcome = session.run_post_detect(
        PostDetectInput {
            detections: &detections,
            oriented: &oriented,
            h_view_to_surface: None,
            anchor_id,
            from_lang: &from_lang,
            to_lang: &to_lang,
            is_auto_source: false,
            available_codes: &available_codes,
            font_provider,
            matted_strips: &[],
            rec_batch_size: 8,
            canonical_quadrant: Some(q),
        },
        recognizer,
        translator,
        &|| false,
    );
    eprintln!(
        "[{label}] post_detect: detected={} rec_ok={} rec_empty={} cache_hits={} canceled={}",
        outcome.detected_count,
        outcome.rec_ok_count,
        outcome.rec_empty_count,
        outcome.cache_hits,
        outcome.canceled
    );

    // Composite + dump the overlay so we can eyeball it.
    let composite_bytes = composite_overlay(&session, anchor_id, IDENTITY_H, &rgba);
    let composite_path = Path::new(DUMP_DIR).join(format!("composite_{label}.png"));
    RgbaImage::from_raw(w, h, composite_bytes)
        .ok_or_else(|| "compose rgba".to_string())?
        .save(&composite_path)
        .map_err(|e| format!("save composite: {e}"))?;

    // Inspect the surface-map state for this anchor: what lines did we
    // ingest, and in what order do they appear when grouped into blocks?
    let surface_lines: Vec<translator::surface_map::SurfaceLine> = {
        let states = session
            .anchor_states
            .lock()
            .expect("anchor_states poisoned");
        match states.get(&anchor_id) {
            Some(state) => state.map.lines().to_vec(),
            None => Vec::new(),
        }
    };
    // `translated_text` is set block-wide once the router runs, so every
    // surface line in a block shares the same joined string. For
    // per-line ordering assertions we want the raw OCR output, which
    // lives in `source_text`.
    let line_texts: Vec<String> = surface_lines
        .iter()
        .map(|l| l.source_text.clone())
        .collect();
    eprintln!("[{label}] surface lines (raw order, source_text): {line_texts:?}");

    // Block ordering is what the user actually sees in the rendered
    // overlay. Use the canonical-quadrant grouper — same path as
    // run_post_detect — so the assertion reflects what the live
    // pipeline produces.
    let blocks =
        translator::live_session::group_surface_lines_into_blocks_in_quadrant(&surface_lines, q);
    let block_orders: Vec<Vec<String>> = blocks
        .iter()
        .map(|idxs| {
            idxs.iter()
                .map(|&i| {
                    surface_lines
                        .get(i)
                        .map(|l| l.source_text.clone())
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect();
    eprintln!("[{label}] block orderings: {block_orders:?}");
    std::fs::write(
        Path::new(DUMP_DIR).join(format!("blocks_{label}.txt")),
        block_orders
            .iter()
            .map(|b| b.join(" | "))
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .ok();

    // Assertion: recognised text contains each source line snippet.
    let all_text: String = line_texts.join(" ").to_ascii_lowercase();
    let mut missing_text: Vec<&str> = Vec::new();
    for needle in LINES {
        let key = needle.to_ascii_lowercase();
        let primary = key.split_whitespace().next().unwrap_or(&key);
        if !all_text.contains(primary) {
            missing_text.push(needle);
        }
    }
    if !missing_text.is_empty() {
        return Err(format!(
            "missing recognised text: {missing_text:?} in {line_texts:?}"
        ));
    }

    // Assertion: at least one block has the source lines in original order.
    let expected_order: Vec<String> = LINES
        .iter()
        .map(|s| {
            s.split_whitespace()
                .next()
                .unwrap_or(*s)
                .to_ascii_lowercase()
        })
        .collect();
    let has_correct_order = block_orders.iter().any(|block| {
        let normalized: Vec<String> = block
            .iter()
            .map(|t| {
                t.split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_ascii_lowercase()
            })
            .collect();
        let mut iter = normalized.iter();
        expected_order
            .iter()
            .all(|exp| iter.any(|got| got.contains(exp)))
    });
    if !has_correct_order {
        return Err(format!(
            "no block contains lines in original reading order. expected first-words {expected_order:?}, got blocks {block_orders:?}"
        ));
    }

    Ok(())
}

fn render_text(font: &FontArc, lines: &[&str]) -> RgbaImage {
    let scale = PxScale::from(56.0);
    let line_h = 80_i32;
    let pad_x = 30_i32;
    let pad_y = 40_i32;
    let max_chars: usize = lines.iter().map(|l| l.len()).max().unwrap_or(20);
    let approx_w = max_chars as i32 * 28;
    let width = (approx_w + 2 * pad_x).max(360) as u32;
    let height = (line_h * lines.len() as i32 + 2 * pad_y) as u32;

    let bg = Rgba([255u8, 255, 255, 255]);
    let fg = Rgba([20u8, 20, 20, 255]);

    let mut img = RgbaImage::from_pixel(width, height, bg);
    for (i, line) in lines.iter().enumerate() {
        let y = pad_y + i as i32 * line_h;
        draw_text_mut(&mut img, fg, pad_x, y, scale, font, line);
    }
    img
}

fn composite_overlay(
    session: &LiveSession,
    anchor_id: u64,
    h_surface_to_view: [f32; 9],
    camera: &RgbaImage,
) -> Vec<u8> {
    let w = camera.width();
    let h = camera.height();
    let camera_bytes = camera.as_raw();
    let mut composite = vec![0u8; camera_bytes.len()];

    let Ok(items_guard) = session.overlay_items.lock() else {
        return composite;
    };
    let items: Vec<CompositeOverlayItem<'_>> = {
        let live = items_guard
            .iter()
            .filter(|it| it.anchor_id == anchor_id)
            .collect::<Vec<_>>();
        let mut out: Vec<CompositeOverlayItem<'_>> = Vec::with_capacity(live.len() * 2);
        for it in &live {
            out.push(CompositeOverlayItem {
                bitmap_rgba: &it.bg_bitmap,
                bitmap_width: it.width,
                bitmap_height: it.height,
                bitmap_origin_surface_x: it.surface_origin_x,
                bitmap_origin_surface_y: it.surface_origin_y,
                row_extents: &it.bg_row_extents,
            });
        }
        for it in &live {
            out.push(CompositeOverlayItem {
                bitmap_rgba: &it.text_bitmap,
                bitmap_width: it.width,
                bitmap_height: it.height,
                bitmap_origin_surface_x: it.surface_origin_x,
                bitmap_origin_surface_y: it.surface_origin_y,
                row_extents: &it.text_row_extents,
            });
        }
        out
    };
    composite_frame_into(
        &mut composite,
        w,
        h,
        camera_bytes,
        &h_surface_to_view,
        &items,
    )
    .expect("composite");
    composite
}

fn scale_detected_box(
    mut b: DetectedTextBox,
    scale: f32,
    max_w: u32,
    max_h: u32,
) -> DetectedTextBox {
    b.rect = scale_rect(&b.rect, scale, max_w, max_h);
    b.oriented_box = scale_oriented(&b.oriented_box, scale);
    b.tight_box = scale_oriented(&b.tight_box, scale);
    for v in &mut b.contour {
        *v *= scale;
    }
    b
}

fn scale_rect(r: &Rect, scale: f32, max_w: u32, max_h: u32) -> Rect {
    let left = ((r.left as f32) * scale).floor().clamp(0.0, max_w as f32) as u32;
    let top = ((r.top as f32) * scale).floor().clamp(0.0, max_h as f32) as u32;
    let right = ((r.right as f32) * scale).ceil().clamp(0.0, max_w as f32) as u32;
    let bottom = ((r.bottom as f32) * scale).ceil().clamp(0.0, max_h as f32) as u32;
    Rect {
        left: left.min(right.saturating_sub(1)),
        top: top.min(bottom.saturating_sub(1)),
        right: right.max(left.saturating_add(1)),
        bottom: bottom.max(top.saturating_add(1)),
    }
}

fn scale_oriented(r: &OrientedRect, scale: f32) -> OrientedRect {
    OrientedRect {
        cx: r.cx * scale,
        cy: r.cy * scale,
        width: r.width * scale,
        height: r.height * scale,
        angle_radians: r.angle_radians,
    }
}

struct PpocrLiveRecognizer<'a> {
    engine: &'a PpocrEngine,
    script: PpocrScript,
}

impl LiveRecognizer for PpocrLiveRecognizer<'_> {
    fn recognize(
        &self,
        oriented: &OrientedImage,
        boxes: &[DetectedTextBox],
        _source_selection: &OcrSourceSelection,
        canonical_quadrant: Option<Quadrant>,
    ) -> Result<Vec<RecognizedTextLine>, String> {
        let rgb = oriented.rgb.as_ref().ok_or_else(|| "no rgb".to_string())?;
        let gray = image::imageops::grayscale(&rgb.to_rgb8());
        let scripts = vec![self.script; boxes.len()];
        self.engine
            .recognize_text_in_boxes_image(
                rgb,
                &gray,
                boxes,
                &scripts,
                PpocrProfile::Still,
                canonical_quadrant,
            )
            .map_err(|e| format!("{e:?}"))
    }
}

struct HostFonts {
    path: PathBuf,
}

impl HostFonts {
    fn new() -> Self {
        Self {
            path: PathBuf::from(FONT_PATH),
        }
    }
}

impl FontProvider for HostFonts {
    fn locate(&self, _request: &FontRequest) -> Vec<FontHandle> {
        vec![FontHandle::from(self.path.clone())]
    }
}

struct PpocrPaths {
    det: PathBuf,
    rec: PathBuf,
    keys: PathBuf,
    textline_ori: PathBuf,
}

fn ppocr_paths() -> Option<PpocrPaths> {
    let det = env_path("OCR_SINGLE_DET")
        .unwrap_or_else(|| Path::new(MODEL_DIR).join("PP-OCRv5_mobile_det.mnn"));
    let rec = env_path("OCR_SINGLE_REC")
        .unwrap_or_else(|| Path::new(MODEL_DIR).join("latin_PP-OCRv5_mobile_rec_infer.mnn"));
    let keys = env_path("OCR_SINGLE_KEYS")
        .unwrap_or_else(|| Path::new(MODEL_DIR).join("latin_PP-OCRv5_keys.txt"));
    let textline_ori = env_path("OCR_SINGLE_TEXTLINE")
        .unwrap_or_else(|| Path::new(MODEL_DIR).join("textline_ori_x0_25_wq8.mnn"));
    if det.exists() && rec.exists() && keys.exists() && textline_ori.exists() {
        Some(PpocrPaths {
            det,
            rec,
            keys,
            textline_ori,
        })
    } else {
        None
    }
}

fn env_path(key: &str) -> Option<PathBuf> {
    let raw = std::env::var(key).ok()?;
    if let Some(rest) = raw.strip_prefix("~/") {
        Some(PathBuf::from(std::env::var("HOME").ok()?).join(rest))
    } else {
        Some(PathBuf::from(raw))
    }
}
