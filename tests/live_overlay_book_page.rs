//! Integration test for the live-overlay block-grouping algorithm on a
//! real dense-paragraph book photo. PP-OCR detects lines, we run
//! `group_live_lines_into_blocks`, and dump the result so we can see
//! which lines went into which block + why same-paragraph lines might
//! be splitting across multiple blocks.
//!
//! Run with:
//!   cargo test --release --features ppocr live_overlay_book_page -- --nocapture
//!
//! Does NOT assert anything — purely diagnostic output for debugging.

#![cfg(feature = "ppocr")]

use std::path::{Path, PathBuf};

use image::ImageReader;
use translator::DetectedTextBox;
use translator::live_compositor::{OverlayItem, composite_frame_into};
use translator::live_frame::OrientedImage;
use translator::live_session::{BlockSpec, render_anchor_canvas};
use translator::ocr::{
    OrientedRect, Rect, TextLine, group_live_lines_into_blocks, live_lines_should_merge,
};
use translator::ppocr::{PpocrEngine, PpocrProfile};

const DET_MAX_PIXELS: u32 = 650_000;
const MODEL_DIR: &str = "/home/david/AndroidStudioProjects/bucket/ocr/1/PP-OCRv5";

#[test]
fn book_page_grouping_dump() {
    let _ = env_logger::builder().is_test(true).try_init();

    let image_path = PathBuf::from("files/live-overlay/book page.jpg");
    if !image_path.exists() {
        eprintln!("fixture {} missing; skipping", image_path.display());
        return;
    }
    let det_path = Path::new(MODEL_DIR).join("PP-OCRv5_mobile_det.mnn");
    if !det_path.exists() {
        eprintln!("PPOCR det model missing; skipping");
        return;
    }

    let rgba = ImageReader::open(&image_path)
        .expect("open fixture")
        .decode()
        .expect("decode fixture")
        .to_rgba8();
    let (sensor_w, sensor_h) = (rgba.width(), rgba.height());
    eprintln!("image (sensor orient): {}x{} px", sensor_w, sensor_h);
    // EXIF Orientation: 6 (RightTop) → sensor pixels are landscape
    // but the page reads correctly only after a 90° CW rotation.
    // Pass that to `build_with_rgb` so PP-OCR sees the page in its
    // natural reading orientation. Crop is in display coords (sensor
    // dims swapped under R90).
    let rotation_degrees: i32 = 90;
    let (display_w, display_h) = (sensor_h, sensor_w);
    let crop = Rect {
        left: 0,
        top: 0,
        right: display_w,
        bottom: display_h,
    };
    let frame = OrientedImage::build_with_rgb(
        &rgba.into_raw(),
        sensor_w,
        sensor_h,
        rotation_degrees,
        crop,
        DET_MAX_PIXELS,
    )
    .expect("build frame");
    let rgb_det = frame.rgb_det.as_ref().expect("with_rgb populated rgb_det");

    let engine = PpocrEngine::load(&det_path, None, None, vec![], 1).expect("load ppocr");
    let det_boxes = engine
        .detect_only_image(rgb_det, PpocrProfile::Live)
        .expect("detection succeeds");

    eprintln!("raw detections: {}", det_boxes.len());

    let scale = frame.det_to_full_scale;
    let rgb = frame.rgb.as_ref().expect("with_rgb populated rgb");
    let max_w = rgb.width();
    let max_h = rgb.height();
    let boxes: Vec<_> = det_boxes
        .into_iter()
        .map(|b| scale_detected_box(b, scale, max_w, max_h))
        .collect();

    let text_lines: Vec<TextLine> = boxes
        .iter()
        .map(|b| TextLine {
            text: String::new(),
            bounding_box: b.rect.clone(),
            oriented_box: b.oriented_box.clone(),
            tight_box: b.tight_box.clone(),
            word_rects: vec![b.rect.clone()],
        })
        .collect();

    eprintln!("\n--- Lines (sorted by tight_top after AABB-aware fix) ---");
    let mut sorted_by_top: Vec<&TextLine> = text_lines.iter().collect();
    sorted_by_top.sort_by(|a, b| {
        line_top(a)
            .partial_cmp(&line_top(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, line) in sorted_by_top.iter().enumerate() {
        let b = &line.tight_box;
        let angle_deg = b.angle_radians.to_degrees();
        let (half_w, half_h) = aabb_half_extents(b);
        eprintln!(
            "  [{i:>3}] cx={:7.1} cy={:7.1} w={:6.1} h={:5.1} ang={:+5.1}° aabb_w={:6.1} aabb_h={:5.1}",
            b.cx,
            b.cy,
            b.width,
            b.height,
            angle_deg,
            2.0 * half_w,
            2.0 * half_h,
        );
    }

    eprintln!("\n--- Pairwise consecutive merge decisions ---");
    for win in sorted_by_top.windows(2) {
        let (a, b) = (win[0], win[1]);
        let merge = live_lines_should_merge(a, b);
        let (decision, reasons) = explain_merge_decision(a, b);
        eprintln!(
            "  [{:>3}] cy={:7.1} -> [{:>3}] cy={:7.1}   merge={}  reasons: {}  ({})",
            sorted_by_top
                .iter()
                .position(|x| std::ptr::eq(*x, a))
                .unwrap(),
            a.tight_box.cy,
            sorted_by_top
                .iter()
                .position(|x| std::ptr::eq(*x, b))
                .unwrap(),
            b.tight_box.cy,
            if merge { "YES" } else { "NO " },
            reasons,
            decision,
        );
    }

    let blocks = group_live_lines_into_blocks(text_lines.clone());
    eprintln!("\n--- Final blocks: {} ---", blocks.len());
    for (bi, block) in blocks.iter().enumerate() {
        eprintln!("  block {} ({} lines):", bi, block.lines.len());
        for line in &block.lines {
            let b = &line.tight_box;
            eprintln!(
                "    cx={:7.1} cy={:7.1} w={:6.1} h={:5.1} ang={:+5.1}°",
                b.cx,
                b.cy,
                b.width,
                b.height,
                b.angle_radians.to_degrees(),
            );
        }
    }

    // -- Render the anchor's full overlay (bg-only here — empty
    // display_text per block) and composite onto the page. The
    // anchor canvas merges all blocks' bg fills into one bitmap so
    // overlapping pills don't darken at the overlap.
    eprintln!("\n--- Rendering composite ---");
    struct FontProviderStub;
    impl translator::font_provider::FontProvider for FontProviderStub {
        fn locate(
            &self,
            _: &translator::font_provider::FontRequest,
        ) -> Vec<translator::font_provider::FontHandle> {
            Vec::new()
        }
    }
    let font_provider = FontProviderStub;
    let mut block_specs: std::collections::BTreeMap<u64, BlockSpec> =
        std::collections::BTreeMap::new();
    for (bi, block) in blocks.iter().enumerate() {
        let strips: Vec<OrientedRect> = block.lines.iter().map(|l| l.tight_box.clone()).collect();
        if strips.is_empty() {
            continue;
        }
        let matted_strips: Vec<Option<translator::color_matting::MattedStrip>> =
            vec![None; strips.len()];
        block_specs.insert(
            bi as u64,
            BlockSpec {
                strips,
                matted_strips,
                display_text: String::new(),
                language: "en".to_string(),
                content_hash: bi as u64,
            },
        );
    }
    let canvas = render_anchor_canvas(&block_specs, &font_provider);
    if let Some(c) = &canvas {
        eprintln!(
            "  anchor canvas: {}x{} at ({}, {})",
            c.width, c.height, c.surface_origin_x, c.surface_origin_y,
        );
    }
    let items: Vec<OverlayItem<'_>> = canvas
        .as_ref()
        .map(|c| {
            vec![OverlayItem {
                bitmap_rgba: &c.bitmap,
                bitmap_width: c.width,
                bitmap_height: c.height,
                bitmap_origin_surface_x: c.surface_origin_x,
                bitmap_origin_surface_y: c.surface_origin_y,
                row_extents: &c.row_extents,
            }]
        })
        .unwrap_or_default();

    // Camera RGBA in display orientation = the rotated `rgb` from
    // OrientedImage. Pad to 4 bytes per pixel (RGB → RGBA with α=255).
    let rgb_dims = (rgb.width(), rgb.height());
    let rgb8 = rgb.to_rgb8();
    let mut camera_rgba = Vec::with_capacity((rgb_dims.0 * rgb_dims.1 * 4) as usize);
    for px in rgb8.pixels() {
        camera_rgba.extend_from_slice(&[px[0], px[1], px[2], 0xFF]);
    }

    // Composite into a fresh RGBA buffer. Identity H since overlay
    // items are stored in this same display-coord frame for the test.
    let mut composed = vec![0u8; (rgb_dims.0 * rgb_dims.1 * 4) as usize];
    let identity_h = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    composite_frame_into(
        &mut composed,
        rgb_dims.0,
        rgb_dims.1,
        &camera_rgba,
        &identity_h,
        &items,
    )
    .expect("composite");

    let out_path = "/tmp/book_page_composite.png";
    image::save_buffer(
        out_path,
        &composed,
        rgb_dims.0,
        rgb_dims.1,
        image::ColorType::Rgba8,
    )
    .expect("save composite");
    eprintln!("Composited image saved to {}", out_path);
}

fn line_top(line: &TextLine) -> f32 {
    line.tight_box.cy - line.tight_box.height * 0.5
}

fn line_bottom(line: &TextLine) -> f32 {
    line.tight_box.cy + line.tight_box.height * 0.5
}

fn line_left(line: &TextLine) -> f32 {
    line.tight_box.cx - line.tight_box.width * 0.5
}

fn line_right(line: &TextLine) -> f32 {
    line.tight_box.cx + line.tight_box.width * 0.5
}

/// AABB half-extents — kept for the dump only (to show how much the
/// rotated AABB exaggerates the line's vertical reach vs the
/// perpendicular `height` we actually want for gap math).
fn aabb_half_extents(b: &OrientedRect) -> (f32, f32) {
    let abs_cos = b.angle_radians.cos().abs();
    let abs_sin = b.angle_radians.sin().abs();
    let hw = b.width * 0.5;
    let hh = b.height * 0.5;
    (hw * abs_cos + hh * abs_sin, hw * abs_sin + hh * abs_cos)
}

/// Mirror the gates inside `live_lines_should_merge` so we can report
/// which one rejected a candidate pair. Keep this in sync with the
/// implementation in `src/ocr.rs`.
fn explain_merge_decision(prev: &TextLine, next: &TextLine) -> (&'static str, String) {
    let prev_h = prev.tight_box.height.max(1.0);
    let next_h = next.tight_box.height.max(1.0);
    let big_h = prev_h.max(next_h);
    let small_h = prev_h.min(next_h);
    let height_ratio = big_h / small_h;
    let gap = line_top(next) - line_bottom(prev);
    let gap_too_close = gap < -big_h * 0.75;
    let gap_too_far = gap > big_h * 4.25;

    let max_w = prev.tight_box.width.max(next.tight_box.width).max(1.0);
    let min_w = prev.tight_box.width.min(next.tight_box.width).max(1.0);
    let cx_dist = (prev.tight_box.cx - next.tight_box.cx).abs();
    let center_aligned = cx_dist <= max_w * 0.25;
    let edge_tol = big_h * 2.0;
    let similar_width = max_w / min_w <= 1.8;
    let left_dist = (line_left(prev) - line_left(next)).abs();
    let right_dist = (line_right(prev) - line_right(next)).abs();
    let left_aligned = similar_width && left_dist <= edge_tol;
    let right_aligned = similar_width && right_dist <= edge_tol;
    let strongly_centered = cx_dist <= max_w * 0.12;
    let very_close = gap <= big_h * 1.25;
    let height_compatible =
        height_ratio <= 1.8 || (height_ratio <= 2.2 && strongly_centered && very_close);

    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("gap={:.1}", gap));
    parts.push(format!("hratio={:.2}", height_ratio));
    parts.push(format!("|Δcx|={:.1}/{:.1}", cx_dist, max_w * 0.25));
    parts.push(format!("wratio={:.2}", max_w / min_w));
    parts.push(format!("|Δleft|={:.1}/{:.1}", left_dist, edge_tol));
    parts.push(format!("|Δright|={:.1}/{:.1}", right_dist, edge_tol));

    let decision = if gap_too_close {
        "REJ: gap too negative"
    } else if gap_too_far {
        "REJ: gap too large"
    } else if !height_compatible {
        "REJ: height incompatible"
    } else if !(center_aligned || left_aligned || right_aligned) {
        "REJ: no alignment match"
    } else {
        "OK"
    };
    (decision, parts.join(" "))
}

fn scale_detected_box(b: DetectedTextBox, scale: f32, max_w: u32, max_h: u32) -> DetectedTextBox {
    let left = ((b.rect.left as f32) * scale).max(0.0) as u32;
    let top = ((b.rect.top as f32) * scale).max(0.0) as u32;
    let right = ((b.rect.right as f32) * scale).min(max_w as f32) as u32;
    let bottom = ((b.rect.bottom as f32) * scale).min(max_h as f32) as u32;
    let rect = Rect {
        left: left.min(right.saturating_sub(1)),
        top: top.min(bottom.saturating_sub(1)),
        right: right.max(left + 1),
        bottom: bottom.max(top + 1),
    };
    let oriented = OrientedRect {
        cx: b.oriented_box.cx * scale,
        cy: b.oriented_box.cy * scale,
        width: b.oriented_box.width * scale,
        height: b.oriented_box.height * scale,
        angle_radians: b.oriented_box.angle_radians,
    };
    let tight = OrientedRect {
        cx: b.tight_box.cx * scale,
        cy: b.tight_box.cy * scale,
        width: b.tight_box.width * scale,
        height: b.tight_box.height * scale,
        angle_radians: b.tight_box.angle_radians,
    };
    let mut contour = Vec::with_capacity(b.contour.len());
    for v in &b.contour {
        contour.push(v * scale);
    }
    DetectedTextBox {
        rect,
        oriented_box: oriented,
        tight_box: tight,
        contour,
        score: b.score,
    }
}
