//! Tier-1 end-to-end test for the per-box screen monitor
//! (`SCREEN_CHANGE_DETECTION.md`): the real PP-OCR detect + recognize models on a
//! real frame, with the monitor fed through the real GPU pinhole recovery — our
//! opaque pill is composited over the source and the [`LatticeProbe`] shader
//! reads the holes back, exactly as on a MediaProjection capture.
//!
//! It validates the two cases the design exists for:
//!   * background motion *between* a subtitle's strokes is ignored (the stroke
//!     mask, derived by binarizing a real text crop, keeps the box Quiet);
//!   * replacing the text under the box trips a re-OCR, and the real recognizer
//!     confirms the text genuinely changed.
//!
//! Requires the PP-OCR model files and the fixture images — it hard-fails rather
//! than skipping, so the test can't silently rot. Models are pinned; fixtures are
//! repo-relative (set them up locally / in CI).

#![cfg(all(feature = "ppocr", feature = "gpu"))]

use std::path::{Path, PathBuf};

use image::imageops::FilterType;
use image::{GrayImage, ImageReader, RgbaImage};
use khronos_egl as egl;
use translator::DetectedTextBox;
use translator::PpocrScript;
use translator::live_frame::OrientedImage;
use translator::ocr::{OrientedRect, Rect};
use translator::ppocr::{PpocrEngine, PpocrProfile, PpocrRecognizerSpec};
use translator::screen_monitor::{FrameClassification, Lattice, MonitorConfig, ScreenMonitor};
use translator::screen_monitor_gpu::{LatticeProbe, PillRegion};

const DET_MAX_PIXELS: u32 = 1_000_000;
const MODEL_DIR: &str = "/home/david/AndroidStudioProjects/bucket/ocr/1/PP-OCRv5";
const FIXTURE_DIR: &str = "files/live-overlay";
const LATTICE_SPACING: f32 = 5.0;
const PILL_LUMA: u8 = 0; // opaque black pill, matching the screen overlay
/// Luma distance from the box mean that marks a lattice hole as on-ink.
const INK_CONTRAST: f32 = 35.0;
const PLATFORM_SURFACELESS_MESA: egl::Enum = 0x31DD;

fn load_engine() -> PpocrEngine {
    let det = Path::new(MODEL_DIR).join("PP-OCRv5_mobile_det.mnn");
    let rec = Path::new(MODEL_DIR).join("latin_PP-OCRv5_mobile_rec_infer.mnn");
    let keys = Path::new(MODEL_DIR).join("latin_PP-OCRv5_keys.txt");
    for p in [&det, &rec, &keys] {
        assert!(p.exists(), "required PP-OCR model missing: {}", p.display());
    }
    let spec = PpocrRecognizerSpec {
        script: PpocrScript::Latin,
        model_path: rec,
        keys_path: keys,
    };
    PpocrEngine::load(&det, None, None, vec![spec], 1).expect("load ppocr")
}

fn fixture(name: &str) -> RgbaImage {
    let path = PathBuf::from(FIXTURE_DIR).join(name);
    assert!(
        path.exists(),
        "required fixture missing: {}",
        path.display()
    );
    ImageReader::open(&path)
        .expect("open fixture")
        .decode()
        .expect("decode fixture")
        .to_rgba8()
}

fn make_probe() -> LatticeProbe {
    let lib = unsafe { egl::DynamicInstance::<egl::EGL1_5>::load_required() }
        .expect("load libEGL (install Mesa/llvmpipe for headless GL)");
    let lib: &'static egl::DynamicInstance<egl::EGL1_5> = Box::leak(Box::new(lib));
    let display = unsafe {
        lib.get_platform_display(
            PLATFORM_SURFACELESS_MESA,
            egl::DEFAULT_DISPLAY,
            &[egl::ATTRIB_NONE],
        )
    }
    .expect("surfaceless display");
    lib.initialize(display).expect("eglInitialize");
    lib.bind_api(egl::OPENGL_ES_API).expect("bind GLES");
    let config = lib
        .choose_first_config(
            display,
            &[
                egl::SURFACE_TYPE,
                egl::PBUFFER_BIT,
                egl::RENDERABLE_TYPE,
                egl::OPENGL_ES2_BIT,
                egl::RED_SIZE,
                8,
                egl::NONE,
            ],
        )
        .expect("choose_config")
        .expect("a matching config");
    let ctx = lib
        .create_context(
            display,
            config,
            None,
            &[egl::CONTEXT_CLIENT_VERSION, 3, egl::NONE],
        )
        .expect("create GLES3 context");
    lib.make_current(display, None, None, Some(ctx))
        .expect("make current");
    LatticeProbe::new(|name| {
        lib.get_proc_address(name)
            .map(|p| p as *const std::ffi::c_void)
            .unwrap_or(std::ptr::null())
    })
    .expect("build LatticeProbe")
}

/// Real detect + recognize on a CPU-built frame. Returns the full-res canonical
/// `gray` (the screen we composite the pill over) plus per-box recognized text.
fn detect_and_rec(
    engine: &PpocrEngine,
    rgba: &RgbaImage,
) -> (GrayImage, Vec<DetectedTextBox>, Vec<String>) {
    let (w, h) = (rgba.width(), rgba.height());
    let crop = Rect {
        left: 0,
        top: 0,
        right: w,
        bottom: h,
    };
    let frame = OrientedImage::build_with_rgb(rgba.as_raw(), w, h, 0, crop, DET_MAX_PIXELS)
        .expect("build frame");
    let rgb_det = frame.rgb_det.as_ref().expect("rgb_det populated");
    let det = engine
        .detect_only_image(rgb_det, PpocrProfile::Live)
        .expect("detection");
    let (sx, sy) = frame.det_to_full;
    let boxes: Vec<DetectedTextBox> = det.into_iter().map(|b| b.scaled_xy(sx, sy)).collect();
    let rgb = frame.rgb.as_ref().expect("rgb populated");
    let scripts = vec![PpocrScript::Latin; boxes.len()];
    let lines = engine
        .recognize_text_in_boxes_image(rgb, &frame.gray, &boxes, &scripts, PpocrProfile::Live, None)
        .expect("recognition");
    assert_eq!(lines.len(), boxes.len(), "rec output aligns with boxes");
    let texts = lines.into_iter().map(|l| l.text).collect();
    (frame.gray, boxes, texts)
}

/// Composite our opaque pill + 50%-alpha hole grid over a luma source, producing
/// the frame a MediaProjection capture hands us: pill colour everywhere in the
/// box, except the lattice hole pixels (a half blend of pill and screen).
fn composite_capture(source: &[u8], w: u32, h: u32, lat: &Lattice, pill: &PillRegion) -> Vec<u8> {
    let inside =
        |x: f32, y: f32| (x - pill.cx).abs() <= pill.half_w && (y - pill.cy).abs() <= pill.half_h;
    let mut cap = source.to_vec();
    for y in 0..h {
        for x in 0..w {
            if inside(x as f32 + 0.5, y as f32 + 0.5) {
                cap[(y * w + x) as usize] = PILL_LUMA;
            }
        }
    }
    for p in lat.points() {
        if !inside(p.x, p.y) {
            continue;
        }
        let (px, py) = ((p.x as u32).min(w - 1), (p.y as u32).min(h - 1));
        let s = source[(py * w + px) as usize] as u16;
        cap[(py * w + px) as usize] = ((PILL_LUMA as u16 + s + 1) / 2) as u8;
    }
    cap
}

fn paste_into_box(dst: &mut RgbaImage, src: &RgbaImage, b: &OrientedRect) {
    let bw = (b.width.round() as u32).max(1).min(
        dst.width()
            .saturating_sub((b.cx - b.width / 2.0).max(0.0) as u32),
    );
    let bh = (b.height.round() as u32).max(1).min(
        dst.height()
            .saturating_sub((b.cy - b.height / 2.0).max(0.0) as u32),
    );
    let left = (b.cx - b.width / 2.0).round().max(0.0) as u32;
    let top = (b.cy - b.height / 2.0).round().max(0.0) as u32;
    if bw == 0 || bh == 0 {
        return;
    }
    let resized = image::imageops::resize(src, bw, bh, FilterType::Triangle);
    for (dx, dy, px) in resized.enumerate_pixels() {
        dst.put_pixel(left + dx, top + dy, *px);
    }
}

const MIN_HOLES: usize = 4;

fn cfg() -> MonitorConfig {
    MonitorConfig {
        warmup_frames: 4,
        // Real-photo fixtures (book/sign) are mid-contrast, not black-on-white, so the
        // hard threshold is lower here than the screen default (110); the jitter below
        // is kept well under it.
        hard_threshold: 55,
        hard_frac: 0.15,
        scroll_frac: 0.7,
        scroll_min_boxes: 2,
    }
}

#[test]
fn subtitle_change_detected_background_motion_ignored() {
    let _ = env_logger::builder().is_test(true).try_init();
    let engine = load_engine();
    let base_rgba = fixture("book.jpg");
    let other_rgba = fixture("sign.jpg");
    let mut probe = make_probe();

    // Real detect + rec on the base frame.
    let (gray_a, boxes_a, texts_a) = detect_and_rec(&engine, &base_rgba);
    assert!(!boxes_a.is_empty(), "expected detections on the base frame");
    let (w, h) = (gray_a.width(), gray_a.height());
    let lat = Lattice::build(w, h, LATTICE_SPACING);

    // Monitor the densest recognized box, sampling its CONTOUR (tight to the text run)
    // so the hard-swing fraction isn't diluted by background margin.
    let (idx, holes) = boxes_a
        .iter()
        .enumerate()
        .filter(|(i, _)| !texts_a[*i].trim().is_empty())
        .map(|(i, b)| {
            let c = lat.holes_in_polygon(&b.contour);
            let h = if c.len() >= MIN_HOLES {
                c
            } else {
                lat.holes_in_rect(&b.tight_box)
            };
            (i, h)
        })
        .max_by_key(|(_, h)| h.len())
        .expect("a non-empty recognized box");
    let text_a = texts_a[idx].trim().to_string();
    let monitored_box = boxes_a[idx].tight_box.clone();
    let pill = PillRegion::from_oriented(&monitored_box);
    assert!(!text_a.is_empty(), "the monitored box recognized some text");
    assert!(
        holes.len() >= MIN_HOLES,
        "box lattice coverage: {}",
        holes.len()
    );
    eprintln!("monitored box text: {text_a:?} ({} holes)", holes.len());

    // Baseline: recover screen_est through the pill holes from the composited base.
    let src_a = gray_a.as_raw().clone();
    let (grid_cols, grid_rows) = (lat.cols(), lat.rows());
    let cap_base = composite_capture(&src_a, w, h, &lat, &pill);
    let baseline = probe.recover(&cap_base, w, h, &lat, PILL_LUMA, &[pill]);

    // Binarize bootstrap on the recovered baseline.
    let box_lumas: Vec<f32> = holes.iter().map(|&i| baseline[i][0] as f32).collect();
    let mean = box_lumas.iter().sum::<f32>() / box_lumas.len() as f32;
    let bootstrap: Vec<bool> = box_lumas
        .iter()
        .map(|&l| (l - mean).abs() > INK_CONTRAST)
        .collect();
    let ink = bootstrap.iter().filter(|b| **b).count();
    assert!(ink >= MIN_HOLES, "ink holes via recovery: {ink}");

    let mut mon = ScreenMonitor::new(lat, cfg());
    mon.set_box(1, holes.clone(), &baseline);

    // Warmup + "video between the strokes": jitter the off-ink hole pixels in the
    // source, composite the pill over it, recover through the holes. Stays Quiet.
    let hole_points: Vec<(usize, u32, u32)> = holes
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            let p = mon.lattice().points()[i];
            (k, (p.x as u32).min(w - 1), (p.y as u32).min(h - 1))
        })
        .collect();
    for f in 0..6u32 {
        let mut src = src_a.clone();
        for &(k, px, py) in &hole_points {
            if !bootstrap[k] {
                src[(py * w + px) as usize] = if (f + px) % 2 == 0 { 90 } else { 120 };
            }
        }
        let rec = probe.recover(
            &composite_capture(&src, w, h, mon.lattice(), &pill),
            w,
            h,
            mon.lattice(),
            PILL_LUMA,
            &[pill],
        );
        assert_eq!(
            mon.observe(&rec),
            FrameClassification::Quiet,
            "frame {f}: background jitter under the pill must not trip the box"
        );
    }

    // Replace the text under the box and re-run the real pipeline.
    let mut changed_rgba = base_rgba.clone();
    paste_into_box(&mut changed_rgba, &other_rgba, &monitored_box);
    let (gray_b, boxes_b, texts_b) = detect_and_rec(&engine, &changed_rgba);
    assert_eq!(
        (gray_b.width(), gray_b.height()),
        (w, h),
        "frame dims stable"
    );

    let cap_changed = composite_capture(gray_b.as_raw(), w, h, mon.lattice(), &pill);
    let rec_b = probe.recover(&cap_changed, w, h, mon.lattice(), PILL_LUMA, &[pill]);
    match mon.observe(&rec_b) {
        FrameClassification::BoxesChanged(ids) => assert!(ids.contains(&1), "{ids:?}"),
        other => panic!("expected the monitored box to change, got {other:?}"),
    }

    // One-shot artifact dump for inspection (set SCREEN_MONITOR_DUMP_DIR).
    if let Ok(dir) = std::env::var("SCREEN_MONITOR_DUMP_DIR") {
        std::fs::create_dir_all(&dir).expect("create dump dir");
        let save = |name: &str, buf: &[u8], iw: u32, ih: u32| {
            GrayImage::from_raw(iw, ih, buf.to_vec())
                .expect("gray buffer")
                .save(PathBuf::from(&dir).join(name))
                .expect("save png");
        };
        save("01_input.png", &src_a, w, h);
        save("02_pills_holes.png", &cap_base, w, h);
        save("03_pills_holes_new_text.png", &cap_changed, w, h);
        // What the shader reads back (recovered screen_est at the lattice). RGB → take
        // a channel for the gray dump (the synthetic fixtures are grayscale).
        let chan = |v: &[[u8; 3]]| v.iter().map(|p| p[0]).collect::<Vec<u8>>();
        save(
            "04_recovered_baseline.png",
            &chan(&baseline),
            grid_cols,
            grid_rows,
        );
        save(
            "05_recovered_changed.png",
            &chan(&rec_b),
            grid_cols,
            grid_rows,
        );
        eprintln!("dumped screen-monitor artifacts to {dir}");
    }

    // The recognizer confirms the text under the box is no longer what it was.
    let center = (monitored_box.cx, monitored_box.cy);
    let text_b = boxes_b
        .iter()
        .zip(&texts_b)
        .find(|(b, _)| {
            center.0 >= b.tight_box.cx - b.tight_box.width / 2.0
                && center.0 <= b.tight_box.cx + b.tight_box.width / 2.0
                && center.1 >= b.tight_box.cy - b.tight_box.height / 2.0
                && center.1 <= b.tight_box.cy + b.tight_box.height / 2.0
        })
        .map(|(_, t)| t.trim().to_string())
        .unwrap_or_default();
    eprintln!("text under box after change: {text_b:?}");
    assert_ne!(
        text_b, text_a,
        "recognizer should read different text under the box"
    );
}
